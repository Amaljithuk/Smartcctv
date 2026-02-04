import cv2
import torch
import numpy as np
import shutil
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import sys

# Ensure we can import detection_ops from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import detection_ops
from backend.alert_utils import AlertManager
from backend import api

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startups ensure directories exist
if not os.path.exists("captured_faces"):
    os.makedirs("captured_faces")
if not os.path.exists("trusted_faces"):
    os.makedirs("trusted_faces")

app.mount("/captured_faces", StaticFiles(directory="captured_faces"), name="captured_faces")
app.mount("/trusted_faces", StaticFiles(directory="trusted_faces"), name="trusted_faces")

app.include_router(api.router)

# Global State
class VideoState:
    def __init__(self):
        self.video_source = "test_video.mp4" # Default
        self.using_webcam = False
        self.cap = None
        self.reload_cap = True
        
        # Models
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        print("Loading YOLO...")
        self.yolo_model = YOLO("yolov8s.pt")
        print("Loading DeepSort...")
        self.deepsort_tracker = DeepSort(
            max_age=30, n_init=1, nms_max_overlap=1.0, embedder_gpu=self.use_cuda
        )
        
        # Face Recognition
        try:
            from facenet_pytorch import MTCNN, InceptionResnetV1
            from backend import database
            print("Loading Face Recognition Models...")
            self.mtcnn = MTCNN(keep_all=True, device=self.device)
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            
            database.init_db()
            self.known_faces = database.get_trusted_faces()
            print(f"Loaded {len(self.known_faces)} trusted faces.")
        except Exception as e:
            print(f"Face Recognition Init Error: {e}")
            self.mtcnn = None
            self.resnet = None
            self.known_faces = []

        # State trackers
        self.track_history = defaultdict(list)
        self.loitering_saved = defaultdict(lambda: False)
        self.saved_untrusted_session = set()
        self.frame_count = 0
        
        # Settings
        self.settings = {
            'loitering_threshold': 10,
            'crowd_threshold': 60,     
            'confidence_threshold': 0.15, 
            'trespassing_zone': [200, 300, 300, 350], # List for JSON compatibility
            'trespassing_enabled': True,
            'loitering_enabled': True,
            'crowd_enabled': True
        }
        
        # Alert Manager
        self.alert_manager = AlertManager()

state = VideoState()

def get_video_capture():
    if state.reload_cap or state.cap is None or not state.cap.isOpened():
        if state.cap:
            state.cap.release()
            
        source = 0 if state.using_webcam else state.video_source
        if not state.using_webcam and not os.path.exists(str(source)):
             print(f"File not found: {source}")
             return None
             
        state.cap = cv2.VideoCapture(source)
        state.reload_cap = False
        state.frame_count = 0
        state.track_history.clear()
        state.loitering_saved.clear()
        state.saved_untrusted_session.clear()
        # Reset tracker on new source otherwise it gets confused
        state.deepsort_tracker.delete_all_tracks()
        
        # Refresh trusted faces on restart/reload
        if state.mtcnn:
            from backend import database
            state.known_faces = database.get_trusted_faces()
        
    return state.cap

def generate_frames():
    while True:
        cap = get_video_capture()
        if not cap:
            break
            
        ret, frame = cap.read()
        if not ret:
            # If it's a file, loop it
            if not state.using_webcam:
                state.frame_count = 0
                state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
                
        state.frame_count += 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        current_time = state.frame_count / fps
        
        # Run Detection
        detections_list = []
        results = state.yolo_model(
            frame, stream=True, conf=state.settings['confidence_threshold'], 
            device=state.device if state.device == 'cpu' else 0, imgsz=640, verbose=False
        )
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls_id = int(box.cls[0])
                if cls_id == 0: # Person
                    w = x2 - x1
                    h = y2 - y1
                    detections_list.append([[x1, y1, w, h], conf, 0])
        
        # Tracker
        tracks = state.deepsort_tracker.update_tracks(detections_list, frame=frame)
        
        # Annotate
        # Convert list to tuple for detection_ops if needed, though list is usually fine
        # Ensure zone is tuple
        current_settings = state.settings.copy()
        current_settings['trespassing_zone'] = tuple(state.settings['trespassing_zone'])
        
        final_frame, alerts, state.saved_untrusted_session = detection_ops.process_frame_annotations(
            frame, tracks, current_time, 
            state.track_history, state.loitering_saved, current_settings,
            mtcnn=state.mtcnn,
            resnet=state.resnet,
            known_faces=state.known_faces,
            device=state.device,
            saved_untrusted_session=state.saved_untrusted_session
        )
        
        # Process Alerts
        state.alert_manager.process_alerts(alerts)
        
        # Encoding
        ret, buffer = cv2.imencode('.jpg', final_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/settings")
def get_settings():
    return state.settings

@app.post("/settings")
def update_settings(new_settings: dict):
    state.settings.update(new_settings)
    return {"status": "updated", "settings": state.settings}

@app.post("/set_source")
def set_source(source_type: str = Form(...)): # 'webcam' or 'file'
    if source_type == 'webcam':
        state.using_webcam = True
    else:
        state.using_webcam = False # Will use last uploaded file
    state.reload_cap = True
    return {"status": "source_changed", "type": source_type}

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    file_location = f"uploaded_{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    state.video_source = file_location
    state.using_webcam = False
    state.reload_cap = True
    return {"status": "file_uploaded", "filename": file_location}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
