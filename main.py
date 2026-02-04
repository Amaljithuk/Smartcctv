import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import torch
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import detection_ops 

# Configuration
VIDEO_SOURCE = "test_video.mp4" 
SETTINGS = {
    'loitering_threshold': 10,
    'crowd_threshold': 60,     
    'confidence_threshold': 0.15, 
    'trespassing_zone': (200, 300, 300, 350), # x1, y1, x2, y2
    'trespassing_enabled': True,
    'loitering_enabled': True,
    'crowd_enabled': True
}

def main():
    use_cuda = torch.cuda.is_available()
    device = '0' if use_cuda else 'cpu'
    print(f"System Check: {'GPU Detected (CUDA)' if use_cuda else 'CPU Only'}")

    print("Initializing Models...")
    yolo_model = YOLO("yolov8s.pt") 
    
    deepsort_tracker = DeepSort(
        max_age=30,
        n_init=1, 
        nms_max_overlap=1.0, 
        embedder_gpu=use_cuda
    )
    print("Models Initialized.")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open source '{VIDEO_SOURCE}'.")
        return

    track_history = defaultdict(list)
    loitering_saved = defaultdict(lambda: False)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    frame_count = 0

    window_name = "Smart CCTV"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"Processing... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream.")
            break

        frame_count += 1
        current_time = frame_count / fps
        process_frame = frame 

        detections_list = []
        
        # YOLO Detection
        results = yolo_model(
            process_frame, 
            stream=True, 
            conf=SETTINGS['confidence_threshold'], 
            device=device,
            imgsz=640, 
            verbose=False
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
        
        # Deepsort Tracker Update
        outputs = deepsort_tracker.update_tracks(detections_list, frame=process_frame)

        final_frame, alerts = detection_ops.process_frame_annotations(
            process_frame, 
            outputs, 
            current_time, 
            track_history,
            loitering_saved,
            SETTINGS
        )

        # Console Output
        # if alerts['count'] > 0:
        #     console_msg = [f"Frame {frame_count}: Detected {alerts['count']} people."]
        #     if alerts['trespassing']: console_msg.append("Trespassing Alert")
        #     if alerts['crowd']: console_msg.append("Crowd Alert")
        #     if alerts['loitering']: console_msg.append("Loitering Alert")
        #     print("\n".join(console_msg))

        cv2.imshow(window_name, final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()