from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import io
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from backend import database
import os

router = APIRouter()

# Initialize Models for API usage (CPU is fine for single uploads)
# Using 'cpu' to avoid VRAM conflicts if main loop is heavy, unless we want to share the model
# For now having a separate small instance for API is safer for concurrency than sharing the global state one without locks
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

@router.get("/")
def read_root():
    return {"message": "SmartCCTV API is running"}

@router.post("/trusted")
async def create_trusted_face(name: str = Form(...), file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Save image for display
        import uuid
        filename = f"{uuid.uuid4().hex}.jpg"
        save_path = os.path.join("trusted_faces", filename)
        image.save(save_path)
        
        # Detect face
        # MTCNN returns cropped face tensor if return_prob=False (default)
        # But we want the cropped face directly to feed into resnet
        # mtcnn(img) returns a tensor of shape (3, h, w) if a face is found, else None
        
        face_tensor = mtcnn(image)
        
        if face_tensor is None:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        # Calculate embedding
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0) # Add batch dimension -> (1, 3, h, w)
            
        with torch.no_grad():
            embedding = resnet(face_tensor.to(device)).detach().cpu().numpy()[0].tolist()
            
        # Save to DB
        face_id = database.add_trusted_face(name, embedding, save_path)
        
        return {"id": face_id, "name": name, "message": "Trusted face added successfully", "image_path": save_path}
        
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trusted")
def list_trusted_faces():
    return database.get_trusted_faces()

@router.delete("/trusted/{face_id}")
def delete_trusted_face_endpoint(face_id: int):
    database.delete_trusted_face(face_id)
    return {"message": "Deleted successfully"}

@router.get("/untrusted")
def list_untrusted_faces():
    faces = database.get_untrusted_faces()
    # Add full URL for image path if needed, or let frontend handle relative path
    # Assuming frontend is on same host or can access /captured_faces
    # Fix: remove directory prefix since frontend appends it
    for face in faces:
         face['image_path'] = os.path.basename(face['image_path'])
    return faces

