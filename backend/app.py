# app.py
import os
import io
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from pathlib import Path
import uvicorn
from pydantic import BaseModel

# --- Configuration ---
# Adjust this path to where your 'best_model.pth' is saved
# Assuming 'app.py' is in 'pneumonia-detection-local/pneumonia-detection/test_ml/'
# and 'best_model.pth' is in 'pneumonia-detection-local/'
PROJECT_ROOT = Path(__file__).resolve().parents[2] # Go up 3 levels from app.py
MODEL_PATH = PROJECT_ROOT / "best_model.pth"

# Define the device for inference
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables to store model and transform
# These will be initialized once when the app starts
model = None
inference_transform = None

# --- FastAPI App Instance ---
app = FastAPI(
    title="Pneumonia Detection API",
    description="API for classifying chest X-ray images as Normal or Pneumonia.",
    version="1.0.0",
)

# --- Pydantic Models for Response ---
class PredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    pneumonia_probability: float
    confidence_score: float

# --- Model Loading and Preprocessing Functions (Copied from predict.py) ---

def load_inference_model(model_path, device):
    """
    Loads a pre-trained EfficientNet-B0 model and prepares it for inference.
    """
    print(f"Loading model from: {model_path} on device: {device}")
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 1)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model

def get_inference_transforms():
    """
    Returns the validation/inference transforms.
    """
    # These should match the val_transforms in your preprocess.py
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def perform_inference(image_bytes: bytes, model: torch.nn.Module, transform: transforms.Compose, device: torch.device):
    """
    Performs inference on a single image byte stream.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob > 0.5 else 0
        confidence_score = abs(prob - 0.5) * 2

    return pred, prob, confidence_score

# --- FastAPI Startup Event ---
@app.on_event("startup")
async def startup_event():
    """
    Load the model and transforms when the FastAPI application starts.
    This ensures the model is loaded only once, not on every request.
    """
    global model, inference_transform
    print("Application startup: Loading model and transforms...")
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Please ensure training is complete and the model is saved.")
    
    model = load_inference_model(str(MODEL_PATH), DEVICE)
    inference_transform = get_inference_transforms()
    print("Model and transforms loaded. API is ready.")

# --- API Endpoint ---
@app.post("/predict", response_model=PredictionResponse)
async def predict_image_endpoint(file: UploadFile = File(...)):
    """
    Receives an image file, performs pneumonia prediction, and returns the result.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        image_bytes = await file.read()
        predicted_label, pneumonia_prob, confidence = perform_inference(image_bytes, model, inference_transform, DEVICE)
        
        predicted_class_name = "Pneumonia" if predicted_label == 1 else "Normal"

        return PredictionResponse(
            filename=file.filename,
            predicted_class=predicted_class_name,
            pneumonia_probability=pneumonia_prob,
            confidence_score=confidence
        )
    except HTTPException as e:
        raise e # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# --- Run the application (for local testing) ---
if __name__ == "__main__":
    # To run this locally:
    # 1. Save this code as 'app.py' (e.g., in your test_ml folder)
    # 2. Make sure 'best_model.pth' is in your 'pneumonia-detection-local' directory
    # 3. Install FastAPI and Uvicorn: pip install fastapi uvicorn python-multipart
    # 4. Run from your terminal in the directory containing app.py: uvicorn app:app --reload
    #    (The --reload flag is useful for development, automatically restarts on code changes)
    # 5. Access the API documentation at http://127.0.0.1:8000/docs
    uvicorn.run(app, host="0.0.0.0", port=8000)

