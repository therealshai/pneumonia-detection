import os
import io
import base64
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from starlette.middleware.cors import CORSMiddleware
import uvicorn

# --- Import modules as part of the 'ml' package ---
from ml.efficientnet_model import get_model # We will use this get_model directly
from ml.preprocess import ChestXRayPreprocessor 
from ml.grad_cam_utils import GradCAM, show_cam_on_image, get_efficientnet_target_layer, ClassifierOutputTarget

# --- Configuration ---
MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml', 'best_model_V0.pth')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on device: {DEVICE}")

# --- Global instances (will be initialized in startup_event) ---
# We will load the model directly from get_model, not wrap it in PneumoniaDetectionModel
model_backbone = None # This will hold the efficientnet_b0 model
image_preprocessor = None
grad_cam_instance = None

# --- FastAPI App ---
app = FastAPI(
    title="Pneumonia Detection API",
    description="API for detecting pneumonia from chest X-ray images using EfficientNet-B0 with Grad-CAM visualization.",
    version="1.0.0",
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Replace with your frontend's actual origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """
    Loads the model, preprocessor, and Grad-CAM when the FastAPI app starts.
    """
    global model_backbone, image_preprocessor, grad_cam_instance

    print(f"Loading model on device: {DEVICE}")
    try:
        # 1. Load the model architecture using get_model from efficientnet_model.py
        # This returns the efficientnet_b0 model with the modified classifier.
        model_backbone = get_model(num_classes=1) 
        
        # 2. Load the trained weights into the model_backbone directly
        print(f"Attempting to load model weights from: {os.path.abspath(MODEL_WEIGHTS_PATH)}")
        if not os.path.exists(MODEL_WEIGHTS_PATH):
            raise FileNotFoundError(f"Model weights not found at: {MODEL_WEIGHTS_PATH}")
        
        # Load state dict directly into the efficientnet_b0 model
        model_backbone.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
        model_backbone.to(DEVICE)
        model_backbone.eval() # Set model to evaluation mode
        print(f"Model '{MODEL_WEIGHTS_PATH}' loaded successfully.")

        # 3. Initialize the preprocessor
        image_preprocessor = ChestXRayPreprocessor()
        print("Image preprocessor initialized.")

        # 4. Initialize Grad-CAM
        # Pass the model_backbone (the efficientnet_b0 instance) to get_efficientnet_target_layer
        target_layer_for_cam = get_efficientnet_target_layer(model_backbone) 
        
        if not target_layer_for_cam:
            raise RuntimeError("Could not find a suitable target layer for Grad-CAM.")
        
        # GradCAM expects target_layers as a list
        if not isinstance(target_layer_for_cam, list):
            target_layer_for_cam = [target_layer_for_cam]

        # Pass the model_backbone (the efficientnet_b0 instance) to GradCAM
        grad_cam_instance = GradCAM(model=model_backbone, target_layers=target_layer_for_cam, use_cuda=DEVICE.type == 'cuda')
        print("Grad-CAM instance initialized.")

    except Exception as e:
        print(f"Error during application startup: {e}")
        raise # Re-raise the exception to prevent the FastAPI app from starting if loading fails

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Analyzes a chest X-ray image for pneumonia and returns a prediction and confidence.
    """
    if model_backbone is None or image_preprocessor is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded. Server is not ready.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        input_tensor = image_preprocessor.get_transforms()['val'](image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Get logits from the model, then apply sigmoid to get probability
            logits = model_backbone(input_tensor)
            prediction_prob = torch.sigmoid(logits).item()
            
        label = "Pneumonia" if prediction_prob >= 0.5 else "Normal"
        confidence = f"{prediction_prob * 100:.2f}"

        return JSONResponse(content={
            "filename": file.filename,
            "prediction": label,
            "confidence": confidence
        })
    except Exception as e:
        print(f"Error in /predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_with_cam")
async def predict_with_cam(file: UploadFile = File(...), target_class: int = None):
    """
    Analyzes a chest X-ray image for pneumonia, returns a prediction, confidence,
    and a base64 encoded Grad-CAM heatmap image.
    """
    if model_backbone is None or image_preprocessor is None or grad_cam_instance is None:
        raise HTTPException(status_code=503, detail="Model, preprocessor, or Grad-CAM not loaded. Server is not ready.")

    try:
        image_bytes = await file.read()
        image_pil_original = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        input_tensor = image_preprocessor.get_transforms()['val'](image_pil_original).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Get logits from the model, then apply sigmoid for probability
            logits = model_backbone(input_tensor)
            prediction_prob = torch.sigmoid(logits).item()
            
        label = "Pneumonia" if prediction_prob >= 0.5 else "Normal"
        confidence = f"{prediction_prob * 100:.2f}"

        # Determine the target for Grad-CAM
        # For a single-output model, the target category is always 0 (the index of the output neuron).
        # The ClassifierOutputTarget helps GradCAM understand which output to backpropagate from.
        targets = [ClassifierOutputTarget(0)] 

        # Generate Grad-CAM heatmap
        # grayscale_cam will be (1, H, W) from our custom GradCAM
        grayscale_cam = grad_cam_instance(input_tensor=input_tensor, targets=targets)
        
        # Ensure grayscale_cam is 2D (H, W) for show_cam_on_image.
        grayscale_cam = grayscale_cam[0, :] # Take the first (and only) image in the batch

        # Convert original PIL image to numpy array for show_cam_on_image
        original_image_np = np.array(image_pil_original)
        cam_image = show_cam_on_image(original_image_np, grayscale_cam, use_rgb=True)

        # Convert the CAM image (NumPy array) to PIL Image and then to base64
        cam_image_pil = Image.fromarray(cam_image)
        buffered = io.BytesIO()
        cam_image_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return JSONResponse(content={
            "filename": file.filename,
            "prediction": label,
            "confidence": confidence,
            "grad_cam_image_base64": img_str,
            "grad_cam_image_format": "image/png"
        })
    except Exception as e:
        print(f"Error in predict_with_cam: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction or Grad-CAM failed: {str(e)}")

# This block is for running the app directly with `python main.py`
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)
