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
from starlette.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()

from ml.efficientnet_model import get_model 
from ml.preprocess import ChestXRayPreprocessor
from ml.llm_utils import get_llm_diagnosis_summary, describe_heatmap

# Import Grad-CAM specific libraries from the installed 'grad-cam' package
import cv2 # Still needed for show_cam_on_image and general image processing
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# --- Configuration ---
# Define paths for both models and the SimCLR backbone
MODEL_V1_BALANCED_PATH = os.path.join(os.path.dirname(__file__), 'ml', 'best_model_V1.pth')
MODEL_HIGH_RECALL_PATH = os.path.join(os.path.dirname(__file__), 'ml', 'best_model.pth') # This was your best recall model
SIMCLR_BACKBONE_PATH = os.path.join(os.path.dirname(__file__), 'outputs', 'simclr_backbone_best.pth') # Assuming SimCLR is in outputs

DOWNLOAD_PATH_V1 = os.path.join(os.path.dirname(__file__), 'ml', 'downloaded_model_V1.pth')
DOWNLOAD_PATH_HIGH_RECALL = os.path.join(os.path.dirname(__file__), 'ml', 'downloaded_model_high_recall.pth')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading models on device: {DEVICE}")
KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Pneumonia_Indicators_Relevance.csv')
# --- Global instances (will be initialized in startup_event) ---
model_v1_balanced = None
model_high_recall = None
image_preprocessor = None
grad_cam_instance = None # Grad-CAM will be for the balanced model
pneumonia_indicators_df = None

# --- FastAPI App ---
app = FastAPI(
    title="Pneumonia Detection API",
    description="API for detecting pneumonia from chest X-ray images using a multi-model pipeline (Balanced & High Recall) with Grad-CAM visualization and LLM diagnosis summary.",
    version="1.0.0",
)

# Configure CORS Middleware
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Pneumonia Detection API is running!"}

@app.get("/health")
async def health():
    return {"status": "healthy"}


# Pydantic Model for the /summarize_diagnosis request body
class PneumoniaDiagnosisRequest(BaseModel):
    model_v1_prediction: str = Field(..., description="Prediction from Model V1 (Balanced).")
    model_v1_probability: float = Field(..., ge=0.0, le=1.0, description="Probability from Model V1 (Balanced).")
    model_high_recall_prediction: str = Field(..., description="Prediction from High Recall Model.")
    model_high_recall_probability: float = Field(..., ge=0.0, le=1.0, description="Probability from High Recall Model.")
    findings: str | None = Field(None, description="Optional additional clinical findings provided by the user.")
    grad_cam_description: str | None = Field(None, description="Textual description of the Grad-CAM heatmap.") # NEW: Add Grad-CAM description

# Helper function to get the last convolutional layer for Grad-CAM
def get_efficientnet_last_conv_layer(model):
    """
    Dynamically finds the last convolutional layer in an EfficientNet model.
    For EfficientNet-B0, this is typically within `model.features[-1]`.
    """
    last_conv_layer = None
    if hasattr(model, 'features') and len(model.features) > 0:
        # Iterate through the modules of the last feature block in reverse
        for module in reversed(list(model.features[-1].modules())): # Use .modules() to go deeper
            if isinstance(module, torch.nn.Conv2d):
                last_conv_layer = module
                break # Found the last one, break
    
    if last_conv_layer:
        print(f"Identified target layer for Grad-CAM: {type(last_conv_layer).__name__} at {last_conv_layer}")
        return last_conv_layer
    else:
        # Fallback: iterate through all named modules if the above structure is not found
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv_layer = module # Keep updating to get the very last one
        if last_conv_layer:
            print(f"Fallback: Identified last Conv2d layer for Grad-CAM: {last_conv_layer}")
            return last_conv_layer

    raise ValueError("Could not find a suitable convolutional layer for Grad-CAM in the model.")

# start event to load models and preprocessor
@app.on_event("startup")
async def startup_event():
    global model_v1_balanced, model_high_recall, image_preprocessor, grad_cam_instance, pneumonia_indicators_df
    
    try:
        print("Starting model initialization...")
        
        # Initialize preprocessor first
        image_preprocessor = ChestXRayPreprocessor()
        print("Image preprocessor initialized.")
        
        # Load knowledge base if it exists
        if os.path.exists(KNOWLEDGE_BASE_PATH):
            pneumonia_indicators_df = pd.read_csv(KNOWLEDGE_BASE_PATH)
            print("Knowledge base loaded.")
        else:
            print("Warning: Knowledge base file not found. Continuing without it.")
            pneumonia_indicators_df = None
        
        # Check if models exist locally first
        if os.path.exists(MODEL_V1_BALANCED_PATH) and os.path.exists(MODEL_HIGH_RECALL_PATH):
            print("Loading models from local files...")
            
            # Load Model V1 (Balanced)
            model_v1_balanced = get_model()
            model_v1_balanced.load_state_dict(torch.load(MODEL_V1_BALANCED_PATH, map_location=DEVICE))
            model_v1_balanced.to(DEVICE)
            model_v1_balanced.eval()
            print("Model V1 (Balanced) loaded successfully.")
            
            # Load High Recall Model
            model_high_recall = get_model()
            model_high_recall.load_state_dict(torch.load(MODEL_HIGH_RECALL_PATH, map_location=DEVICE))
            model_high_recall.to(DEVICE)
            model_high_recall.eval()
            print("High Recall Model loaded successfully.")
            
        else:
            # Try to download models
            model_v1_url = os.environ.get('MODEL_V1_URL')
            model_high_recall_url = os.environ.get('MODEL_HIGH_RECALL_URL')
            
            if model_v1_url and model_high_recall_url:
                print("Downloading models from Azure Blob Storage...")
                
                async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout
                    # Download Model V1
                    response_v1 = await client.get(model_v1_url)
                    if response_v1.status_code == 200:
                        with open(DOWNLOAD_PATH_V1, 'wb') as f:
                            f.write(response_v1.content)
                        
                        model_v1_balanced = get_model()
                        model_v1_balanced.load_state_dict(torch.load(DOWNLOAD_PATH_V1, map_location=DEVICE))
                        model_v1_balanced.to(DEVICE)
                        model_v1_balanced.eval()
                        print("Model V1 downloaded and loaded successfully.")
                    else:
                        raise Exception(f"Failed to download Model V1. Status: {response_v1.status_code}")
                    
                    # Download High Recall Model
                    response_hr = await client.get(model_high_recall_url)
                    if response_hr.status_code == 200:
                        with open(DOWNLOAD_PATH_HIGH_RECALL, 'wb') as f:
                            f.write(response_hr.content)
                        
                        model_high_recall = get_model()
                        model_high_recall.load_state_dict(torch.load(DOWNLOAD_PATH_HIGH_RECALL, map_location=DEVICE))
                        model_high_recall.to(DEVICE)
                        model_high_recall.eval()
                        print("High Recall Model downloaded and loaded successfully.")
                    else:
                        raise Exception(f"Failed to download High Recall Model. Status: {response_hr.status_code}")
            else:
                raise Exception("Model files not found locally and MODEL_V1_URL/MODEL_HIGH_RECALL_URL not set")
        
        # Initialize Grad-CAM for Model V1
        if model_v1_balanced is not None:
            target_layer = get_efficientnet_last_conv_layer(model_v1_balanced)
            grad_cam_instance = GradCAM(model=model_v1_balanced, target_layers=[target_layer])
            print("Grad-CAM initialized successfully.")
        
        print("All models and components initialized successfully!")
        
    except Exception as e:
        print(f"Error during startup: {e}")
        # Don't raise the exception - let the app start but mark models as unavailable
        print("App will start but models may not be available.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Analyzes a chest X-ray image for pneumonia using a multi-model pipeline
    and returns predictions and probabilities from both models.
    """
    if model_v1_balanced is None or model_high_recall is None or image_preprocessor is None:
        raise HTTPException(status_code=503, detail="Models or preprocessor not loaded. Server is not ready.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        input_tensor = image_preprocessor.get_transforms()['val'](image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Predict with Model V1 (Balanced)
            logits_v1 = model_v1_balanced(input_tensor)
            prob_v1 = torch.sigmoid(logits_v1).item()
            pred_v1 = "Pneumonia" if prob_v1 >= 0.5 else "Normal"

            # Predict with High Recall Model
            logits_high_recall = model_high_recall(input_tensor)
            prob_high_recall = torch.sigmoid(logits_high_recall).item()
            pred_high_recall = "Pneumonia" if prob_high_recall >= 0.5 else "Normal"

        return JSONResponse(content={
            "filename": file.filename,
            "model_v1_prediction": pred_v1,
            "model_v1_probability": f"{prob_v1:.4f}",
            "model_high_recall_prediction": pred_high_recall,
            "model_high_recall_probability": f"{prob_high_recall:.4f}"
        })
    except Exception as e:
        print(f"Error in /predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_with_cam")
async def predict_with_cam(file: UploadFile = File(...)):
    """
    Analyzes a chest X-ray image for pneumonia, returns predictions from both models,
    a base64 encoded Grad-CAM heatmap image (from Model V1), and a textual description
    of the heatmap for LLM consumption.
    """
    if model_v1_balanced is None or model_high_recall is None or image_preprocessor is None or grad_cam_instance is None:
        raise HTTPException(status_code=503, detail="Models, preprocessor, or Grad-CAM not loaded. Server is not ready.")

    try:
        image_bytes = await file.read()
        image_pil_original = Image.open(io.BytesIO(image_bytes))

        if image_pil_original.mode != 'RGB':
            image_pil_original = image_pil_original.convert('RGB')

        # Preprocess the image for the model's input
        input_tensor = image_preprocessor.get_transforms()['val'](image_pil_original).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Predict with Model V1 (Balanced)
            logits_v1 = model_v1_balanced(input_tensor)
            prob_v1 = torch.sigmoid(logits_v1).item()
            pred_v1 = "Pneumonia" if prob_v1 >= 0.5 else "Normal"

            # Predict with High Recall Model
            logits_high_recall = model_high_recall(input_tensor)
            prob_high_recall = torch.sigmoid(logits_high_recall).item()
            pred_high_recall = "Pneumonia" if prob_high_recall >= 0.5 else "Normal"

        # Generate Grad-CAM for Model V1 (Balanced)
        # The input_tensor needs to have requires_grad=True for Grad-CAM
        # We create a new tensor for this to avoid modifying the original
        cam_input_tensor = image_preprocessor.get_transforms()['val'](image_pil_original).unsqueeze(0).to(DEVICE).requires_grad_(True)
        
        targets = [ClassifierOutputTarget(0)] # For a single output neuron binary classifier
        grayscale_cam = grad_cam_instance(input_tensor=cam_input_tensor, targets=targets)
        
        if grayscale_cam is None or grayscale_cam.size == 0:
            print("Warning: Grad-CAM returned empty or None heatmap. Cannot generate description.")
            grad_cam_description_text = "Grad-CAM heatmap could not be generated."
            img_str = None # No CAM image to return
        else:
            grayscale_cam = grayscale_cam[0, :] # Take the first (and only) image in the batch

            # Generate textual description of the heatmap
            grad_cam_description_text = describe_heatmap(grayscale_cam)

            # Resize original PIL image to 224x224 for CAM overlay
            image_pil_resized_for_cam = image_pil_original.resize((224, 224), Image.LANCZOS)
            original_image_np = np.array(image_pil_resized_for_cam).astype(np.float32) / 255.0

            cam_image = show_cam_on_image(original_image_np, grayscale_cam, use_rgb=True)

            if cam_image is None or cam_image.size == 0:
                print("Warning: show_cam_on_image returned empty or None image. Cannot return CAM image.")
                img_str = None
            else:
                cam_image_pil = Image.fromarray(cam_image) # show_cam_on_image already returns uint8 [0,255]
                buffered = io.BytesIO()
                cam_image_pil.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return JSONResponse(content={
            "filename": file.filename,
            "model_v1_prediction": pred_v1,
            "model_v1_probability": f"{prob_v1:.4f}",
            "model_high_recall_prediction": pred_high_recall,
            "model_high_recall_probability": f"{prob_high_recall:.4f}",
            "grad_cam_image_base64": img_str, # Will be None if CAM failed
            "grad_cam_image_format": "image/png" if img_str else None,
            "grad_cam_description": grad_cam_description_text # NEW: Textual CAM description
        })
    except Exception as e:
        print(f"Error in predict_with_cam: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction or Grad-CAM failed: {str(e)}")

@app.post("/summarize_diagnosis")
async def summarize_diagnosis(
    request: PneumoniaDiagnosisRequest
):
    """
    Uses an LLM to generate a concise, professional summary of the diagnosis for a medical doctor,
    based on multi-model predictions and optionally augmented with knowledge from pneumonia indicators.
    """
    try:
        summary = await get_llm_diagnosis_summary(
            model_v1_prediction=request.model_v1_prediction,
            model_v1_probability=request.model_v1_probability,
            model_high_recall_prediction=request.model_high_recall_prediction,
            model_high_recall_probability=request.model_high_recall_probability,
            findings=request.findings,
            pneumonia_indicators_df=pneumonia_indicators_df, # Pass the loaded knowledge base
            grad_cam_description=request.grad_cam_description # NEW: Pass Grad-CAM description to LLM
        )
        return JSONResponse(content={
            "diagnosis_summary": summary,
            "model_v1_prediction": request.model_v1_prediction,
            "model_v1_probability": f"{request.model_v1_probability:.2f}",
            "model_high_recall_prediction": request.model_high_recall_prediction,
            "model_high_recall_probability": f"{request.model_high_recall_probability:.2f}"
        })
    except Exception as e:
        print(f"Error in /summarize_diagnosis endpoint: {e}")
        if isinstance(e, httpx.HTTPStatusError) and e.response:
            print(f"Perplexity API Response Body (Error): {e.response.text}")
        raise HTTPException(status_code=500, detail=f"Failed to generate diagnosis summary: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
