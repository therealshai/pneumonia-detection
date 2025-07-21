#%% 
import os
import pathlib # Used for robust path handling
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 
from .preprocess import ChestXRayPreprocessor 

#%% Model loading and prediction functions
def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads a pre-trained EfficientNet-B0 model and prepares it for inference.

    Args:
        model_path (str): Path to the saved model state dictionary (e.g., 'best_model.pth').
        device (torch.device): The device to load the model onto ('cuda' or 'cpu').

    Returns:
        torch.nn.Module: The loaded model in evaluation mode.
    """
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    import torch.nn as nn

    print(f"Loading model from: {model_path} on device: {device}")
    
    # Initialize EfficientNet-B0 with default pre-trained weights
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Modify the classifier head to match the training setup (binary classification)
    # This must be identical to how the model was defined in efficientnet_model.py
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3), 
        nn.Linear(in_features, 1) # Output a single logit for binary classification
    )

    # Load the trained state dictionary onto the specified device
    # map_location ensures compatibility whether trained on GPU/CPU and loaded on GPU/CPU
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device) 
    model.eval() # Set model to evaluation mode (disables dropout, batch norm updates)
    print("Model loaded successfully and set to evaluation mode.")
    return model

def predict_image(image_path: str, model: torch.nn.Module, transform: transforms.Compose, device: torch.device) -> tuple[int, float, float]:
    """
    Predicts the class and probability for a single image.

    Args:
        image_path (str): Path to the input image file.
        model (torch.nn.Module): The loaded model.
        transform (callable): The transformation to apply to the image (e.g., validation transforms).
        device (torch.device): The device to perform inference on.

    Returns:
        tuple: (predicted_label (int), pneumonia_probability (float), confidence_score (float))
               - predicted_label: 1 for Pneumonia, 0 for Normal.
               - pneumonia_probability: Probability of the image being Pneumonia (0 to 1).
               - confidence_score: A score from 0 to 1 indicating confidence (1 is most confident).
                                   Calculated as |probability - 0.5| * 2.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}. Returning dummy data.")
        return -1, 0.0, 0.0 

    # Apply transformations and add a batch dimension (unsqueeze(0))
    # Images are processed in batches by PyTorch models, even for a single image
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad(): # Disable gradient calculations for faster inference and less memory usage
        output = model(input_tensor)
        # Apply sigmoid to convert logits to probabilities 
        prob = torch.sigmoid(output).item() # .item() gets the scalar value from a 0-dim tensor
        
        # Determine the binary prediction based on a 0.5 threshold
        pred = 1 if prob > 0.5 else 0

        # Calculate confidence score: how far is the probability from the 0.5 threshold?
        # A probability of 0.5 yields 0 confidence, 0 or 1 yields 1 confidence.
        confidence_score = abs(prob - 0.5) * 2

    return pred, prob, confidence_score

def predict_images_batch(image_paths: list[str], model: torch.nn.Module, transform: transforms.Compose, device: torch.device) -> list[dict]:
    """
    Predicts the class and probability for a list of images, processing them individually.
    For true batch processing, images would be stacked into a single tensor.

    Args:
        image_paths (list[str]): A list of paths to input image files.
        model (torch.nn.Module): The loaded model.
        transform (callable): The transformation to apply to the images.
        device (torch.device): The device to perform inference on.

    Returns:
        list[dict]: A list of dictionaries, each containing 'image', 'prediction',
                    'probability', and 'confidence_score' for each image.
    """
    results = []
    print(f"\n--- Starting prediction for {len(image_paths)} images ---")
    
    for i, path in enumerate(tqdm(image_paths, desc="Predicting images")):
        pred, prob, confidence = predict_image(path, model, transform, device)
        results.append({
            'image': path,
            'prediction': pred,
            'probability': prob,
            'confidence_score': confidence
        })
        
        if (i + 1) % 10 == 0: 
            predicted_label_text = "Pneumonia" if pred == 1 else "Normal"
            print(f"  Processed {i+1} images. Last prediction for {os.path.basename(path)}: {predicted_label_text} (Prob: {prob:.4f}, Conf: {confidence:.4f})")
    print("--- Prediction finished ---")
    return results

#%% Example usage
if __name__ == "__main__":
    print("Starting prediction script...")
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    current_script_path = pathlib.Path(__file__).resolve()
    
    project_root = current_script_path.parents[2] 
    print(f"Project root determined as: {project_root}")

    model_path = current_script_path.parent / "best_model.pth" 
    print(f"Model path: {model_path}")  


    sample_image_name = "image.png" 
    sample_image_path = current_script_path.parent / sample_image_name
    print(f"Sample image path: {sample_image_path}")
    
    save_path = project_root / "reports" / "figures" / "transforms_visualization.png"
    print(f"Visualization save path: {save_path}")

    # Initialize Preprocessor and get transforms
    preprocessor = ChestXRayPreprocessor()
    transforms_dict = preprocessor.get_transforms()
    # Use validation transforms for inference as they don't include augmentations
    inference_transform = transforms_dict['val']

    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please ensure training has completed and 'best_model.pth' exists in the '{current_script_path.parent.name}' folder.")
    model = load_model(str(model_path), device) # Convert Path object to string for torch.load


    print("\n--- Predicting on a single sample image ---")
    if not sample_image_path.exists():
        print(f"Error: Sample image '{sample_image_path}' not found. Skipping single prediction.")
    else:
        pred, prob, confidence = predict_image(str(sample_image_path), model, inference_transform, device)
        
        
        if pred == -1: 
            print(f"Could not process sample image: {sample_image_path}. Check previous error messages.")
        else:
            label = "Pneumonia" if pred == 1 else "Normal"
            print(f"Prediction for {os.path.basename(sample_image_path)}:")
            print(f"  Predicted Class: {label}")
            print(f"  Probability of Pneumonia: {prob:.4f}")
            print(f"  Confidence Score: {confidence:.4f} (0=Least Confident, 1=Most Confident)")

    # --- Visualize transforms (optional, for debugging/understanding data processing) ---
    if sample_image_path.exists():
        print(f"\nVisualizing transforms for: {sample_image_path}")
        # Ensure the save directory exists for visualization
        os.makedirs(save_path.parent, exist_ok=True) 
        preprocessor.visualize_transforms(
            image_path=str(sample_image_path),
            save_path=str(save_path)
        )
        print(f"Visualization saved to: {save_path}")
    else:
        print(f"Cannot visualize transforms: Sample image {sample_image_path} not found.")

    print("\nPrediction script finished.")
# %%
