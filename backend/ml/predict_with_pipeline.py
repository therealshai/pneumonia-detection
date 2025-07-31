import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
import sys
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# --- Self-contained Helper Functions (from your ml/ directory) ---

# From ml/preprocess.py
class ChestXRayPreprocessor:
    def __init__(self):
        self.val_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.simclr_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_transforms(self):
        return {
            'test': self.val_transforms,
            'simclr': self.simclr_transforms
        }

# From ml/efficientnet_model.py (Modified for loading saved models)
def get_model(num_classes=1, pretrain_path=None):
    """
    Initializes a pre-trained EfficientNet-B0 model and modifies its classifier
    head for binary classification.
    Optionally loads SimCLR pre-trained weights.
    """
    # print("Initializing EfficientNet-B0 model...") # Suppress for cleaner output
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    if pretrain_path and os.path.exists(pretrain_path):
        # print(f"Loading SimCLR pre-trained backbone weights from: {pretrain_path}") # Suppress
        simclr_state_dict = torch.load(pretrain_path, map_location='cpu')
        
        encoder_state_dict = {k.replace('encoder.', ''): v for k, v in simclr_state_dict.items() if k.startswith('encoder.')}
        
        model.features.load_state_dict(encoder_state_dict, strict=False) 
        # print("SimCLR pre-trained backbone weights loaded successfully into model.features.") # Suppress
    # else:
        # print("No SimCLR pre-trained weights specified or found. Using ImageNet pre-trained weights.") # Suppress

    # Reconstruct the classifier to match the expected saved structure.
    # This addresses the "classifier.1.1.weight" issue.
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        model.classifier[0], # Original Dropout layer
        nn.Sequential(       # This Sequential wrapper creates the 'classifier.1.1' path
            nn.Dropout(0.5), # This becomes classifier.1.0 (if you saved with this)
            nn.Linear(in_features, num_classes) # This becomes classifier.1.1
        )
    )
    # print(f"Model loaded with modified classifier for {num_classes} output class(es).") # Suppress
    return model

# --- Main Prediction Pipeline ---

def predict_pneumonia_pipeline(image_path, model_v1_path, model_high_recall_path, simclr_pretrain_path, device):
    """
    Runs an image through a pipeline of two models and provides their predictions.

    Args:
        image_path (str): Path to the input X-ray image.
        model_v1_path (str): Path to the best_model_V1.pth (balanced/higher precision).
        model_high_recall_path (str): Path to the best_model.pth (higher recall).
        simclr_pretrain_path (str): Path to the SimCLR pre-trained backbone.
        device (torch.device): 'cuda' or 'cpu'.

    Returns:
        dict: A dictionary containing predictions and confidence scores from both models.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    if not os.path.exists(model_v1_path):
        raise FileNotFoundError(f"Model V1 not found at: {model_v1_path}")
    if not os.path.exists(model_high_recall_path):
        raise FileNotFoundError(f"High Recall Model not found at: {model_high_recall_path}")
    if not os.path.exists(simclr_pretrain_path):
        print(f"Warning: SimCLR pre-trained backbone not found at: {simclr_pretrain_path}. Models will use ImageNet weights.")

    # Load models
    model_v1 = get_model(num_classes=1, pretrain_path=simclr_pretrain_path).to(device)
    model_v1.load_state_dict(torch.load(model_v1_path, map_location=device), strict=False)
    model_v1.eval()

    model_high_recall = get_model(num_classes=1, pretrain_path=simclr_pretrain_path).to(device)
    model_high_recall.load_state_dict(torch.load(model_high_recall_path, map_location=device), strict=False)
    model_high_recall.eval()

    # Preprocess image
    preprocessor = ChestXRayPreprocessor()
    transform = preprocessor.get_transforms()['test']
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension

    results = {}
    with torch.no_grad():
        # Predict with Model V1 (Balanced)
        output_v1 = model_v1(input_tensor)
        prob_v1 = torch.sigmoid(output_v1).item()
        pred_v1 = 1 if prob_v1 > 0.5 else 0
        results['model_V1'] = {
            'prediction': 'Pneumonia' if pred_v1 == 1 else 'Normal',
            'confidence': prob_v1
        }

        # Predict with High Recall Model
        output_high_recall = model_high_recall(input_tensor)
        prob_high_recall = torch.sigmoid(output_high_recall).item()
        pred_high_recall = 1 if prob_high_recall > 0.5 else 0
        results['model_HighRecall'] = {
            'prediction': 'Pneumonia' if pred_high_recall == 1 else 'Normal',
            'confidence': prob_high_recall
        }
    
    return results

def llm_decision_logic(model_v1_results, model_high_recall_results):
    """
    Placeholder for LLM decision logic.
    In a real scenario, you'd send these results to an LLM API.
    """
    print("\n--- LLM Decision Logic (Placeholder) ---")
    print(f"Model V1 (Balanced) Prediction: {model_v1_results['prediction']} (Confidence: {model_v1_results['confidence']:.4f})")
    print(f"Model High Recall Prediction: {model_high_recall_results['prediction']} (Confidence: {model_high_recall_results['confidence']:.4f})")

    # Example simple rule-based decision (replace with actual LLM call later)
    if model_v1_results['prediction'] == 'Pneumonia' and model_v1_results['confidence'] > 0.7:
        decision = "HIGH CONFIDENCE: Pneumonia (Model V1 strong positive)"
        reasoning = "The balanced model (V1) confidently detected pneumonia."
    elif model_high_recall_results['prediction'] == 'Pneumonia' and model_high_recall_results['confidence'] > 0.6 and model_v1_results['prediction'] == 'Normal':
        decision = "POSSIBLE PNEUMONIA: Further Review Recommended"
        reasoning = "The balanced model (V1) indicated Normal, but the high-recall model detected possible pneumonia. Consider additional diagnostics."
    elif model_v1_results['prediction'] == 'Normal' and model_high_recall_results['prediction'] == 'Normal':
        decision = "NORMAL: No Pneumonia Detected"
        reasoning = "Both models indicated normal findings."
    else:
        decision = "UNCERTAIN: Ambiguous Results, Further Review Recommended"
        reasoning = "Model predictions are conflicting or not highly confident. Clinical correlation advised."
    
    print(f"LLM's Simulated Decision: {decision}")
    print(f"Reasoning: {reasoning}")
    print("---------------------------------------")
    # In a real application, you'd call the Gemini API here:
    # prompt = f"Given Model A predicted {model_v1_results['prediction']} with {model_v1_results['confidence']:.2f} confidence and Model B predicted {model_high_recall_results['prediction']} with {model_high_recall_results['confidence']:.2f} confidence, provide a medical assessment for pneumonia detection. Model A is balanced, Model B is highly sensitive."
    # response = await call_gemini_api(prompt)
    # return response

# --- Main Execution ---

if __name__ == "__main__":
    # --- Configuration ---
    project_root_in_drive = '/content/drive/Othercomputers/My Laptop/pneumonia-detection'
    data_root_for_colab = '/content/drive/My Drive/my_xray_project_data' # Your image data root

    # Model paths
    model_v1_path = os.path.join(project_root_in_drive, 'ml', 'best_model_V1.pth')
    model_high_recall_path = os.path.join(project_root_in_drive, 'ml', 'best_model.pth') # This was your highest recall model
    simclr_pretrain_path = os.path.join(project_root_in_drive, 'outputs', 'simclr_backbone_best.pth') # SimCLR backbone path

    # Example image path (replace with an actual image path from your dataset)
    # You might need to pick one from your test_split.csv for a real test
    # For example, let's try to get a pneumonia image from the test set if possible
    example_image_path = None
    try:
        test_df_path = os.path.join(project_root_in_drive, 'test_split.csv')
        if os.path.exists(test_df_path):
            test_df = pd.read_csv(test_df_path)
            # Try to find a pneumonia image first
            pneumonia_images = test_df[test_df['Finding Labels'] == 'Pneumonia']['Image Index'].tolist()
            if pneumonia_images:
                example_image_path = "/content/drive/Othercomputers/My Laptop/pneumonia-detection/ml/p-sample.jpg"
                print(f"Using example pneumonia image: {example_image_path}")
            else:
                # If no pneumonia images, pick a normal one
                normal_images = test_df[test_df['Finding Labels'] == 'Normal']['Image Index'].tolist()
                if normal_images:
                    example_image_path = os.path.join(data_root_for_colab, 'all_images_for_supervised_training', normal_images[0])
                    print(f"Using example normal image: {example_image_path}")
                else:
                    print("No images found in test_split.csv for example prediction.")
        else:
            print(f"Warning: test_split.csv not found at {test_df_path}. Cannot auto-select example image.")
            # Fallback to a dummy path if test_split.csv is not available
            example_image_path = '/content/drive/My Drive/my_xray_project_data/all_images_for_supervised_training/00000001_000.png' # Replace with a known image path if you have one
            print(f"Using fallback example image path: {example_image_path}")

    except Exception as e:
        print(f"Error auto-selecting example image: {e}. Please manually set 'example_image_path'.")
        # Fallback to a dummy path if any error occurs during auto-selection
        example_image_path = '/content/drive/My Drive/my_xray_project_data/all_images_for_supervised_training/00000001_000.png' # Replace with a known image path if you have one
        print(f"Using fallback example image path: {example_image_path}")


    # --- End Configuration ---

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if not example_image_path or not os.path.exists(example_image_path):
            print("Error: No valid example image path set or image not found. Please provide a valid image path.")
            sys.exit(1)

        print(f"\nRunning pipeline for image: {example_image_path}")
        pipeline_results = predict_pneumonia_pipeline(
            example_image_path, model_v1_path, model_high_recall_path, simclr_pretrain_path, device
        )

        print("\n--- Pipeline Results ---")
        print(f"Model V1 (Balanced) Prediction: {pipeline_results['model_V1']['prediction']} (Confidence: {pipeline_results['model_V1']['confidence']:.4f})")
        print(f"Model High Recall Prediction: {pipeline_results['model_HighRecall']['prediction']} (Confidence: {pipeline_results['model_HighRecall']['confidence']:.4f})")
        print("------------------------")

        # Call the LLM decision logic (simulated for now)
        llm_decision_logic(pipeline_results['model_V1'], pipeline_results['model_HighRecall'])

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your model or image paths.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

