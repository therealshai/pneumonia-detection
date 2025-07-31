# ml/evaluate_only.py
import torch
import os
import sys

# Add the parent directory (pneumonia-detection) to the Python path
# This allows relative imports like 'ml.efficientnet_model' to work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary functions from your existing modules
from ml.efficientnet_model import get_model, evaluate_model, config # Import config too
from ml.data_loaders import create_stratified_loaders
# No explicit import for ChestXRayPreprocessor needed here as it's handled by data_loaders

def main():
    """
    Loads a pre-trained model and evaluates it on the test set.
    This version is simplified for direct evaluation.
    """
    # --- Configuration for Evaluation ---
    # Reusing the 'config' dictionary from efficientnet_model.py
    # Ensure data_dir points to your 50GB 'external-dataset' in Google Drive
    config['data_dir'] = '/content/drive/My Drive/external-dataset'
    config['metadata_path'] = os.path.join(config['data_dir'], "meta_data.csv")
    
    # Path to the model weights you want to evaluate
    # Confirmed: 'best_model.pth' exists in the 'outputs' folder
    MODEL_TO_EVALUATE_PATH = os.path.join(os.path.dirname(__file__), 'outputs', 'best_model.pth')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    # 1. Create data loaders (necessary to get the test_loader)
    print("Creating data loaders...")
    loaders = create_stratified_loaders(config)
    test_loader = loaders['test']
    print("Test data loader ready.")

    # 2. Load the model architecture
    model = get_model(num_classes=1)
    
    # 3. Load the saved model state dictionary
    print(f"Loading model weights from: {os.path.abspath(MODEL_TO_EVALUATE_PATH)}")
    if not os.path.exists(MODEL_TO_EVALUATE_PATH):
        raise FileNotFoundError(f"Model weights not found at: {MODEL_TO_EVALUATE_PATH}. Please ensure the model is trained and saved.")
    
    # Load state dict directly into the model, mapping to the correct device
    model.load_state_dict(torch.load(MODEL_TO_EVALUATE_PATH, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode
    print("Model loaded and ready for evaluation.")

    # 4. Perform evaluation
    evaluate_model(model, test_loader, device)

    print("\nEvaluation script finished.")

if __name__ == "__main__":
    main()
