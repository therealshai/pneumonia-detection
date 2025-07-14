#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
from data_loaders import create_stratified_loaders # Import the data loader function
import os

#%%
# Configuration - Corrected paths based on previous discussions
config = {
    # Corrected data_dir to point to the 'external-dataset' folder
    'data_dir': r"C:\Users\SyedShaistaTabassum\Capstone project\pneumonia-detection-local\external-dataset",
    # Corrected metadata_path to point directly to meta_data.csv within 'external-dataset'
    'metadata_path': r"C:\Users\SyedShaistaTabassum\Capstone project\pneumonia-detection-local\external-dataset\meta_data.csv",
    'batch_size': 32,
    'num_epochs': 7
}

#%%
def get_model(num_classes=1):
    """
    Initializes a pre-trained EfficientNet-B0 model and modifies its classifier
    head for binary classification.

    Args:
        num_classes (int): The number of output classes. Default is 1 for binary
                           classification (e.g., pneumonia vs. normal).
    Returns:
        torch.nn.Module: The configured EfficientNet-B0 model.
    """
    print("Initializing EfficientNet-B0 model...")
    # Load pre-trained weights
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Replace the classifier head for binary classification
    # The original classifier[1] is a Linear layer. We replace it with a Sequential
    # block containing Dropout and a new Linear layer for the desired num_classes.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3), # Dropout for regularization
        nn.Linear(in_features, num_classes) # Output layer for binary classification
    )
    print(f"Model loaded with modified classifier for {num_classes} output class(es).")
    return model

#%%
def train_epoch(model, loader, criterion, optimizer, device, epoch_num):
    """
    Performs one training epoch.

    Args:
        model (torch.nn.Module): The neural network model.
        loader (torch.utils.data.DataLoader): DataLoader for the training set.
        criterion (torch.nn.Module): Loss function (e.g., BCEWithLogitsLoss).
        optimizer (torch.optim.Optimizer): Optimizer (e.g., AdamW).
        device (torch.device): Device to run the training on ('cuda' or 'cpu').
        epoch_num (int): Current epoch number for logging.

    Returns:
        tuple: Average epoch loss and accuracy.
    """
    model.train() # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    print(f"\n--- Epoch {epoch_num} Training Started ---")
    # Use tqdm for a progress bar
    for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch_num} Training", leave=False)):
        # Handle dummy data from data_loaders.py if image loading failed
        valid_mask = labels != -1
        if not valid_mask.any(): # If no valid samples in batch, skip
            print(f"  Batch {batch_idx}: No valid samples found, skipping.")
            continue

        inputs = inputs[valid_mask].to(device)
        labels = labels[valid_mask].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        # Ensure labels are float and have the correct shape (N, 1) for BCEWithLogitsLoss
        loss = criterion(outputs, labels.float().unsqueeze(1))

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update statistics
        running_loss += loss.item() * inputs.size(0) # Accumulate batch loss
        
        # Convert logits to probabilities and then to binary predictions
        preds = (torch.sigmoid(outputs) > 0.5).long() # Apply sigmoid and threshold at 0.5
        
        correct_predictions += (preds.squeeze(1) == labels).sum().item()
        total_samples += labels.size(0)

        # Print batch-level progress every N batches (e.g., 100 batches)
        if (batch_idx + 1) % 100 == 0:
            current_loss = running_loss / total_samples
            current_acc = correct_predictions / total_samples
            print(f"  Batch {batch_idx+1}/{len(loader)} - Loss: {current_loss:.4f}, Acc: {current_acc:.4f}")

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"--- Epoch {epoch_num} Training Finished ---")
    print(f"  Total Samples Processed in Training: {total_samples}")
    return epoch_loss, epoch_acc

#%%
def validate_epoch(model, loader, criterion, device, epoch_num):
    """
    Performs one validation epoch.

    Args:
        model (torch.nn.Module): The neural network model.
        loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the validation on.
        epoch_num (int): Current epoch number for logging.

    Returns:
        tuple: Average epoch loss and accuracy.
    """
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    print(f"\n--- Epoch {epoch_num} Validation Started ---")
    # Disable gradient calculation for validation
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch_num} Validation", leave=False)):
            # Handle dummy data from data_loaders.py
            valid_mask = labels != -1
            if not valid_mask.any():
                print(f"  Validation Batch {batch_idx}: No valid samples found, skipping.")
                continue

            inputs = inputs[valid_mask].to(device)
            labels = labels[valid_mask].to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))

            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            
            correct_predictions += (preds.squeeze(1) == labels).sum().item()
            total_samples += labels.size(0)

            # Print batch-level progress every N batches (e.g., 50 batches for validation)
            if (batch_idx + 1) % 50 == 0:
                current_loss = running_loss / total_samples
                current_acc = correct_predictions / total_samples
                print(f"  Validation Batch {batch_idx+1}/{len(loader)} - Loss: {current_loss:.4f}, Acc: {current_acc:.4f}")

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"--- Epoch {epoch_num} Validation Finished ---")
    print(f"  Total Samples Processed in Validation: {total_samples}")
    return epoch_loss, epoch_acc

#%%
def evaluate_model(model, test_loader, device):
    """
    Evaluates the trained model on the test set and prints a classification report
    and confusion matrix.

    Args:
        model (torch.nn.Module): The trained neural network model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        device (torch.device): Device to run the evaluation on.
    """
    model.eval() # Set model to evaluation mode
    all_preds, all_labels = [], []

    print("\n--- Model Evaluation on Test Set Started ---")
    with torch.no_grad(): # Disable gradient calculation
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Testing Model")):
            # Handle dummy data from data_loaders.py
            valid_mask = labels != -1
            if not valid_mask.any():
                # print(f"  Test Batch {batch_idx}: No valid samples found, skipping.")
                continue # Skip batches with no valid samples

            inputs = inputs[valid_mask].to(device)
            labels = labels[valid_mask].to(device)

            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).long() # Get binary predictions

            all_preds.extend(preds.cpu().squeeze().tolist()) # Ensure it's a list of single values
            all_labels.extend(labels.cpu().squeeze().tolist()) # Ensure it's a list of single values
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx+1} test batches...")

    print("\n--- Evaluation Results ---")
    if not all_labels:
        print("No valid samples processed for evaluation. Cannot generate report.")
        return

    print("\nClassification Report:")
    # target_names should match the order of labels (0: Normal, 1: Pneumonia)
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Pneumonia'], zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # Calculate individual metrics for more detailed printing
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    test_recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)

    print(f"\nOverall Test Accuracy: {test_accuracy:.4f}")
    print(f"Pneumonia Precision:   {test_precision:.4f}")
    print(f"Pneumonia Recall:      {test_recall:.4f}")
    print(f"Pneumonia F1-Score:    {test_f1:.4f}")
    print("\n--- Model Evaluation Finished ---")


#%%
def main():
    """
    Main function to orchestrate the training and evaluation process.
    """
    start_time = time.time()
    print("Starting training script...")

    # Determine device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    print("\nCreating data loaders...")
    loaders = create_stratified_loaders(config)
    print("Data loaders created successfully.")

    # Get the model
    model = get_model().to(device)
    print(f"Model moved to device: {device}")

    # Define loss function and optimizer
    # Use class weights from the data loader for BCEWithLogitsLoss
    # The class_weights tensor from data_loaders.py is [weight_for_normal, weight_for_pneumonia]
    # For BCEWithLogitsLoss, pos_weight should be the weight for the positive class (Pneumonia)
    pos_weight = loaders['class_weights'][1] # Weight for Pneumonia class
    print(f"Using positive class weight (for Pneumonia): {pos_weight.item():.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # Scheduler with verbose=True for more logging
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True) 

    best_val_loss = float('inf')
    best_epoch = -1
    model_save_path = 'best_model.pth'
    early_stop_patience = 7   # Number of epochs to wait after LR stops decreasing
    lr_stagnant_epochs = 0
    last_lr = optimizer.param_groups[0]['lr']

    print(f"\n--- Training for {config['num_epochs']} Epochs ---")
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        # Train epoch
        train_loss, train_acc = train_epoch(model, loaders['train'], criterion, optimizer, device, epoch+1)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validate epoch
        val_loss, val_acc = validate_epoch(model, loaders['val'], criterion, device, epoch+1)
        print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Step the learning rate scheduler
        scheduler.step(val_loss)

        # Check if learning rate is stagnant for early stopping
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr == last_lr:
            lr_stagnant_epochs += 1
            print(f"Learning rate stagnant for {lr_stagnant_epochs} epoch(s). Current LR: {current_lr:.6f}")
        else:
            lr_stagnant_epochs = 0
            last_lr = current_lr
            print(f"Learning rate changed to: {current_lr:.6f}")


        if lr_stagnant_epochs >= early_stop_patience:
            print(f"Early stopping: Learning rate has not decreased for {early_stop_patience} epochs. Stopping training.")
            break

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with validation loss {best_val_loss:.4f} at epoch {best_epoch}")
        else:
            print(f"Validation loss did not improve. Current best: {best_val_loss:.4f} (Epoch {best_epoch})")

        print(f"Epoch {epoch+1} time: {time.time() - epoch_start:.2f} seconds")
        print("-" * 50) # Separator for epochs

    print("\n--- Training Complete ---")
    print(f"Loading best model from '{model_save_path}' (from Epoch {best_epoch}) for final evaluation...")
    # Ensure best_model.pth exists before loading
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
    else:
        print(f"Warning: Best model '{model_save_path}' not found. Using the model from the last epoch.")
    
    print("\n--- Starting Final Model Evaluation ---")
    evaluate_model(model, loaders['test'], device)
    
    total_training_duration = time.time() - start_time
    print(f"\nTotal script execution time: {total_training_duration:.2f} seconds")
    print("Script finished.")

#%%
if __name__ == "__main__":
    main()
# %%
