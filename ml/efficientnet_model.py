#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
from .data_loaders import create_stratified_loaders 
import os

#%%
# Configuration for Azure ML (these paths will be overridden by mounted data)
# For local testing, ensure these paths point to your local data setup
# In Azure ML, 'data_input' will be the name of the mounted input dataset
data_root_for_azure_ml = './data_input' # This is the path where Azure ML will mount your dataset

config = {
    'data_dir': data_root_for_azure_ml,
    'metadata_path': os.path.join(data_root_for_azure_ml, "meta_data.csv"),
    'batch_size': 32,
    'num_epochs': 50, # Increased epochs for better generalization
    'model_save_dir': './outputs' # Azure ML automatically uploads contents of this folder
}

#%%
def get_model(num_classes=1):
    """
    Initializes a pre-trained EfficientNet-B0 model and modifies its classifier
    head for binary classification.
    """
    print("Initializing EfficientNet-B0 model...")
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    print(f"Model loaded with modified classifier for {num_classes} output class(es).")
    return model

#%%
def train_epoch(model, loader, criterion, optimizer, device, epoch_num):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    print(f"\n--- Epoch {epoch_num} Training Started ---")
    for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch_num} Training", leave=False)):
        valid_mask = labels != -1
        if not valid_mask.any():
            # print(f"  Batch {batch_idx}: No valid samples found, skipping.") # Suppress for cleaner logs
            continue

        inputs = inputs[valid_mask].to(device)
        labels = labels[valid_mask].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float().unsqueeze(1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).long()
        
        correct_predictions += (preds.squeeze(1) == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"--- Epoch {epoch_num} Training Finished ---")
    print(f"  Total Samples Processed in Training: {total_samples}")
    return epoch_loss, epoch_acc

#%%
def validate_epoch(model, loader, criterion, device, epoch_num):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    print(f"\n--- Epoch {epoch_num} Validation Started ---")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch_num} Validation", leave=False)):
            valid_mask = labels != -1
            if not valid_mask.any():
                # print(f"  Validation Batch {batch_idx}: No valid samples found, skipping.") # Suppress for cleaner logs
                continue

            inputs = inputs[valid_mask].to(device)
            labels = labels[valid_mask].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))

            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            
            correct_predictions += (preds.squeeze(1) == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"--- Epoch {epoch_num} Validation Finished ---")
    print(f"  Total Samples Processed in Validation: {total_samples}")
    return epoch_loss, epoch_acc

#%%
def evaluate_model(model, test_loader, device, run=None): # Added run parameter for Azure ML logging
    model.eval()
    all_preds, all_labels = [], []

    print("\n--- Model Evaluation on Test Set Started ---")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Testing Model")):
            valid_mask = labels != -1
            if not valid_mask.any():
                continue

            inputs = inputs[valid_mask].to(device)
            labels = labels[valid_mask].to(device)

            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).long()

            all_preds.extend(preds.cpu().squeeze().tolist())
            all_labels.extend(labels.cpu().squeeze().tolist())
            
    print("\n--- Evaluation Results ---")
    if not all_labels:
        print("No valid samples processed for evaluation. Cannot generate report.")
        return

    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=['Normal', 'Pneumonia'], zero_division=0)
    print(report)
    print("\nConfusion Matrix:")
    matrix = confusion_matrix(all_labels, all_preds)
    print(matrix)

    test_accuracy = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    test_recall = recall_recall(all_labels, all_preds, pos_label=1, zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)

    print(f"\nOverall Test Accuracy: {test_accuracy:.4f}")
    print(f"Pneumonia Precision:   {test_precision:.4f}")
    print(f"Pneumonia Recall:      {test_recall:.4f}")
    print(f"Pneumonia F1-Score:    {test_f1:.4f}")
    print("\n--- Model Evaluation Finished ---")

    # Log metrics to Azure ML (if run context is provided)
    if run:
        run.log("test_accuracy", test_accuracy)
        run.log("test_precision", test_precision)
        run.log("test_recall", test_recall)
        run.log("test_f1_score", test_f1)
        run.log("classification_report", report)
        run.log("confusion_matrix", str(matrix.tolist())) # Log as string/list for easier viewing

#%%
def main():
    """
    Main function to orchestrate the training and evaluation process.
    This function will be called by the Azure ML entry script.
    """
    start_time = time.time()
    print("Starting training script...")

    # Get Azure ML run context if running in Azure ML
    run = None
    try:
        from azureml.core import Run
        run = Run.get_context()
        print("Running in Azure ML context.")
    except Exception:
        print("Not running in Azure ML context.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nCreating data loaders...")
    loaders = create_stratified_loaders(config)
    print("Data loaders created successfully.")

    model = get_model().to(device)
    print(f"Model moved to device: {device}")

    pos_weight = loaders['class_weights'][1]
    print(f"Using positive class weight (for Pneumonia): {pos_weight.item():.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True) 

    best_val_loss = float('inf')
    best_epoch = -1
    
    # Model will be saved to the 'outputs' directory, which Azure ML automatically uploads
    os.makedirs(config['model_save_dir'], exist_ok=True)
    model_save_path = os.path.join(config['model_save_dir'], 'best_model.pth')

    early_stop_patience = 7
    lr_stagnant_epochs = 0
    last_lr = optimizer.param_groups[0]['lr']

    print(f"\n--- Training for {config['num_epochs']} Epochs ---")
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        train_loss, train_acc = train_epoch(model, loaders['train'], criterion, optimizer, device, epoch+1)
        if run: # Log training metrics
            run.log(f"train_loss_epoch_{epoch+1}", train_loss)
            run.log(f"train_acc_epoch_{epoch+1}", train_acc)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_loss, val_acc = validate_epoch(model, loaders['val'], criterion, device, epoch+1)
        if run: # Log validation metrics
            run.log(f"val_loss_epoch_{epoch+1}", val_loss)
            run.log(f"val_acc_epoch_{epoch+1}", val_acc)
        print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with validation loss {best_val_loss:.4f} at epoch {best_epoch}")
        else:
            print(f"Validation loss did not improve. Current best: {best_val_loss:.4f} (Epoch {best_epoch})")

        print(f"Epoch {epoch+1} time: {time.time() - epoch_start:.2f} seconds")
        print("-" * 50)

    print("\n--- Training Complete ---")
    print(f"Loading best model from '{model_save_path}' (from Epoch {best_epoch}) for final evaluation...")
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
    else:
        print(f"Warning: Best model '{model_save_path}' not found. Using the model from the last epoch.")
    
    print("\n--- Starting Final Model Evaluation ---")
    evaluate_model(model, loaders['test'], device, run) # Pass run context for logging
    
    total_training_duration = time.time() - start_time
    print(f"\nTotal script execution time: {total_training_duration:.2f} seconds")
    print("Script finished.")

#%%
if __name__ == "__main__":
    main()
