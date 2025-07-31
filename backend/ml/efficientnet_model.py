#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # Needed for Focal Loss
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
import os
import numpy as np
import sys # Import sys for explicit exit
import json # Import json to load class_weights for bias calculation
import pandas as pd # Import pandas for bias calculation

# Explicitly import DataLoader and Dataset from torch.utils.data
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Import your custom modules using absolute paths
from .data_loaders import create_stratified_loaders, PneumoniaOnlyDataset
from .preprocess import ChestXRayPreprocessor

# Removed the old NumPy compatibility warning block as it's no longer relevant with fixed versions.

#%%
# Configuration for Colab (or local testing)
data_root_for_colab = '/content/drive/My Drive/my_xray_project_data' # Example for Colab
pneumonia_data_path_colab = os.path.join(data_root_for_colab, 'pneumonia_only_for_simclr')

project_root_in_drive='/content/drive/Othercomputers/My Laptop/pneumonia-detection'

config = {
    'data_dir': os.path.join(data_root_for_colab, 'all_images_for_supervised_training'), # Path to supervised training images (used by dataset class)
    'metadata_path': os.path.join(project_root_in_drive, "actual_meta_data.csv"), # Updated to use actual_meta_data.csv
    'project_root_in_drive': project_root_in_drive, # New: Pass project root for data_loaders to find split files
    'batch_size': 32, # Batch size for supervised training (Phase 2)
    'num_epochs': 20, # Max epochs for supervised training (Phase 2)
    'model_save_dir': os.path.join(project_root_in_drive, 'ml'), # CRITICAL FIX: Save models directly in ml/ as per user's file structure
    'early_stop_patience': 10, # Number of epochs to wait for F1 improvement

    # NEW: SimCLR Phase 1 Configuration
    'simclr_batch_size': 256, # Ideal batch size for SimCLR
    'simclr_epochs': 150, # Number of epochs for SimCLR pre-training (was 200, reduced for time)
    'simclr_lr': 1e-3, # Learning rate for SimCLR
    'simclr_temperature': 0.1, # Temperature for NT-Xent loss
    'simclr_model_save_path': os.path.join(project_root_in_drive, 'ml', 'simclr_backbone_best.pth'), # CRITICAL FIX: Save SimCLR model directly in ml/
    'freeze_backbone_epochs': 5 # Number of epochs to freeze backbone in Phase 2 initial training
}

#%%
# NEW: Focal Loss Implementation (already present, ensuring correct usage)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction='mean', pos_weight=None): # Adjusted alpha to 0.75
        super(FocalLoss, self).__init__()
        self.alpha = alpha # Alpha parameter for balancing classes
        self.gamma = gamma # Gamma parameter for focusing on hard examples
        self.reduction = reduction
        self.pos_weight = pos_weight # This will be a tensor for BCEWithLogitsLoss style

    def forward(self, inputs, targets):
        # inputs are logits (raw outputs from the model before sigmoid)
        # targets are binary (0 or 1)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Calculate pt (probability of the true class)
        # For binary classification, pt is p for target=1 else 1-p
        # sigmoid(inputs) gives p
        pt = torch.exp(-BCE_loss) # Equivalent to pt = p if target=1 else 1-p

        # Apply alpha weighting
        if self.alpha is not None:
            # alpha_t is alpha for positive class (targets=1) and 1-alpha for negative class (targets=0)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            BCE_loss = alpha_t * BCE_loss

        # Apply gamma focusing term
        # This line must always execute to define focal_loss
        focal_loss = (1 - pt)**self.gamma * BCE_loss 

        # Apply pos_weight if provided (for BCEWithLogitsLoss style)
        if self.pos_weight is not None:
            # pos_weight is a tensor [weight_normal, weight_pneumonia]
            weight_for_samples = torch.ones_like(targets)
            weight_for_samples[targets == 1] = self.pos_weight[1] # Weight for Pneumonia
            weight_for_samples[targets == 0] = self.pos_weight[0] # Weight for Normal
            focal_loss = weight_for_samples * focal_loss # Apply this on the already calculated focal_loss


        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss

#%%
# NEW: SimCLR Model Definition
class SimCLRModel(nn.Module):
    def __init__(self, base_model, out_dim):
        super(SimCLRModel, self).__init__()
        self.encoder = base_model
        # Remove the original classifier head from EfficientNet-B0
        self.encoder.classifier = nn.Identity()

        # Projection head (g)
        # EfficientNet-B0's last convolutional layer output features before global pooling is 1280
        # The default classifier input features are 1280
        s_in_features = 1280 # EfficientNet-B0's last layer output features
        self.projection_head = nn.Sequential(
            nn.Linear(s_in_features, s_in_features), # First layer (e.g., 1280 -> 1280)
            nn.ReLU(),
            nn.Linear(s_in_features, out_dim) # Output dimension (e.g., 128)
        )

    def forward(self, x):
        # Encoder forward pass (features before pooling)
        features = self.encoder.features(x)
        # Global average pooling
        out = self.encoder.avgpool(features)
        # Flatten for the projection head
        out = torch.flatten(out, 1)
        # Projection head forward pass
        return self.projection_head(out)

# NEW: NT-Xent Loss for SimCLR
class NTXentLoss(nn.Module):
    def __init__(self, temperature, device):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        # Create a mask to remove self-similarities and duplicate pairs
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool).to(self.device)
        mask = mask.fill_diagonal_(0) # Remove self-similarity
        for i in range(batch_size):
            mask[i, batch_size + i] = 0 # Remove (i, i+batch_size)
            mask[batch_size + i, i] = 0 # Remove (i+batch_size, i)
        return mask

    def forward(self, z_i, z_j):
        """
        Calculates NT-Xent loss given two sets of projected features.
        z_i: (batch_size, feature_dim)
        z_j: (batch_size, feature_dim)
        """
        batch_size = z_i.size(0)
        # Concatenate both sets of features
        z = torch.cat((z_i, z_j), dim=0) # (2*batch_size, feature_dim)

        # Calculate cosine similarity matrix
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature # (2*batch_size, 2*batch_size)

        # Mask out self-similarities and duplicate pairs
        mask = self.mask_correlated_samples(batch_size)
        sim_i_j = torch.diag(sim, batch_size) # Similarities between z_i and z_j
        sim_j_i = torch.diag(sim, -batch_size) # Similarities between z_j and z_i

        # Positive pairs (target labels for CrossEntropyLoss)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2 * batch_size, 1) # (2*batch_size, 1)
        
        # All samples except self-similarities
        negative_samples = sim[mask].reshape(2 * batch_size, -1) # (2*batch_size, N-2)

        # Combine positive and negative samples for logits
        logits = torch.cat((positive_samples, negative_samples), dim=1) # (2*batch_size, N-1)

        # Labels for CrossEntropyLoss are 0 for the positive pair
        labels = torch.zeros(2 * batch_size).long().to(self.device)

        loss = self.criterion(logits, labels)
        return loss / (2 * batch_size) # Average loss over all samples

#%%
def get_model(num_classes=1, pretrain_path=None, freeze_backbone=False, initial_bias_val=None): # Added initial_bias_val
    """
    Initializes a pre-trained EfficientNet-B0 model and modifies its classifier
    head for binary classification.
    Optionally loads SimCLR pre-trained weights and freezes backbone.
    """
    print("Initializing EfficientNet-B0 model...")
    # Load pre-trained EfficientNet-B0 weights
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights) # Load the base model with its default classifier

    if pretrain_path and os.path.exists(pretrain_path):
        print(f"Loading SimCLR pre-trained backbone weights from: {pretrain_path}")
        simclr_state_dict = torch.load(pretrain_path, map_location='cpu')

        # Filter out projection head weights if they exist in the state_dict
        encoder_state_dict = {k.replace('encoder.', ''): v for k, v in simclr_state_dict.items() if k.startswith('encoder.')}
        
        # Load the filtered state_dict into the EfficientNet-B0 model's features
        # This assumes the SimCLR backbone is just the 'features' part of EfficientNet
        model.features.load_state_dict(encoder_state_dict, strict=False) 
        print("SimCLR pre-trained backbone weights loaded successfully into model.features.")
    else:
        print("No SimCLR pre-trained weights specified or found. Using ImageNet pre-trained weights.")

    # CRITICAL FIX: Modify the classifier head to match the structure of your saved models.
    # The default EfficientNet-B0 classifier is Sequential(Dropout(0), Linear(1))
    # The error implies your saved models have a Sequential *inside* the original Linear layer.
    # This means the original Linear layer (model.classifier[1]) was replaced by a new Sequential.
    in_features = model.classifier[1].in_features # Get input features from the original Linear layer

    final_linear_layer = nn.Linear(in_features, num_classes)

    # Initialize bias for imbalanced datasets
    if initial_bias_val is not None:
        print(f"Initializing final layer bias to {initial_bias_val:.4f} for imbalance handling.")
        with torch.no_grad():
            final_linear_layer.bias.fill_(initial_bias_val)

    # Replace the original Linear layer (model.classifier[1]) with our new Sequential
    model.classifier[1] = nn.Sequential( # This is the key change to match saved models
        nn.Dropout(0.5), # This becomes model.classifier[1][0]
        final_linear_layer # This becomes model.classifier[1][1]
    )

    if freeze_backbone:
        print("Freezing EfficientNet-B0 backbone layers.")
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the new classifier head
        for param in model.classifier[1].parameters(): # Unfreeze the parameters of the new Sequential block
            param.requires_grad = True
        # Unfreeze batch norm layers for better fine-tuning (common practice)
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = True
                module.requires_grad_(True)
        print("Only classifier head parameters and BatchNorm layers are trainable.")
    else:
        print("Backbone layers are not frozen (will be fine-tuned).")

    print(f"Model loaded with modified classifier for {num_classes} output class(es).")
    return model

def evaluate_model(model, dataloader, device, criterion, epoch_num=None):
    """
    Evaluates the model on a given dataloader, calculating loss and key metrics
    like accuracy, precision, recall, and F1-score for the Pneumonia class.
    """
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Disable gradient calculations
        for inputs, labels in tqdm(dataloader, desc=f"Evaluating {'Validation' if epoch_num is not None else 'Test'} (Epoch {epoch_num if epoch_num is not None else 'Final'})"):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1) # Ensure labels are float and have shape (batch_size, 1)

            outputs = model(inputs)
            loss = criterion(outputs, labels) # Use logits directly with BCEWithLogitsLoss or FocalLoss

            running_loss += loss.item() * inputs.size(0)

            # Convert logits to probabilities, then to binary predictions
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long() # Convert to long for sklearn metrics

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    # Flatten lists for sklearn metrics
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    # Use zero_division=0 to avoid warnings when no positive predictions are made
    precision_pneumonia = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    recall_pneumonia = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1_pneumonia = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)

    # Generate classification report and confusion matrix for detailed view
    class_names = ['Normal', 'Pneumonia']
    report = classification_report(all_labels, all_preds, target_names=class_names, labels=[0, 1], zero_division=0) # Added labels=[0, 1]
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        'loss': epoch_loss,
        'accuracy': accuracy,
        'precision_pneumonia': precision_pneumonia,
        'recall_pneumonia': recall_pneumonia,
        'f1_pneumonia': f1_pneumonia,
        'classification_report': report,
        'confusion_matrix': cm
    }
    return metrics

# NEW: Phase 1 Training Function (SimCLR)
def train_simclr(simclr_model, simclr_loader, criterion_simclr, optimizer_simclr, device, config):
    """
    Trains the SimCLR model for self-supervised pre-training.
    """
    simclr_epochs = config['simclr_epochs']
    simclr_model_save_path = config['simclr_model_save_path']

    os.makedirs(os.path.dirname(simclr_model_save_path), exist_ok=True)

    print("\n--- Starting SimCLR Pre-training (Phase 1) ---")
    best_loss = float('inf')

    # Cosine Annealing LR scheduler for SimCLR
    scheduler_simclr = CosineAnnealingLR(optimizer_simclr, T_max=simclr_epochs, eta_min=1e-7)

    for epoch in range(simclr_epochs):
        epoch_start = time.time()
        simclr_model.train()
        running_loss = 0.0

        for view1, view2 in tqdm(simclr_loader, desc=f"SimCLR Epoch {epoch+1}/{simclr_epochs}"):
            view1 = view1.to(device)
            view2 = view2.to(device)

            optimizer_simclr.zero_grad()

            # Get projected features
            z_i = simclr_model(view1)
            z_j = simclr_model(view2)

            loss = criterion_simclr(z_i, z_j)

            loss.backward()
            optimizer_simclr.step()

            running_loss += loss.item() * view1.size(0) # Multiply by batch size

        epoch_loss = running_loss / (len(simclr_loader.dataset) * 2) # Divide by total number of views

        scheduler_simclr.step() # Step the scheduler after each epoch

        print(f"\nSimCLR Epoch {epoch+1}/{simclr_epochs}")
        print(f"SimCLR Loss: {epoch_loss:.4f}")
        print(f"Current SimCLR LR: {optimizer_simclr.param_groups[0]['lr']:.6f}")

        # Save the best SimCLR model backbone based on loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # Save only the encoder (EfficientNet-B0 backbone) state_dict
            torch.save(simclr_model.encoder.state_dict(), simclr_model_save_path)
            print(f"Saved best SimCLR backbone model with loss {best_loss:.4f} at epoch {epoch+1}")

        print(f"SimCLR Epoch {epoch+1} time: {time.time() - epoch_start:.2f} seconds")
        print("-" * 50)

    print("\n--- SimCLR Pre-training Complete ---")
    print(f"Best SimCLR backbone saved to '{simclr_model_save_path}'")
    return simclr_model.encoder # Return the pre-trained encoder

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, config):
    """
    Trains the EfficientNet-B0 model for binary classification (Phase 2).
    Monitors Pneumonia F1-score for early stopping.
    Implements partial freezing.
    """
    num_epochs = config['num_epochs']
    model_save_dir = config['model_save_dir']
    early_stop_patience = config['early_stop_patience']
    freeze_backbone_epochs = config['freeze_backbone_epochs']

    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, 'best_model.pth')

    best_val_f1 = -1.0 # Initialize with a value that any F1 will improve upon
    f1_stagnant_epochs = 0
    best_epoch = 0

    print("\n--- Starting Supervised Fine-tuning (Phase 2) ---")
    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Handle partial freezing/unfreezing
        if epoch == 0:
            # Initially freeze the backbone as per strategy 'C'
            print(f"Epoch {epoch+1}: Freezing backbone for initial {freeze_backbone_epochs} epochs.")
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze the new classifier head
            for param in model.classifier[1].parameters(): # Unfreeze the parameters of the new Sequential block
                param.requires_grad = True
            # Unfreeze batch norm layers for better fine-tuning (common practice)
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = True
                    module.requires_grad_(True)
            print("Only classifier head parameters and BatchNorm layers are trainable.")
        elif epoch == freeze_backbone_epochs:
            # After initial frozen epochs, unfreeze the entire model
            print(f"Epoch {epoch+1}: Unfreezing entire model for fine-tuning.")
            for param in model.parameters():
                param.requires_grad = True
            # Re-initialize optimizer to include newly unfrozen parameters with a potentially lower LR
            optimizer = optim.Adam(model.parameters(), lr=optimizer.param_groups[0]['lr'] / 10) # Reduce LR for full fine-tuning
            print(f"Optimizer re-initialized with LR: {optimizer.param_groups[0]['lr']:.6f}")


        model.train() # Set model to training mode
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1) # Ensure labels are float and have shape (batch_size, 1)

            optimizer.zero_grad() # Zero the gradients

            outputs = model(inputs)
            loss = criterion(outputs, labels) # Calculate loss using logits

            loss.backward() # Backpropagation
            optimizer.step() # Update model parameters

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Evaluate on validation set
        val_metrics = evaluate_model(model, val_loader, device, criterion, epoch_num=epoch+1)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val Pneumonia Precision: {val_metrics['precision_pneumonia']:.4f}")
        print(f"Val Pneumonia Recall: {val_metrics['recall_pneumonia']:.4f}")
        print(f"Val Pneumonia F1-Score: {val_metrics['f1_pneumonia']:.4f}")
        print("\nValidation Classification Report:\n", val_metrics['classification_report'])
        print("Validation Confusion Matrix:\n", val_metrics['confusion_matrix'])


        # Adjust learning rate based on validation Pneumonia F1-score
        scheduler.step(val_metrics['f1_pneumonia']) # Scheduler steps on F1-score

        # Early stopping logic based on validation Pneumonia F1-score
        if val_metrics['f1_pneumonia'] > best_val_f1:
            best_val_f1 = val_metrics['f1_pneumonia']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with validation Pneumonia F1-Score {best_val_f1:.4f} at epoch {best_epoch}")
            f1_stagnant_epochs = 0 # Reset stagnation counter
        else:
            f1_stagnant_epochs += 1
            print(f"Validation Pneumonia F1-Score did not improve. Current best F1: {best_val_f1:.4f} (Epoch {best_epoch}). Stagnant for {f1_stagnant_epochs} epoch(s).")

        if f1_stagnant_epochs >= early_stop_patience:
            print(f"Early stopping: Validation Pneumonia F1-Score has not improved for {early_stop_patience} epochs. Stopping training.")
            break

        print(f"Epoch {epoch+1} time: {time.time() - epoch_start:.2f} seconds")
        print("-" * 50)

    print("\n--- Training Complete ---")
    print(f"Loading best model from '{model_save_path}' (from Epoch {best_epoch}) for final evaluation...")
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
    else:
        print(f"Warning: Best model '{model_save_path}' not found. Using the model from the last epoch.")

    # Final evaluation on the test set
    print("\n--- Final Evaluation on Test Set ---")
    # Re-create loaders to ensure fresh state, especially for test set
    # (create_stratified_loaders is robust and will re-read metadata)
    # Ensure config passed here also contains 'project_root_in_drive'
    test_loaders = create_stratified_loaders({
            'data_dir': config['data_dir'], # This is the image root for dataset classes
            'project_root_in_drive': config['project_root_in_drive'], # Pass project root for split files
            'batch_size': config['batch_size']
        })
    test_metrics = evaluate_model(model, test_loaders['test'], device, criterion_supervised) # Use criterion_supervised
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Pneumonia Precision: {test_metrics['precision_pneumonia']:.4f}")
    print(f"Test Pneumonia Recall: {test_metrics['recall_pneumonia']:.4f}")
    print(f"Test Pneumonia F1-Score: {test_metrics['f1_pneumonia']:.4f}")
    print("\nTest Classification Report:\n", test_metrics['classification_report'])
    print("Test Confusion Matrix:\n", test_metrics['confusion_matrix'])
    print("--- Final Evaluation Complete ---")

    # Corrected return value for final_best_epoch
    # If training was skipped, final_best_epoch would be -1.
    # If training completed, it would be the best epoch found.
    # For evaluation context, we can return the last epoch or the best_epoch if training happened.
    # Let's return the final_best_epoch found during training, or -1 if no training occurred.
    if 'final_best_epoch' not in locals(): # If training was skipped
        final_best_epoch = -1
    
    print("--- Final Evaluation Complete ---")

    return model, final_val_f1, final_best_epoch

#%%
if __name__ == "__main__":
    try:
        # Check for CUDA availability
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("CUDA is available. Using GPU.")
        else:
            device = torch.device("cpu")
            print("CUDA is NOT available. Using CPU. This will be very slow.")

        # Ensure outputs directory exists
        os.makedirs(config['model_save_dir'], exist_ok=True)
        print(f"Output directory ensured: {config['model_save_dir']}")

        # --- Phase 1: SimCLR Pre-training ---
        simclr_saved_model_path = config['simclr_model_save_path']
        pre_trained_encoder = None # Initialize to None

        if os.path.exists(simclr_saved_model_path):
            print(f"\nSkipping Phase 1: SimCLR pre-trained model already found at '{simclr_saved_model_path}'.")
            # Load the pre-trained encoder to pass to Phase 2
            # Initialize a dummy SimCLRModel to load the encoder state_dict
            dummy_base_encoder = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            simclr_model_for_loading = SimCLRModel(dummy_base_encoder, out_dim=128)
            
            simclr_state_dict = torch.load(simclr_saved_model_path, map_location='cpu')
            simclr_model_for_loading.encoder.load_state_dict(simclr_state_dict, strict=False)
            pre_trained_encoder = simclr_model_for_loading.encoder
            print("Loaded pre-trained SimCLR encoder for Phase 2.")
        else:
            print("\nStarting Phase 1: SimCLR Pre-training on Pneumonia-Only Data")
            print(f"Checking if pneumonia data path exists: {pneumonia_data_path_colab}")
            if not os.path.exists(pneumonia_data_path_colab):
                print(f"Error: Pneumonia data path '{pneumonia_data_path_colab}' not found. Please ensure this directory exists and contains pneumonia images.")
                sys.exit(1) # Exit if data path is missing

            preprocessor = ChestXRayPreprocessor()
            simclr_transforms = preprocessor.get_transforms()['simclr']
            print("SimCLR transforms initialized.")

            pneumonia_dataset = PneumoniaOnlyDataset(pneumonia_data_path_colab, transform=simclr_transforms)
            print(f"PneumoniaOnlyDataset created with {len(pneumonia_dataset)} images.")

            simclr_loader = DataLoader(
                pneumonia_dataset,
                batch_size=config['simclr_batch_size'],
                shuffle=True,
                drop_last=True, # Drop last batch if smaller to avoid issues with NTXentLoss
                num_workers=os.cpu_count() // 2,
                pin_memory=True
            )
            print(f"SimCLR DataLoader created with batch size {config['simclr_batch_size']}.")

            # Initialize EfficientNet-B0 as the encoder for SimCLR
            # Load default ImageNet weights first
            base_encoder = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            simclr_model = SimCLRModel(base_encoder, out_dim=128).to(device) # out_dim is typically 128 or 256
            print("SimCLRModel initialized.")

            optimizer_simclr = optim.Adam(simclr_model.parameters(), lr=config['simclr_lr'])
            criterion_simclr = NTXentLoss(temperature=config['simclr_temperature'], device=device)
            print("SimCLR optimizer and criterion initialized.")

            # Train SimCLR model
            pre_trained_encoder = train_simclr(
                simclr_model, simclr_loader, criterion_simclr, optimizer_simclr, device, config
            )
            print("Phase 1 (SimCLR) Complete. Pre-trained encoder saved.")

        # --- Phase 2: Supervised Fine-tuning or Evaluation ---
        # CRITICAL FIX: Supervised model path also points to ml/ directory
        supervised_saved_model_path = os.path.join(config['model_save_dir'], 'best_model.pth')
        
        # Calculate initial bias for the final layer based on training set imbalance
        # This helps the model start with a more reasonable output probability for the minority class
        # We need the class counts from the training split
        train_df_path = os.path.join(config['project_root_in_drive'], 'train_split.csv')
        initial_bias = 0.0 # Default value
        if os.path.exists(train_df_path):
            train_df = pd.read_csv(train_df_path)
            pos_count = (train_df['Finding Labels'] == 'Pneumonia').sum()
            neg_count = (train_df['Finding Labels'] == 'Normal').sum()
            if pos_count > 0 and neg_count > 0:
                initial_bias = np.log(pos_count / neg_count)
                print(f"Calculated initial bias for final layer: {initial_bias:.4f}")
            else:
                print("Warning: Cannot calculate initial bias due to missing class in training data. Setting to 0.")
        else:
            print("Warning: train_split.csv not found for initial bias calculation. Setting bias to 0.")

        # Initialize model for Phase 2 - this will be used whether training or evaluating
        # Pass the pre_trained_encoder (from Phase 1 or loaded) AND the initial_bias_val
        model = get_model(num_classes=1, pretrain_path=config['simclr_model_save_path'], freeze_backbone=True, initial_bias_val=initial_bias)
        model.to(device)
        print("Phase 2 model initialized with pre-trained weights (or fresh for training).")

        # Define loss function and optimizer for Phase 2 (needed for evaluation too)
        # Create data loaders for supervised training
        # Pass the project_root_in_drive to create_stratified_loaders
        loaders = create_stratified_loaders({
                'data_dir': config['data_dir'], # This is the image root for dataset classes
                'project_root_in_drive': config['project_root_in_drive'], # Pass project root for split files
                'batch_size': config['batch_size']
            })
        train_loader = loaders['train']
        val_loader = loaders['val']
        class_weights = loaders['class_weights'].to(device)
        print("Supervised DataLoaders and class weights created.")

        # Pass the class_weights (which are balanced for Normal and Pneumonia) to FocalLoss
        criterion_supervised = FocalLoss(alpha=0.75, gamma=2, pos_weight=class_weights) # Alpha adjusted to 0.75
        print(f"Using Focal Loss with alpha=0.75 and class weights for Phase 2: {class_weights.cpu().numpy()}")

        # Optimizer and scheduler for Phase 2 (initialized even if skipping training)
        optimizer_supervised = optim.Adam(model.parameters(), lr=1e-4)
        scheduler_supervised = ReduceLROnPlateau(
            optimizer_supervised, mode='max', factor=0.1, patience=config['early_stop_patience'], verbose=True, min_lr=1e-7
        )
        print("Phase 2 optimizer and scheduler initialized.")


        if os.path.exists(supervised_saved_model_path):
            print(f"\nSkipping Phase 2 Training: Supervised model already found at '{supervised_saved_model_path}'.")
            model.load_state_dict(torch.load(supervised_saved_model_path, map_location=device))
            print("Loaded pre-trained supervised model for evaluation.")
            
            # Since we skipped training, we don't have final_val_f1 or final_best_epoch from train_model
            # For simplicity, we'll just proceed to final evaluation.
            final_val_f1 = -1.0 # Placeholder
            final_best_epoch = -1 # Placeholder

        else:
            print("\nStarting Phase 2: Supervised Fine-tuning on Full Dataset")
            # Train the model for Phase 2
            trained_model, final_val_f1, final_best_epoch = train_model(
                model, train_loader, val_loader, criterion_supervised, optimizer_supervised, scheduler_supervised, device, config
            )
            print(f"\nPhase 2 (Supervised Fine-tuning) finished. Best Validation Pneumonia F1-Score: {final_val_f1:.4f} at Epoch {final_best_epoch}")

        # --- Final Evaluation on Test Set (Common to both paths) ---\
        print("\n--- Final Evaluation on Test Set ---")
        # Re-create loaders to ensure fresh state, especially for test set
        # (create_stratified_loaders is robust and will re-read metadata)
        # Ensure config passed here also contains 'project_root_in_drive'
        test_loaders = create_stratified_loaders({
            'data_dir': config['data_dir'], # This is the image root for dataset classes
            'project_root_in_drive': config['project_root_in_drive'], # Pass project root for split files
            'batch_size': config['batch_size']
        })
        test_metrics = evaluate_model(model, test_loaders['test'], device, criterion_supervised) # Use criterion_supervised
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Pneumonia Precision: {test_metrics['precision_pneumonia']:.4f}")
        print(f"Test Pneumonia Recall: {test_metrics['recall_pneumonia']:.4f}")
        print(f"Test Pneumonia F1-Score: {test_metrics['f1_pneumonia']:.4f}")
        print("\nTest Classification Report:\n", test_metrics['classification_report'])
        print("Test Confusion Matrix:\n", test_metrics['confusion_matrix'])
        print("--- Final Evaluation Complete ---")

    except Exception as e:
        print(f"\nAn unexpected error occurred during model training: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        sys.exit(1) # Exit with an error code
