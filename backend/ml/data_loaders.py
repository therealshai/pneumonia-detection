# ml/data_loaders.py
#%%
import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split # Still needed for local testing __main__ block
from sklearn.utils.class_weight import compute_class_weight # Still needed for local testing __main__ block
from .preprocess import ChestXRayPreprocessor
from pathlib import Path
import time
import json # New import for loading class weights

#%%
class BalancedChestDataset(Dataset):
    """
    Custom Dataset for Chest X-Ray images, designed to work with a DataFrame
    where the 'Path' column contains the full, absolute path to each image.
    """
    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image metadata for this split.
                               Must contain 'Image Index', 'Finding Labels', and 'Path' columns.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.transform = transform
        self.df = df.copy()

        # Map 'Pneumonia' to 1 and 'Normal' to 0 for numerical labels
        self.df['label'] = self.df['Finding Labels'].apply(lambda x: 1 if x == 'Pneumonia' else 0)

        # Filter out rows where the file does not exist (important for robustness)
        initial_count = len(self.df)
        self.df = self.df[self.df['Path'].apply(lambda p: Path(p).exists())]
        if len(self.df) < initial_count:
            print(f"Warning: Removed {initial_count - len(self.df)} non-existent image paths from dataset during BalancedChestDataset init.")

        self.image_paths = self.df['Path'].tolist()
        self.labels = self.df['label'].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

#%%
class PneumoniaOnlyDataset(Dataset):
    """
    Custom Dataset to load only pneumonia images for SimCLR pre-training.
    It returns two augmented views of the same image.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root directory containing ONLY pneumonia images.
                            e.g., '/content/drive/MyDrive/my_xray_project_data/pneumonia_only_for_simclr'
            transform (callable, optional): SimCLR-specific transform to be applied.
                                            This transform should return two augmented views.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []

        for ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
            self.image_paths.extend(list(self.root_dir.rglob(f'*.{ext}')))

        if not self.image_paths:
            raise ValueError(f"No image files found in the specified pneumonia-only directory: {root_dir}")
        print(f"Found {len(self.image_paths)} pneumonia images for SimCLR pre-training.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            view1 = self.transform(image)
            view2 = self.transform(image)
            return view1, view2
        else:
            return image, image

#%%
def create_stratified_loaders(config):
    """
    Creates stratified train, validation, and test data loaders by loading
    pre-saved split CSVs and class weights.

    Args:
        config (dict): Configuration dictionary containing:
            'data_dir' (str): Root directory of the dataset (e.g., '/content/drive/My Drive/my_xray_project_data/all_images_for_supervised_training').
                              This is used by the dataset classes to construct full paths.
            'project_root_in_drive' (str): Base path for loading split CSVs and class weights.
            'batch_size' (int): Batch size for data loaders.

    Returns:
        dict: A dictionary containing 'train', 'val', 'test' DataLoaders
              and 'class_weights' for the loss function.
    """
    start_time = time.time()
    batch_size = config['batch_size']
    project_root_in_drive = config['project_root_in_drive'] # New config parameter

    # Define paths for pre-saved splits and weights
    train_csv_path = Path(project_root_in_drive) / 'train_split.csv'
    val_csv_path = Path(project_root_in_drive) / 'val_split.csv'
    test_csv_path = Path(project_root_in_drive) / 'test_split.csv'
    class_weights_json_path = Path(project_root_in_drive) / 'class_weights.json'

    # Check if all necessary pre-saved files exist
    if not (train_csv_path.exists() and val_csv_path.exists() and
            test_csv_path.exists() and class_weights_json_path.exists()):
        raise FileNotFoundError(
            f"Pre-saved data split files or class weights not found. "
            f"Please ensure you have run the 'Pre-generate and Save Data Splits' Colab cell successfully.\n"
            f"Missing files: {[f for f in [train_csv_path, val_csv_path, test_csv_path, class_weights_json_path] if not f.exists()]}"
        )

    try:
        # Load pre-saved DataFrames
        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)
        test_df = pd.read_csv(test_csv_path)

        # Load pre-saved class weights
        with open(class_weights_json_path, 'r') as f:
            class_weights_list = json.load(f)
        class_weights = torch.tensor(class_weights_list, dtype=torch.float)

        preprocessor = ChestXRayPreprocessor()
        train_transforms = preprocessor.get_transforms()['train']
        val_transforms = preprocessor.get_transforms()['val']

        train_dataset = BalancedChestDataset(train_df, transform=train_transforms)
        val_dataset = BalancedChestDataset(val_df, transform=val_transforms)
        test_dataset = BalancedChestDataset(test_df, transform=val_transforms)

        # --- Implement WeightedRandomSampler for Training Data ---
        train_labels_for_sampler = train_dataset.labels
        class_counts = np.bincount(train_labels_for_sampler)

        weights_for_sampler = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        weights_for_sampler[class_counts == 0] = 0.0 # Assign 0 weight if class count is zero

        sample_weights = weights_for_sampler[train_labels_for_sampler]

        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=os.cpu_count() // 2,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=os.cpu_count() // 2,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=os.cpu_count() // 2,
            pin_memory=True
        )

        loaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'class_weights': class_weights
        }

        print("\n=== Dataset Statistics (Loaded from Pre-generated Splits) ===")
        for name, df in zip(['Train', 'Validation', 'Test'], [train_df, val_df, test_df]):
            pos = sum(df['binary_label'] == 1)
            neg = sum(df['binary_label'] == 0)
            ratio_str = f"{neg / max(1, pos):.1f}:1" if pos > 0 else "N/A (No Pneumonia cases)"
            print(f"{name:<12} - Normal: {neg:>5} | Pneumonia: {pos:>5} | Ratio: {ratio_str}")
        print(f"\nClass weights for Loss (Normal, Pneumonia): {class_weights.numpy()}")
        print(f"Data loaders created from pre-saved splits in {time.time() - start_time:.2f} seconds")

        return loaders

    except Exception as e:
        print(f"Error creating data loaders from pre-saved splits: {str(e)}")
        raise

#%%
if __name__ == "__main__":
    # This __main__ block is for local testing/debugging of data_loaders.py
    # It will still perform the full split if the pre-saved files are not found locally.
    local_data_root = r"C:\Users\SyedShaistaTabassum\Capstone project\pneumonia-detection-local\my_xray_project_data"
    local_project_root = r"C:\Users\SyedShaistaTabassum\Capstone project\pneumonia-detection-local\pneumonia-detection" # Assuming project root is here

    config = {
        'data_dir': os.path.join(local_data_root, 'all_images_for_supervised_training'),
        'metadata_path': os.path.join(local_data_root, "Data_Entry_2017.csv"), # Original metadata for local splitting
        'project_root_in_drive': local_project_root, # Used to find/save local split files
        'batch_size': 32,
        'num_epochs': 50,
        'model_save_dir': './outputs'
    }
    
    # Check if pre-saved split files exist locally for testing
    train_csv_path_local = Path(local_project_root) / 'train_split.csv'
    if not train_csv_path_local.exists():
        print("\nPre-saved local split files not found. Performing full local split for testing...")
        # If pre-saved splits don't exist locally, perform the full split and save them
        original_meta_df = pd.read_csv(config['metadata_path'])
        original_meta_df = original_meta_df[['Image Index', 'Finding Labels']].copy()
        
        supervised_images_dir_local = Path(config['data_dir'])
        data = []
        image_extensions = ['jpeg', 'png', 'jpg', 'gif', 'bmp', 'tiff']
        label_map = original_meta_df.set_index('Image Index')['Finding Labels'].to_dict()

        for entry in os.listdir(supervised_images_dir_local):
            file_path = Path(supervised_images_dir_local) / entry
            if file_path.is_file() and file_path.suffix.lower().replace('.', '') in image_extensions:
                image_index = file_path.name
                finding_label = label_map.get(image_index, 'Unknown')
                if 'No Finding' in finding_label:
                    finding_label_standardized = 'Normal'
                elif 'Pneumonia' in finding_label:
                    finding_label_standardized = 'Pneumonia'
                else:
                    continue
                data.append({
                    'Image Index': image_index,
                    'Finding Labels': finding_label_standardized,
                    'Path': str(file_path),
                    'Split': 'unknown'
                })
        all_data_df = pd.DataFrame(data)
        all_data_df['binary_label'] = all_data_df['Finding Labels'].apply(lambda x: 1 if x == 'Pneumonia' else 0)

        train_df, temp_df = train_test_split(all_data_df, test_size=0.2, stratify=all_data_df['binary_label'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['binary_label'], random_state=42)

        # Calculate class weights locally
        train_labels_for_weights = train_df['binary_label'].tolist()
        classes = np.unique(train_labels_for_weights)
        if not (0 in classes and 1 in classes):
            class_weights_np = np.array([1.0, 1.0])
        else:
            class_weights_np = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels_for_weights)
        class_weights_list = class_weights_np.tolist()

        # Save locally
        os.makedirs(Path(local_project_root), exist_ok=True)
        train_df.to_csv(Path(local_project_root) / 'train_split.csv', index=False)
        val_df.to_csv(Path(local_project_root) / 'val_split.csv', index=False)
        test_df.to_csv(Path(local_project_root) / 'test_split.csv', index=False)
        with open(Path(local_project_root) / 'class_weights.json', 'w') as f:
            json.dump(class_weights_list, f)
        print("Local split files generated and saved.")

    try:
        loaders = create_stratified_loaders(config)
        print("\nSuccessfully created data loaders for testing (using pre-saved or newly generated local splits).")

        # Test PneumoniaOnlyDataset
        pneumonia_data_path_colab = os.path.join(local_data_root, 'pneumonia_only_for_simclr')
        preprocessor = ChestXRayPreprocessor()
        simclr_transforms = preprocessor.get_transforms()['simclr']
        pneumonia_dataset = PneumoniaOnlyDataset(pneumonia_data_path_colab, transform=simclr_transforms)
        pneumonia_loader = DataLoader(pneumonia_dataset, batch_size=4, shuffle=True)
        print("\nSuccessfully created PneumoniaOnlyDataset for testing.")
        for i, (view1, view2) in enumerate(pneumonia_loader):
            print(f"PneumoniaOnlyDataset Batch {i}: View1 shape: {view1.shape}, View2 shape: {view2.shape}")
            if i == 2: break
    except Exception as e:
        print(f"Failed to create data loaders: {e}")
