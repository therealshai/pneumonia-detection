#%%
import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from preprocess import ChestXRayPreprocessor 
from pathlib import Path
import time

#%%
class BalancedChestDataset(Dataset):
    """
    Custom Dataset for Chest X-Ray images, designed to handle the
    'train/normal', 'train/pneumonia' etc. directory structure and
    oversample the pneumonia class during __getitem__.
    """
    def __init__(self, df, root_dir, split, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image metadata for this split.
            root_dir (str): Root directory of the dataset (e.g., 'external-dataset').
                            The actual images are expected under root_dir/split/class/.
            split (str): The current split ('train', 'val', or 'test').
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.df = df.copy() # Use a copy to avoid modifying the original DataFrame

        # Ensure 'class' column exists and is 'pneumonia' or 'normal'
        # This is important if the input df doesn't already have it
        self.df['class'] = self.df['Finding Labels'].apply(
            lambda x: 'pneumonia' if 'Pneumonia' in x else 'normal'
        )

        # Construct full paths to images based on the new directory structure
        self.df['path'] = self.df.apply(
            lambda row: self.root_dir / self.split / row['class'] / row['Image Index'], axis=1
        )

        # Filter out rows where the image file does not exist on disk
        # This is a crucial step to ensure only valid paths are included
        initial_len = len(self.df)
        self.df = self.df[self.df['path'].apply(lambda x: x.exists())].reset_index(drop=True)
        if len(self.df) < initial_len:
            print(f"Warning: {initial_len - len(self.df)} images not found in {self.root_dir/self.split} and were removed from {split} dataset.")

        self.pneumonia_idx = self.df[self.df['class'] == 'pneumonia'].index.tolist()

        if len(self.df) == 0:
            raise ValueError(f"No valid images found in {self.root_dir/self.split} after filtering by path existence.")

        print(f"{split} dataset: {len(self.df)} images ({len(self.pneumonia_idx)} pneumonia)")

    def __len__(self):
        
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label.
        Handles potential image loading errors by returning a dummy tensor and -1 label.
        """
        row = self.df.iloc[idx]

        try:
            image = Image.open(row['path']).convert('RGB')
            label = 1 if row['class'] == 'pneumonia' else 0 # 1 for pneumonia, 0 for normal
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image at {row['path']}: {e}. Returning dummy data.")
        
            return torch.zeros(3, 224, 224), -1 
#%%
def create_stratified_loaders(config):
    """
    Creates stratified PyTorch DataLoaders for training, validation, and testing.
    Handles metadata loading, splitting, class weight calculation, and sampler creation.

    Args:
        config (dict): Configuration dictionary containing:
            'data_dir' (str): Root directory for the dataset (e.g., 'external-dataset').
            'metadata_path' (str): Full path to the metadata CSV file (e.g., 'meta_data.csv').
            'batch_size' (int): Batch size for DataLoaders.

    Returns:
        dict: A dictionary containing 'train', 'val', 'test' DataLoaders and 'class_weights'.
    """
    try:
        start_time = time.time()
        print("Loading metadata...")
        metadata = pd.read_csv(config['metadata_path'])

        # Standardize 'No Finding' to 'Normal'
        metadata['Finding Labels'] = metadata['Finding Labels'].replace('No Finding', 'Normal')

        # Filter to include only 'Pneumonia' and 'Normal' for this task
        metadata = metadata[
            (metadata['Finding Labels'] == 'Pneumonia') |
            (metadata['Finding Labels'] == 'Normal')
        ].copy()

        print(f"Metadata loaded and filtered: {len(metadata)} entries")
        print("Class distribution in full metadata:")
        print(metadata['Finding Labels'].value_counts())

        # Perform stratified train/val/test split on the metadata DataFrame
        # First split into train and temp (for val/test)
        train_df, temp_df = train_test_split(
            metadata,
            test_size=0.3, # 30% for temp (val+test)
            stratify=metadata['Finding Labels'],
            random_state=42
        )

        # Then split temp into val and test (50/50 of temp, so 15% each of original)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5, # 50% of temp_df goes to test, 50% to val
            stratify=temp_df['Finding Labels'],
            random_state=42
        )

        # Initialize the preprocessor to get image transforms
        preprocessor = ChestXRayPreprocessor()
        transforms = preprocessor.get_transforms()

        print("Creating datasets...")
        
        train_dataset = BalancedChestDataset(train_df, config['data_dir'], 'train', transform=transforms['train'])
        val_dataset = BalancedChestDataset(val_df, config['data_dir'], 'val', transform=transforms['val'])
        test_dataset = BalancedChestDataset(test_df, config['data_dir'], 'test', transform=transforms['val'])

        # Calculate class weights for the training set for BCEWithLogitsLoss
      
        labels_for_weights = [1 if lbl == 'Pneumonia' else 0 for lbl in train_df['Finding Labels']]
        classes = np.array([0, 1]) # 0 for Normal, 1 for Pneumonia
        class_weights = compute_class_weight('balanced', classes=classes, y=labels_for_weights)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        # Create a WeightedRandomSampler for the training set to handle imbalance during batching
        # The weights for each sample are based on their class weight
        sample_weights = [class_weights[1] if lbl == 'Pneumonia' else class_weights[0]
                          for lbl in train_df['Finding Labels']]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        loaders = {
            'train': DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler,
                                num_workers=0, pin_memory=True),
            'val': DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, # No shuffling for val/test with sampler
                              num_workers=0, pin_memory=True),
            'test': DataLoader(test_dataset, batch_size=1, shuffle=False, # Batch size 1 for test for easier individual prediction analysis
                               num_workers=0, pin_memory=True),
            'class_weights': class_weights # Pass class weights to the training script
        }

        print("\n=== Dataset Statistics ===")
        for name, df in zip(['Train', 'Validation', 'Test'], [train_df, val_df, test_df]):
            pos = sum(df['Finding Labels'] == 'Pneumonia')
            neg = sum(df['Finding Labels'] == 'Normal')
            # Handle division by zero if pos is 0
            ratio_str = f"{neg / max(1, pos):.1f}:1" if pos > 0 else "N/A (No Pneumonia cases)"
            print(f"{name:<12} - Normal: {neg:>5} | Pneumonia: {pos:>5} | Ratio: {ratio_str}")
        print(f"\nClass weights (Normal, Pneumonia): {class_weights.numpy()}")
        print(f"Data loaders created in {time.time() - start_time:.2f} seconds")

        return loaders

    except Exception as e:
        print(f"Error creating data loaders: {str(e)}")
        raise # Re-raise the exception after printing

#%%
if __name__ == "__main__":
    # Example usage: Define your configuration
    config = {
        'data_dir': r"C:\Users\SyedShaistaTabassum\Capstone project\pneumonia-detection-local\external-dataset",
        'metadata_path': r"C:\Users\SyedShaistaTabassum\Capstone project\pneumonia-detection-local\external-dataset\meta_data.csv", # Ensure this points to your meta_data.csv
        'batch_size': 32
    }
    
    try:
        loaders = create_stratified_loaders(config)
        print("\nData loaders created successfully!")
        # You can now access loaders['train'], loaders['val'], loaders['test'], loaders['class_weights']
    except Exception as e:
        print(f"Failed to create data loaders: {e}")

# %%
