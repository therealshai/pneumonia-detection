# ml/data_loaders.py
import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split # Still used for metadata split
from sklearn.utils.class_weight import compute_class_weight
from .preprocess import ChestXRayPreprocessor 
from pathlib import Path
import time

#%%
class BalancedChestDataset(Dataset):
    """
    Custom Dataset for Chest X-Ray images, designed to handle an already
    organized 'train/normal', 'train/pneumonia' etc. directory structure.
    """
    def __init__(self, df, root_dir, split, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image metadata for this split.
            root_dir (str): Root directory of the dataset. This will be the mounted path
                            to the base folder containing 'train', 'val', 'test' subdirectories.
            split (str): The current split ('train', 'val', or 'test').
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = Path(root_dir) # This will be the mounted path from Azure ML
        self.split = split
        self.transform = transform
        self.df = df.copy() # Use a copy to avoid modifying the original DataFrame

        # Ensure 'class' column exists and is 'pneumonia' or 'normal'
        self.df['class'] = self.df['Finding Labels'].apply(
            lambda x: 'pneumonia' if 'Pneumonia' in x else 'normal'
        )

        # Construct full paths to images based on the pre-organized structure
        # Example: root_dir/train/normal/image1.png
        self.df['path'] = self.df.apply(
            lambda row: self.root_dir / self.split / row['class'] / row['Image Index'], axis=1
        )

        # Filter out rows where the image file does not exist on disk
        initial_len = len(self.df)
        self.df = self.df[self.df['path'].apply(lambda x: x.exists())].reset_index(drop=True)
        if len(self.df) < initial_len:
            print(f"Warning: {initial_len - len(self.df)} images not found in {self.root_dir/self.split} and were removed from {split} dataset.")

        self.pneumonia_idx = self.df[self.df['class'] == 'pneumonia'].index.tolist()

        if len(self.df) == 0:
            raise ValueError(f"No valid images found in {self.root_dir/self.split} after filtering by path existence for {split} split.")

        print(f"{split} dataset: {len(self.df)} images ({len(self.pneumonia_idx)} pneumonia)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
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
    Reads data from the mounted Azure ML dataset (already pre-organized).
    """
    try:
        start_time = time.time()
        print("Loading metadata...")
        # metadata_path will now point to the mounted meta_data.csv
        metadata = pd.read_csv(config['metadata_path'])

        metadata['Finding Labels'] = metadata['Finding Labels'].replace('No Finding', 'Normal')
        metadata = metadata[
            (metadata['Finding Labels'] == 'Pneumonia') |
            (metadata['Finding Labels'] == 'Normal')
        ].copy()

        print(f"Metadata loaded and filtered: {len(metadata)} entries")
        print("Class distribution in full metadata:")
        print(metadata['Finding Labels'].value_counts())

        # Perform stratified train/val/test split on the metadata DataFrame
        # The split happens on the metadata, not the physical files, as they are already organized.
        # This ensures the dataframes passed to BalancedChestDataset are correct for each split.
        train_df, temp_df = train_test_split(
            metadata, test_size=0.3, stratify=metadata['Finding Labels'], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df['Finding Labels'], random_state=42
        )

        preprocessor = ChestXRayPreprocessor()
        transforms = preprocessor.get_transforms()

        print("Creating datasets...")
        # Pass config['data_dir'] as the root_dir, which is the mounted path to the base folder
        # containing 'train', 'val', 'test' subdirectories.
        train_dataset = BalancedChestDataset(train_df, config['data_dir'], 'train', transform=transforms['train'])
        val_dataset = BalancedChestDataset(val_df, config['data_dir'], 'val', transform=transforms['val'])
        test_dataset = BalancedChestDataset(test_df, config['data_dir'], 'test', transform=transforms['val'])

        labels_for_weights = [1 if lbl == 'Pneumonia' else 0 for lbl in train_df['Finding Labels']]
        classes = np.array([0, 1])
        class_weights = compute_class_weight('balanced', classes=classes, y=labels_for_weights)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        sample_weights = [class_weights[1] if lbl == 'Pneumonia' else class_weights[0]
                          for lbl in train_df['Finding Labels']]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        loaders = {
            'train': DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler,
                                num_workers=os.cpu_count() // 2, pin_memory=True),
            'val': DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                              num_workers=os.cpu_count() // 2, pin_memory=True),
            'test': DataLoader(test_dataset, batch_size=1, shuffle=False,
                               num_workers=os.cpu_count() // 2, pin_memory=True),
            'class_weights': class_weights
        }

        print("\n=== Dataset Statistics ===")
        for name, df in zip(['Train', 'Validation', 'Test'], [train_df, val_df, test_df]):
            pos = sum(df['Finding Labels'] == 'Pneumonia')
            neg = sum(df['Finding Labels'] == 'Normal')
            ratio_str = f"{neg / max(1, pos):.1f}:1" if pos > 0 else "N/A (No Pneumonia cases)"
            print(f"{name:<12} - Normal: {neg:>5} | Pneumonia: {pos:>5} | Ratio: {ratio_str}")
        print(f"\nClass weights (Normal, Pneumonia): {class_weights.numpy()}")
        print(f"Data loaders created in {time.time() - start_time:.2f} seconds")

        return loaders

    except Exception as e:
        print(f"Error creating data loaders: {str(e)}")
        raise 

#%%
if __name__ == "__main__":
    # For local testing, this config needs to point to your local data
    # For Azure ML, these paths will be overridden by the mounted data
    local_data_root = r"C:\Users\SyedShaistaTabassum\Capstone project\pneumonia-detection-local\external-dataset"
    config = {
        'data_dir': local_data_root,
        'metadata_path': os.path.join(local_data_root, "meta_data.csv"),
        'batch_size': 32
    }
    
    try:
        loaders = create_stratified_loaders(config)
        print("\nData loaders created successfully!")
    except Exception as e:
        print(f"Failed to create data loaders: {e}")

