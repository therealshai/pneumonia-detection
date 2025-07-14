"""
The data segregated in the load_data file is will go through thr preprocessing steps defined in the preprocess file.
-- on-the-fly preprocessing.


"""
#A custom Dataset class must implement three functions: __init__, __len__, and __getitem__. 
# %%
import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from data.preprocess import ChestXRayPreprocessor
# %%
class ChestXRayDataset(Dataset):
    """Custom dataset for chest X-ray images with metadata support"""
    
    def __init__(self, root_dir, metadata_path="\meta_data.csv", transform=None):
        """
        Args:
            root_dir (string): Directory with subfolders 'pneumonia' and 'normal'
            metadata_path (string): Path to metadata CSV file
            transform (callable, optional): Optional transform to be applied
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['normal', 'pneumonia']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Load image paths and labels
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
        
        # Load metadata if provided
        self.metadata = None
        try:
            if metadata_path and os.path.exists(metadata_path):
                self.metadata = pd.read_csv(metadata_path).set_index('Image Index')
        except Exception as e:
            print(f"Error loading metadata: {e}")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # Get metadata if available
        metadata = torch.zeros(4)  # Default values
        if self.metadata is not None:
            img_name = os.path.basename(img_path)
            if img_name in self.metadata.index:
                meta = self.metadata.loc[img_name]
                metadata = torch.tensor([
                    meta['OriginalImage[Width]'],
                    meta['OriginalImage[Height]'],
                    meta['OriginalImagePixelSpacing[x]'],
                    meta['OriginalImagePixelSpacing[y]']
                ], dtype=torch.float32)
        
        return image, label, metadata
# %%
def create_data_loaders(config):
    """
    Creates data loaders for training, validation, and testing
    
    Args:
        config (dict): Configuration dictionary containing:
            - data_dir (str): Root directory containing train/val/test folders
            - batch_size (int): Batch size for data loaders
            - metadata_path (str): Path to metadata CSV file
            - augment (bool): Whether to use data augmentation
            
    Returns:
        dict: Dictionary containing data loaders and class weights
    """
    # Initialize preprocessor
    preprocessor = ChestXRayPreprocessor()
    transforms_dict = preprocessor.get_transforms()
    
    # Create datasets
    train_dataset = ChestXRayDataset(
        root_dir=os.path.join(config['data_dir'], 'train'),
        metadata_path=config.get('metadata_path'),
        transform=transforms_dict['train'] if config.get('augment', True) else transforms_dict['val']
    )
    
    val_dataset = ChestXRayDataset(
        root_dir=os.path.join(config['data_dir'], 'val'),
        metadata_path=config.get('metadata_path'),
        transform=transforms_dict['val']
    )
    
    test_dataset = ChestXRayDataset(
        root_dir=os.path.join(config['data_dir'], 'test'),
        metadata_path=config.get('metadata_path'),
        transform=transforms_dict['val']
    )
    
    # Calculate class weights for imbalance
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_dataset.labels),
        y=np.array(train_dataset.labels)
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    sample_weights = class_weights[torch.tensor(train_dataset.labels)]
    
    # Create samplers and loaders
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=1,  # Batch size 1 for precise evaluation
            shuffle=False,
            num_workers=4,
            pin_memory=True
        ),
        'class_weights': class_weights,
        'class_names': train_dataset.classes
    }
    
    return loaders

config = {
    'data_dir': r"C:\Users\SyedShaistaTabassum\Capstone project\pneumonia-detection-local\external-dataset",
    'metadata_path': os.path.join(
        r"C:\Users\SyedShaistaTabassum\Capstone project\pneumonia-detection-local\external-dataset",
        "meta_data.csv"
    ),
    'batch_size': 32,
    'augment': True
}



# %%
if __name__ == "__main__":
    # Example configuration
    config = {
        'data_dir': r"C:\Users\SyedShaistaTabassum\Capstone project\pneumonia-detection-local\external-dataset",
        'metadata_path': os.path.join(
            r"C:\Users\SyedShaistaTabassum\Capstone project\pneumonia-detection-local\external-dataset",
            "meta_data.csv"
        ),
        'batch_size': 32,
        'augment': True
    }
    
    # Create and test data loaders
    data_loaders = create_data_loaders(config)
    
    # Print dataset statistics
    print(f"Class names: {data_loaders['class_names']}")
    print(f"Class weights: {data_loaders['class_weights']}")
    print(f"Training batches: {len(data_loaders['train'])}")
    print(f"Validation batches: {len(data_loaders['val'])}")
    print(f"Test batches: {len(data_loaders['test'])}")
    
    # Test one batch
    images, labels, metadata = next(iter(data_loaders['train']))
    print(f"\nBatch image shape: {images.shape}")
    print(f"Batch labels: {labels[:5]}...")  # Show first 5 labels
    print(f"Batch metadata shape: {metadata.shape}")


#%%
# Create and test data loaders
data_loaders = create_data_loaders(config)

# Print dataset statistics
print(f"Class names: {data_loaders['class_names']}")
print(f"Class weights: {data_loaders['class_weights']}")
print(f"Training batches: {len(data_loaders['train'])}")
print(f"Validation batches: {len(data_loaders['val'])}")
print(f"Test batches: {len(data_loaders['test'])}")

# Test one batch
images, labels, metadata = next(iter(data_loaders['train']))
print(f"\nBatch image shape: {images.shape}")
print(f"Batch labels: {labels[:5]}...")  # Show first 5 labels
print(f"Batch metadata shape: {metadata.shape}")
# %%

# %%
