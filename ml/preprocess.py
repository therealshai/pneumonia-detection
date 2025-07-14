# ml/data/preprocess.py
#%%
import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
#%%
class ChestXRayPreprocessor:
    def __init__(self):
        self.train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

        self.inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

    def visualize_transforms(self, image_path, save_path=None):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        orig_img = Image.open(image_path).convert('RGB')
        train_img = self.train_transforms(orig_img)
        val_img = self.val_transforms(orig_img)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_img)
        axes[0].set_title("Original")
        axes[0].axis('off')

        axes[1].imshow(self.inv_normalize(train_img).permute(1, 2, 0).clamp(0, 1))
        axes[1].set_title("Training Transform")
        axes[1].axis('off')

        axes[2].imshow(self.inv_normalize(val_img).permute(1, 2, 0).clamp(0, 1))
        axes[2].set_title("Validation Transform")
        axes[2].axis('off')

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def get_transforms(self):
        return {
            'train': self.train_transforms,
            'val': self.val_transforms
        }

    @staticmethod
    def calculate_mean_std(data_loader):
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        for data, _ in data_loader:
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
            num_batches += 1
        mean = channels_sum / num_batches
        std = (channels_squared_sum / num_batches - mean**2)**0.5
        return mean, std
#%%
if __name__ == "__main__":
    preprocessor = ChestXRayPreprocessor()
    try:
        sample_path = "sample_chest_xray.png"
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Sample image not found at {sample_path}")

        save_dir = os.path.join("..", "..", "reports", "figures")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "preprocessing_sample.png")

        print("Visualizing transformations...")
        preprocessor.visualize_transforms(
            image_path=sample_path,
            save_path=save_path
        )

        tfms = preprocessor.get_transforms()
        print("Training Transform:", tfms['train'])
        print("Validation Transform:", tfms['val'])


    except Exception as e:
        print(f"Error during visualization: {str(e)}")


# %%
