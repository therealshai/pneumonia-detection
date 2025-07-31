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
        # OPTIMIZATION: Increased degrees for RandomRotation and ranges for ColorJitter
        self.train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=25), # Increased from 20 to 25
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Increased ranges
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

        # NEW: Transforms specifically for SimCLR pre-training
        # These are typically more aggressive than supervised training transforms
        self.simclr_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)), # Random crop and resize
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8), # Aggressive color jitter
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5), # Gaussian blur
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        self.inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

    def get_transforms(self):
        return {
            'train': self.train_transforms,
            'val': self.val_transforms,
            'simclr': self.simclr_transforms, # Added SimCLR transforms
            'inv_normalize': self.inv_normalize
        }

    def visualize_transforms(self, image_path, save_path, transform_type='train'):
        """
        Visualizes the effect of transformations on a sample image.
        Added transform_type to visualize different sets of transforms.
        """
        original_image = Image.open(image_path).convert('RGB')

        if transform_type == 'train':
            selected_transform = self.train_transforms
            title_prefix = "Augmented (Train)"
        elif transform_type == 'simclr':
            selected_transform = self.simclr_transforms
            title_prefix = "Augmented (SimCLR)"
        else:
            selected_transform = self.val_transforms
            title_prefix = "Augmented (Val)" # Fallback, though val usually doesn't need visualization

        transformed_images = [selected_transform(original_image) for _ in range(5)] # 5 augmented versions

        fig, axes = plt.subplots(1, 6, figsize=(18, 4))
        axes[0].imshow(original_image)
        axes[0].set_title("Original")
        axes[0].axis('off')

        for i, img_tensor in enumerate(transformed_images):
            # Denormalize for visualization
            img_denorm = self.inv_normalize(img_tensor)
            # Convert to numpy array and transpose from (C, H, W) to (H, W, C)
            img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
            # Clip values to [0, 1] for proper display
            img_np = np.clip(img_np, 0, 1)
            axes[i+1].imshow(img_np)
            axes[i+1].set_title(f"{title_prefix} {i+1}")
            axes[i+1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig) # Close the figure to free memory
        print(f"Transform visualization saved to {save_path}")

    def calculate_mean_std(self, dataloader):
        """
        Calculates mean and standard deviation for normalization.
        This function is typically run once on the training data.
        """
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        for data, _ in dataloader:
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
            # Attempt to create a dummy image if not found for local testing
            print(f"Sample image not found at {sample_path}. Creating a dummy image for visualization test.")
            dummy_image = Image.new('RGB', (224, 224), color = 'red')
            dummy_image.save(sample_path)
            print(f"Dummy image saved to {sample_path}")

        save_dir = os.path.join("..", "..", "reports", "figures")
        os.makedirs(save_dir, exist_ok=True)
        save_path_train = os.path.join(save_dir, "preprocessing_sample_train.png")
        save_path_simclr = os.path.join(save_dir, "preprocessing_sample_simclr.png")


        print("Visualizing training transformations...")
        preprocessor.visualize_transforms(
            image_path=sample_path,
            save_path=save_path_train,
            transform_type='train'
        )
        print("Visualizing SimCLR transformations...")
        preprocessor.visualize_transforms(
            image_path=sample_path,
            save_path=save_path_simclr,
            transform_type='simclr'
        )
        print("Visualization complete.")
        # Clean up dummy image if created
        if "dummy_image" in locals() and os.path.exists(sample_path):
            os.remove(sample_path)
            print(f"Cleaned up dummy image at {sample_path}")

    except Exception as e:
        print(f"Error during preprocessing visualization: {str(e)}")
