    # ml/grad_cam_utils.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2 # OpenCV for image processing (resizing, colormapping, overlay)
from PIL import Image
from torchvision import transforms

class GradCAM:
        """
        Implements Grad-CAM (Gradient-weighted Class Activation Mapping) for model interpretability.
        This is a custom implementation, as the pytorch-grad-cam library was not installable.
        Designed for a single-output binary classification model (sigmoid output).
        """
        def __init__(self, model, target_layers, use_cuda=False):
            self.model = model
            self.target_layers = target_layers
            self.use_cuda = use_cuda
            self.gradients = {}
            self.activations = {}

            for i, layer in enumerate(target_layers):
                layer.register_forward_hook(self._save_activations_hook(layer))
                layer.register_backward_hook(self._save_gradients_hook(layer))

        def _save_activations_hook(self, layer):
            def hook(module, input, output):
                self.activations[layer] = output
            return hook

        def _save_gradients_hook(self, layer):
            def hook(module, grad_input, grad_output):
                self.gradients[layer] = grad_output[0]
            return hook

        def __call__(self, input_tensor, targets=None):
            self.model.eval()
            self.model.zero_grad()

            if self.use_cuda:
                input_tensor = input_tensor.cuda()

            model_output = self.model(input_tensor)

            target_category = 0 # Always target the 0th output neuron for single output models

            target_output_for_backward = model_output[:, target_category]
            target_output_for_backward.backward(torch.ones_like(target_output_for_backward), retain_graph=True)

            batch_cam = []

            for i in range(input_tensor.shape[0]):
                # Get gradients and activations for the current image in the batch
                # Assuming a single target layer for simplicity in this custom implementation
                gradients = self.gradients[self.target_layers[0]][i].cpu().data.numpy()
                activations = self.activations[self.target_layers[0]][i].cpu().data.numpy()

                weights = np.mean(gradients, axis=(1, 2))

                cam = np.zeros(activations.shape[1:], dtype=np.float32)
                for j, w in enumerate(weights):
                    cam += w * activations[j]

                cam = np.maximum(cam, 0)

                cam_min, cam_max = np.min(cam), np.max(cam)
                if cam_max - cam_min > 1e-8:
                    cam = (cam - cam_min) / (cam_max - cam_min)
                else:
                    cam = 0

                batch_cam.append(cam)
            
            return np.array(batch_cam)

class ClassifierOutputTarget:
        def __init__(self, category):
            self.category = category

        def __call__(self, model_output):
            return model_output[:, self.category]

def show_cam_on_image(img: np.ndarray,
                          mask: np.ndarray,
                          use_rgb: bool = False,
                          colormap: int = cv2.COLORMAP_JET,
                          alpha: float = 0.4) -> np.ndarray:
        if use_rgb:
            img = img.astype(np.float32) / 255.0
            
        if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        if img.max() <= 1.0 + 1e-8:
            heatmap = heatmap.astype(np.float32) / 255.0
        else:
            heatmap = heatmap.astype(np.float32)

        overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        
        if img.max() > 1.0 + 1e-8:
            return np.uint8(overlayed_img)
        else:
            return np.uint8(overlayed_img * 255.0)

def get_efficientnet_target_layer(model):
        """
        Identifies the last convolutional layer in an EfficientNet-B0 model's features.
        This is typically the last block of the feature extractor.
        """
        last_conv_layer = None
        # Iterate through all named modules to find the last Conv2d
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv_layer = module
        
        if last_conv_layer:
            print(f"Identified specific target Conv2d layer for Grad-CAM: {last_conv_layer}")
            return last_conv_layer
        
        raise ValueError("Could not find a suitable Conv2d target layer for Grad-CAM in the provided model.")

    