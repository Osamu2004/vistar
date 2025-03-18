import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Union, Optional, Dict


# 类别对应的颜色字典
CLASS_COLORS = {
    0: (255, 255, 255),  # 背景，白色
    1: (70, 181, 121),   # Intact，绿色
    2: (228, 189, 139),  # Damaged，浅棕色
    3: (182, 70, 69),    # Destroyed，红色
}

class Visualizer:
    def __init__(self, image: Optional[np.ndarray] = None):
        self._image = image  # The image to be visualized

    def set_image(self, image: np.ndarray) -> None:
        """
        Set the image to be visualized.
        
        Args:
            image (np.ndarray): Image to be visualized.
        """
        self._image = image

    def get_image(self) -> np.ndarray:
        """
        Get the image after visualization.
        
        Returns:
            np.ndarray: The image after visualization.
        """
        return self._image

    def show(self, wait_time: float = 0, backend: str = "matplotlib") -> None:
        """ Show the image.

        Args:
            wait_time (float): Time to wait before closing the window.
            backend (str): Backend to use for displaying the image, options are 'matplotlib' or 'cv2'.
        """
        if backend == "matplotlib":
            plt.imshow(self._image)
            plt.axis('off')
            plt.show()
        elif backend == "cv2":
            cv2.imshow('Image', self._image)
            cv2.waitKey(int(wait_time * 1000))  # Wait in milliseconds
            cv2.destroyAllWindows()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def draw_images(self, tensor: torch.Tensor, mode: str = 'color', logits: bool = False) -> np.ndarray:
        """
        Visualizes the segmentation result from a tensor of shape (B, C, H, W).
        
        Args:
            tensor (torch.Tensor): The input tensor of shape (B, C, H, W).
            mode (str): The mode for visualization, 'color' (color image), 'labels' (0, 1, 2, 3), or 'logits' (logits values).
            logits (bool): If True, the input tensor is considered as logits (before softmax).
        
        Returns:
            np.ndarray: The visualized image in numpy format (RGB).
        """
        assert tensor.ndimension() == 4, "Tensor must have 4 dimensions (B, C, H, W)"
        B, C, H, W = tensor.shape

        # Convert tensor to numpy for visualization
        tensor = tensor.cpu().detach().numpy()

        output_images = []

        for b in range(B):
            if mode == 'color':
                color_image = np.zeros((H, W, 3), dtype=np.uint8)
                pred_image = np.argmax(tensor[b], axis=0)  # Assuming tensor[b] is (C, H, W)
                for label, color in CLASS_COLORS.items():
                    color_image[pred_image == label] = color
                output_images.append(color_image)

            elif mode == 'labels':
                # Simply assign labels as integers (0, 1, 2, 3...)
                label_image = np.argmax(tensor[b], axis=0)
                output_images.append(label_image)

            elif mode == 'logits':
                # If logits are required, output the raw logits values (probabilities are also an option after softmax)
                output_images.append(tensor[b])  # Just return the raw logits

            else:
                raise ValueError(f"Unsupported mode: {mode}")

        # Assuming we're showing just the first sample in the batch for simplicity.
        # You can return all if needed in a different format.
        return output_images[0]


