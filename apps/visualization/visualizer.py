import os
import numpy as np
import torch
import cv2
from typing import Dict, Tuple, Optional, List, Union
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, output_path: str, class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None) -> None:
        """
        Visualizer initialization.

        Args:
            output_path (str): The directory path where the images will be saved.
            class_colors (dict, optional): Dictionary to map class ids to RGB colors. 
                If None, it will use the default color scheme.
        """
        self.output_path = output_path
        self.CLASS_COLORS = class_colors
        # Ensure the output path exists
        os.makedirs(output_path, exist_ok=True)

    def save_npy(self, filename: str, data: np.ndarray, save_path: str) -> None:
        """ Save the numpy array as a .npy file.

        Args:
            filename (str): The name of the file to save the numpy array.
            data (np.ndarray): The numpy array to save.
            mode (str): Mode of the image (logits, labels, or color).
        """

        save_path = os.path.join(save_path, f"{filename}.npy")
        np.save(save_path, data)

    def save_image(self, filename: str, image: np.ndarray, save_path: str, file_format: Optional[str] = 'png') -> None:
        """ Save the visualized image to the specified path with the given format.

        Args:
            filename (str): The name of the file to save the image.
            image (np.ndarray): The image to save.
            mode (str): Mode of the image (logits, labels, or color).
            file_format (str, optional): The format of the saved image (e.g., 'png', 'jpg', 'bmp').
        """
        save_path = os.path.join(save_path, f"{filename}.{file_format}")
        cv2.imwrite(save_path, image)

    def show(self, drawn_img: np.ndarray, wait_time: float = 0, backend: str = "matplotlib") -> None:
        """ Show the drawn image.

        Args:
            drawn_img (np.ndarray): The image to show.
            wait_time (float): Time to wait before closing the window.
            backend (str): Backend to use for displaying the image, options are 'matplotlib' or 'cv2'.
        """
        if backend == "matplotlib":
            plt.imshow(drawn_img)
            plt.axis('off')
            plt.show()
        elif backend == "cv2":
            cv2.imshow('Image', drawn_img)
            cv2.waitKey(int(wait_time * 1000))  # Wait in milliseconds
            cv2.destroyAllWindows()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def draw_segmentation(self, tensor: torch.Tensor, names: Optional[List[str]] = None, modes: Union[str, List[str]] = 'color', formats: Optional[List[str]] = None) -> np.ndarray:
        """
        Visualizes the segmentation result from a tensor of shape (B, C, H, W).
        
        Args:
            tensor (torch.Tensor): The input tensor of shape (B, C, H, W).
            names (List[str], optional): List of sample names for each image.
            modes (Union[str, List[str]]): The mode(s) for visualization, 'color' (color image), 
                                           'labels' (0, 1, 2, 3), or 'logits' (logits values).
                                           Can be a single mode or a list of modes.
            formats (List[str], optional): The formats for each image, e.g., ['png', 'jpg'].
        
        Returns:
            np.ndarray: The visualized image in numpy format (RGB).
        """
        assert tensor.ndimension() == 4, "Tensor must have 4 dimensions (B, C, H, W)"
        B, C, H, W = tensor.shape

        # Convert tensor to numpy for visualization
        tensor = tensor.cpu().detach().numpy()

        output_images = []

        # If modes is a single string, convert it to a list for consistency
        if isinstance(modes, str):
            modes = [modes]

        for b in range(B):
            for mode in modes:
                if mode == 'color':
                    color_image = np.zeros((H, W, 3), dtype=np.uint8)
                    pred_image = np.argmax(tensor[b], axis=0)  # Assuming tensor[b] is (C, H, W)
                    for label, color in self.CLASS_COLORS.items():
                        color_image[pred_image == label] = color
                    output_images.append(color_image)

                elif mode == 'labels':
                    # Simply assign labels as integers (0, 1, 2, 3...)
                    label_image = np.argmax(tensor[b], axis=0)
                    output_images.append(label_image)

                elif mode == 'logits':
                    # If logits are required, output the raw logits values (probabilities are also an option after softmax)
                    logits_image = tensor[b]
                    output_images.append(logits_image)  # Just return the raw logits

                else:
                    raise ValueError(f"Unsupported mode: {mode}")

        # Save all output images with the name if provided
        for idx, img in enumerate(output_images):
            name = names[idx] if names is not None else f"{idx}"
            file_format = formats[idx] if formats is not None else 'png'

            # Save the image based on the mode
            if 'logits' in modes:
                # For logits, save as a numpy array
                mode_folder = os.path.join(self.output_path, 'logits')
                os.makedirs(mode_folder, exist_ok=True) 
                self.save_npy(name, img, mode_folder)  # Save as .npy file
            elif 'labels' in modes:
                # For labels, save as a grayscale image
                mode_folder = os.path.join(self.output_path, 'labels')
                os.makedirs(mode_folder, exist_ok=True) 
                self.save_image(name, img.astype(np.uint8), mode_folder, file_format)
            elif 'color' in modes:
                mode_folder = os.path.join(self.output_path, 'color')
                os.makedirs(mode_folder, exist_ok=True) 
                self.save_image(name, img, mode_folder, file_format)
