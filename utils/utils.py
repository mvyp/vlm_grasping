"""
Utilities.
"""
import cv2
import numpy as np
from PIL import Image
import yaml


class BoundingBox2D:
    """
    Represents a 2D bounding box defined by its top-left and bottom-right coordinates.

    Attributes:
        y_min (int): Top boundary (row index) of the box.
        x_min (int): Left boundary (column index) of the box.
        y_max (int): Bottom boundary (row index) of the box.
        x_max (int): Right boundary (column index) of the box.
    """
    def __init__(self, y_min: int, x_min: int, y_max: int, x_max: int):
        """
        Initialize the bounding box with given coordinates.

        Args:
            y_min (int): Minimum y-coordinate (top).
            x_min (int): Minimum x-coordinate (left).
            y_max (int): Maximum y-coordinate (bottom).
            x_max (int): Maximum x-coordinate (right).
        """
        self.y_min = y_min
        self.x_min = x_min
        self.y_max = y_max
        self.x_max = x_max

    def __repr__(self) -> str:
        """
        Official string representation of the BoundingBox2D.
        """
        return (f'bbox(y_min={self.y_min}, x_min={self.x_min}, '
                f'y_max={self.y_max}, x_max={self.x_max})')
    
    def __str__(self) -> str:
        """
        Informal string representation, same as __repr__.
        """
        return self.__repr__()
    
    def __iter__(self):
        """
        Allow unpacking of the bounding box coordinates.

        Returns:
            iterator: An iterator over (y_min, x_min, y_max, x_max).
        """
        return iter((self.y_min, self.x_min, self.y_max, self.x_max))


def load_config_from_yaml(yaml_path: str) -> dict:
    """
    Load a YAML configuration file into a Python dictionary.

    Args:
        yaml_path (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML content.
    """
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def draw_bbox_on_image(image: Image.Image,
                       bbox_dict: dict) -> np.ndarray:
    """
    Draw labeled bounding boxes on a PIL image.

    Args:
        image (PIL.Image.Image): Input image in RGB mode.
        bbox_dict (dict): Mapping from label (str) to BoundingBox2D.

    Returns:
        np.ndarray: Image (H, W, 3) array with colored boxes and labels.
    """
    # Convert PIL image to NumPy array (RGB) and then to BGR for OpenCV
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Iterate over all labels and their bounding boxes
    for label, bbox in bbox_dict.items():
        # Draw the rectangle with specified thickness
        bbox_thickness = 8
        cv2.rectangle(
            image_bgr,
            (bbox.x_min, bbox.y_min),
            (bbox.x_max, bbox.y_max),
            color=(255, 0, 0),  # Blue channel in BGR
            thickness=bbox_thickness
        )
        # Prepare text parameters
        font_scale = 2
        text_thickness = 4
        # Place the label text slightly above the top-left corner
        text_org = (bbox.x_min, max(bbox.y_min - 10, 0))
        cv2.putText(
            image_bgr,
            text=label,
            org=text_org,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(255, 0, 0),  # Blue channel in BGR
            thickness=text_thickness
        )

    # Convert back to RGB before returning
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def draw_mask_on_image(image: Image.Image,
                       mask: np.ndarray) -> np.ndarray:
    """
    Overlay a binary mask onto an image as a colored semi-transparent layer.

    Args:
        image (PIL.Image.Image): Input image in RGB mode.
        mask (np.ndarray): Binary 2D array (H, W) or boolean mask.

    Returns:
        np.ndarray: Image (H, W, 3) array with mask overlay in red.
    """
    # Convert PIL image to NumPy array (RGB)
    img_np = np.array(image).astype(np.uint8)

    # Create a red mask layer where mask is True
    red_layer = (mask[..., None] * np.array([0, 0, 255], dtype=np.uint8))

    # Blend original image and mask layer (50% opacity each)
    blended = ((img_np.astype(float) * 0.5) + (red_layer.astype(float) * 0.5)).astype(np.uint8)

    return blended
