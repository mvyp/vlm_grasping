import cv2
import numpy as np
from PIL import Image

class BoundingBox2D:
    def __init__(self, y_min, x_min, y_max, x_max):
        self.y_min = y_min
        self.x_min = x_min
        self.y_max = y_max
        self.x_max = x_max

    def __repr__(self):
        return f'bbox(y_min={self.y_min}, x_min={self.x_min}, y_max={self.y_max}, x_max={self.x_max})'
    
    def __str__(self):
        return f'bbox(y_min={self.y_min}, x_min={self.x_min}, y_max={self.y_max}, x_max={self.x_max})'
    
    def __iter__(self):
        return iter((self.y_min, self.x_min, self.y_max, self.x_max))


def draw_bbox_on_image(image: Image.Image,
                       bbox_dict: dict):
    # Open the image using PIL
    actual_width, actual_height = image.size
    # Convert the PIL image to a NumPy array (which OpenCV can use)
    image = np.array(image)
    # Convert the image from RGB to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for label, bbox in bbox_dict.items():
        # Draw the bounding box with increased thickness
        bbox_thickness = 8  # Adjust thickness
        cv2.rectangle(image, (bbox.x_min, bbox.y_min), (bbox.x_max, bbox.y_max),
                      (255, 0, 0), bbox_thickness)
        # Adjust font size and thickness for better visibility
        font_scale = 2  # Adjusted font size
        thickness = 4   # Adjusted text thickness
        # Put label text above the bounding box
        cv2.putText(image, label, (bbox.x_min, bbox.y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
    # Convert the image back to RGB for displaying with matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def draw_mask_on_image(image: Image.Image,
                       mask: np.ndarray):
    image = np.array(image)
    mask_layer = (mask[..., None] * np.array([0, 0, 255], dtype=np.uint8))  # (H, W, 3)
    image = ((image.astype(float) * 0.5) + (mask_layer.astype(float) * 0.5)).astype(np.uint8)
    return image

