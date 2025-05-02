"""
main.py

Script to detect a specified object in an image by first using a vision-language model
(Gemini) to localize via bounding box, then refining with Segment Anything Model v2 (SAM2)
to obtain a precise segmentation mask. Results are displayed and saved as PNG files.
"""
import os
from PIL import Image
import matplotlib.pyplot as plt

from vlm_interfaces.gemini import GeminiInterface
from foundation_models.sam2 import SAM2Interface
from utils.utils import (
    load_config_from_yaml,
    draw_bbox_on_image,
    draw_mask_on_image
)


def detect(
    target_object: str,
    vlm_interface: GeminiInterface,
    sam2_interface: SAM2Interface,
    image: Image.Image
) -> None:
    """
    Perform object detection and segmentation on an image.

    1. Use the vision-language model to predict a bounding box for `target_object`.
    2. Draw and display the bounding box.
    3. Use SAM2 to predict a segmentation mask within that box.
    4. Draw and display the mask overlay.

    Args:
        target_object (str): Name of the object to detect (e.g., 'cat').
        vlm_interface (GeminiInterface): Initialized Gemini VLM interface.
        sam2_interface (SAM2Interface): Initialized SAM2 segmentation interface.
        image (PIL.Image.Image): Input image to process.
    """
    # Step 1: Detect bounding box for target object
    bbox_dict = vlm_interface.detect_object(image, target_object)
    print(f"Detected bounding boxes: {bbox_dict}")

    # Step 2: Draw bounding boxes on the original image
    image_with_bbox = draw_bbox_on_image(image, bbox_dict)
    plt.imshow(image_with_bbox)
    plt.axis('off')  # Turn off axes for cleaner display
    plt.show()

    # Step 3: Extract the box for the target object and predict a mask
    bbox = bbox_dict[target_object]
    mask = sam2_interface.predict_mask_with_bbox(image, bbox)

    # Step 4: Overlay the segmentation mask on the image
    image_with_mask = draw_mask_on_image(image, mask)
    plt.imshow(image_with_mask)
    plt.axis('off')
    plt.show()


def main() -> None:
    """
    Entry point for the detection pipeline.

    1. Load configuration from YAML.
    2. Initialize Gemini and SAM2 interfaces.
    3. Load the demo image.
    4. Run the detect() function on the target object.
    """
    # Construct path to YAML config relative to this script file
    config_path = os.path.join(
        os.path.dirname(__file__),
        'config',
        'config.yaml'
    )
    config = load_config_from_yaml(config_path)

    # Initialize the vision-language model interface with API key
    vlm_interface = GeminiInterface(
        api_key=config['google_gemini_api_key']
    )

    # Initialize the SAM2 interface with checkpoint and config paths
    sam2_interface = SAM2Interface(
        checkpoint_path=os.path.join(
            config['sam2_directory'],
            'checkpoints',
            'sam2.1_hiera_large.pt'
        ),
        model_config_path="configs/sam2.1/sam2.1_hiera_l.yaml"
    )

    # Load the input image for detection
    image = Image.open('demo.jpeg')

    # Run detection on the specified object (hardcoded here)
    detect(
        target_object='vita lemon tea',
        vlm_interface=vlm_interface,
        sam2_interface=sam2_interface,
        image=image
    )


if __name__ == '__main__':
    main()
