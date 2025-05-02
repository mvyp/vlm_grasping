import matplotlib.pyplot as plt
from PIL import Image
import os

from vlm_interfaces.gemini import GeminiInterface
from foundation_models.sam2 import SAM2Interface
from utils.utils import load_config_from_yaml,\
    draw_bbox_on_image, draw_mask_on_image


def detect(target_object_str: str,
           vlm_interface: GeminiInterface,
           sam2_interface: SAM2Interface,
           image: Image.Image):
    """
    Detect the target object in the image using VLM and SAM2.
    Args:
        target_object_str (str): The target object to detect.
        vlm_interface (GeminiInterface): The VLM interface.
        sam2_interface (SAM2Interface): The SAM2 interface.
        img (Image.Image): The input image.
    """
    # Detect objects in the image
    bbox_dict = vlm_interface.detect_object(image, target_object_str)
    print(bbox_dict)

    # Draw the bounding boxes on the image
    image_with_bbox = draw_bbox_on_image(image, bbox_dict)

    # Display the image with bounding boxes
    plt.imshow(image_with_bbox)
    plt.axis('off')  # Hide axis
    plt.show()
    plt.imsave('demo_detect_object_bbox.png', image_with_bbox)

    # Get the mask with SAM2
    bbox = bbox_dict[target_object_str]
    mask = sam2_interface.predict_mask_with_bbox(image, bbox)

    # Draw the mask on the image
    image_with_mask = draw_mask_on_image(image, mask)

    # Display the image with mask
    plt.imshow(image_with_mask)
    plt.axis('off')  # Hide axis
    plt.show()
    plt.imsave('demo_detect_object_mask.png', image_with_mask)


def main():
    # load yaml config
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    config = load_config_from_yaml(config_path)

    # Set the API key and initialize the VLM interface
    vlm_interface = GeminiInterface(api_key=config['google_gemini_api_key'])

    # Initialize SAM2 interface
    sam2_interface = SAM2Interface(checkpoint_path=os.path.join(
        config['sam2_directory'],
        'checkpoints',
        'sam2.1_hiera_large.pt'),
        model_config_path="configs/sam2.1/sam2.1_hiera_l.yaml")

    # Load the image
    image = Image.open('demo.jpeg')

    # Detect the target object
    detect('vita lemon tea', vlm_interface, sam2_interface, image)

if __name__ == "__main__":
    main()
