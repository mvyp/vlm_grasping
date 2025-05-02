import matplotlib.pyplot as plt
from PIL import Image

from vlm_interfaces.gemini import GeminiInterface
from foundation_models.sam2 import SAM2Interface
from utils.utils import draw_mask_on_image, draw_marks_on_image, overlay_masks_with_contours


# Set the API key for Google Generative AI
vlm = GeminiInterface(api_key="<your-google-gemini-api-key>")
# Initialize SAM2
sam2 = SAM2Interface(checkpoint_path="<path-to-sam2>/checkpoints/sam2.1_hiera_large.pt",
                     model_config_path="configs/sam2.1/sam2.1_hiera_l.yaml")
# Load the image
img = Image.open('demo.jpeg')


def main(target_object_str: str):
    # Claim the global variables
    global vlm, sam2, img

    # Detect objects in the image
    bbox_dict = vlm.detect_object(img, target_object_str)
    print(bbox_dict)

    # Draw the bounding boxes on the image
    image_with_bbox = draw_marks_on_image(img, bbox_dict)

    # Display the image with bounding boxes
    plt.imshow(image_with_bbox)
    plt.axis('off')  # Hide axis
    plt.show()
    plt.imsave('demo_detect_object_bbox.png', image_with_bbox)

    # Get the mask with SAM2
    bbox = bbox_dict[target_object_str]
    mask = sam2.predict_mask_with_bbox(img, bbox)

    # Draw the mask on the image
    image_with_mask = draw_mask_on_image(img, mask)

    # Display the image with mask
    plt.imshow(image_with_mask)
    plt.axis('off')  # Hide axis
    plt.show()
    plt.imsave('demo_detect_object_mask.png', image_with_mask)

if __name__ == "__main__":
    main('vita lemon tea')
