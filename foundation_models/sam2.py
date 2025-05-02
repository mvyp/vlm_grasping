from PIL import Image
import numpy as np

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from utils.utils import BoundingBox2D

class SAM2Interface:
    def __init__(self,
                 checkpoint_path: str,
                 model_config_path: str):
        self._checkpoint_path = checkpoint_path
        self._model_config_path = model_config_path
        self._sam2 = build_sam2(model_config_path, checkpoint_path)
        self._image_predictor   = SAM2ImagePredictor(self._sam2)
        self._mask_generator = SAM2AutomaticMaskGenerator(self._sam2)

    def predict_mask_with_bbox(self,
                          image: Image.Image,
                          bbox: BoundingBox2D):
        bbox_sam = np.array([bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max])
        self._image_predictor.set_image(image)
        masks, scores, logits = self._image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox_sam,
            multimask_output=False
            )
        mask = masks[0]
        return mask
    
    def generate_masks(self,
                       image: Image.Image):
        image = np.array(image.convert("RGB"))
        masks = self._mask_generator.generate(image)
        return masks
