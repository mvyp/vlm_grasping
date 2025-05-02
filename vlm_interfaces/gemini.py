"""
Module providing an interface to Google Gemini via the Generative AI API.
This wraps both text-based queries and image-based object detection tasks.
"""
import google.generativeai as genai
import numpy as np
from PIL import Image
import ast

from utils.utils import BoundingBox2D


class GeminiInterface:
    """
    Wrapper class for interacting with Google Gemini generative model.

    Attributes:
        _safety_settings (list of dict): Content filtering thresholds for various harm categories.
        _generation_config (dict): Parameters controlling text generation (temperature, max tokens).
        _gemini_model (genai.GenerativeModel): Configured Gemini model instance.
        _image_size (tuple[int, int]): Height and width used to resize images before sending to the model.
    """
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-pro-latest",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        image_size: list[int] = [512, 512]
    ):
        """
        Initialize the GeminiInterface.

        Args:
            api_key (str): API key for Google Generative AI.
            model_name (str): The specific Gemini model to use.
            temperature (float): Sampling temperature for generation; higher => more random.
            max_tokens (int): Maximum number of tokens to generate in the response.
            image_size (list[int]): [height, width] to resize input images for detection.
        """
        # Authenticate with Gemini API
        genai.configure(api_key=api_key)

        # Define safety settings (no blocking for these categories)
        self._safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # Define generation parameters\        
        self._generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # Initialize the Gemini generative model
        self._gemini_model = genai.GenerativeModel(
            model_name=model_name,
            safety_settings=self._safety_settings
        )

        # Save target image size for detection
        self._image_size = image_size

    def ask_gemini(self, prompt_elements: list) -> str:
        """
        Send structured prompt elements to Gemini and return its text response.

        Args:
            prompt_elements (list): A sequence of text or image objects to include in the prompt.

        Returns:
            str: The generated text response from the model.
        """
        # Combine all prompt parts into a single message entry
        parts = []
        for element in prompt_elements:
            parts.append(element)
        messages = [{"role": "user", "parts": parts}]

        # Call the Gemini API
        response = self._gemini_model.generate_content(
            messages,
            generation_config=self._generation_config,
        )

        # Print any prompt feedback for debugging
        print(response.prompt_feedback)

        return response.text

    def detect_object(
        self,
        image: Image.Image,
        object_name: str
    ) -> dict[str, BoundingBox2D]:
        """
        Detect instances of a named object in an image using Gemini’s vision capabilities.

        Args:
            image (PIL.Image.Image): The original input image.
            object_name (str): The object class name to detect (e.g., 'cat', 'car').

        Returns:
            dict[str, BoundingBox2D]: Mapping from object label to its bounding box.
        """
        # Resize the image for efficient API processing
        target_h, target_w = self._image_size
        resized = image.resize((target_w, target_h))

        # Construct the query: pass image and a textual instruction for detection
        query = [
            resized,
            (
                f"""
                Analyze the following image and provide the bounding\
                box of the [{object_name}]. Bounding boxes should be\
                in the format [ymin, xmin, ymax, xmax]. Additional\
                notes:\n
                * Please ensure the coordinates are relative to\
                  the original image size.\n
                * If an object is partially out of frame, estimate\
                  the bounding box as best as possible.\n
                * Return your answer as a single dict object where\
                  each key is an object name and each value is the\
                  corresponding bounding box coordinates. For example,\
                  {object_name}: [ymin, xmin, ymax, xmax]. Do not\
                  use Markdown. where H and W are the original image\
                  size.\n
                * If there are mutiple same object, you can name\
                  the the second object as {object_name}_2.
                  """)
        ]

        # Ask Gemini and capture its raw response string
        raw_bbox = self.ask_gemini(query)

        # Extract the JSON-like dict substring from first '{' to matching '}'
        start = raw_bbox.find('{')
        end = raw_bbox.rfind('}') + 1
        bbox_str = raw_bbox[start:end]

        # Safely evaluate the string into a Python dict
        bbox_coords = ast.literal_eval(bbox_str)

        # Convert normalized coords back to original image dimensions
        orig_w, orig_h = image.size
        result: dict[str, BoundingBox2D] = {}
        for label, coords in bbox_coords.items():
            ymin, xmin, ymax, xmax = coords
            # Scale values (assuming Gemini returned percent*10, e.g., out of 1000)
            x_min = int(xmin * orig_w / 1000)
            x_max = int(xmax * orig_w / 1000)
            y_min = int(ymin * orig_h / 1000)
            y_max = int(ymax * orig_h / 1000)
            result[label] = BoundingBox2D(y_min, x_min, y_max, x_max)

        return result
