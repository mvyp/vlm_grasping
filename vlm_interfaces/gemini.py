import google.generativeai as genai
import numpy as np
from PIL import Image
import ast

from utils.utils import BoundingBox2D

class GeminiInterface:
    def __init__(self,
                 api_key: str,
                 model_name: str = "gemini-1.5-pro-latest",
                 temperature: float = 0.0,
                 max_tokens: int = 2048,
                 image_size: list = [512, 512]):
        # Configure the API key for Gemini
        genai.configure(api_key=api_key)

        self._safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
                },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
                },
            ]
        self._generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            }
        self._gemini_model = genai.GenerativeModel(
            model_name = model_name,
            safety_settings = self._safety_settings
            )
        self._image_size = image_size
    
    def ask_gemini(self,
                           prompt_elements: list):
        parts = []
        for prompt_element in prompt_elements:
            parts.append(prompt_element)
        messages = [{"role": "user", "parts": parts}]
        response = self._gemini_model.generate_content(
            messages,
            generation_config = self._generation_config,
            )
        print(response.prompt_feedback)
        return response.text
    
    def detect_object(self,
                      image: Image.Image,
                      object_name: str):
        # resize image to the specified size for faster processing
        H, W = self._image_size
        resized_image = image.resize((H, W))

        # ask Gemini to detect the object and provide the bounding boxes
        query_list = [resized_image]
        query_list.append(f"""
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
        bbox_str = self.ask_gemini(query_list)
        for i in range(0, len(bbox_str)):
            if bbox_str[i] == '{':
                bbox_str = bbox_str[i:]
                break
        for i in range(0, len(bbox_str)):
            if bbox_str[i] == '}':
                bbox_str = bbox_str[:(i+1)]
                break
        bbox_dict = ast.literal_eval(bbox_str)

        # resize the bounding box coordinates to the original image size
        _actual_width, _actual_height = image.size
        for label, bbox in bbox_dict.items():
            y_min, x_min, y_max, x_max = bbox
            x_min = int(x_min * _actual_width / 1000)
            x_max = int(x_max * _actual_width / 1000)
            y_min = int(y_min * _actual_height / 1000)
            y_max = int(y_max * _actual_height / 1000)
            bbox = BoundingBox2D(y_min, x_min, y_max, x_max)
            bbox_dict[label] = bbox

        return bbox_dict
