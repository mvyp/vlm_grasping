## Installation
Create a Python vistual environment.
```sh
python -m venv ~/venvs/vlm
```

Install [Segment Anything Model 2 (SAM 2)](https://github.com/facebookresearch/sam2)
```sh
git clone https://github.com/facebookresearch/sam2.git
cd sam2

# Make sure installing SAM 2 in the Python virtual environment.
source ~/venvs/vlm/bin/activate
pip install -e .

# Download checkpoints
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

Install this package
```sh
# Make sure installing dependencies in the Python virtual environment.
source ~/venvs/vlm/bin/activate

# Install Dependencies
pip install -r requirements.txt
```

## Demo
Before running the demo, setup `<your-google-gemini-api-key>` and `<path-to-sam2>` in `demo.py`:
```python
# Set the API key for Google Generative AI
vlm = GeminiInterface(api_key="<your-google-gemini-api-key>")
# Initialize SAM2
sam2 = SAM2Interface(checkpoint_path="<path-to-sam2>/checkpoints/sam2.1_hiera_large.pt",
                     model_config_path="configs/sam2.1/sam2.1_hiera_l.yaml")
```

To run the demo, simply run `demo.py`
```sh
# Make sure using the Python virtual environment.
source ~/venvs/vlm/bin/activate
python demo.py
```
