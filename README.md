# advertima-task

## Installation

Create a virtualenvironment to work in:
```
python3 -m venv venv
source venv/bin/activate
```
Install dependencies:
```
pip install -r requirements.txt
```

## Run training

Change hyperparameters in train.py and network architecture in networks.py. Models and training files will be stored in models directory.

## Run object detection

To run object detection pipeline first we will need a pretrained classifier:
```
usage: detection.py [-h] --root_dir ROOT_DIR --target_dir TARGET_DIR
                    --frame_ext FRAME_EXT --model_path MODEL_PATH

optional arguments:
  -h, --help            show this help message and exit
  --root_dir ROOT_DIR   Directory containing frames.
  --target_dir TARGET_DIR
                        Directory containing frames.
  --frame_ext FRAME_EXT
                        Frame extension.
  --model_path MODEL_PATH
                        Path to the pretrained classifier model
```
