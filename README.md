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

## Run frame object detection

To run object detection pipeline first we will need a pretrained classifier and a directory containing the frames:
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

## Run video object detection

To run object detection on video and store results in an output video file run:
```
usage: video_detection.py [-h] [--video_path VIDEO_PATH]
                          [--video_target_path VIDEO_TARGET_PATH]
                          [--model_path MODEL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --video_path VIDEO_PATH
                        Path to video file.
  --video_target_path VIDEO_TARGET_PATH
                        Path to output video file.
  --model_path MODEL_PATH
                        Path to the pretrained classifier model
```
