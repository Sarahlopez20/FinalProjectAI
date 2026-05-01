# Risk-Aware Navigation System

## Project Overview

This project implements a multimodal artificial intelligence system that estimates the danger level of road scenes.
The system analyzes visual, environmental, and optional audio signals to generate a risk score between 0 and 1, along with a classification
of the risk level as Low, Medium, or High.

The goal is to simulate an AI safety layer that could be integrated into navigation platforms such as Waze to provide safer route recommendations.

---

## Project Structure

ROAD FINAL PROJECT/

- config.py
- main.py
- requirements.txt
- README.md

src/
- pipeline.py
- scoring.py
- segmentation.py
- yolo_detection.py
- weather.py
- audio.py
- visualization.py

data/
- input_images/
- input_audio/

models/
- segmentation_model_improved.pth
- weather_model.pth
- weather_classes.npy
- yolov8n.pt

outputs/
- results.csv
- demo_images/
- predicted_masks/

---

## Installation

Open a terminal inside the project folder and run:

```bash
pip install -r requirements.txt
```

## How to Run
1: Add road images to:
data/input_images/
2: (Optional) Add audio file to:
data/input_audio/
3: Run the project:

```bash
python main.py
```

---

## Dataset Note

The full Cityscapes dataset is not included in this repository because of its large file size. The prototype can be run using the trained model files already included in the `models/` folder, together with the demo input images in `data/input_images/`.

If you want to download the original Cityscapes dataset used for segmentation training, it is available from the official Cityscapes website:

https://www.kaggle.com/datasets/electraawais/cityscape-dataset

The dataset may require account registration before downloading. The main Cityscapes folders used for training were `leftImg8bit` for the road images and `gtFine` for the segmentation annotations.
