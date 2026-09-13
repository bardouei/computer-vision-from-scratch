
# Computer Vision From Scratch

A portfolio repository containing three computer-vision projects built and trained from scratch using TensorFlow and OpenCV.

## Projects

| Project | Task | Description |
|---|---|---|
| `01-face-detection` | Object Detection | Detects faces in images and live webcam video by drawing bounding boxes around each face. |
| `02-pose-coach` | Pose Estimation | Detects human body keypoints and analyzes exercise form, such as squat repetition counting and knee-angle feedback. |
| `03-road-defect-segmentation` | Semantic Segmentation | Identifies road cracks and potholes pixel by pixel to support road-condition monitoring. |

## Technologies

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

## Repository Structure

```text
computer-vision-from-scratch/
├── 01-face-detection/
│   ├── data/
│   ├── models/
│   ├── notebooks/
│   ├── src/
│   ├── outputs/
│   └── README.md
├── 02-pose-coach/
│   ├── data/
│   ├── models/
│   ├── notebooks/
│   ├── src/
│   ├── outputs/
│   └── README.md
├── 03-road-defect-segmentation/
│   ├── data/
│   ├── models/
│   ├── notebooks/
│   ├── src/
│   ├── outputs/
│   └── README.md
└── README.md
```

## Goals

- Build computer-vision models without relying on pretrained model weights.
- Prepare and augment image datasets.
- Train and evaluate models using suitable metrics.
- Deploy each model on images, videos, or live webcam input.
- Document limitations and improvements for future versions.

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/computer-vision-from-scratch.git
cd computer-vision-from-scratch
pip install -r requirements.txt
```

## Results

Each project includes its own README with:

- Dataset details
- Model architecture
- Training process
- Evaluation metrics
- Sample predictions
- Instructions for running the project

## Author

Your Name  
GitHub: [YOUR-USERNAME](https://github.com/YOUR-USERNAME)
