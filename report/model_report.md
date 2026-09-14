# SignalScope – AI-Generated Image Detection

## 1. Overview

SignalScope is a web-based AI-generated image detection system. It analyzes an uploaded image and predicts whether the image is real or AI-generated.

## 2. Technologies Used

- Python
- FastAPI
- PyTorch
- ResNet18
- React
- Grad-CAM
- Pillow
- JavaScript

## 3. Model Architecture

The system uses a ResNet18-based binary image classifier.

- Class 0: Real image
- Class 1: AI-generated image

The model returns a probability score indicating how likely the uploaded image is AI-generated.

## 4. System Workflow

1. The user uploads an image through the frontend.
2. The frontend sends the image to the FastAPI backend.
3. The backend preprocesses the image.
4. The trained ResNet18 model analyzes the image.
5. The system returns the predicted class and confidence score.
6. Grad-CAM provides visual evidence for the prediction.

## 5. Explainability

Grad-CAM is used to identify the image regions that influence the model's prediction. This helps users understand why the model classified an image as real or AI-generated.

## 6. Limitations

The prediction is probabilistic and should not be treated as conclusive forensic proof. Performance may vary depending on image quality, compression, unseen AI generators, and image manipulation.

## 7. Project Structure

```text
SignalScope-main/
├── backend/
├── frontend/
├── gradcam/
├── report/
│   └── model_report.md
├── README.md
├── requirements.txt
└── .gitignore