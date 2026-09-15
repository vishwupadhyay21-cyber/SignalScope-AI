# SignalScope

### AI-Powered Media Forensics & Image Authenticity Detection

SignalScope is an AI-powered media forensics platform designed to analyze digital images and estimate whether they are real or AI-generated.

## 🚨 Problem

Generative AI can create highly realistic images, making it increasingly difficult to distinguish authentic media from synthetic content.

This creates challenges for:
- Journalists
- Fact-checkers
- Investigators
- Digital platforms
- Everyday users

## 💡 Solution

SignalScope analyzes an uploaded image and provides:

- Real / AI-generated prediction
- Confidence score
- Visual forensic analysis
- AI-generated image indicators
- Explainable analysis using Grad-CAM
- Privacy-focused image processing

## ✨ Key Features

### 1. AI Image Detection
Upload an image and SignalScope predicts whether it is likely real or AI-generated.

### 2. Confidence Score
The system provides a confidence score along with the prediction.

### 3. Explainable AI
Grad-CAM based visualization helps identify regions that influenced the model's prediction.

### 4. Forensic Analysis
The system analyzes visual characteristics and other available image information to provide additional forensic insights.

### 5. Privacy
Images are processed for analysis without unnecessary persistent storage.

## 🏗️ Architecture

```text
User
  ↓
SignalScope Frontend
  ↓
Image Upload
  ↓
Backend API
  ↓
Preprocessing
  ↓
AI / Computer Vision Model
  ↓
Prediction + Confidence
  ↓
Forensic Analysis
  ↓
Grad-CAM Explanation
  ↓
Result Dashboard

## 🎥 Project Demo

https://drive.google.com/file/d/1M9TGCaRGB0K1hLCXnsxAIo1ETQI0BpJP/view?usp=drive_link