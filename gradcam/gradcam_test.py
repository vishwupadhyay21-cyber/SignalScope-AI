import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

# -----------------------------
# 1. Device
# -----------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# -----------------------------
# 2. Load ResNet50 architecture
# -----------------------------
model = models.resnet50(weights=None)

# 2 classes
model.fc = nn.Linear(
    model.fc.in_features,
    2
)

# -----------------------------
# 3. Load YOUR trained model
# -----------------------------
model.load_state_dict(
    torch.load(
        "resnet50_real_ai.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Trained model loaded!")

# -----------------------------
# 4. Image
# -----------------------------
# Change this filename if needed
image_path = "dataset/real/real1.png"

image = Image.open(image_path).convert("RGB")

# -----------------------------
# 5. Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

input_tensor = transform(
    image
).unsqueeze(0).to(device)

# -----------------------------
# 6. Prediction
# -----------------------------
with torch.no_grad():
    output = model(input_tensor)

probabilities = torch.softmax(
    output,
    dim=1
)

predicted_class = output.argmax(
    dim=1
).item()

# ImageFolder alphabetical order:
# 0 = ai
# 1 = real
classes = ["AI-GENERATED", "REAL"]

confidence = probabilities[
    0,
    predicted_class
].item() * 100

print()
print("Prediction:", classes[predicted_class])
print(f"Confidence: {confidence:.2f}%")

# -----------------------------
# 7. Grad-CAM
# -----------------------------
target_layers = [
    model.layer4[-1]
]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)

targets = [
    ClassifierOutputTarget(
        predicted_class
    )
]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)[0]

# -----------------------------
# 8. Original image
# -----------------------------
rgb_image = np.array(
    image.resize((224, 224))
).astype(np.float32) / 255.0

# -----------------------------
# 9. Heatmap
# -----------------------------
visualization = show_cam_on_image(
    rgb_image,
    grayscale_cam,
    use_rgb=True
)

# -----------------------------
# 10. Save
# -----------------------------
output_path = "trained_gradcam_result.jpg"

Image.fromarray(
    visualization
).save(output_path)

print()
print("Grad-CAM saved to:")
print(output_path)
