import os
from PIL import Image
from model.inference import SignalScopeDetector

detector = SignalScopeDetector()

test_images = [
    (r"D:\my_dataset\real\0000.jpg", "Real Photo #1"),
    (r"D:\my_dataset\real\4988.jpg", "Real Photo #2"),
    (r"D:\my_dataset\AI-generated\1000.jpg", "AI Image #1"),
    (r"D:\my_dataset\AI-generated\1000 (5).jpg", "AI Image #2"),
    (r"D:\my_dataset\AI-generated\4990.jpg", "AI Image #3"),
]

print("=" * 70)
print("       LIVE MULTI-IMAGE FORENSIC PREDICTION TEST")
print("=" * 70)

for path, desc in test_images:
    if os.path.exists(path):
        with Image.open(path) as img:
            res = detector.predict(img)
            folder = os.path.basename(os.path.dirname(path))
            fname = os.path.basename(path)
            print(f"Testing {desc} [{folder}/{fname}]:")
            print(f"  -> Predicted Verdict:  {res['verdict']['label']}")
            print(f"  -> Real Probability:   {res['probabilities']['real']*100:.2f}%")
            print(f"  -> AI Gen Probability: {res['probabilities']['ai_generated']*100:.2f}%")
            print("-" * 70)

print("=" * 70 + "\n")
