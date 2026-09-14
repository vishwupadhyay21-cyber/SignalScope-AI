import pillow_heif
pillow_heif.register_heif_opener()

from PIL import Image
from model.inference import SignalScopeDetector

def test_heic():
    detector = SignalScopeDetector()
    path = r"E:\iphone photo\202401_a\IMG_0651.HEIC"

    with Image.open(path) as img:
        res = detector.predict(img)
        print("=" * 60)
        print("         IPHONE HEIC IMAGE PREDICTION RESULT")
        print("=" * 60)
        print("Image File:      ", path)
        print("Dimensions:      ", img.size)
        print("Verdict:         ", res['verdict']['label'])
        print(f"Confidence:       {res['verdict']['confidence']*100:.2f}%")
        print(f"Real Probability: {res['probabilities']['real']*100:.2f}%")
        print(f"AI Probability:   {res['probabilities']['ai_generated']*100:.2f}%")
        print("=" * 60)
        if "explanation" in res:
            print("\nEXPLAINABILITY CUES (Grad-CAM):")
            print(" Summary:", res["explanation"]["summary"])
            for cue in res["explanation"]["cues"]:
                print("  •", cue)
        print("=" * 60 + "\n")

if __name__ == "__main__":
    test_heic()
