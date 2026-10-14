
import sys
import joblib
import sklearn
from pathlib import Path

print("=" * 50)
print("HOUSE PRICE MODEL DIAGNOSTICS")
print("=" * 50)

print("Python version:", sys.version)
print("Scikit-learn version:", sklearn.__version__)

model_path = Path(__file__).resolve().parent / "house_price_predict.joblib"

print("Model path:", model_path)
print("Model file exists:", model_path.exists())

if not model_path.exists():
    print("ERROR: Model file not found!")
    sys.exit(1)

print("\nAttempting to load model...")

try:
    artifact = joblib.load(model_path)

    print("\nSUCCESS: Model loaded successfully!")
    print("Artifact type:", type(artifact))

    if isinstance(artifact, dict):
        print("Artifact keys:", list(artifact.keys()))

        model = artifact.get("model")

        if model is not None:
            print("Model type:", type(model))
            print("Has predict method:", hasattr(model, "predict"))

        for key in ("feature", "features", "feature_columns"):
            if key in artifact:
                print(f"{key}:", artifact[key])

    else:
        print("Has predict method:", hasattr(artifact, "predict"))

except Exception as error:
    print("\nFAILED: Could not load model.")
    print("Error type:", type(error).__name__)
    print("Error message:", str(error))
    print("\nDo not retrain or delete the model yet.")
