import os
import joblib
import pandas as pd

# Path to the saved model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "titanic_model.pkl")

# Example passenger (edit these values to test different cases)
example = {
    "pclass": 3,
    "sex": "male",       # 'male' or 'female'
    "age": 22,
    "fare": 7.25,
    "family_size": 1,    # sibsp + parch + 1
    "is_alone": 1,       # 1 if alone else 0
    "embarked": "S"      # 'S', 'C', or 'Q' (most common ports)
}

def predict(example_dict):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run `python src/train.py` first.")
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([example_dict])[['pclass','sex','age','fare','family_size','is_alone','embarked']]
    pred = model.predict(X)[0]
    proba = getattr(model, "predict_proba", lambda x: None)(X)
    if proba is not None:
        proba = proba[0][1]
    return int(pred), (float(proba) if proba is not None else None)

if __name__ == "__main__":
    yhat, p = predict(example)
    if p is not None:
        print(f"Prediction: {'Survived' if yhat==1 else 'Did NOT Survive'}  |  Probability of survival: {p:.3f}")
    else:
        print(f"Prediction: {'Survived' if yhat==1 else 'Did NOT Survive'}")
