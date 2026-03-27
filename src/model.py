"""
Iris Flower Classifier — Demo ML Model
Used to teach CI/CD pipeline concepts in ML projects.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Class names for human-readable output
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def load_data():
    """Load the Iris dataset and return X, y."""
    iris = load_iris()
    return iris.data, iris.target


def preprocess(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train(X_train, y_train, max_iter=200, C=1.0):
    """Train a Logistic Regression classifier."""
    model = LogisticRegression(max_iter=max_iter, C=C, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    """Return accuracy and full classification report."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    return acc, report


def predict(model, scaler, features: list):
    """
    Predict a single sample.
    features: list of 4 floats [sepal_length, sepal_width, petal_length, petal_width]
    Returns predicted class name.
    """
    x = np.array(features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    return CLASS_NAMES[pred]


def save_model(model, scaler, path="models/"):
    """Save model and scaler to disk."""
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, "model.pkl"))
    joblib.dump(scaler, os.path.join(path, "scaler.pkl"))
    print(f"✅ Model saved to {path}")


def load_model(path="models/"):
    """Load model and scaler from disk."""
    model = joblib.load(os.path.join(path, "model.pkl"))
    scaler = joblib.load(os.path.join(path, "scaler.pkl"))
    return model, scaler


if __name__ == "__main__":
    print("🌸 Training Iris Classifier...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_s, X_test_s, scaler = preprocess(X_train, X_test)
    model = train(X_train_s, y_train)
    acc, report = evaluate(model, X_test_s, y_test)

    print(f"\n📊 Accuracy: {acc:.4f}")
    print(report)

    save_model(model, scaler)
