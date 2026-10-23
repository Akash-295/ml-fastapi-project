
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Load data and its metadata
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build and train the pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000))
])

pipeline.fit(X_train, y_train)

# Evaluate on the held-out test set
predictions = pipeline.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("F1 score:", round(f1_score(y_test, predictions), 4))

# Save model and metadata together
artifact = {
    "model": pipeline,
    "feature_names": list(X.columns),
    "target_names": data.target_names.tolist()
}

joblib.dump(artifact, "model_artifact.joblib")

print("Saved model_artifact.joblib")
print("Number of expected features:", len(X.columns))
