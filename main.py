import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import time

# For PMML export
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline


class DDoSDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            "source_request_rate",
            "system_request_rate",
            "payload_size",
            "cpu_demand",
            "bw_demand",
        ]

    def load_data(self, file_path):
        """Load and prepare dataset from CSV file"""
        # Assume CSV has no header, so we provide column names
        columns = self.feature_names + ["is_attack"]
        data = pd.read_csv(file_path, header=None, names=columns)

        # Split features and target
        X = data[self.feature_names]
        y = data["is_attack"]

        return X, y

    def train(self, X, y, test_size=0.3, random_state=42):
        """Train the DDoS detection model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Standardize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model - using Random Forest as it handles this type of data well
        start_time = time.time()
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Model trained in {training_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        return {
            "accuracy": accuracy,
            "training_time": training_time,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    def predict(self, input_data):
        """Make predictions on new data"""
        if isinstance(input_data, pd.DataFrame):
            # If input is a DataFrame, ensure it has the correct features
            input_data = input_data[self.feature_names]
        elif isinstance(input_data, (list, np.ndarray)):
            # Convert list to appropriate format
            if len(np.array(input_data).shape) == 1:
                # Single sample
                input_data = np.array(input_data).reshape(1, -1)
            else:
                input_data = np.array(input_data)

        # Scale the input data
        scaled_data = self.scaler.transform(input_data)

        # Perform prediction
        start_time = time.time()
        predictions = self.model.predict(scaled_data)
        prediction_time = time.time() - start_time

        # For a batch of 500 requests
        print(f"Prediction time for this batch: {prediction_time:.5f} seconds")
        print(
            f"Average prediction time per request: {prediction_time/len(input_data):.7f} seconds"
        )

        return predictions

    def save_model(self, model_path="ddos_model.joblib"):
        """Save the model to disk using joblib"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Save with joblib (Python only)
        joblib.dump({"model": self.model, "scaler": self.scaler}, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path="ddos_model.joblib"):
        """Load a saved model from disk"""
        saved_model = joblib.load(model_path)
        self.model = saved_model["model"]
        self.scaler = saved_model["scaler"]
        print(f"Model loaded from {model_path}")

    def export_pmml(self, export_path="ddos_model.pmml"):
        """Export model to PMML format for Java compatibility"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet")

        # Create a PMML pipeline with preprocessing and model
        pipeline = PMMLPipeline([("scaler", self.scaler), ("classifier", self.model)])

        # We need to fit the pipeline with some data
        # Create a small dummy dataset with the right features
        dummy_X = pd.DataFrame(
            np.zeros((1, len(self.feature_names))), columns=self.feature_names
        )
        dummy_y = pd.Series([0])

        # Fit the pipeline with dummy data (since our components are already fitted)
        pipeline.fit(dummy_X, dummy_y)

        # Export to PMML
        sklearn2pmml(pipeline, export_path)
        print(f"Model exported to PMML format at {export_path}")


# Example usage
if __name__ == "__main__":
    # Create sample dataset file from your examples
    sample_data = """0.77,29.85,1513,0.28,0.22,0
0.45,19.38,1471,0.57,0.20,0
0.44,18.45,1305,0.53,0.13,0
0.99,23.08,1654,0.60,0.16,0
0.65,23.09,1546,0.48,0.08,0
0.62,29.84,2058,0.51,0.12,0
1.73,24.49,1709,0.42,0.17,0
0.35,15.32,1431,0.41,0.11,0
1.54,17.99,1780,0.45,0.08,0
0.82,18.30,996,0.30,0.08,0
0.34,18.41,965,0.59,0.23,0
1.73,26.52,1984,0.36,0.09,0
0.59,20.64,1082,0.50,0.22,0
1.25,12.12,777,0.39,0.07,0
0.58,25.19,1955,0.52,0.22,0
19.36,118.30,10443,0.95,0.60,1
13.31,120.51,9890,0.95,0.96,1
22.13,98.54,3211,0.81,0.64,1
10.78,113.78,8736,0.63,0.86,1
7.60,72.33,9949,0.80,0.86,1
22.42,127.13,8490,0.66,0.48,1
17.72,82.93,5882,0.70,0.48,1
19.04,111.87,4509,0.68,0.99,1
12.86,50.38,6228,0.87,0.54,1
8.83,129.11,3726,0.60,0.38,1
5.95,38.78,9333,0.79,0.86,1
18.25,118.65,6956,0.93,0.96,1
8.06,126.00,2278,0.73,0.86,1
5.24,107.87,10546,0.67,0.84,1
20.41,101.87,8710,0.98,0.65,1"""

    with open("sample_ddos_data.csv", "w") as f:
        f.write(sample_data)

    # Create and train model
    detector = DDoSDetectionModel()
    X, y = detector.load_data("sample_ddos_data.csv")

    # Train the model
    results = detector.train(X, y)

    # Save the model in different formats
    detector.save_model()

    try:
        # These might fail if libraries aren't installed
        detector.export_pmml()
        detector.export_onnx()
    except Exception as e:
        print(f"Export failed: {e}")
        print("Make sure you have sklearn2pmml and skl2onnx installed.")

    # Test with some sample attack data
    test_data = np.array(
        [
            [0.6, 25.0, 1500, 0.3, 0.2],  # Normal traffic
            [18.0, 100.0, 8000, 0.9, 0.7],  # Attack traffic
        ]
    )

    predictions = detector.predict(test_data)
    print(f"Predictions: {predictions}")

    # Simulate batch processing (500 requests)
    import numpy as np

    batch_size = 500
    # Generate random data within the ranges seen in the dataset
    batch_data = np.random.rand(batch_size, 5)
    # Scale to appropriate ranges
    batch_data[:, 0] = batch_data[:, 0] * 25  # source_request_rate
    batch_data[:, 1] = batch_data[:, 1] * 130 + 10  # system_request_rate
    batch_data[:, 2] = batch_data[:, 2] * 10000 + 700  # payload_size
    batch_data[:, 3] = batch_data[:, 3] * 0.7 + 0.2  # cpu_demand
    batch_data[:, 4] = batch_data[:, 4] * 0.9 + 0.05  # bw_demand

    # Measure batch prediction performance
    batch_predictions = detector.predict(batch_data)
    print(f"Made {batch_size} predictions")
