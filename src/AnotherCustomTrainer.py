import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from ModelUtility import save_custom_trainer
from collections import Counter


class AnotherCustomTrainer:
    def __init__(self):
        self.class_metrics = {}  # To store metric values (mean, median, etc.) for each class

    def fit(self, features, labels):
        """
        Train the model by calculating metrics for each class.
        """
        self.class_metrics = {}
        unique_labels = set(labels)

        for label in unique_labels:
            # Filter features by class label
            class_features = [features[i] for i in range(len(features)) if labels[i] == label]
            
            # Convert to a NumPy array for easier manipulation
            class_features = np.array(class_features)
            
            # Calculate metrics (mean, median, std dev, min, max) for the class
            self.class_metrics[label] = {
                "mean": np.mean(class_features, axis=0),
                "median": np.median(class_features, axis=0),
                "std": np.std(class_features, axis=0),
                "min": np.min(class_features, axis=0),
                "max": np.max(class_features, axis=0),
            }

    def predict(self, feature_vector):
        """
        Predict the class of a single feature vector using majority voting from multiple metrics.
        """
        predictions = []

        for label, metrics in self.class_metrics.items():
            # Compare the feature vector to each metric
            distances = {
                "mean": np.linalg.norm(feature_vector - metrics["mean"]),
                "median": np.linalg.norm(feature_vector - metrics["median"]),
                "std": np.linalg.norm(feature_vector - metrics["std"]),
                "min": np.linalg.norm(feature_vector - metrics["min"]),
                "max": np.linalg.norm(feature_vector - metrics["max"]),
            }
            
            # Predict the label for each metric (label with the smallest distance)
            for metric, distance in distances.items():
                predictions.append((metric, label, distance))
        
        # Sort predictions by distance for each metric
        sorted_predictions = sorted(predictions, key=lambda x: x[2])

        # Take the top prediction for each metric
        metric_votes = [pred[1] for pred in sorted_predictions[:len(self.class_metrics)]]

        # Majority voting
        final_prediction = Counter(metric_votes).most_common(1)[0][0]
        return final_prediction

    def evaluate(self, features, labels):
        """
        Evaluate the model on a test set.
        """
        predictions = [self.predict(feature) for feature in features]

    # Print classification report
        print("\nClassification Report:")
        print(classification_report(labels, predictions))

        # Calculate accuracy
        correct = sum(1 for i in range(len(labels)) if predictions[i] == labels[i])
        accuracy = correct / len(labels)
        #print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy
    

def load_data_from_json(folder_path, n_first=0, n_last=0):
        features = []
        labels = []

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.json'):
                file_path = os.path.join(folder_path, file_name)

                base_name = os.path.splitext(file_name)[0]  # Remove .json extension
                label = '_'.join(base_name.split('_')[:2])  # Taking only the first two words

                with open(file_path, 'r') as file:
                    data = json.load(file)
                    
                    for image_name, parts_data in data.items():
                        # putting all the parts of the image frequency vectors into one feature vector
                        image_features = []
                        for part_key, freq_vector in parts_data.items():
                            trimmed_vector = freq_vector[n_first:len(freq_vector) - n_last if n_last > 0 else None]
                            image_features.extend(trimmed_vector)
                        
                        features.append(image_features)
                        labels.append(label)

        return features, labels


folder_path = "C://Users//Trust_pc_dz//Documents//IMED//DATASET//Frequencies//Data"
#trainer = CustomTrainer()
features, labels = load_data_from_json(folder_path, 1, 0)


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train custom trainer
trainer = AnotherCustomTrainer()
trainer.fit(X_train, y_train)

# Evaluate model
accuracy = trainer.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")