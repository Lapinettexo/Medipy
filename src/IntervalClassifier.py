import os
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

class IntervalClassifier:
    def __init__(self):
        #self.use_median = use_median  # Whether to use median for pixel assignment
        self.class_intervals = {}  # Store min, max, and median for each class

    def fit(self, X, y):
        """
        Compute min, max, and optionally median for each pixel intensity
        across all images in each class.
        """
        class_data = {}
        for features, label in zip(X, y):
            if label not in class_data:
                class_data[label] = []
            class_data[label].append(features)
        
        # Calculate min, max, and median for each class
        for label, feature_vectors in class_data.items():
            feature_array = np.array(feature_vectors)  # Convert to NumPy array for easier manipulation
            min_vals = np.min(feature_array, axis=0)
            max_vals = np.max(feature_array, axis=0)
            median_vals = np.median(feature_array, axis=0)
            self.class_intervals[label] = {
                'min': min_vals,
                'max': max_vals,
                'median': median_vals
            }

    def predict(self, X):
        """
        Predict class labels based on pixel frequency assignments.
        """
        predictions = []
        for features in X:
            # Assign each pixel to a class based on min-max and median
            pixel_class_votes = {label: 0 for label in self.class_intervals.keys()}
            
            for i, pixel_value in enumerate(features):
                # Track ambiguous classes
                ambiguous_classes = []
                for label, intervals in self.class_intervals.items():
                    if intervals['min'][i] <= pixel_value <= intervals['max'][i]:
                        ambiguous_classes.append(label)

                if len(ambiguous_classes) == 1:
                    # Unambiguous case: Increment vote for the matching class
                    pixel_class_votes[ambiguous_classes[0]] += 1
                elif len(ambiguous_classes) > 1:
                    # Ambiguous case: Resolve using the closest median
                    closest_class = min(
                        ambiguous_classes,
                        key=lambda label: abs(pixel_value - self.class_intervals[label]['median'][i])
                    )
                    pixel_class_votes[closest_class] += 1

            # Predict class based on majority vote
            predicted_class = max(pixel_class_votes, key=pixel_class_votes.get)
            predictions.append(predicted_class)
        
        return predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate the classifier on test data and print classification metrics.
        """
        from sklearn.metrics import classification_report, accuracy_score

        y_pred = self.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))


# Load data from JSON with optional trimming
def load_data_from_json(folder_path, n_first=0, n_last=0, normalize=True):
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
                    # Concatenate all part frequency vectors into one feature vector
                    image_features = []
                    for part_key, freq_vector in parts_data.items():
                        trimmed_vector = freq_vector[n_first:len(freq_vector) - n_last if n_last > 0 else None]
                        image_features.extend(trimmed_vector)

                    if normalize:
                        # Normalize the frequencies to sum to 1
                        total = sum(image_features)
                        if total > 0:
                            image_features = [value / total for value in image_features]


                    features.append(image_features)
                    labels.append(label)

    return features, labels


# Example usage
folder_path = "C://Users//Trust_pc_dz//Documents//IMED//DATASET//Frequencies//16 Data"
features, labels = load_data_from_json(folder_path, 0, 0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train and evaluate the interval classifier
classifier = IntervalClassifier()
classifier.fit(X_train, y_train)
classifier.evaluate(X_test, y_test)
