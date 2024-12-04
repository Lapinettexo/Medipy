import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from ModelUtility import save_custom_trainer

class CustomTrainer:
    def __init__(self):
        self.class_means = {}


    def calculate_means(self, X, y):
        
        class_data = {}
        
        # Group frequency vectors by class
        for features, label in zip(X, y):
            if label not in class_data:
                class_data[label] = []
            class_data[label].append(features)
        
        # Calculate mean vector for each class
        self.class_means = {label: np.mean(data, axis=0) for label, data in class_data.items()}
    
    def predict(self, X):
        
        predictions = []
        for features in X:
            # Calculate distances to each class mean
            distances = {label: np.linalg.norm(features - mean) for label, mean in self.class_means.items()}
            
            # Assign the class with the smallest distance
            predicted_label = min(distances, key=distances.get)
            predictions.append(predicted_label)
        
        return predictions
    

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
features, labels = load_data_from_json(folder_path, 1, 235)

#trainer.calculate_means(features, labels)

# Print calculated means
#print("Class Means:", trainer.class_means)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

trainer = CustomTrainer()
trainer.calculate_means(X_train, y_train)

# Predict on testing data
predictions = trainer.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2%}")

print("Classification Report:\n", classification_report(y_test, predictions))

#rf_model_path = "./Models/custom_model_10_5_1.pkl"
#save_custom_trainer(trainer, rf_model_path)