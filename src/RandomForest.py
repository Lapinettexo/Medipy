import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the data
def load_data_from_json(folder_path):
    features = []
    labels = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)

            # Extract label from the first two words of the file name
            base_name = os.path.splitext(file_name)[0]  # Remove .json extension
            label = '_'.join(base_name.split('_')[:2])  # Take the first two words

            with open(file_path, 'r') as file:
                data = json.load(file)
                
                # Flatten the nested dictionary structure
                for image_name, parts_data in data.items():
                    # Concatenate all part frequency vectors into a single feature vector
                    image_features = []
                    for part_key, freq_vector in parts_data.items():
                        image_features.extend(freq_vector)
                    
                    features.append(image_features)
                    labels.append(label)

    return features, labels

folder_path = "C://Users//Trust_pc_dz//Documents//IMED//DATASET//Frequencies//Data"
X, y = load_data_from_json(folder_path)

# Step 2: Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4:  Random Forest Classifier training
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 5: model eval
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)  # Full probabilities for multi-class AUC-ROC

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Calculate AUC-ROC per class (if multi-class)
if len(le.classes_) > 2:
    from sklearn.metrics import roc_auc_score
    aucs = []
    for i in range(len(le.classes_)):
        auc = roc_auc_score((y_test == i).astype(int), y_prob[:, i])
        aucs.append((le.classes_[i], auc))
    print("AUC-ROC Scores by Class:")
    for label, auc in aucs:
        print(f"{label}: {auc:.4f}")
else:
    auc_roc = roc_auc_score(y_test, y_prob[:, 1])
    print("AUC-ROC:", auc_roc)


