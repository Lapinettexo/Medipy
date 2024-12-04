import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from ModelUtility import save_model

# Step 1: Loading the data
def load_data_from_json(folder_path):
    features = []
    labels = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)

            
            base_name = os.path.splitext(file_name)[0]  # Removing .json extension
            label = '_'.join(base_name.split('_')[:2])  # Taking only the first two words

            with open(file_path, 'r') as file:
                data = json.load(file)
                
                for image_name, parts_data in data.items():
                    # putting all the parts of the image frequency vectors into one feature vector
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
clf = RandomForestClassifier(n_estimators=120, random_state=42)
clf.fit(X_train, y_train)

# Step 5: model eval
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)  # Full probabilities for multi-class AUC-ROC

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#rf_model_path = "./Models/random_forest_model_120_42_4.pkl"
#save_model(clf, rf_model_path)


