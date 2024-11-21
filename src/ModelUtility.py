import pickle
import os


def save_model(model, file_path):
    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the model
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {file_path}")

def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {file_path}")
    return model


def save_custom_trainer(trainer, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(trainer, file)
    print(f"Custom Trainer saved to {file_path}")

# Load Custom Trainer model
def load_custom_trainer(file_path):
    with open(file_path, 'rb') as file:
        trainer = pickle.load(file)
    print(f"Custom Trainer loaded from {file_path}")
    return trainer
