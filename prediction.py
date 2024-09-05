import numpy as np
import joblib  # Use `import joblib` if `sklearn.externals` is deprecated
from scipy.stats import entropy
import os

# Calculate entropy of data
def calculate_entropy(data):
    probability_distribution = np.bincount(data) / len(data)
    return entropy(probability_distribution, base=2)

# Extract features from ciphertext
def extract_features(data):
    features = []
    for item in data:
        item_bytes = np.frombuffer(item, dtype=np.uint8)
        features.append([len(item_bytes), calculate_entropy(item_bytes)])
    return np.array(features)

# Load the trained model and label encoder
def load_model_and_encoder(model_filename='model.joblib', encoder_filename='label_encoder.joblib'):
    model = joblib.load(model_filename)
    label_encoder = joblib.load(encoder_filename)
    return model, label_encoder

# Predict the algorithm for a given ciphertext
def predict_algorithm(ciphertext, model, label_encoder):
    features = extract_features([ciphertext])
    prediction = model.predict(features)
    # Reverse mapping from numeric label to algorithm name
    reverse_encoder = {v: k for k, v in label_encoder.items()}
    return reverse_encoder[prediction[0]]

if __name__ == "__main__":
    # Load the trained model and label encoder
    model, label_encoder = load_model_and_encoder('model.joblib', 'label_encoder.joblib')

    # Example ciphertexts (replace these with actual ciphertexts you want to test)
    test_ciphertexts = [
        os.urandom(64),  # Simulated ciphertext for testing
        os.urandom(64),  # Simulated ciphertext for testing
        os.urandom(64)   # Simulated ciphertext for testing
    ]

    # Predict and display results
    for i, ciphertext in enumerate(test_ciphertexts):
        prediction = predict_algorithm(ciphertext, model, label_encoder)
        print(f"Ciphertext {i+1}: Predicted algorithm - {prediction}")
