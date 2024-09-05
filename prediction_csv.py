import numpy as np
import pandas as pd
import joblib
from scipy.stats import entropy


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
    reverse_encoder = {v: k for k, v in label_encoder.items()}
    return reverse_encoder[prediction[0]]


# Predict algorithms for ciphertexts in a CSV file and print results
def predict_from_csv(input_csv, model, label_encoder):
    # Load the input CSV file
    df = pd.read_csv(input_csv)

    if 'ciphertext' not in df.columns:
        raise ValueError("CSV file must contain a 'ciphertext' column")

    # Remove spaces and convert ciphertext from hex to bytes
    df['ciphertext'] = df['ciphertext'].apply(lambda x: bytes.fromhex(x.replace(" ", "")))

    # Predict algorithm for each ciphertext and print results
    df['predicted_algorithm'] = df['ciphertext'].apply(lambda x: predict_algorithm(x, model, label_encoder))

    for index, row in df.iterrows():
        print(f"Ciphertext {index + 1}: Predicted algorithm - {row['predicted_algorithm']}")


if __name__ == "__main__":
    # Load the trained model and label encoder
    model, label_encoder = load_model_and_encoder('model.joblib', 'label_encoder.joblib')

    # Specify the input CSV with ciphertexts
    input_csv = 'sample_dataset.csv'  # Path to your input CSV file with ciphertexts

    # Perform predictions and print results
    predict_from_csv(input_csv, model, label_encoder)
