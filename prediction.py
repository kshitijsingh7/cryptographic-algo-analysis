import numpy as np
import pandas as pd
import binascii
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import re

# Function to extract features from ciphertext
def extract_features(ciphertext):
    # Remove any whitespace from the ciphertext
    ciphertext = re.sub(r'\s+', '', ciphertext)

    # Ensure the ciphertext has an even length
    if len(ciphertext) % 2 != 0:
        ciphertext = '0' + ciphertext

    # Convert hex back to bytes
    ciphertext_bytes = binascii.unhexlify(ciphertext)

    # Byte distribution
    byte_distribution = np.bincount(np.frombuffer(ciphertext_bytes, dtype=np.uint8), minlength=256)

    # Normalized byte distribution
    normalized_distribution = byte_distribution / byte_distribution.sum()

    # Entropy
    entropy_value = entropy(normalized_distribution, base=2)

    # Statistical features
    mean = np.mean(byte_distribution)
    std_dev = np.std(byte_distribution)
    max_value = np.max(byte_distribution)
    min_value = np.min(byte_distribution)

    # Additional features
    ciphertext_length = len(ciphertext_bytes)
    first_byte = ciphertext_bytes[0]
    last_byte = ciphertext_bytes[-1]
    unique_bytes = len(set(ciphertext_bytes))

    # Combine all features into a single array
    features = np.concatenate((
        normalized_distribution,
        [entropy_value, mean, std_dev, max_value, min_value,
         ciphertext_length, first_byte, last_byte, unique_bytes, 0, 0]  # Key and IV length are unknown
    ))

    return features


# Preprocess the input CSV for prediction
def preprocess_input(input_file):
    # Load the ciphertext data
    df = pd.read_csv(input_file)

    # Extract features from ciphertexts
    features = df['ciphertext'].apply(extract_features).tolist()

    # Convert to numpy array
    X = np.array(features)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


# Make predictions and save the results
def predict_and_save(input_file, output_file='predictions.csv'):
    # Load the trained model and label encoder classes
    model = load_model('best_model.keras')
    label_encoder_classes = np.load('label_encoder_classes.npy', allow_pickle=True)

    # Preprocess input data
    X = preprocess_input(input_file)

    # Predict using the loaded model
    predictions = model.predict(X)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels = label_encoder_classes[predicted_classes]

    # Save predictions to a CSV file
    df = pd.DataFrame({'ciphertext': pd.read_csv(input_file)['ciphertext'], 'predicted_algorithm': predicted_labels})
    df.to_csv(output_file, index=False)

    # Print predictions to the terminal
    print(df)


if __name__ == "__main__":
    predict_and_save('sample_dataset.csv')
