import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib  # Use `import joblib` if `sklearn.externals` is deprecated
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

# Load dataset from CSV and extract features
def load_and_extract_features(filename='cryptography_dataset.csv'):
    df = pd.read_csv(filename)
    df['ciphertext'] = df['ciphertext'].apply(bytes.fromhex)
    features = extract_features(df['ciphertext'])
    return features, df['algorithm'].values

# Train and save model
def train_and_save_model(features, labels, model_filename='model.joblib', encoder_filename='label_encoder.joblib'):
    label_encoder = {label: idx for idx, label in enumerate(set(labels))}
    y = np.array([label_encoder[label] for label in labels])
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_encoder.keys()))
    joblib.dump(model, model_filename)
    joblib.dump(label_encoder, encoder_filename)
    print(f"Model saved to {model_filename}.")
    print(f"Label encoder saved to {encoder_filename}.")

if __name__ == "__main__":
    features, labels = load_and_extract_features('cryptography_dataset.csv')
    train_and_save_model(features, labels)
