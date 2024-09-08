import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Data Preprocessing
def preprocess_data(filename='cipher_dataset.csv'):
    # Load dataset
    df = pd.read_csv(filename)

    # Separate features and labels
    X = df.drop(['algorithm', 'ciphertext'], axis=1).values
    y = df['algorithm'].values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Save label encoder classes for future use
    np.save('label_encoder_classes.npy', le.classes_)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, le

# Model Training
def train_model(X_train, X_test, y_train, y_test, label_encoder):
    # Build the model
    model = Sequential([
        Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('best_model.keras', save_best_only=True, save_weights_only=False)
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {test_accuracy:.4f}')

    # Predict and evaluate
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    return model

if __name__ == "__main__":
    # Preprocess the data
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data()

    # Train the model
    model = train_model(X_train, X_test, y_train, y_test, label_encoder)
