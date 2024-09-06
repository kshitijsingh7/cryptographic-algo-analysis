# Cryptographic Algorithm Prediction

This project aims to generate datasets for various cryptographic algorithms, train a machine learning model to predict the algorithm used for given ciphertexts, and predict algorithms for ciphertexts from a CSV file.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup and Installation](#setup-and-installation)
3. [Generating the Dataset](#generating-the-dataset)
4. [Training the Model](#training-the-model)
5. [Predicting Algorithm from Ciphertext](#predicting-algorithm-from-ciphertext)
6. [Usage](#usage)
7. [Explanation of How the Code Works](#explanation-of-how-the-code-works)
8. [Notes](#notes)

## Prerequisites

Make sure you have the following installed:

- Python 3.10 or above
- Required Python libraries (listed in `requirements.txt`)

## Setup and Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/kshitijsingh7/cryptographic-algo-analysis.git
    ```
    ```bash
    cd cryptographic-algo-analysis
    ```
    ```bash
    git checkout -b neural-net
    ```
    ```bash
    git pull origin neural-net
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment:**

    - On Windows: Run the Set-ExecutionPolicy command to bypass UnauthorizedAccess error on Windows.

        ```bash
        Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted
        ```
        ```bash
        .\venv\Scripts\activate
        ```

4. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Generating the Dataset

1. **Run the multithreaded dataset generation script:**

    This script generates a dataset containing ciphertexts encrypted with different cryptographic algorithms using multithreading to speed up the process.

    ```bash
    python generate_dataset_multithread.py
    ```

2. **Check the generated dataset:**

    The dataset will be saved as `cipher_dataset.csv` in the current directory.

## Training the Model

1. **Train the model using the generated dataset:**

    ```bash
    python train_model_neural_net.py
    ```

2. **Check the output:**

    The trained model (`best_model.keras`) and label encoder classes (`label_encoder_classes.npy`) will be saved in the current directory.

## Predicting Algorithm from Ciphertext

1. **Prepare a CSV file containing ciphertexts:**

    Ensure the CSV file (e.g., `sample_dataset.csv`) has a column named `ciphertext` with the hex-encoded ciphertexts.

2. **Run the prediction script:**

    ```bash
    python prediction.py
    ```

## Usage

- **Generate Dataset**: To create a dataset using multithreading, run:

    ```bash
    python generate_dataset_multithread.py
    ```

- **Train Model**: To train the neural network model, use:

    ```bash
    python train_model_neural_net.py
    ```

- **Predict from CSV**: To predict algorithms from a CSV file:

    ```bash
    python prediction.py
    ```

## Explanation of How the Code Works

### 1. Dataset Generation (`generate_dataset_multithread.py`)

- This script generates ciphertexts using various cryptographic algorithms like AES, DES, 3DES, RSA, Blowfish, RC4, Salsa20, CAST5, etc.
- Multithreading is used to speed up the generation process.
- For each algorithm, a function generates a random key, initializes the cipher, and encrypts a random plaintext.
- The extracted features from the ciphertexts are saved in a CSV file (`cipher_dataset.csv`) with their corresponding algorithm labels.

### 2. Feature Extraction and Model Training (`train_model_neural_net.py`)

- **Feature Extraction**: The script reads the dataset from the CSV file and extracts features from each ciphertext. The features include byte distribution, entropy, and statistical properties like mean, standard deviation, max, and min values.
- **Model Training**: A neural network model is built using TensorFlow's Keras API, and it is trained with the extracted features. The model includes several dense layers with dropout for regularization.
- The trained model (`best_model.keras`) and label encoder classes are saved to disk for future use.

### 3. Predicting the Algorithm (`prediction.py`)

- This script loads the trained neural network model and the label encoder classes.
- It reads a CSV file containing ciphertexts, removes any spaces from the ciphertexts, and extracts the same features as used in training.
- The model predicts which cryptographic algorithm was likely used to generate each ciphertext, and the results are printed on the screen and saved in a new CSV file (`predictions.csv`).

## Notes

- **CSV Format**: The input CSV file must contain a `ciphertext` column with hexadecimal-encoded ciphertexts. Any spaces in the hexadecimal strings will be removed automatically.
- **Model Files**: Make sure `best_model.keras` and `label_encoder_classes.npy` are in the same directory as `prediction.py` for predictions to work.
