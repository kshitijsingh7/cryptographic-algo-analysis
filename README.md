# Cryptographic Algorithm Prediction

This project aims to generate datasets for various cryptographic algorithms, train a machine learning model to predict the algorithm used for given ciphertexts, and predict algorithms for ciphertexts from a CSV file.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup and Installation](#setup-and-installation)
3. [Generating the Dataset](#generating-the-dataset)
4. [Training the Model](#training-the-model)
5. [Predicting Algorithm from Ciphertext](#predicting-algorithm-from-ciphertext)
6. [Usage](#usage)
7. [Notes](#notes)

## Prerequisites

Make sure you have the following installed:

- Python 3.10 or above
- Required Python libraries (listed in `requirements.txt`)

## Setup and Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/kshitijsingh7/cryptographic-algo-analysis.git
    cd cryptographic-algo-analysis
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment:**

    - On Windows: Run the Set-ExecutionPolicy command to to bypass UnauthorizedAccess error on windows

        ```bash
        Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted
        .\venv\Scripts\activate
        ```

4. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Generating the Dataset

1. **Run the dataset generation script:**

    This script generates a dataset containing ciphertexts encrypted with different cryptographic algorithms.

    ```bash
    python generate_dataset.py
    ```

2. **Check the generated dataset:**

    The dataset will be saved as `cryptography_dataset.csv` in the current directory.

## Training the Model

1. **Train the model using the generated dataset:**

    ```bash
    python train_model.py
    ```

2. **Check the output:**

    The trained model (`model.joblib`) and label encoder (`label_encoder.joblib`) will be saved in the current directory.

## Predicting Algorithm from Ciphertext

1. **Prepare a CSV file containing ciphertexts:**

    Ensure the CSV filename example `sample_dataset.csv` has a column named `ciphertext` with the hex-encoded ciphertexts.

2. **Run the prediction script:**

    ```bash
    python prediction_csv.py
    ```

## Usage

- **Generate Dataset**: To create a dataset, run:

    ```bash
    python generate_dataset.py
    ```

- **Train Model**: To train the model, use:

    ```bash
    python train_model.py
    ```

- **Predict from CSV**: To predict algorithms from a CSV file:

    ```bash
    python prediction_csv.py
    ```

## Notes

- **CSV Format**: The input CSV file must contain a `ciphertext` column with hexadecimal-encoded ciphertexts. Any spaces in the hexadecimal strings will be removed automatically.
- **Model Files**: Make sure `model.joblib` and `label_encoder.joblib` are in the same directory as `predict_algorithm.py` for predictions to work.
