import os
import numpy as np
import pandas as pd
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from Crypto.Cipher import DES, DES3, Blowfish, ARC4, ChaCha20, CAST
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP, Salsa20
from scipy.stats import entropy
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import binascii

# Function to extract features from ciphertext
def extract_features(ciphertext, key, iv):
    # Byte distribution
    byte_distribution = np.bincount(np.frombuffer(ciphertext, dtype=np.uint8), minlength=256)

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
    ciphertext_length = len(ciphertext)
    first_byte = ciphertext[0]
    last_byte = ciphertext[-1]
    unique_bytes = len(set(ciphertext))

    # Key and IV lengths
    key_length = len(key)
    iv_length = len(iv) if iv is not None else 0

    # Convert ciphertext to hexadecimal representation
    ciphertext_hex = binascii.hexlify(ciphertext).decode('utf-8')

    # Combine all features into a single array
    features = np.concatenate((
        normalized_distribution,
        [entropy_value, mean, std_dev, max_value, min_value,
         ciphertext_length, first_byte, last_byte, unique_bytes,
         key_length, iv_length]
    ))

    return features, ciphertext_hex

# Ciphertext generation functions for different algorithms
def generate_aes_data():
    key = os.urandom(16)
    iv = os.urandom(16)
    aes_cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = aes_cipher.encryptor()
    plaintext = os.urandom(64)
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return ciphertext, key, iv, 'AES'

def generate_des_data():
    key = os.urandom(8)
    iv = os.urandom(8)
    des_cipher = DES.new(key, DES.MODE_CFB, iv)
    plaintext = os.urandom(64)
    ciphertext = des_cipher.encrypt(plaintext)
    return ciphertext, key, iv, 'DES'

def generate_rsa_data():
    key = RSA.generate(2048)
    rsa_cipher = PKCS1_OAEP.new(key)
    plaintext = os.urandom(190)  # RSA encryption with 2048 bits allows up to 190 bytes for PKCS1_OAEP padding
    ciphertext = rsa_cipher.encrypt(plaintext)
    return ciphertext, key.export_key(), None, 'RSA'

def generate_3des_data():
    key = DES3.adjust_key_parity(os.urandom(24))
    iv = os.urandom(8)
    des3_cipher = DES3.new(key, DES3.MODE_CFB, iv)
    plaintext = os.urandom(64)
    ciphertext = des3_cipher.encrypt(plaintext)
    return ciphertext, key, iv, '3DES'

def generate_chacha20_data():
    key = os.urandom(32)
    nonce = os.urandom(12)
    chacha20_cipher = ChaCha20.new(key=key, nonce=nonce)
    plaintext = os.urandom(64)
    ciphertext = chacha20_cipher.encrypt(plaintext)
    return ciphertext, key, nonce, 'ChaCha20'

def generate_blowfish_data():
    key = os.urandom(16)
    iv = os.urandom(8)
    blowfish_cipher = Blowfish.new(key, Blowfish.MODE_CFB, iv)
    plaintext = os.urandom(64)
    ciphertext = blowfish_cipher.encrypt(plaintext)
    return ciphertext, key, iv, 'Blowfish'

def generate_rc4_data():
    key = os.urandom(16)
    cipher = ARC4.new(key)
    plaintext = os.urandom(64)
    ciphertext = cipher.encrypt(plaintext)
    return ciphertext, key, None, 'RC4'

def generate_salsa20_data():
    key = os.urandom(32)
    nonce = os.urandom(8)
    salsa20_cipher = Salsa20.new(key=key, nonce=nonce)
    plaintext = os.urandom(64)
    ciphertext = salsa20_cipher.encrypt(plaintext)
    return ciphertext, key, nonce, 'Salsa20'

def generate_cast5_data():
    key = os.urandom(16)
    iv = os.urandom(8)
    cast5_cipher = CAST.new(key, CAST.MODE_CFB, iv)
    plaintext = os.urandom(64)
    ciphertext = cast5_cipher.encrypt(plaintext)
    return ciphertext, key, iv, 'CAST5'

# Generate dataset and save to CSV
def generate_dataset(size=100, filename='cipher_dataset.csv'):
    data = []
    labels = []
    ciphertexts = []

    # List of generation functions
    generators = [
        generate_aes_data,
        generate_des_data,
        generate_rsa_data,
        generate_3des_data,
        generate_chacha20_data,
        generate_blowfish_data,
        generate_rc4_data,
        generate_salsa20_data,
        generate_cast5_data
    ]

    # Multithreading to speed up data generation
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(generator) for generator in generators for _ in range(size)]

        # Collect results as they are completed
        for future in as_completed(futures):
            ciphertext, key, iv, label = future.result()
            features, ciphertext_hex = extract_features(ciphertext, key, iv)
            data.append(features)
            labels.append(label)
            ciphertexts.append(ciphertext_hex)

    # Define feature names
    feature_names = [f'byte_dist_{i}' for i in range(256)] + [
        'entropy', 'mean', 'std_dev', 'max_value', 'min_value',
        'ciphertext_length', 'first_byte', 'last_byte', 'unique_bytes',
        'key_length', 'iv_length'
    ]

    df = pd.DataFrame(data, columns=feature_names)
    df['algorithm'] = labels
    df['ciphertext'] = ciphertexts  # Add ciphertext column
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}.")

if __name__ == "__main__":
    start_time = time.time()

    generate_dataset(size=100, filename='cipher_dataset.csv')

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Time taken to generate dataset: {elapsed_time:.2f} minutes")
