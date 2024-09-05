import os
import numpy as np
import pandas as pd
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from Crypto.Cipher import DES, DES3, Blowfish, ARC4, ChaCha20, CAST
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP, Salsa20
from scipy.stats import entropy

# Calculate entropy of data
def calculate_entropy(data):
    probability_distribution = np.bincount(data) / len(data)
    return entropy(probability_distribution, base=2)

# Generate ciphertext for AES
def generate_aes_data():
    key = os.urandom(16)
    iv = os.urandom(16)
    aes_cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = aes_cipher.encryptor()
    plaintext = os.urandom(64)
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return ciphertext

# Generate ciphertext for DES
def generate_des_data():
    key = os.urandom(8)
    iv = os.urandom(8)
    des_cipher = DES.new(key, DES.MODE_CFB, iv)
    plaintext = os.urandom(64)
    ciphertext = des_cipher.encrypt(plaintext)
    return ciphertext

# Generate ciphertext for RSA
def generate_rsa_data():
    key = RSA.generate(2048)
    rsa_cipher = PKCS1_OAEP.new(key)
    plaintext = os.urandom(64)
    ciphertext = rsa_cipher.encrypt(plaintext)
    return ciphertext

# Generate ciphertext for 3DES
def generate_3des_data():
    key = DES3.adjust_key_parity(os.urandom(24))
    iv = os.urandom(8)
    des3_cipher = DES3.new(key, DES3.MODE_CFB, iv)
    plaintext = os.urandom(64)
    ciphertext = des3_cipher.encrypt(plaintext)
    return ciphertext

# Generate ciphertext for ChaCha20
def generate_chacha20_data():
    key = os.urandom(32)
    nonce = os.urandom(12)
    chacha20_cipher = ChaCha20.new(key=key, nonce=nonce)
    plaintext = os.urandom(64)
    ciphertext = chacha20_cipher.encrypt(plaintext)
    return ciphertext

# Generate ciphertext for Blowfish
def generate_blowfish_data():
    key = os.urandom(16)
    iv = os.urandom(8)
    blowfish_cipher = Blowfish.new(key, Blowfish.MODE_CFB, iv)
    plaintext = os.urandom(64)
    ciphertext = blowfish_cipher.encrypt(plaintext)
    return ciphertext


# Generate ciphertext for RC4
def generate_rc4_data():
    key = os.urandom(16)
    cipher = ARC4.new(key)
    plaintext = os.urandom(64)
    ciphertext = cipher.encrypt(plaintext)
    return ciphertext


# Generate ciphertext for Salsa20
def generate_salsa20_data():
    key = os.urandom(32)
    nonce = os.urandom(8)
    salsa20_cipher = Salsa20.new(key=key, nonce=nonce)
    plaintext = os.urandom(64)
    ciphertext = salsa20_cipher.encrypt(plaintext)
    return ciphertext

# Generate ciphertext for CAST5
def generate_cast5_data():
    key = os.urandom(16)
    iv = os.urandom(8)
    cast5_cipher = CAST.new(key, CAST.MODE_CFB, iv)
    plaintext = os.urandom(64)
    ciphertext = cast5_cipher.encrypt(plaintext)
    return ciphertext

# Generate dataset and save to CSV
def generate_dataset(size=100, filename='cryptography_dataset.csv'):
    data = []
    labels = []

    for _ in range(size):
        data.append(generate_aes_data())
        labels.append('AES')

        data.append(generate_des_data())
        labels.append('DES')

        data.append(generate_rsa_data())
        labels.append('RSA')

        data.append(generate_3des_data())
        labels.append('3DES')

        data.append(generate_chacha20_data())
        labels.append('ChaCha20')

        data.append(generate_blowfish_data())
        labels.append('Blowfish')

        data.append(generate_rc4_data())
        labels.append('RC4')

        data.append(generate_salsa20_data())
        labels.append('Salsa20')

        data.append(generate_cast5_data())
        labels.append('CAST5')

    hex_data = [cipher.hex() for cipher in data]
    df = pd.DataFrame({'ciphertext': hex_data, 'algorithm': labels})
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}.")

if __name__ == "__main__":
    generate_dataset(size=100, filename='cryptography_dataset.csv')
