from cryptography.fernet import Fernet

# Generate a key and save it for later use
key = Fernet.generate_key()
print(f"Your encryption key: {key.decode()}")
