import os
from PIL import Image

dataset_path = "dataset"  # Change this to your dataset folder path

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            with Image.open(file_path) as img:
                img.verify()  # Check for corruption
        except (IOError, SyntaxError) as e:
            print(f"Corrupt file found: {file_path}")
