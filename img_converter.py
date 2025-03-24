import imghdr
import os

dataset_path = "dataset"

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)
        ext = imghdr.what(file_path)  # Detect the real format
        if ext is None:
            print(f"Invalid image file: {file_path}")
        elif not file.lower().endswith(ext):
            new_path = file_path.rsplit(".", 1)[0] + "." + ext
            os.rename(file_path, new_path)
            print(f"Renamed {file_path} to {new_path}")
