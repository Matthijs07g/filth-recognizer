import os
from PIL import Image

# Update this path to point to your dataset folder
DATASET_DIR = "dataset"

def convert_webp_to_png(dataset_dir):
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(".webp") or file.lower().endswith(".avif"):
                file_path = os.path.join(root, file)
                try:
                    # Open the WebP file
                    with Image.open(file_path) as img:
                        # Define the new filename with .png extension
                        new_file_path = os.path.splitext(file_path)[0] + ".png"
                        # Save the image as PNG
                        img.save(new_file_path, "PNG")
                        print(f"Converted: {file_path} -> {new_file_path}")
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error converting {file_path}: {e}")

if __name__ == "__main__":
    convert_webp_to_png(DATASET_DIR)
