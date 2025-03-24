import os

# Data & image configuration
DATA_DIR = "dataset"   # Update to your dataset directory
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Model save path (check if the model file exists here)
MODEL_SAVE_PATH = os.path.join("saved_models", "plate_classifier_model.keras")
