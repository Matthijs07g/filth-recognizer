# evaluate.py
import tensorflow as tf
from data_loader import load_datasets
from config import MODEL_SAVE_PATH

def main():
    _, val_ds, _ = load_datasets()
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    loss, accuracy = model.evaluate(val_ds)
    print(f"Validation Loss: {loss:.2f}")
    print(f"Validation Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
