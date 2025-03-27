import os
import shutil  # For removing directories
import tensorflow as tf
from data_loader import load_datasets
from model import create_model
from config import NUM_EPOCHS, MODEL_SAVE_PATH

def main():
    train_ds, val_ds, class_names = load_datasets()
    num_classes = len(class_names)

    # If a saved model already exists, delete it
    if os.path.exists(MODEL_SAVE_PATH):
        print("Old model found. Deleting it...")
        # Remove the directory (if model is saved in the native Keras format as a directory)
        os.remove(MODEL_SAVE_PATH)

    model = create_model(num_classes)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=NUM_EPOCHS
    )

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    # Save the new model. Ensure you use a supported file extension.
    model.save(MODEL_SAVE_PATH)
    print("Training complete. New model saved.")

if __name__ == "__main__":
    main()
