# train.py
import os
import tensorflow as tf
from data_loader import load_datasets
from model import create_model
from config import NUM_EPOCHS, MODEL_SAVE_PATH

def main():
    train_ds, val_ds, class_names = load_datasets()
    num_classes = len(class_names)

    # Check if model already exists
    if os.path.exists(MODEL_SAVE_PATH):
        print("Model already exists. Skipping training.")
        return

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
    model.save(MODEL_SAVE_PATH + ".keras")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
