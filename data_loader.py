# data_loader.py
import tensorflow as tf
from config import DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE

def load_datasets():
    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    raw_val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    # Extract class_names BEFORE transformations
    class_names = raw_train_ds.class_names

    # Apply caching, shuffling, prefetching, etc. 
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (
        raw_train_ds
        .cache()
        .shuffle(1000)
        .prefetch(buffer_size=AUTOTUNE)
    )
    val_ds = raw_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Return the new datasets plus the class names
    return train_ds, val_ds, class_names
