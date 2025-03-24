# predict.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from config import IMG_HEIGHT, IMG_WIDTH, MODEL_SAVE_PATH

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

def main(img_path):
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    # Apply softmax if your model doesn't already include it
    score = tf.nn.softmax(predictions[0])
    # Replace these class names with your actual classes:
    class_names = ["clean", "dirty"]
    predicted_class = class_names[np.argmax(score)]
    print(f"This image is most likely: {predicted_class} with {(100 * np.max(score)):.2f}% confidence.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python predict.py <path_to_image>")
