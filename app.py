import os
import subprocess
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import shutil

# OPTIONAL: If you want to load your model in memory for fast inference:
# import tensorflow as tf
# model = tf.keras.models.load_model("saved_models/plate_classifier_model.keras")

app = Flask(__name__)
app.secret_key = "some_secret_key_for_sessions"  # required for flash messages

# Path to store uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    # Renders a simple page with buttons for training, evaluation, and image upload
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    # Option A: Directly run "python train.py" in a blocking call
    #   - Not recommended for large training because it can block the web server
    try:
        # This calls your existing train.py script
        return_code = os.system("python train.py")
        if return_code != 0:
            flash("Training failed or returned a non-zero exit code.")
        else:
            flash("Training completed successfully!")
    except Exception as e:
        flash(f"Error running train.py: {e}")
    return redirect(url_for("index"))

@app.route("/evaluate", methods=["POST"])
def evaluate():
    # Similarly, run "python evaluate.py"
    try:
        return_code = os.system("python evaluate.py")
        if return_code != 0:
            flash("Evaluation failed or returned a non-zero exit code.")
        else:
            flash("Evaluation completed successfully!")
    except Exception as e:
        flash(f"Error running evaluate.py: {e}")
    return redirect(url_for("index"))

@app.route("/predict", methods=["POST"])
def predict():
    # Handle the uploaded file from the form
    if "image_file" not in request.files:
        flash("No file part in request.")
        return redirect(url_for("index"))

    file = request.files["image_file"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if file:
        # Save the file to UPLOAD_FOLDER
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Option A: call your predict.py with the image path
        # We capture the output to show on the webpage
        try:
            # use subprocess to capture output
            result = subprocess.check_output(["python", "predict.py", filepath], text=True)
            flash(f"Prediction result: {result}")
        except subprocess.CalledProcessError as e:
            flash(f"Predict script error: {e.output}")
        except Exception as e:
            flash(f"Error running predict.py: {e}")

        # Optionally remove the uploaded file afterwards
        # os.remove(filepath)

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
