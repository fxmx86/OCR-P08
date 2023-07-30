# Access via http://fxa-ocrp08-flaskapi.azurewebsites.net:5000

# Start importing relevant librairies
import mimetypes
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, send_file
from PIL import Image
import glob, os
import numpy as np
from matplotlib import colors

# Path to the Keras model
model_path = "./model"

# Input dimensions expected by your Keras model
MODEL_INPUT_WIDTH = 512
MODEL_INPUT_HEIGHT = 256

# Load the Keras model
MODEL = load_model(model_path, compile=False)


def generate_img_from_mask(mask, colors_palette=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']):
    '''Generate a color image array from a segmented mask
    Args:
      mask - numpy array of dimension
      colors_palette - list - color to be assigned to each class
    Returns
      Image array'''

    id2category = {0: 'void',
                   1: 'flat',
                   2: 'construction',
                   3: 'object',
                   4: 'nature',
                   5: 'sky',
                   6: 'human',
                   7: 'vehicle'}

    # Start initializing the output image
    img_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype='float')

    # Assign RGB channels
    for cat in id2category.keys():
        img_seg[:, :, 0] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[0]
        img_seg[:, :, 1] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[1]
        img_seg[:, :, 2] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[2]

    return img_seg


def predict_segmentation(image_array, image_width, image_height):
    '''Generate a color mask from a model
    Args:
      image_array - Input image numpy array
      model_path - Path to your Keras model
      image_width - int - Width (pixels) of the input image expected by the model
      image_height - int - Height (pixels) of the input image expected by the model
    Returns
      Image array'''

    # First, resize your input image at the size expected by your model
    # Otherwise, it will throw an Exception
    image_array = Image.fromarray(image_array).resize((image_width, image_height))

    # Expand dimension of the image. For example (width, height, 3) --> (1, width, height, 3)
    # This is needed by your model to work properly
    # Otherwise, it will throw an Exception
    image_array = np.expand_dims(np.array(image_array), axis=0)

    # Predict the mask as an output of the model
    mask_predict = MODEL.predict(image_array)

    # Squeeze the first dimension of the mask.
    # For example (1, x, y, z) -> (x, y, z)
    mask_predict = np.squeeze(mask_predict, axis=0)

    # Finally, generate a color image (RGB image) from the mask array
    # For example (width, height, 8) -> (width, height, 3)
    mask_color = generate_img_from_mask(mask_predict) * 255

    return mask_color


app = Flask(__name__)

# This is the route to the welcome page of the segmentation API
@app.route("/")
def hello():
    return "Hello, welcome on the segmentation API"

# This is the route to the API
@app.route("/predict_mask", methods=["POST"])
def segment_image():
    # Get image file included in the request
    file = request.files['image']
    # Read the image via file.stream
    img = Image.open(file.stream)

    # Call predict_segmentation to grab the output mask color from the Keras model
    mask_color = predict_segmentation(image_array=np.array(img), image_width=MODEL_INPUT_WIDTH,
                                      image_height=MODEL_INPUT_HEIGHT)

    # Write the mask as a temporary file
    # Image.fromarray(mask_color).save("tmp.png", "PNG")
    Image.fromarray(mask_color.astype(np.uint8)).save("tmp.png")

    # Return the response. The response is the mask image
    return send_file("tmp.png", mimetype="image/png")