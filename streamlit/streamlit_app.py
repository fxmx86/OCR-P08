# Pour test API Flask
# Import relevant Python libraries
import streamlit as st
import requests
import cv2
import numpy as np
import os.path

# Var initialisation
url = 'https://fxa-ocrp08-flaskapi.azurewebsites.net/predict_mask'

REP = "./streamlit/images/"
images = []
END = "_leftImg8bit.png"
MEND = "_gtFine_color.png"
WIDTH = 1024

# Loop in the image folder
for i, file in enumerate(os.listdir(REP)):
	if file.endswith(END):
		filename = file.replace(END, "")
		images.append(filename)

st.title("P08 - Participez à la conception d'une voiture autonome")
image = st.selectbox( "Select the image you want to try :", images )

image_path = REP+image+END
image_mask = REP+image+MEND
legend = REP+'Legende.jpg'

# Create the image data to be used in the request
image_data = {'image': open(image_path, 'rb')}

# Send the request containing the image to the API and grab the response r
# Then convert the bytecodes of the mask color image to numpy array
r = requests.post(url, files=image_data)
img_array = cv2.imdecode(np.frombuffer(r.content, np.uint8), -1)

# Display the initial color and mask images
st.image(image_path, caption='Initial color image', width=WIDTH)
st.image(image_mask, caption='Initial mask color image', width=WIDTH)

# Display the image array of the mask color image, and legend
st.image(img_array, caption='Predicted mask color image', width=WIDTH)
st.image(legend, caption='Légende', width=WIDTH)