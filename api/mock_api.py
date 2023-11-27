from fastapi import FastAPI, UploadFile, File

import numpy as np
import cv2
import io

from datetime import datetime

# Initializing the API
app = FastAPI()

# Defining root endpoint
@app.get("/")
def index():
    return {"status": "ok"}

# Defining upload endpoint
@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()

    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

    current_datetime = datetime.now()
    filename = f"{current_datetime.year}-{current_datetime.month}-{current_datetime.day}-{current_datetime.hour}-{current_datetime.minute}-{current_datetime.second}"

    # Saving the uploaded images
    cv2.imwrite(f"uploaded_images/{filename}.png", cv2_img)
