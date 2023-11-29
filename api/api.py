from fastapi import FastAPI, UploadFile, File
from starlette.responses import Response
import numpy as np
import cv2
import io
from datetime import datetime
import random
import pandas as pd

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

    # Define maximum file size (1048576B equals 1MB)
    max_file_size = 1048576

    # Ensure the file is not too big
    if len(contents) >= max_file_size:
        return {"status": f"Error - uploaded file exceeds max_file_size of {max_file_size} Bytes"}

    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

    current_datetime = datetime.now()
    timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    # Saving the uploaded images
    cv2.imwrite(f"uploaded_images/{timestamp}.png", cv2_img)

    ### define random bird species to return

    # load bird species csv
    bird_species_df = pd.read_csv('bird_data/bird_species.csv').set_index('species_no')

    ## First likely species
    # load random number within the range of bird species numbers
    first_random_species_no = random.randint(1, 10982)
    # load scientific name of random bird species
    first_random_scientific_name = bird_species_df[bird_species_df.index == first_random_species_no]['scientific_name'].iloc[0]
    # generate random probability
    first_prob = "%.2f" % round(random.uniform(0.7, 1), 2)

    ## Second likely species
    # load random number within the range of bird species numbers
    second_random_species_no = random.randint(1, 10982)
    # load scientific name of random bird species
    second_random_scientific_name = bird_species_df[bird_species_df.index == second_random_species_no]['scientific_name'].iloc[0]
    # generate random probability
    second_prob = "%.2f" % round(random.uniform(0.55, 0.69), 2)

    ## Third likely species
    # load random number within the range of bird species numbers
    third_random_species_no = random.randint(1, 10982)
    # load scientific name of random bird species
    third_random_scientific_name = bird_species_df[bird_species_df.index == third_random_species_no]['scientific_name'].iloc[0]
    # generate random probability
    third_prob = "%.2f" % round(random.uniform(0, 0.54), 2)

    response = {
        "status": "ok",
        "bird_detected": True,
        "timestamp": timestamp,
        "first_likely_bird_species": {
            "species_no": first_random_species_no,
            "scientific_name": first_random_scientific_name,
            "probability": first_prob
        },
        "second_likely_bird_species": {
            "species_no": second_random_species_no,
            "scientific_name": second_random_scientific_name,
            "probability": second_prob
        },
        "third_likely_bird_species": {
        "species_no": third_random_species_no,
        "scientific_name": third_random_scientific_name,
        "probability": third_prob
        }
    }

    return response
