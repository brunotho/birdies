from fastapi import FastAPI, UploadFile, File
from starlette.responses import Response
import numpy as np
import cv2
import io
from datetime import datetime
import random
import pandas as pd
from birdies_code.ml_logic.prediction import load_model_
from birdies_code.ml_logic.prediction import prediction

model = load_model_()

# loading cached data warehouse from csv file
warehouse_df = pd.read_csv('bird_data/warehouse_231201-1523.csv').set_index('id')

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

    result = prediction(model, cv2_img)

    print(result)

    current_datetime = datetime.now()
    timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    # Saving the uploaded images
    cv2.imwrite(f"uploaded_images/{timestamp}.png", cv2_img)

    ### define random bird species to return

    # load bird species csv
    bird_species_df = pd.read_csv('bird_data/bird_species.csv').set_index('species_no')

## First likely species
    # load random number within the range of bird species numbers
    # first_random_species_no = random.randint(1, 10982)
    # load first prediction from model
    first_species_no = int(result.get('pred_1')[0])
    # load first probability from model
    first_prob = result.get('pred_1')[1]
    # load scientific name of random bird species
    first_random_scientific_name = bird_species_df[bird_species_df.index == first_species_no]['scientific_name'].iloc[0]
    # generate random probability
    # first_prob = "%.2f" % round(random.uniform(0.7, 1), 2)
    # load description
    first_description = warehouse_df[warehouse_df.index == first_species_no]['General_Describtion'].iloc[0]
    # load common name
    first_common_name = warehouse_df[warehouse_df.index == first_species_no]['Common_Name'].iloc[0]
    # load size
    first_size = warehouse_df[warehouse_df.index == first_species_no]['size'].iloc[0]
    # load size category
    first_size_category = warehouse_df[warehouse_df.index == first_species_no]['Size_category'].iloc[0]
    # load mass
    first_mass = warehouse_df[warehouse_df.index == first_species_no]['Mass'].iloc[0]
    # load habitat
    first_habitat = warehouse_df[warehouse_df.index == first_species_no]['Habitat'].iloc[0]
    # load habitat category
    first_habitat_category = warehouse_df[warehouse_df.index == first_species_no]['Habitat_Category'].iloc[0]
    # load migration
    first_migration = warehouse_df[warehouse_df.index == first_species_no]['Migration'].iloc[0]
    # load trophic level feeding habits
    first_trophic_level_feeding_habits = warehouse_df[warehouse_df.index == first_species_no]['Trophic_Level__Feeding_Habits_'].iloc[0]
    # load min latitude
    first_min_latitude = warehouse_df[warehouse_df.index == first_species_no]['Min_Latitude'].iloc[0]
    # load max latitude
    first_max_latitude = warehouse_df[warehouse_df.index == first_species_no]['Max_Latitude'].iloc[0]
    # load centroid latitude
    first_centroid_latitude = warehouse_df[warehouse_df.index == first_species_no]['Centroid_Latitude'].iloc[0]
    # load centroid longitude
    first_centroid_longitude = warehouse_df[warehouse_df.index == first_species_no]['Centroid_Longitude'].iloc[0]
    # load range size
    first_range_size = warehouse_df[warehouse_df.index == first_species_no]['Range_Size'].iloc[0]
    # load species status (e. g., extinct or endangered)
    first_species_status = warehouse_df[warehouse_df.index == first_species_no]['species_status'].iloc[0]
    # load data status (fallback vs. enriched)
    first_status = warehouse_df[warehouse_df.index == first_species_no]['status'].iloc[0]


    ## Second likely species
    # load random number within the range of bird species numbers
    #second_random_species_no = random.randint(1, 10982)
    # load first prediction from model
    second_species_no = int(result.get('pred_2')[0])
    # load first probability from model
    second_prob = result.get('pred_2')[1]
    # load scientific name of random bird species
    second_random_scientific_name = bird_species_df[bird_species_df.index == second_species_no]['scientific_name'].iloc[0]
    # generate random probability
    #second_prob = "%.2f" % round(random.uniform(0.55, 0.69), 2)
    # load description
    second_description = warehouse_df[warehouse_df.index == second_species_no]['General_Describtion'].iloc[0]
    # load common name
    second_common_name = warehouse_df[warehouse_df.index == second_species_no]['Common_Name'].iloc[0]
    # load size
    second_size = warehouse_df[warehouse_df.index == second_species_no]['size'].iloc[0]
    # load size category
    second_size_category = warehouse_df[warehouse_df.index == second_species_no]['Size_category'].iloc[0]
    # load mass
    second_mass = warehouse_df[warehouse_df.index == second_species_no]['Mass'].iloc[0]
    # load habitat
    second_habitat = warehouse_df[warehouse_df.index == second_species_no]['Habitat'].iloc[0]
    # load habitat category
    second_habitat_category = warehouse_df[warehouse_df.index == second_species_no]['Habitat_Category'].iloc[0]
    # load migration
    second_migration = warehouse_df[warehouse_df.index == second_species_no]['Migration'].iloc[0]
    # load trophic level feeding habits
    second_trophic_level_feeding_habits = warehouse_df[warehouse_df.index == second_species_no]['Trophic_Level__Feeding_Habits_'].iloc[0]
    # load min latitude
    second_min_latitude = warehouse_df[warehouse_df.index == second_species_no]['Min_Latitude'].iloc[0]
    # load max latitude
    second_max_latitude = warehouse_df[warehouse_df.index == second_species_no]['Max_Latitude'].iloc[0]
    # load centroid latitude
    second_centroid_latitude = warehouse_df[warehouse_df.index == second_species_no]['Centroid_Latitude'].iloc[0]
    # load centroid longitude
    second_centroid_longitude = warehouse_df[warehouse_df.index == second_species_no]['Centroid_Longitude'].iloc[0]
    # load range size
    second_range_size = warehouse_df[warehouse_df.index == second_species_no]['Range_Size'].iloc[0]
    # load species status (e. g., extinct or endangered)
    second_species_status = warehouse_df[warehouse_df.index == second_species_no]['species_status'].iloc[0]
    # load data status (fallback vs. enriched)
    second_status = warehouse_df[warehouse_df.index == second_species_no]['status'].iloc[0]


    ## Third likely species
    # load random number within the range of bird species numbers
    #third_random_species_no = random.randint(1, 10982)
    # load first prediction from model
    third_species_no = int(result.get('pred_3')[0])
    # load first probability from model
    third_prob = result.get('pred_3')[1]
    # load scientific name of random bird species
    third_random_scientific_name = bird_species_df[bird_species_df.index == third_species_no]['scientific_name'].iloc[0]
    # generate random probability
    #third_prob = "%.2f" % round(random.uniform(0, 0.54), 2)
    # load description
    third_description = warehouse_df[warehouse_df.index == third_species_no]['General_Describtion'].iloc[0]
    # load common name
    third_common_name = warehouse_df[warehouse_df.index == third_species_no]['Common_Name'].iloc[0]
    # load size
    third_size = warehouse_df[warehouse_df.index == third_species_no]['size'].iloc[0]
    # load size category
    third_size_category = warehouse_df[warehouse_df.index == third_species_no]['Size_category'].iloc[0]
    # load mass
    third_mass = warehouse_df[warehouse_df.index == third_species_no]['Mass'].iloc[0]
    # load habitat
    third_habitat = warehouse_df[warehouse_df.index == third_species_no]['Habitat'].iloc[0]
    # load habitat category
    third_habitat_category = warehouse_df[warehouse_df.index == third_species_no]['Habitat_Category'].iloc[0]
    # load migration
    third_migration = warehouse_df[warehouse_df.index == third_species_no]['Migration'].iloc[0]
    # load trophic level feeding habits
    third_trophic_level_feeding_habits = warehouse_df[warehouse_df.index == third_species_no]['Trophic_Level__Feeding_Habits_'].iloc[0]
    # load min latitude
    third_min_latitude = warehouse_df[warehouse_df.index == third_species_no]['Min_Latitude'].iloc[0]
    # load max latitude
    third_max_latitude = warehouse_df[warehouse_df.index == third_species_no]['Max_Latitude'].iloc[0]
    # load centroid latitude
    third_centroid_latitude = warehouse_df[warehouse_df.index == third_species_no]['Centroid_Latitude'].iloc[0]
    # load centroid longitude
    third_centroid_longitude = warehouse_df[warehouse_df.index == third_species_no]['Centroid_Longitude'].iloc[0]
    # load range size
    third_range_size = warehouse_df[warehouse_df.index == third_species_no]['Range_Size'].iloc[0]
    # load species status (e. g., extinct or endangered)
    third_species_status = warehouse_df[warehouse_df.index == third_species_no]['species_status'].iloc[0]
    # load data status (fallback vs. enriched)
    third_status = warehouse_df[warehouse_df.index == third_species_no]['status'].iloc[0]

    # prepare output json file
    response = {
        "status": "ok",
        "bird_detected": True,
        "timestamp": timestamp,
        "first_likely_bird_species": {
            "species_no": first_species_no,
            "scientific_name": first_random_scientific_name,
            "probability": first_prob,
            "description": first_description,
            "common_name": first_common_name,
            "size": first_size,
            "size_category": first_size_category,
            "mass": first_mass,
            "habitat": first_habitat,
            "habitat_category": first_habitat_category,
            "migration": first_migration,
            "trophic_level_feeding_habits": first_trophic_level_feeding_habits,
            "min_latitude": first_min_latitude,
            "max_latitude": first_max_latitude,
            "centroid_latitude": first_centroid_latitude,
            "centroid_longitude": first_centroid_longitude,
            "range_size": first_range_size,
            "species_status": first_species_status,
            "status": first_status
        },
        "second_likely_bird_species": {
            "species_no": second_species_no,
            "scientific_name": second_random_scientific_name,
            "probability": second_prob,
            "description": second_description,
            "common_name": second_common_name,
            "size": second_size,
            "size_category": second_size_category,
            "mass": second_mass,
            "habitat": second_habitat,
            "habitat_category": second_habitat_category,
            "migration": second_migration,
            "trophic_level_feeding_habits": second_trophic_level_feeding_habits,
            "min_latitude": second_min_latitude,
            "max_latitude": second_max_latitude,
            "centroid_latitude": second_centroid_latitude,
            "centroid_longitude": second_centroid_longitude,
            "range_size": second_range_size,
            "species_status": second_species_status,
            "status": second_status
        },
        "third_likely_bird_species": {
            "species_no": third_species_no,
            "scientific_name": third_random_scientific_name,
            "probability": third_prob,
            "description": third_description,
            "common_name": third_common_name,
            "size": third_size,
            "size_category": third_size_category,
            "mass": third_mass,
            "habitat": third_habitat,
            "habitat_category": third_habitat_category,
            "migration": third_migration,
            "trophic_level_feeding_habits": third_trophic_level_feeding_habits,
            "min_latitude": third_min_latitude,
            "max_latitude": third_max_latitude,
            "centroid_latitude": third_centroid_latitude,
            "centroid_longitude": third_centroid_longitude,
            "range_size": third_range_size,
            "species_status": third_species_status,
            "status": third_status
        }
    }

    return response
