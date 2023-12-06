### IMPORTS
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import os
import sys
from tqdm import tqdm

### SELECTED SPECIES SELECTION
if len(sys.argv) > 1:
    csvpath = sys.argv[1]
else:
    csvpath = input("Please enter the path to the CSV file with selected bird species for image retrieval\n(format: index,species_no,scientific_name)")

## Importing the selected species CSV per command-line into a data frame
selected_birds_df = pd.read_csv(csvpath)

### CREATING A CSV FOR DOWNLOAD LOGGING [if non-existent]
bird_data_dir = "../bird_data/"
tracking_csv_path = os.path.join(bird_data_dir, "tracking.csv")

## Check if the directory exists, if not, create it
if not os.path.exists(bird_data_dir):
    os.makedirs(bird_data_dir)

## Check if the tracking CSV file exists
if os.path.exists(tracking_csv_path):
    tracking_df = pd.read_csv(tracking_csv_path)
else:
    tracking_df = pd.DataFrame(columns=['species_no', 'downloaded', 'downloaded_no', 'skipped', 'total'])

### BIRD MASTER IMPORT
# bird_master_df = pd.read_csv('../bird_data/test_script.csv')
bird_master_df = pd.read_csv('../data_sourcing/bird_master_df.csv') # THIS NEEDS TO BE ACTIVATED WHEN RUNNING A NON-TEST DOWNLOAD
bird_master_df = bird_master_df.drop(columns='Unnamed: 0')

## Conversion of all selected species_no into an iterable list
species_no_to_download = selected_birds_df['species_no'].tolist()

### PRIMARY IMAGE RETRIEVAL FUNCTION | THE HEART OF THIS SCRIPT
def image_retrieval(bird_master_df, selected_birds_df, mydir=None, size=(256, 256), number=1):
    '''Download images based on master URL file (~10Mio. entries for ~11,000 species) for selected species'''

    temp_df = bird_master_df[bird_master_df['species_no'] == selected_birds_df['species_no'].iloc[0]].drop(columns=['Unnamed: 0'], errors='ignore')
    temp_dict = {}  # Dictionary to track downloaded image_ids

    total_images = temp_df.shape[0]
    downloaded_count = 0
    skipped_count = 0

    with tqdm(total=total_images, desc=f"Species {selected_birds_df['species_no'].iloc[0]} Progress", position=0, leave=True) as species_progress_bar:
        for index, row in temp_df.iterrows():
            image_id = row['image_id']
            if image_id in temp_dict:
                species_progress_bar.update(1)  # Skip duplicate image_ids
                skipped_count += 1
                continue

            url_im = row['image_url']

            try:
                response = requests.get(url_im, timeout=5)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                species_progress_bar.update(1)  # Update progress bar for each iteration
                skipped_count += 1
                continue

            species_progress_bar.update(1)  # Update progress bar for each iteration

            if response.status_code == 200:
                try:
                    im = Image.open(BytesIO(response.content))
                    im.thumbnail(size, Image.LANCZOS)

                    if mydir is None:
                        mydir = os.getcwd()
                        species_folder = os.path.join(mydir, f"{row['species_no']}")
                        os.makedirs(species_folder, exist_ok=True)

                    im_dir = os.path.join(species_folder, f"{row['species_no']}_{number:04d}.png")

                    if not os.path.exists(im_dir):
                        im.save(im_dir)
                        temp_dict[image_id] = True  # Mark image_id as downloaded
                        downloaded_count += 1
                        number += 1
                    else:
                        skipped_count += 1

                except Exception as e:
                    skipped_count += 1  # Do nothing and continue with the next iteration

    return number, downloaded_count, skipped_count, total_images

# ... (previous code)

for i in range(len(species_no_to_download)):
    current_species_no = species_no_to_download[i]
    print('\n')

    # Check if species_no already exists in tracking_df
    if current_species_no in tracking_df['species_no'].values:
        print(f"The species [species_no: {current_species_no}] has already been downloaded successfully.")
        continue

    print(f"Starting to work on species {i+1}/{len(species_no_to_download)} [species_no: {current_species_no}].")

    function_input_df = selected_birds_df.loc[selected_birds_df['species_no'] == current_species_no]

    number, downloaded_count, skipped_count, total_images = image_retrieval(bird_master_df, function_input_df, number=1)

    downloaded_images_count = len([f for f in os.listdir(os.path.join(os.getcwd(), f"{current_species_no}")) if os.path.isfile(os.path.join(os.getcwd(), f"{current_species_no}", f))])

    new_row = {'species_no': current_species_no, 'downloaded': 1, 'downloaded_no': downloaded_images_count, 'skipped': skipped_count, 'total': total_images}
    tracking_df.loc[len(tracking_df)] = new_row

    tracking_df.to_csv(tracking_csv_path, index=False)

    print(f"Downloaded: {downloaded_count} | Skipped: {skipped_count} | Total: {total_images}")
    print(f"Completed work on species {i+1}/{len(species_no_to_download)} [species_no: {current_species_no}].")

print(f"\nFinished download of {len(species_no_to_download)} specified species. Script ended.")
