### IMPORTS
import pandas as pd
import requests
from io import StringIO, BytesIO
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
## Specify the directory path
bird_data_dir = "../bird_data/"
tracking_csv_path = os.path.join(bird_data_dir, "test_tracking.csv")  # ADJUST FILENAME IN THE REAL THING

## Check if the directory exists, if not, create it
if not os.path.exists(bird_data_dir):
    os.makedirs(bird_data_dir)

## Check if the tracking CSV file exists
if os.path.exists(tracking_csv_path):
    # If it exists, read the DataFrame from the CSV
    tracking_df = pd.read_csv(tracking_csv_path)
else:
    # If it doesn't exist, create a new DataFrame
    tracking_df = pd.DataFrame(columns=['species_no', 'downloaded', 'downloaded_no'])

### BIRD MASTER IMPORT
bird_master_df = pd.read_csv('../bird_data/test_script.csv') # THIS NEEDS CHANGING ONCE WE WORK WITH THE COMPLETE SET OF BIRDS
# bird_master_df = pd.read_csv('../data_sourcing/bird_master_df.csv')
bird_master_df = bird_master_df.drop(columns='Unnamed: 0')  # drop weird column
## Conversion of all selected species_no into an iterable list
species_no_to_download = selected_birds_df['species_no'].tolist()

### PRIMARY IMAGE RETRIEVAL FUNCTION | THE HEART OF THIS SCRIPT
def image_retrieval(bird_master_df, selected_birds_df, mydir=None, size=(256, 256), number=1, progress_bar=None):
    '''Download images based on the master URL file (~10Mio. entries for ~11,000 species) for selected species'''

    temp_df = bird_master_df[bird_master_df['species_no'] == selected_birds_df['species_no'].iloc[0]].drop(columns=['Unnamed: 0'], errors='ignore')
    temp_list = []

    itercount = 0
    downloaded_count = 0
    skipped_count = 0

    total_image_urls = temp_df.shape[0]  # Move the total_image_urls calculation outside the loop

    for index, row in temp_df.iterrows():
        itercount += 1

        if row['image_id'] in temp_list:
            continue

        url_im = row['image_url']

        try:
            response = requests.get(url_im, timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            skipped_count += 1
            continue

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
                    temp_list.append(row['image_id'])
                    downloaded_count += 1
                else:
                    skipped_count += 1

                number += 1

            except Exception as e:
                skipped_count += 1

        # if progress_bar:
        #     progress_bar.update(downloaded_count + skipped_count - progress_bar.n)
        #     progress_bar.set_postfix(downloaded=downloaded_count, skipped=skipped_count, total=total_image_urls, refresh=True)

        if progress_bar:
            progress_bar.last_print_n = downloaded_count + skipped_count
            progress_bar.update(downloaded_count + skipped_count - progress_bar.n)
            progress_bar.set_postfix(downloaded=downloaded_count, skipped=skipped_count, total=total_image_urls, refresh=True)

    return number, downloaded_count, skipped_count, total_image_urls

# print(species_no_to_download)

# Calculate the total number of image URLs to check for all species
total_image_urls_all_species = sum(bird_master_df.groupby('species_no').size())

# Display the total number of image URLs
# print(f"Total number of image URLs to download: {total_image_urls_all_species}")

# Iterate through each selected species
for i in range(len(species_no_to_download)):
    print('\n')
    print(f"Starting to work on species {i+1}/{len(species_no_to_download)} [species_no: {species_no_to_download[i]}].")

    # Create a mini-DF as input for the download retrieval function
    function_input_df = selected_birds_df.loc[selected_birds_df['species_no'] == species_no_to_download[i]]

    # Calculate the total number of image URLs for the current species_no
    total_image_urls = sum(bird_master_df['species_no'] == species_no_to_download[i])

    # Initialize species progress bar
    with tqdm(total=total_image_urls, desc=f"Species {species_no_to_download[i]} progress", position=0) as species_progress_bar:

        # Tracking the specific species_no (i.e., downloaded = 1) and the number of images downloaded per species_no
        species_rows = tracking_df[tracking_df['species_no'] == species_no_to_download[i]]
        initial_downloaded_no = species_rows['downloaded_no'].iloc[0] if not species_rows.empty else 0

        # Call image retrieval function and update numbering variable
        number, downloaded_count, skipped_count, total_image_urls = image_retrieval(bird_master_df, function_input_df, progress_bar=species_progress_bar, number=initial_downloaded_no + 1)

        # Update the progress bar description
        species_progress_bar.set_description(
            f"Species {species_no_to_download[i]} Progress")

        # Print completion message for the current species
        # print(f'Completed working on species {i+1}/{len(species_no_to_download)} [species_no: {species_no_to_download[i]}].')

        # Check if species_no_to_download[i] exists in tracking_df
        if species_no_to_download[i] not in tracking_df['species_no'].values:
            # Update tracking_df based on the downloaded images
            species_folder_path = os.path.join(os.getcwd(), f"{species_no_to_download[i]}")

            # Create folder if it doesn't exist
            if not os.path.exists(species_folder_path):
                os.makedirs(species_folder_path, exist_ok=True)

            # Count the number of downloaded images
            downloaded_images_count = len([f for f in os.listdir(species_folder_path) if os.path.isfile(os.path.join(species_folder_path, f))])

            # Update tracking_df with downloaded and downloaded_no information
            new_row = {'species_no': species_no_to_download[i], 'downloaded': 1, 'downloaded_no': downloaded_images_count}
            tracking_df = tracking_df.append(new_row, ignore_index=True)

            # print(f"Species {species_no_to_download[i]} - After update - Tracking DataFrame:")
            # print(tracking_df)

        # Update the total progress bar after processing each species
        # total_progress_bar.update(species_progress_bar.n)

        # TRACKING_DF EXPORT
        tracking_df.to_csv(tracking_csv_path, index=False)

print(f"\nFinished download of {len(species_no_to_download)} specified species. Script ended.")
