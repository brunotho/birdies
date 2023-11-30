### IMPORTS
import pandas as pd
import requests
from io import StringIO, BytesIO
from PIL import Image
import os
import sys

### SELECTED SPECIES SELECTION
if len(sys.argv) > 1:
    csvpath = sys.argv[1]
else:
    csvpath = input("Please enter the path to the CSV file with selected bird species for image retrieval\n(format: index,species_no,scientific_name)")

## Importing the selected species CSV per command-line into a data frame
selected_birds_df = pd.read_csv(csvpath)

### CREATING A CSV FOR DOWNLOAD LOGGING [if non-existent]
## Specify the directory path
bird_data_dir = "../bird_data/" # THIS MAY CALL FOR AN ADJUSTMENT IF DEPLOYED ON A DIFFERENT MACHINE / ENIVRONMENT
tracking_csv_path = os.path.join(bird_data_dir, "tracking.csv")

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

### PRIMARY IMAGE RETRIEVAL FUNCTION | THE HEART OF THIS SCRIPT
def image_retrieval(bird_master_df, selected_birds_df, mydir=None, size=(256, 256), number=1):
    '''Download images based on master URL file (~10Mio. entries for ~11,000 species) for selected species'''

    ## Temporary merged data frame based on a filtered version of bird_master_df
    # ONLY NECCESSARY AS LONG AS WORKING WITH A SELECTION OF BIRDS
    temp_df = bird_master_df[bird_master_df['species_no'] == selected_birds_df['species_no'].iloc[0]]

    temp_df_row_count = temp_df.shape[0]

    temp_list = []

    print(f"temp_df created with {temp_df_row_count} rows")

    itercount = 0

    ## Iterate over rows in temp_df
    for index, row in temp_df.iterrows():

        itercount += 1

        print(f"Now working on row {itercount} / {temp_df_row_count} in temp_df. There are {temp_df_row_count - (itercount)} rows left.")

        if row['image_id'] in temp_list:
            print("Already successfully downloaded a picture from this list of duplicate URLs.")
            continue

        # Locate download links
        url_im = row['image_url']

        # Attempt download via link ('image_url') from temp_df
        try:
            response = requests.get(url_im, timeout=5)
            # Raise an HTTPError for bad responses
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image from {url_im}: {e}")
            # Skip to the next iteration
            continue

        # Check if the download was successful
        if response.status_code == 200:

            try:
                im = Image.open(BytesIO(response.content))

                # Resize image using im.thumbnail(size, Image.ANTIALIAS) with 256 as the maximum dimension while keeping original image proportions
                im.thumbnail(size, Image.LANCZOS)

                # Create folder if it doesn't exist
                if mydir is None:
                    mydir = os.getcwd()
                    species_folder = os.path.join(mydir, f"{row['species_no']}")
                    os.makedirs(species_folder, exist_ok=True)

                # Save resulting image via im.save as [species_no]_[xxxx].png in the corresponding folder created above
                im_dir = os.path.join(species_folder, f"{row['species_no']}_{number:04d}.png")

                # Check if the file already exists, if not, save the image
                if not os.path.exists(im_dir):
                    im.save(im_dir)

                    # Remove duplicate image_ids from temp_df
                    temp_list.append(row['image_id'])

                    print(f"Saved: {im_dir}")
                else:
                    print(f"Skipped (already exists): {im_dir}")

                # Increment the numbering variable for the next image
                number += 1

            except Exception as e:
                print(f"Error processing image: {e}")

    print('Iterating over the rows of temp_df has been completed.')
    return number  # Return the updated numbering variable

print(species_no_to_download)

for i in range(len(species_no_to_download)):
    print(f"Now starting to work on species {i+1} / {len(species_no_to_download)}: species_no {species_no_to_download[i]}.")

    ### CREATING A MINI-DF AS INPUT FOR THE DOWNLOAD RETRIEVAL FUNCTION
    function_input_df = selected_birds_df.loc[selected_birds_df['species_no'] == species_no_to_download[i]]
    print(function_input_df)

    ### THE PART BELOW CREATES A DOWNLOAD LOGGING DATA FRAME & CSV
    ## Tracking the specific species_no (i. e. downloaded = 1) and the number of images downloaded per species_no

    # Check if tracking_df has any matching rows for the current species_no
    species_rows = tracking_df[tracking_df['species_no'] == species_no_to_download[i]]

    # Check if there are any rows before accessing values
    if not species_rows.empty:
        # If there are rows, get the first value
        initial_downloaded_no = species_rows['downloaded_no'].iloc[0]
    else:
        # If no rows are found, set initial_downloaded_no to 0
        initial_downloaded_no = 0

    ### THIS IS WHERE THE ACTUAL DOWNLOAD HAPPENS
    # Call image retrieval function and update numbering variable
    initial_downloaded_no = image_retrieval(bird_master_df, function_input_df, number=initial_downloaded_no + 1)

    # Check if species_no_to_download[i] exists in tracking_df
    if species_no_to_download[i] in tracking_df['species_no'].values:
        print(f"Species {species_no_to_download[i]} already exists in tracking_df.")
    else:
        # Update tracking_df based on the downloaded images
        species_folder_path = os.path.join(os.getcwd(), f"{species_no_to_download[i]}")

        # Create folder if it doesn't exist
        if not os.path.exists(species_folder_path):
            os.makedirs(species_folder_path, exist_ok=True)

        # Count the number of downloaded images
        downloaded_images_count = len([f for f in os.listdir(species_folder_path) if os.path.isfile(os.path.join(species_folder_path, f))])

        # Update tracking_df with downloaded and downloaded_no information
        new_row = {'species_no': species_no_to_download[i], 'downloaded': 1, 'downloaded_no': downloaded_images_count}
        print(f"Adding new row to tracking_df:\n{new_row}")
        tracking_df.loc[len(tracking_df)] = new_row

        print(f"Species {species_no_to_download[i]} - After update - Tracking DataFrame:")
        print(tracking_df)

### TRACKING_DF EXPORT
tracking_df.to_csv(tracking_csv_path, index=False)
print(f"Finished download of {len(species_no_to_download)} specified species. Script ended.")
