import pandas as pd
import os
from io import BytesIO
from PIL import Image
import requests
from tqdm import tqdm
import sys

# Function to download images based on provided CSV
def download_images(csv_path, output_dir='aircraft_images', timeout=5):
    # Import CSV into a DataFrame
    df = pd.read_csv(csv_path, names=['aircraft_model', 'image_url'], skiprows=1)

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize numbering variable
    number = 1

    # Create and display the progress bar for total selection progress
    with tqdm(total=len(df), desc="Image Download Progress") as total_progress_bar:
        # Iterate over rows in the DataFrame
        for index, row in df.iterrows():
            aircraft_model = row['aircraft_model']
            image_url = row['image_url']

            # Extract filename from URL
            filename = os.path.join(output_dir, f"{aircraft_model}_{number:04d}.png")

            # Check if the file already exists, if yes, skip
            if os.path.exists(filename):
                continue

            # Attempt to download the image
            try:
                response = requests.get(image_url, timeout=timeout)
                response.raise_for_status()  # Raise an HTTPError for bad responses

                # Open the image and save it
                im = Image.open(BytesIO(response.content))
                im.save(filename)

                # Increment the numbering variable for the next image
                number += 1
            except requests.exceptions.RequestException as e:
                print(f"Error downloading image for {aircraft_model}: {e}")
                continue
            except Exception as e:
                print(f"Error processing image for {aircraft_model}: {e}")
                continue

            # Update the progress bar
            total_progress_bar.update(1)

if __name__ == "__main__":
    # Check if a CSV file path is provided as a command-line argument
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        print("Please provide the path to the CSV file with aircraft models and image URLs.")
        sys.exit(1)

    # Specify the output directory for images (optional, default is 'aircraft_images')
    output_directory = 'aircraft_images'

    # Set a timeout for image downloads (optional, default is 5 seconds)
    download_timeout = 5

    # Call the function to download images
    download_images(csv_path, output_directory, download_timeout)

    print("Image download completed.")
