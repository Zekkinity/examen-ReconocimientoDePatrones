import requests
import zipfile
import os

SEED = 222216237
A = 7
B = 3

def get_file(url, extract_dir):
    local_filename = "downloaded_file.zip"

    # 1. Send a GET request to the URL
    response = requests.get(url)

    # 2. Check if the request was successful (status code 200)
    if response.status_code == 200:
        # 3. Open a local file in write-binary ('wb') mode and save the content
        with open(local_filename, "wb") as file:
            file.write(response.content)
        print("Download complete!")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

    # Open the ZIP file in read mode ('r')
    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        # Extract all the contents into the destination folder
        zip_ref.extractall(extract_dir)

    print("Extraction complete!")

    # Clean up the downloaded file
    os.remove(local_filename)

