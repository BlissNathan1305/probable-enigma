import requests
from PIL import Image
from io import BytesIO
import os

API_KEY = "52233499-a664b999bd91fc221f7dfa5db"
SEARCH_QUERY = "nature"
IMAGE_TYPE = "photo"
ORIENTATION = "vertical"
PER_PAGE = 5
OUTPUT_DIR = "pixabay_wallpapers"

def fetch_pixabay_images(api_key, query, image_type, orientation, per_page):
    url = "https://pixabay.com/api/"
    params = {
        "key": api_key,
        "q": query,
        "image_type": image_type,
        "orientation": orientation,
        "per_page": per_page,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def download_and_save_images(image_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    for i, hit in enumerate(image_data.get("hits", [])):
        image_url = hit.get("largeImageURL")
        if image_url:
            res = requests.get(image_url)
            img = Image.open(BytesIO(res.content))
            save_path = os.path.join(output_dir, f"wallpaper_{i+1}.png")
            img.save(save_path)
            saved_paths.append(save_path)
    return saved_paths

def main():
    data = fetch_pixabay_images(API_KEY, SEARCH_QUERY, IMAGE_TYPE, ORIENTATION, PER_PAGE)
    saved_images = download_and_save_images(data, OUTPUT_DIR)
    print(f"Downloaded and saved wallpapers: {saved_images}")

if __name__ == "__main__":
    main()

