import requests
import os
from PIL import Image
from io import BytesIO

# Your Pixabay API key
API_KEY = "52233499-a664b999bd91fc221f7dfa5db"

# Pixabay API endpoint for searching images
API_URL = "https://pixabay.com/api/"

# Folder to save downloaded wallpapers
SAVE_FOLDER = "fractal_wallpapers"
os.makedirs(SAVE_FOLDER, exist_ok=True)

def fetch_fractal_wallpapers(api_key, per_page=10, orientation='vertical', category='backgrounds'):
    """Fetches fractal wallpapers from Pixabay and saves them locally in 4K resolution."""
    params = {
        'key': api_key,
        'q': 'fractal',
        'image_type': 'photo',
        'orientation': orientation,
        'category': category,
        'per_page': per_page,
        'safesearch': 'true'
    }
    response = requests.get(API_URL, params=params)
    response.raise_for_status()
    data = response.json()

    print(f"Found {data.get('totalHits', 0)} images.")
    
    for idx, hit in enumerate(data.get('hits', []), 1):
        img_url = hit['largeImageURL']
        try:
            img_response = requests.get(img_url)
            img_response.raise_for_status()
            img = Image.open(BytesIO(img_response.content))
            
            # Resize image to 2160x3840 (4K vertical mobile wallpaper)
            img = img.resize((2160, 3840), resample=Image.Resampling.LANCZOS)

            save_path = os.path.join(SAVE_FOLDER, f'fractal_wallpaper_{idx}.jpg')
            img.save(save_path, 'JPEG', quality=95)
            print(f"Saved wallpaper {save_path}")
        except Exception as e:
            print(f"Failed to download or save image {idx}: {e}")

if __name__ == "__main__":
    fetch_fractal_wallpapers(API_KEY)

