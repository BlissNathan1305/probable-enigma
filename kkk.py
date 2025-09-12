import requests
import os
from PIL import Image
from io import BytesIO

# Add your Pixabay API key here
API_KEY = "52233499-a664b999bd91fc221f7dfa5db"

# Pixabay API endpoint for searching images
API_URL = "https://pixabay.com/api/"

# Folder to save downloaded wallpapers
SAVE_FOLDER = "abstract_wallpapers"
os.makedirs(SAVE_FOLDER, exist_ok=True)

def fetch_abstract_wallpapers(api_key, per_page=10, orientation='vertical', category='backgrounds'):
    """Fetches abstract wallpapers from Pixabay and saves them locally."""
    params = {
        'key': api_key,
        'q': 'abstract',
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
            
            # Resize image to 1080x1920 (mobile HD wallpaper) using new resampling method
            img = img.resize((1080, 1920), resample=Image.Resampling.LANCZOS)

            save_path = os.path.join(SAVE_FOLDER, f'abstract_wallpaper_{idx}.jpg')
            img.save(save_path, 'JPEG', quality=95)
            print(f"Saved wallpaper {save_path}")
        except Exception as e:
            print(f"Failed to download or save image {idx}: {e}")

if __name__ == "__main__":
    if API_KEY == "YOUR_PIXABAY_API_KEY":
        print("Please replace 'YOUR_PIXABAY_API_KEY' with your actual Pixabay API key.")
    else:
        fetch_abstract_wallpapers(API_KEY)

