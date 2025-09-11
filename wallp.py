import requests
import os
from PIL import Image

# Your Pixabay API key
API_KEY = "52233499-a664b999bd91fc221f7dfa5db"

# Search query for wallpapers
QUERY = "beautiful landscape wallpaper"

# Number of wallpapers to download
NUM_WALLPAPERS = 10

# Output folder
OUTPUT_DIR = "pixabay_wallpapers"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pixabay API endpoint
url = "https://pixabay.com/api/"

params = {
    "key": API_KEY,
    "q": QUERY,
    "image_type": "photo",
    "orientation": "vertical",   # optimized for mobile
    "per_page": NUM_WALLPAPERS,
    "safesearch": "true"
}

# Send request
response = requests.get(url, params=params)
data = response.json()

# Download wallpapers
for i, hit in enumerate(data["hits"], start=1):
    img_url = hit["largeImageURL"]
    img_data = requests.get(img_url).content
    file_path = os.path.join(OUTPUT_DIR, f"wallpaper_{i}.jpg")
    with open(file_path, "wb") as f:
        f.write(img_data)
    print(f"Downloaded: {file_path}")

print("✅ Wallpapers downloaded successfully!")
