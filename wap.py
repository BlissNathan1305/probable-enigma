import requests
from PIL import Image
from io import BytesIO
import os

# Your Pixabay API key
API_KEY = "52233499-a664b999bd91fc221f7dfa5db"

# Create a folder for wallpapers
os.makedirs("wallpapers", exist_ok=True)

# Search queries for gorgeous wallpapers
queries = [
    "nature landscape",
    "galaxy stars",
    "ocean waves",
    "mountains sunrise",
    "forest waterfall",
    "night sky",
    "flower closeup",
    "abstract art",
    "desert sunset",
    "city skyline night"
]

# Desired wallpaper size
WIDTH, HEIGHT = 2160, 3840

for i, query in enumerate(queries, 1):
    url = f"https://pixabay.com/api/?key={API_KEY}&q={query.replace(' ', '+')}&image_type=photo&orientation=vertical&per_page=3"
    response = requests.get(url).json()
    
    if response["hits"]:
        image_url = response["hits"][0]["largeImageURL"]
        img_data = requests.get(image_url).content
        img = Image.open(BytesIO(img_data))
        
        # Resize and crop to 4K mobile size
        img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)
        
        # Save as JPEG
        filepath = f"wallpapers/wallpaper_{i}.jpg"
        img.save(filepath, "JPEG", quality=95)
        print(f"Saved: {filepath}")
    else:
        print(f"No results for {query}")
