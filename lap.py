import requests
from PIL import Image
from io import BytesIO
import os

# ===========================
# CONFIGURATION
# ===========================
API_KEY = "52233499-a664b999bd91fc221f7dfa5db"  # ⚠️ Replace with your actual key
OUTPUT_DIR = "mobile_wallpapers"
NUM_WALLPAPERS = 10
# Common mobile resolution (portrait): 1080x1920
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================
# PIXABAY API REQUEST
# ===========================
url = "https://pixabay.com/api/"
params = {
    "key": API_KEY,
    "q": "nature",          # Search term (change as desired: "abstract", "space", etc.)
    "image_type": "photo",
    "per_page": NUM_WALLPAPERS,
    "safesearch": "true",
    "orientation": "vertical",  # Better for mobile wallpapers
    "min_width": 1000,
    "min_height": 1800
}

print("Fetching images from Pixabay...")
response = requests.get(url, params=params)
data = response.json()

if "hits" not in data:
    print("Error fetching images:", data)
    exit()

# ===========================
# PROCESS IMAGES
# ===========================
for i, hit in enumerate(data["hits"][:NUM_WALLPAPERS], 1):
    image_url = hit["largeImageURL"]  # or use 'webformatURL' for smaller/faster
    print(f"Downloading image {i}: {image_url}")

    # Download image
    img_data = requests.get(image_url).content
    img = Image.open(BytesIO(img_data))

    # Resize and crop to fit target aspect ratio
    img_ratio = img.width / img.height
    target_ratio = TARGET_WIDTH / TARGET_HEIGHT

    if img_ratio > target_ratio:
        # Image is wider than target → crop sides
        new_height = img.height
        new_width = int(new_height * target_ratio)
        left = (img.width - new_width) // 2
        img = img.crop((left, 0, left + new_width, img.height))
    else:
        # Image is taller than target → crop top/bottom
        new_width = img.width
        new_height = int(new_width / target_ratio)
        top = (img.height - new_height) // 2
        img = img.crop((0, top, img.width, top + new_height))

    # Resize to exact dimensions
    img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)

    # Save as JPEG
    filename = os.path.join(OUTPUT_DIR, f"wallpaper_{i:02d}.jpg")
    img.save(filename, "JPEG", quality=95)
    print(f"✅ Saved: {filename}")

print(f"\n🎉 Done! {NUM_WALLPAPERS} wallpapers saved in '{OUTPUT_DIR}' folder.")
