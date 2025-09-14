import requests
import json
import os

# Set your Pixabay API key
API_KEY = "52233499-a664b999bd91fc221f7dfa5db"

# Set the wallpaper dimensions (2160x3840 resolution for mobile devices)
WIDTH, HEIGHT = 2160, 3840

# Set the texture categories
TEXTURE_CATEGORIES = ["wood", "stone", "metal", "fabric"]

# Set the number of wallpapers to generate
NUM_WALLPAPERS = 15

def get_texture_image(category):
    url = f"https://pixabay.com/api/?key={API_KEY}&q={category}&image_type=photo&orientation=vertical&min_width={WIDTH}&min_height={HEIGHT}&per_page={NUM_WALLPAPERS}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve data for {category}. Status code: {response.status_code}")
        return []
    try:
        data = json.loads(response.text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON for {category}: {e}")
        return []
    image_urls = [hit["largeImageURL"] for hit in data["hits"]]
    return image_urls

def generate_wallpaper(image_url, category, index):
    response = requests.get(image_url)
    if response.status_code != 200:
        print(f"Failed to download image for {category}. Status code: {response.status_code}")
        return None
    image_path = f"{category}_wallpaper_{index}.jpg"
    with open(image_path, "wb") as file:
        file.write(response.content)
    return image_path

def main():
    for category in TEXTURE_CATEGORIES:
        image_urls = get_texture_image(category)
        for i, image_url in enumerate(image_urls):
            image_path = generate_wallpaper(image_url, category, i+1)
            if image_path:
                print(f"Wallpaper generated for {category} texture: {image_path}")

if __name__ == "__main__":
    main()
