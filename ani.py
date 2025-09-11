import requests
import json
import os

# Your Pixabay API key
api_key = "52233499-a664b999bd91fc221f7dfa5db"  # Replace with your actual API key

# Set the parameters for the API request
params_list = [
    {
        "key": api_key,
        "q": "anime",  # Search query for anime wallpapers
        "image_type": "photo",
        "orientation": "vertical",  # For mobile wallpapers
        "min_width": 2160,  # Minimum width for 4K resolution
        "min_height": 3840,  # Minimum height for 4K resolution
        "per_page": 10  # Number of images to fetch
    },
    {
        "key": api_key,
        "q": "avatar",  # Search query for avatar wallpapers
        "image_type": "photo",
        "orientation": "vertical",  # For mobile wallpapers
        "min_width": 2160,  # Minimum width for 4K resolution
        "min_height": 3840,  # Minimum height for 4K resolution
        "per_page": 10  # Number of images to fetch
    }
]

# Create a directory to store the wallpapers
directory = "wallpapers"
if not os.path.exists(directory):
    os.makedirs(directory)

# Make API requests and download images
for i, params in enumerate(params_list):
    response = requests.get("https://pixabay.com/api/", params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = json.loads(response.text)
        
        # Download the images
        for j, image in enumerate(data["hits"]):
            image_url = image["largeImageURL"]
            image_response = requests.get(image_url)
            if image_response.status_code == 200:
                # Save the image to the directory
                if i == 0:
                    filename = f"anime_wallpaper_{j+1}.jpg"
                else:
                    filename = f"avatar_wallpaper_{j+1}.jpg"
                filepath = os.path.join(directory, filename)
                with open(filepath, "wb") as file:
                    file.write(image_response.content)
                print(f"Downloaded {filename}")
            else:
                print(f"Failed to download image {j+1}")
    else:
        print(f"Failed to retrieve images for query {params['q']}")
