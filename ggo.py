import requests
from PIL import Image, ImageDraw, ImageFont
import io
import os
import random

# --- CONFIGURATION ---
WALLPAPER_WIDTH = 2160  # 4K Ultra HD - Vertical Orientation
WALLPAPER_HEIGHT = 3840
JPEG_QUALITY = 95
FONT_PATH = "path/to/your/font.ttf"  # Replace with the actual path to your TrueType Font file (.ttf)
NUM_WALLPAPERS = 20
OUTPUT_DIR = "wallpapers" #Directory where wallpapers will be saved.

# --- PIXABAY API ---
PIXABAY_API_KEY = "52233499-a664b999bd91fc221f7dfa5db"  # Replace with your actual Pixabay API key
PIXABAY_CATEGORIES = ["nature", "sky", "mountains", "forest", "city", "abstract", "animals", "ocean", "space", "travel"]
PIXABAY_COLORS = ["grayscale", "transparent", "red", "orange", "yellow", "green", "turquoise", "blue", "violet", "pink", "brown", "black", "gray", "white"]


def download_image_from_pixabay(category, color=None):
    """Downloads a random image from Pixabay based on the given category and color."""
    url = "https://pixabay.com/api/"
    params = {
        "key": PIXABAY_API_KEY,
        "q": category,
        "image_type": "photo",
        "safesearch": "true",  # Enable safe search
        "per_page": 200  # Increase number of results per page (max 200)
    }

    if color:
        params["colors"] = color

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if data["totalHits"] == 0:
            print(f"No images found for category: {category} and color: {color}")
            return None

        image_url = random.choice(data["hits"])["largeImageURL"] # Use largeImageURL for higher resolution
        image_response = requests.get(image_url, stream=True)
        image_response.raise_for_status()
        image_response.raw.decode_content = True  # Handle compressed content

        return Image.open(image_response.raw)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None
    except Exception as e:
        print(f"Error processing Pixabay API response: {e}")
        return None


def create_gradient_wallpaper(color1, color2):
    """Creates a gradient wallpaper using two colors."""
    img = Image.new("RGB", (WALLPAPER_WIDTH, WALLPAPER_HEIGHT))
    draw = ImageDraw.Draw(img)
    for i in range(WALLPAPER_HEIGHT):
        # Linear interpolation for the color
        r = int(color1[0] + (color2[0] - color1[0]) * i / WALLPAPER_HEIGHT)
        g = int(color1[1] + (color2[1] - color1[1]) * i / WALLPAPER_HEIGHT)
        b = int(color1[2] + (color2[2] - color1[2]) * i / WALLPAPER_HEIGHT)
        draw.line((0, i, WALLPAPER_WIDTH, i), fill=(r, g, b))
    return img


def create_text_wallpaper(text, bg_color, text_color):
    """Creates a wallpaper with text on a solid background."""
    img = Image.new("RGB", (WALLPAPER_WIDTH, WALLPAPER_HEIGHT), bg_color)
    d = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(FONT_PATH, 150)
    except OSError:
        print(f"Error: Font file not found or invalid: {FONT_PATH}")
        print("Please ensure the path is correct and the file is a valid TrueType font.")
        return None

    text_width, text_height = d.textsize(text, font=font)
    x = (WALLPAPER_WIDTH - text_width) / 2
    y = (WALLPAPER_HEIGHT - text_height) / 2

    d.text((x, y), text, fill=text_color, font=font)
    return img

def create_pixabay_wallpaper(category, color=None):
    """Creates a wallpaper using a Pixabay image and optionally adds text."""
    image = download_image_from_pixabay(category, color)
    if not image:
        return None

    # Resize the image to fit the wallpaper dimensions (cover)
    image = image.resize((WALLPAPER_WIDTH, WALLPAPER_HEIGHT), Image.Resampling.LANCZOS)

    # Optionally add text overlay (example)
    # draw = ImageDraw.Draw(image)
    # font = ImageFont.truetype(FONT_PATH, 80)
    # text = f"From Pixabay: {category}"
    # text_width, text_height = draw.textsize(text, font=font)
    # x = 10
    # y = WALLPAPER_HEIGHT - text_height - 10
    # draw.text((x, y), text, fill=(255, 255, 255), font=font)

    return image


def main():
    """Generates multiple wallpapers using different methods."""

    # 1.  Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for i in range(NUM_WALLPAPERS):
        # 2. Randomly choose a wallpaper creation method
        method = random.choice(["gradient", "text", "pixabay"])

        try:
            if method == "gradient":
                color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                wallpaper = create_gradient_wallpaper(color1, color2)
                filename = f"{OUTPUT_DIR}/gradient_wallpaper_{i+1}.jpg"

            elif method == "text":
                text = f"Wallpaper {i+1}"
                bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                text_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                wallpaper = create_text_wallpaper(text, bg_color, text_color)
                if wallpaper is None:
                    continue #Skip if font error occurred
                filename = f"{OUTPUT_DIR}/text_wallpaper_{i+1}.jpg"

            elif method == "pixabay":
                category = random.choice(PIXABAY_CATEGORIES)
                color = random.choice(PIXABAY_COLORS) if random.random() < 0.5 else None  # 50% chance of using a color filter
                wallpaper = create_pixabay_wallpaper(category, color)
                if wallpaper is None:
                    continue  # Skip if download failed.
                filename = f"{OUTPUT_DIR}/pixabay_wallpaper_{i+1}.jpg"
            else:
                print("Invalid wallpaper creation method.")
                continue

            # 3. Save the wallpaper
            if wallpaper:
                wallpaper.save(filename, "JPEG", quality=JPEG_QUALITY)
                print(f"Wallpaper {i+1} created: {filename}")
            else:
                print(f"Failed to create wallpaper {i+1}")

        except Exception as e:
            print(f"Error creating wallpaper {i+1}: {e}")


if __name__ == "__main__":
    main()
