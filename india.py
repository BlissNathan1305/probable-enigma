from PIL import Image, ImageDraw
import math

# Set the dimensions of the flag
width = 600
height = 400

# Create a new image with the specified dimensions
img = Image.new('RGB', (width, height))
pixels = img.load()

# Define the colors
saffron = (255, 153, 0)
white = (255, 255, 255)
green = (0, 128, 0)
blue = (0, 0, 139)  # Ashoka Chakra color

# Function to draw a wavy line
def draw_wave(y_offset, color):
    for x in range(width):
        y_wave = int(height / 3 + 20 * math.sin(x / 50.0) + y_offset)
        for y in range(height):
            if y < y_wave:
                pixels[x, y] = color
            else:
                break

# Draw the saffron stripe
draw_wave(0, saffron)

# Draw the white stripe
draw_wave(height / 3, white)

# Draw the Ashoka Chakra
chakra_radius = 30
chakra_center_x = width // 2
chakra_center_y = int(height / 2)
for x in range(width):
    for y in range(height):
        distance = math.sqrt((x - chakra_center_x) ** 2 + (y - chakra_center_y) ** 2)
        if distance < chakra_radius:
            pixels[x, y] = blue

# Draw the green stripe
draw_wave(2 * height / 3, green)

# Save the image as a PNG file
img.save('indian_flag_wavy.png')
