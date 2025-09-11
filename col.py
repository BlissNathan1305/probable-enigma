from PIL import Image, ImageDraw

# Set the dimensions of the image
width = 1000
height = 400

# Create a new image with the specified dimensions
img = Image.new('RGB', (width, height))
draw = ImageDraw.Draw(img)

# Define the colors
colors = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),  # Dark Red
    (0, 128, 0),  # Dark Green
    (0, 0, 128),  # Dark Blue
    (128, 128, 0),  # Brown
    (255, 165, 0),  # Orange
    (75, 0, 130),  # Indigo
    (238, 84, 144),  # Pink
    (34, 139, 34),  # Forest Green
    (0, 128, 128),  # Teal
    (128, 0, 128),  # Purple
    (255, 215, 0),  # Golden
    (173, 255, 47),  # Green-Yellow
    (135, 206, 235),  # Sky Blue
    (160, 82, 45)  # Brown-Tan
]

# Draw the colors
box_width = width // len(colors)
box_height = height
for i, color in enumerate(colors):
    draw.rectangle([(i * box_width, 0), ((i + 1) * box_width, box_height)], fill=color)

# Save the image as a PNG file
img.save('colors.png')
