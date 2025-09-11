from PIL import Image, ImageDraw

# Set the dimensions of the wallpaper (mobile device resolution)
width = 1080
height = 1920

# Create a new image with the specified dimensions
img = Image.new('RGB', (width, height), (135, 206, 235))  # Light blue background

# Create a drawing context
draw = ImageDraw.Draw(img)

# Draw a simple shape (e.g., circle)
circle_x = width // 2
circle_y = height // 2
circle_radius = 300
draw.ellipse([(circle_x - circle_radius, circle_y - circle_radius), (circle_x + circle_radius, circle_y + circle_radius)], fill=(255, 255, 255))  # White circle

# Add some subtle gradient effect
for i in range(circle_radius):
    draw.ellipse([(circle_x - (circle_radius - i), circle_y - (circle_radius - i)), (circle_x + (circle_radius - i), circle_y + (circle_radius - i))], outline=(200, 200, 200, int(255 * (1 - i / circle_radius))))

# Save the image as a PNG file
img.save('mobile_wallpaper.png')
