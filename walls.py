from PIL import Image, ImageDraw, ImageFilter

# Set the dimensions of the wallpaper (4K resolution for mobile devices)
width = 2160
height = 3840

# Create a new image with the specified dimensions
img = Image.new('RGB', (width, height), (30, 40, 60))  # Dark blue background

# Create a drawing context
draw = ImageDraw.Draw(img)

# Draw a beautiful gradient
for y in range(height):
    gradient_color = int(50 + 100 * y / height)
    draw.line([(0, y), (width, y)], fill=(gradient_color, gradient_color, 150))

# Add some beautiful shapes
import random
for _ in range(100):
    x = random.randint(0, width)
    y = random.randint(0, height)
    size = random.randint(10, 50)
    draw.ellipse([(x, y), (x + size, y + size)], fill=(200, 200, 255, 128))

# Apply a blur filter to the shapes
img = img.filter(ImageFilter.GaussianBlur(radius=2))

# Save the image as a JPEG file
img.save('mobile_wallpaper_4k.jpg', 'JPEG', quality=95)
