from PIL import Image, ImageDraw

# Set the dimensions of the flag
width = 800
height = 500

# Create a new image with the specified dimensions
img = Image.new('RGB', (width, height))
draw = ImageDraw.Draw(img)

# Define the colors
red = (255, 0, 0)
white = (255, 255, 255)
blue = (0, 0, 255)

# Draw the stripes
stripe_height = height // 13
for i in range(13):
    if i % 2 == 0:
        draw.rectangle((0, i * stripe_height, width, (i + 1) * stripe_height), fill=red)
    else:
        draw.rectangle((0, i * stripe_height, width, (i + 1) * stripe_height), fill=white)

# Draw the blue rectangle
blue_rect_width = width * 0.4
blue_rect_height = stripe_height * 7
draw.rectangle((0, 0, blue_rect_width, blue_rect_height), fill=blue)

# Draw the stars
star_size = 10
star_spacing_x = 30
star_spacing_y = 30
num_stars_x = int(blue_rect_width // star_spacing_x)
num_stars_y = 9
for i in range(num_stars_y):
    for j in range(num_stars_x):
        if (i + j) % 2 == 0:
            x = j * star_spacing_x + star_spacing_x // 2
            y = i * star_spacing_y + star_spacing_y // 2
            draw.polygon([(x, y - star_size // 2), (x + star_size // 2, y + star_size // 2), (x - star_size // 2, y + star_size // 2)], fill=white)

# Save the image as a PNG file
img.save('usa_flag.png')
