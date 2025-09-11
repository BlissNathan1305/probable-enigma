from PIL import Image, ImageDraw

# Set the dimensions of the flag
width = 600
height = 300

# Create a new image with the specified dimensions
img = Image.new('RGB', (width, height))

# Create a drawing context
draw = ImageDraw.Draw(img)

# Define the colors
green = (0, 135, 62)  # Nigeria's green flag color
white = (255, 255, 255)

# Calculate the width of each stripe
stripe_width = width // 3

# Draw the green stripes
draw.rectangle((0, 0, stripe_width, height), fill=green)
draw.rectangle((2 * stripe_width, 0, width, height), fill=green)

# Draw the white stripe
draw.rectangle((stripe_width, 0, 2 * stripe_width, height), fill=white)

# Save the image as a PNG file
img.save('nigeria_flag.png')
