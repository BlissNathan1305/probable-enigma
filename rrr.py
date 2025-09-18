from PIL import Image, ImageDraw, ImageFont

# Canvas size and cell size
cell_width, cell_height = 120, 150
cols, rows = 18, 9
width, height = cols * cell_width, rows * cell_height

# Colors for categories
colors = {
    "alkali": "#FF6666",
    "alkaline_earth": "#FFDEAD",
    "transition": "#FFD700",
    "post_transition": "#90EE90",
    "metalloid": "#66CCCC",
    "nonmetal": "#FFA500",
    "halogen": "#FF69B4",
    "noble_gas": "#9ACD32",
    "lanthanide": "#DA70D6",
    "actinide": "#BA55D3",
    "unknown": "#D3D3D3",
}

# Example elements (symbol, atomic number, category, col, row)
elements = [
    ("H", 1, "nonmetal", 1, 1),
    ("He", 2, "noble_gas", 18, 1),
    ("Li", 3, "alkali", 1, 2),
    ("Be", 4, "alkaline_earth", 2, 2),
    ("B", 5, "metalloid", 13, 2),
    ("C", 6, "nonmetal", 14, 2),
    ("N", 7, "nonmetal", 15, 2),
    ("O", 8, "nonmetal", 16, 2),
    ("F", 9, "halogen", 17, 2),
    ("Ne", 10, "noble_gas", 18, 2),
    ("Na", 11, "alkali", 1, 3),
    ("Mg", 12, "alkaline_earth", 2, 3),
    ("Al", 13, "post_transition", 13, 3),
    ("Si", 14, "metalloid", 14, 3),
    ("P", 15, "nonmetal", 15, 3),
    ("S", 16, "nonmetal", 16, 3),
    ("Cl", 17, "halogen", 17, 3),
    ("Ar", 18, "noble_gas", 18, 3),
    # More elements can be added here
]

# Create image and drawing context
image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)

# Load a font or use default
try:
    font_small = ImageFont.truetype("arial.ttf", 24)
    font_large = ImageFont.truetype("arial.ttf", 60)
except IOError:
    font_small = ImageFont.load_default()
    font_large = ImageFont.load_default()

for symbol, number, category, col, row in elements:
    x = (col - 1) * cell_width
    y = (row - 1) * cell_height
    color = colors.get(category, colors["unknown"])

    # Draw rectangle
    draw.rectangle([x, y, x + cell_width - 1, y + cell_height - 1], fill=color, outline="black")

    # Draw atomic number (top-left)
    draw.text((x + 10, y + 10), str(number), fill="black", font=font_small)

    # Calculate text size with textbbox
    bbox = draw.textbbox((0, 0), symbol, font=font_large)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # Draw element symbol (centered)
    draw.text((x + (cell_width - w) / 2, y + (cell_height - h) / 2), symbol, fill="black", font=font_large)

# Save as JPEG
image.save("periodic_table.jpg", "JPEG")
print("Periodic table image saved as periodic_table.jpg")

