from PIL import Image, ImageDraw, ImageFont

# Set the dimensions of the image
width = 1200
height = 800

# Create a new image with the specified dimensions
img = Image.new('RGB', (width, height), (255, 255, 255))
draw = ImageDraw.Draw(img)

# Define the font
font = ImageFont.load_default()

# Define the periodic table data
periodic_table = [
    ['H', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'He'],
    ['Li', 'Be', '', '', '', '', '', '', '', '', '', '', 'B', 'C', 'N', 'O', 'F', 'Ne'],
    ['Na', 'Mg', '', '', '', '', '', '', '', '', '', '', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'],
    ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'],
    ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe'],
    ['Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'],
    ['Fr', 'Ra', 'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
]

# Draw the periodic table
x_offset = 50
y_offset = 50
cell_width = 50
cell_height = 50
for i, row in enumerate(periodic_table):
    for j, element in enumerate(row):
        if element:
            draw.rectangle([(j * cell_width + x_offset, i * cell_height + y_offset), ((j + 1) * cell_width + x_offset, (i + 1) * cell_height + y_offset)], outline='black')
            draw.text((j * cell_width + x_offset + 10, i * cell_height + y_offset + 10), element, font=font, fill='black')

# Save the image as a PNG file
img.save('periodic_table.png')
