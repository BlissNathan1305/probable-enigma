from PIL import Image, ImageDraw

def generate_gradient_wallpaper(width, height, start_color, end_color, output_filename):
    """
    Generates a 4K vertical wallpaper with a linear gradient.
    
    Args:
        width (int): The width of the wallpaper in pixels.
        height (int): The height of the wallpaper in pixels.
        start_color (tuple): The RGB tuple for the starting color (e.g., (100, 149, 237) for cornflower blue).
        end_color (tuple): The RGB tuple for the ending color (e.g., (255, 255, 255) for white).
        output_filename (str): The name of the output JPEG file.
    """
    # Create a new blank image with the specified dimensions and RGB mode
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)

    # Calculate the difference between the start and end colors for each channel
    r_diff = end_color[0] - start_color[0]
    g_diff = end_color[1] - start_color[1]
    b_diff = end_color[2] - start_color[2]
    
    # Iterate over each row (y-coordinate) to draw the gradient
    for y in range(height):
        # Calculate the interpolation factor based on the current row
        interpolation_factor = y / (height - 1)
        
        # Interpolate the color for the current row
        r = int(start_color[0] + r_diff * interpolation_factor)
        g = int(start_color[1] + g_diff * interpolation_factor)
        b = int(start_color[2] + b_diff * interpolation_factor)
        
        # Draw a horizontal line with the interpolated color
        draw.line((0, y, width, y), fill=(r, g, b))

    # Save the image as a JPEG file with a high quality setting
    img.save(output_filename, "JPEG", quality=95)
    print(f"Successfully generated and saved {output_filename}")

# --- Customize and Run ---

if __name__ == "__main__":
    # Standard 4K resolution for mobile (vertical)
    WIDTH = 2160
    HEIGHT = 3840

    # Choose your colors (RGB tuples from 0-255)
    # Example 1: A cool blue to light gray gradient 🧊
    start_color_1 = (0, 0, 100)      # Deep Navy Blue
    end_color_1 = (200, 200, 255)    # Very Light Blue
    output_filename_1 = "4k_mobile_wallpaper_blue.jpeg"
    generate_gradient_wallpaper(WIDTH, HEIGHT, start_color_1, end_color_1, output_filename_1)
    
    # Example 2: A warm orange to pink gradient 🌅
    start_color_2 = (255, 140, 0)    # Dark Orange
    end_color_2 = (255, 20, 147)     # Deep Pink
    output_filename_2 = "4k_mobile_wallpaper_pink.jpeg"
    generate_gradient_wallpaper(WIDTH, HEIGHT, start_color_2, end_color_2, output_filename_2)
