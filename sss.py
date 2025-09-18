from PIL import Image, ImageDraw, ImageFilter
import math
import os
import colorsys

# 4K Mobile resolution (portrait)
WIDTH = 2160
HEIGHT = 3840

def create_gradient(width, height, colors, direction='vertical', style='linear'):
    """Create various gradient styles"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    if style == 'linear':
        if direction == 'vertical':
            for y in range(height):
                ratio = y / height
                color = interpolate_colors(colors, ratio)
                draw.line([(0, y), (width, y)], fill=color)
        elif direction == 'horizontal':
            for x in range(width):
                ratio = x / width
                color = interpolate_colors(colors, ratio)
                draw.line([(x, 0), (x, height)], fill=color)
        elif direction == 'diagonal':
            for y in range(0, height, 2):  # Skip pixels for performance
                for x in range(0, width, 2):
                    ratio = (x + y) / (width + height)
                    color = interpolate_colors(colors, ratio)
                    draw.rectangle([x, y, x+1, y+1], fill=color)
    
    elif style == 'radial':
        center_x, center_y = width // 2, height // 2
        max_distance = math.sqrt(center_x**2 + center_y**2)
        
        for y in range(0, height, 2):  # Skip pixels for performance
            for x in range(0, width, 2):
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                ratio = min(distance / max_distance, 1.0)
                color = interpolate_colors(colors, ratio)
                draw.rectangle([x, y, x+1, y+1], fill=color)
    
    return img

def interpolate_colors(colors, ratio):
    """Smooth color interpolation between multiple colors"""
    if len(colors) < 2:
        return colors[0]
    
    ratio = max(0, min(1, ratio))
    segment_size = 1.0 / (len(colors) - 1)
    segment = int(ratio / segment_size)
    
    if segment >= len(colors) - 1:
        return colors[-1]
    
    local_ratio = (ratio - segment * segment_size) / segment_size
    
    c1, c2 = colors[segment], colors[segment + 1]
    return (
        int(c1[0] + (c2[0] - c1[0]) * local_ratio),
        int(c1[1] + (c2[1] - c1[1]) * local_ratio),
        int(c1[2] + (c2[2] - c1[2]) * local_ratio)
    )

def create_geometric_circles(width, height, base_color=(50, 50, 50)):
    """Create overlapping circles pattern"""
    img = Image.new('RGB', (width, height), base_color)
    draw = ImageDraw.Draw(img)
    
    # Create multiple overlapping circles
    colors = [
        (255, 100, 150, 100),  # Pink
        (100, 150, 255, 100),  # Blue
        (150, 255, 100, 100),  # Green
        (255, 200, 100, 100),  # Orange
        (200, 100, 255, 100),  # Purple
    ]
    
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    for i, color in enumerate(colors):
        # Calculate circle position and size
        angle = (i / len(colors)) * 2 * math.pi
        radius = min(width, height) // 3
        center_x = width // 2 + int(math.cos(angle) * width // 6)
        center_y = height // 2 + int(math.sin(angle) * height // 8)
        
        bbox = [
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius
        ]
        overlay_draw.ellipse(bbox, fill=color)
    
    # Blend with base image
    result = Image.alpha_composite(img.convert('RGBA'), overlay)
    return result.convert('RGB')

def create_wave_pattern(width, height, wave_colors):
    """Create flowing wave pattern"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    # Create background gradient
    for y in range(height):
        ratio = y / height
        color = interpolate_colors(wave_colors, ratio)
        draw.line([(0, y), (width, y)], fill=color)
    
    # Add wave overlay
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Create multiple wave layers
    for wave in range(3):
        wave_height = height // 6
        wave_y = height // 4 + wave * height // 4
        
        points = []
        for x in range(0, width + 20, 20):
            offset = math.sin((x / width) * 4 * math.pi + wave * math.pi / 3) * wave_height
            y = wave_y + int(offset)
            points.append((x, y))
        
        # Create wave shape
        wave_points = points[:]
        wave_points.extend([(width, height), (0, height)])
        
        wave_color = (*wave_colors[wave + 1], 80)
        overlay_draw.polygon(wave_points, fill=wave_color)
    
    result = Image.alpha_composite(img.convert('RGBA'), overlay)
    return result.convert('RGB')

def create_minimal_bars(width, height, colors, orientation='vertical'):
    """Create minimal color bar design"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    if orientation == 'vertical':
        bar_width = width // len(colors)
        for i, color in enumerate(colors):
            x1 = i * bar_width
            x2 = (i + 1) * bar_width if i < len(colors) - 1 else width
            draw.rectangle([x1, 0, x2, height], fill=color)
    else:  # horizontal
        bar_height = height // len(colors)
        for i, color in enumerate(colors):
            y1 = i * bar_height
            y2 = (i + 1) * bar_height if i < len(colors) - 1 else height
            draw.rectangle([0, y1, width, y2], fill=color)
    
    return img

def create_sunburst(width, height, center_color, outer_color):
    """Create sunburst/radial gradient"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    center_x, center_y = width // 2, height // 2
    max_distance = math.sqrt(center_x**2 + center_y**2)
    
    # Draw from outside in for better performance
    for radius in range(int(max_distance), 0, -5):
        ratio = radius / max_distance
        color = interpolate_colors([center_color, outer_color], ratio)
        
        bbox = [
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius
        ]
        draw.ellipse(bbox, fill=color, outline=color)
    
    return img

def generate_wallpaper(style_id):
    """Generate wallpaper based on style"""
    
    styles = {
        1: {
            'name': 'ocean_gradient',
            'generator': lambda: create_gradient(WIDTH, HEIGHT, 
                [(20, 50, 80), (30, 120, 180), (100, 200, 255), (200, 240, 255)], 
                'vertical')
        },
        2: {
            'name': 'sunset_gradient',
            'generator': lambda: create_gradient(WIDTH, HEIGHT,
                [(80, 30, 120), (255, 100, 50), (255, 200, 100), (255, 255, 200)],
                'vertical')
        },
        3: {
            'name': 'forest_gradient',
            'generator': lambda: create_gradient(WIDTH, HEIGHT,
                [(20, 40, 20), (40, 100, 40), (80, 160, 80), (150, 220, 150)],
                'vertical')
        },
        4: {
            'name': 'radial_purple',
            'generator': lambda: create_gradient(WIDTH, HEIGHT,
                [(100, 50, 150), (200, 100, 255), (255, 200, 255)],
                style='radial')
        },
        5: {
            'name': 'geometric_circles',
            'generator': lambda: create_geometric_circles(WIDTH, HEIGHT, (25, 25, 35))
        },
        6: {
            'name': 'wave_blue',
            'generator': lambda: create_wave_pattern(WIDTH, HEIGHT,
                [(20, 50, 100), (50, 150, 200), (100, 200, 255), (200, 230, 255)])
        },
        7: {
            'name': 'minimal_warm',
            'generator': lambda: create_minimal_bars(WIDTH, HEIGHT,
                [(255, 180, 100), (255, 150, 80), (255, 120, 60), (200, 80, 40)],
                'horizontal')
        },
        8: {
            'name': 'sunburst_gold',
            'generator': lambda: create_sunburst(WIDTH, HEIGHT, (255, 220, 100), (150, 50, 20))
        },
        9: {
            'name': 'diagonal_mint',
            'generator': lambda: create_gradient(WIDTH, HEIGHT,
                [(100, 200, 150), (150, 255, 200), (200, 255, 230)],
                'diagonal')
        },
        10: {
            'name': 'midnight_blue',
            'generator': lambda: create_gradient(WIDTH, HEIGHT,
                [(15, 15, 30), (30, 50, 80), (60, 100, 150)],
                'vertical')
        },
        11: {
            'name': 'coral_pink',
            'generator': lambda: create_gradient(WIDTH, HEIGHT,
                [(255, 150, 150), (255, 200, 180), (255, 230, 200)],
                style='radial')
        },
        12: {
            'name': 'minimalist_gray',
            'generator': lambda: create_minimal_bars(WIDTH, HEIGHT,
                [(240, 240, 240), (200, 200, 200), (160, 160, 160), (120, 120, 120)],
                'vertical')
        }
    }
    
    style = styles[style_id]
    print(f"Generating {style['name']}...")
    
    wallpaper = style['generator']()
    
    # Apply subtle blur for smoothness
    if style_id in [4, 5, 8, 11]:  # Only for certain styles
        wallpaper = wallpaper.filter(ImageFilter.GaussianBlur(radius=1))
    
    return wallpaper, style['name']

def main():
    """Generate 12 simple 4K mobile wallpapers"""
    print("Starting 4K mobile wallpaper generation...")
    print(f"Resolution: {WIDTH}x{HEIGHT} (4K Mobile Portrait)")
    
    # Create output directory
    output_dir = "simple_4k_mobile_wallpapers"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate 12 different styles
    for i in range(1, 13):
        print(f"Generating wallpaper {i}/12...")
        
        try:
            wallpaper, name = generate_wallpaper(i)
            
            # Save as high-quality JPEG
            filename = f"{output_dir}/{i:02d}_{name}_4k_mobile.jpg"
            wallpaper.save(filename, "JPEG", quality=95, optimize=True)
            
            # Get file size
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            print(f"Saved: {filename} ({file_size:.1f} MB)")
            
        except Exception as e:
            print(f"Error generating wallpaper {i}: {e}")
    
    print(f"\n✅ Completed! Generated 12 simple 4K mobile wallpapers in '{output_dir}' directory")
    print(f"📱 Resolution: {WIDTH}x{HEIGHT} (4K Mobile Portrait)")
    print("🎨 Styles: Gradients, geometric patterns, waves, sunbursts, and minimalist designs")
    print("📸 Format: High-quality JPEG (95% quality)")

if __name__ == "__main__":
    main()
