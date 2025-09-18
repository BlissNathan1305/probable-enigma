import math
import random
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import os

# Mobile wallpaper dimensions (common resolutions)
WIDTH = 1080
HEIGHT = 1920

def noise(x, y, seed=0):
    """Simple hash-based noise function for performance"""
    n = int(x + y * 57 + seed * 131)
    n = (n << 13) ^ n
    return (1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0)

def smooth_noise(x, y, seed=0):
    """Smoothed noise using bilinear interpolation"""
    int_x, int_y = int(x), int(y)
    frac_x, frac_y = x - int_x, y - int_y
    
    v1 = noise(int_x, int_y, seed)
    v2 = noise(int_x + 1, int_y, seed)
    v3 = noise(int_x, int_y + 1, seed)
    v4 = noise(int_x + 1, int_y + 1, seed)
    
    i1 = v1 * (1 - frac_x) + v2 * frac_x
    i2 = v3 * (1 - frac_x) + v4 * frac_x
    
    return i1 * (1 - frac_y) + i2 * frac_y

def fractal_noise(x, y, octaves=6, persistence=0.5, scale=0.01, seed=0):
    """Multi-octave fractal noise"""
    value = 0
    amplitude = 1
    frequency = scale
    max_value = 0
    
    for i in range(octaves):
        value += smooth_noise(x * frequency, y * frequency, seed + i) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= 2
    
    return value / max_value

def create_gradient(width, height, colors, direction='vertical'):
    """Create smooth color gradient"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    if direction == 'vertical':
        for y in range(height):
            ratio = y / height
            color = interpolate_colors(colors, ratio)
            draw.line([(0, y), (width, y)], fill=color)
    else:  # horizontal
        for x in range(width):
            ratio = x / width
            color = interpolate_colors(colors, ratio)
            draw.line([(x, 0), (x, height)], fill=color)
    
    return img

def interpolate_colors(colors, ratio):
    """Interpolate between multiple colors"""
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

def generate_mountain_silhouette(width, height, layers=3, seed=42):
    """Generate layered mountain silhouettes"""
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    colors = [
        (20, 30, 50, 200),    # Dark mountains
        (40, 60, 90, 150),    # Mid mountains  
        (80, 100, 130, 100)   # Light mountains
    ]
    
    for layer in range(layers):
        points = []
        base_height = height - (height // 4) + layer * (height // 8)
        
        for x in range(0, width + 50, 20):
            noise_val = fractal_noise(x, layer, octaves=4, scale=0.005, seed=seed + layer)
            y = base_height + int(noise_val * height * 0.3)
            points.append((x, y))
        
        # Close the polygon
        points.append((width, height))
        points.append((0, height))
        
        draw.polygon(points, fill=colors[layer % len(colors)])
    
    return img

def create_terrain_wallpaper(style_id, seed=None):
    """Create a terrain wallpaper based on style"""
    if seed is None:
        seed = random.randint(0, 10000)
    
    random.seed(seed)
    
    styles = {
        1: {  # Dawn Mountains
            'bg_colors': [(25, 25, 40), (100, 50, 80), (255, 150, 100), (255, 200, 150)],
            'terrain_colors': [(20, 30, 60), (40, 50, 80), (80, 90, 120)],
            'name': 'dawn_mountains'
        },
        2: {  # Forest Hills
            'bg_colors': [(30, 60, 30), (60, 120, 60), (120, 180, 120), (200, 255, 200)],
            'terrain_colors': [(20, 40, 20), (40, 80, 40), (80, 120, 80)],
            'name': 'forest_hills'
        },
        3: {  # Desert Dunes
            'bg_colors': [(40, 30, 20), (120, 80, 40), (200, 150, 80), (255, 220, 150)],
            'terrain_colors': [(60, 40, 20), (120, 80, 40), (180, 140, 80)],
            'name': 'desert_dunes'
        },
        4: {  # Arctic Peaks
            'bg_colors': [(40, 50, 70), (80, 120, 150), (150, 200, 255), (255, 255, 255)],
            'terrain_colors': [(60, 80, 120), (100, 130, 180), (200, 220, 255)],
            'name': 'arctic_peaks'
        },
        5: {  # Sunset Canyon
            'bg_colors': [(60, 20, 40), (150, 60, 30), (255, 120, 60), (255, 200, 100)],
            'terrain_colors': [(80, 40, 20), (120, 60, 30), (180, 100, 60)],
            'name': 'sunset_canyon'
        },
        6: {  # Ocean Cliffs
            'bg_colors': [(20, 40, 80), (40, 80, 150), (100, 180, 255), (200, 230, 255)],
            'terrain_colors': [(40, 60, 40), (80, 100, 80), (120, 150, 120)],
            'name': 'ocean_cliffs'
        },
        7: {  # Volcanic Landscape
            'bg_colors': [(40, 20, 20), (100, 40, 20), (200, 80, 40), (255, 150, 100)],
            'terrain_colors': [(60, 20, 20), (120, 40, 20), (180, 80, 40)],
            'name': 'volcanic_landscape'
        },
        8: {  # Misty Valleys
            'bg_colors': [(50, 60, 70), (100, 120, 140), (160, 180, 200), (220, 230, 240)],
            'terrain_colors': [(40, 60, 80), (80, 100, 120), (140, 160, 180)],
            'name': 'misty_valleys'
        },
        9: {  # Autumn Mountains
            'bg_colors': [(60, 40, 30), (150, 100, 50), (255, 150, 80), (255, 200, 120)],
            'terrain_colors': [(80, 50, 30), (120, 80, 40), (180, 120, 60)],
            'name': 'autumn_mountains'
        },
        10: {  # Purple Twilight
            'bg_colors': [(40, 20, 60), (80, 40, 120), (150, 80, 200), (220, 150, 255)],
            'terrain_colors': [(60, 40, 80), (100, 60, 120), (160, 100, 180)],
            'name': 'purple_twilight'
        },
        11: {  # Green Highlands
            'bg_colors': [(40, 60, 40), (80, 140, 80), (120, 200, 120), (180, 255, 180)],
            'terrain_colors': [(30, 50, 30), (60, 100, 60), (100, 160, 100)],
            'name': 'green_highlands'
        },
        12: {  # Golden Hour
            'bg_colors': [(80, 60, 30), (180, 140, 60), (255, 200, 100), (255, 240, 180)],
            'terrain_colors': [(100, 80, 40), (150, 120, 60), (200, 160, 100)],
            'name': 'golden_hour'
        }
    }
    
    style = styles[style_id]
    
    # Create background gradient
    bg = create_gradient(WIDTH, HEIGHT, style['bg_colors'], 'vertical')
    
    # Generate terrain heightmap
    terrain_img = Image.new('RGB', (WIDTH, HEIGHT))
    pixels = terrain_img.load()
    
    print(f"Generating {style['name']} terrain...")
    
    # Generate height map with multiple noise layers
    for x in range(WIDTH):
        for y in range(HEIGHT):
            # Multiple octaves for realistic terrain
            height1 = fractal_noise(x, y, octaves=6, persistence=0.6, scale=0.003, seed=seed)
            height2 = fractal_noise(x, y, octaves=4, persistence=0.4, scale=0.01, seed=seed + 100)
            height3 = fractal_noise(x, y, octaves=2, persistence=0.3, scale=0.05, seed=seed + 200)
            
            combined_height = (height1 * 0.6 + height2 * 0.3 + height3 * 0.1)
            
            # Create elevation-based coloring
            elevation = (combined_height + 1) / 2  # Normalize to 0-1
            
            if elevation < 0.3:
                color = style['terrain_colors'][0]
            elif elevation < 0.6:
                # Interpolate between low and mid colors
                ratio = (elevation - 0.3) / 0.3
                color = interpolate_colors([style['terrain_colors'][0], style['terrain_colors'][1]], ratio)
            else:
                # Interpolate between mid and high colors
                ratio = (elevation - 0.6) / 0.4
                color = interpolate_colors([style['terrain_colors'][1], style['terrain_colors'][2]], ratio)
            
            pixels[x, y] = color
    
    # Apply gaussian blur for smoother terrain
    terrain_img = terrain_img.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Blend terrain with background
    result = Image.blend(bg, terrain_img, 0.7)
    
    # Generate mountain silhouettes
    mountains = generate_mountain_silhouette(WIDTH, HEIGHT, layers=3, seed=seed)
    
    # Composite mountains onto terrain
    result = Image.alpha_composite(result.convert('RGBA'), mountains).convert('RGB')
    
    # Add some atmospheric perspective (subtle brightness gradient)
    enhancer = ImageEnhance.Contrast(result)
    result = enhancer.enhance(1.1)
    
    # Add subtle vignette effect
    vignette = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    vignette_draw = ImageDraw.Draw(vignette)
    
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    max_distance = math.sqrt(center_x**2 + center_y**2)
    
    for x in range(WIDTH):
        for y in range(HEIGHT):
            distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            vignette_strength = int((distance / max_distance) * 30)
            if vignette_strength > 0:
                vignette_draw.point((x, y), (0, 0, 0, vignette_strength))
    
    result = Image.alpha_composite(result.convert('RGBA'), vignette).convert('RGB')
    
    return result, style['name']

def main():
    """Generate 12 beautiful terrain wallpapers"""
    print("Starting terrain wallpaper generation...")
    
    # Create output directory
    output_dir = "terrain_wallpapers"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate 12 different terrain styles
    for i in range(1, 13):
        print(f"Generating wallpaper {i}/12...")
        
        try:
            wallpaper, name = create_terrain_wallpaper(i, seed=i * 1000)
            
            # Save the wallpaper
            filename = f"{output_dir}/{i:02d}_{name}_mobile.png"
            wallpaper.save(filename, "PNG", optimize=True)
            print(f"Saved: {filename}")
            
        except Exception as e:
            print(f"Error generating wallpaper {i}: {e}")
    
    print(f"\nCompleted! Generated 12 terrain wallpapers in '{output_dir}' directory")
    print(f"Resolution: {WIDTH}x{HEIGHT} (optimized for mobile devices)")

if __name__ == "__main__":
    main()
