import requests
from PIL import Image, ImageEnhance, ImageFilter
import os
import time
import random

# Pixabay API configuration
API_KEY = "52233499-a664b999bd91fc221f7dfa5db"
BASE_URL = "https://pixabay.com/api/"

# Mobile 4K resolution (portrait)
MOBILE_WIDTH = 2160
MOBILE_HEIGHT = 3840

def fetch_images_from_pixabay(query, count=3, image_type='photo', category='nature'):
    """Fetch images from Pixabay API"""
    params = {
        'key': API_KEY,
        'q': query,
        'image_type': image_type,
        'category': category,
        'min_width': 1920,  # Ensure high resolution
        'min_height': 1080,
        'per_page': count,
        'safesearch': 'true',
        'order': 'popular'
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['totalHits'] > 0:
            return data['hits']
        else:
            print(f"No images found for query: {query}")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching images for '{query}': {e}")
        return []

def download_image(url):
    """Download image from URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None

def process_image_for_mobile(image_data, enhance_colors=True, add_effects=True):
    """Process image to create mobile-optimized 4K wallpaper"""
    try:
        # Open image from bytes
        img = Image.open(image_data).convert('RGB')
        
        # Get original dimensions
        orig_width, orig_height = img.size
        
        # Calculate crop area for mobile aspect ratio (9:16)
        mobile_ratio = MOBILE_HEIGHT / MOBILE_WIDTH  # 16:9 -> 1.778
        orig_ratio = orig_height / orig_width
        
        if orig_ratio > mobile_ratio:
            # Image is taller - crop height
            new_height = int(orig_width * mobile_ratio)
            top = (orig_height - new_height) // 2
            img = img.crop((0, top, orig_width, top + new_height))
        else:
            # Image is wider - crop width
            new_width = int(orig_height / mobile_ratio)
            left = (orig_width - new_width) // 2
            img = img.crop((left, 0, left + new_width, orig_height))
        
        # Resize to mobile 4K
        img = img.resize((MOBILE_WIDTH, MOBILE_HEIGHT), Image.Resampling.LANCZOS)
        
        if enhance_colors:
            # Enhance colors for mobile display
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.15)  # Boost saturation slightly
            
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.08)  # Slight contrast boost
            
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.02)  # Slight brightness boost
        
        if add_effects:
            # Add subtle sharpening
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
        
        return img
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def create_wallpaper_collection():
    """Create collection of mobile wallpapers from Pixabay"""
    
    # Define search queries with variations
    queries = {
        'sunset': ['sunset', 'golden hour', 'sunrise mountain', 'sunset ocean'],
        'beach': ['tropical beach', 'ocean waves', 'beach paradise', 'sandy beach'],
        'dolphins': ['dolphins jumping', 'dolphin ocean', 'marine dolphins'],
        'forest': ['forest path', 'green forest', 'woodland trees', 'forest sunlight'],
        'amazon': ['amazon rainforest', 'jungle canopy', 'tropical rainforest'],
        'lion': ['lion portrait', 'african lion', 'lion savanna', 'majestic lion']
    }
    
    output_dir = "pixabay_mobile_4k_wallpapers"
    os.makedirs(output_dir, exist_ok=True)
    
    wallpaper_count = 0
    target_count = 13
    
    print(f"Generating {target_count} mobile 4K wallpapers using Pixabay API...")
    print(f"Target resolution: {MOBILE_WIDTH}x{MOBILE_HEIGHT}")
    
    for category, search_terms in queries.items():
        print(f"\n--- Processing {category.upper()} category ---")
        
        for search_term in search_terms:
            if wallpaper_count >= target_count:
                break
                
            print(f"Searching for: {search_term}")
            
            # Fetch images from Pixabay
            images = fetch_images_from_pixabay(search_term, count=2)
            
            if not images:
                continue
            
            for i, img_data in enumerate(images):
                if wallpaper_count >= target_count:
                    break
                
                wallpaper_count += 1
                
                # Get the highest resolution available
                img_url = img_data.get('fullHDURL') or img_data.get('webformatURL') or img_data.get('largeImageURL')
                
                if not img_url:
                    print(f"No suitable image URL found")
                    wallpaper_count -= 1
                    continue
                
                print(f"Downloading and processing wallpaper {wallpaper_count}/{target_count}...")
                print(f"Source: {img_data.get('tags', 'Unknown')}")
                
                # Download image
                image_content = download_image(img_url)
                
                if not image_content:
                    print(f"Failed to download image")
                    wallpaper_count -= 1
                    continue
                
                # Process image for mobile
                from io import BytesIO
                processed_img = process_image_for_mobile(
                    BytesIO(image_content),
                    enhance_colors=True,
                    add_effects=True
                )
                
                if processed_img is None:
                    print(f"Failed to process image")
                    wallpaper_count -= 1
                    continue
                
                # Save as high-quality JPEG
                filename = f"{output_dir}/{wallpaper_count:02d}_{category}_{search_term.replace(' ', '_')}_4k_mobile.jpg"
                processed_img.save(filename, "JPEG", quality=95, optimize=True)
                
                print(f"✅ Saved: {filename}")
                
                # Add delay to respect API rate limits
                time.sleep(1)
            
            if wallpaper_count >= target_count:
                break
        
        if wallpaper_count >= target_count:
            break
    
    print(f"\n🎉 Successfully generated {wallpaper_count} mobile 4K wallpapers!")
    print(f"📁 Location: '{output_dir}' directory")
    print(f"📱 Resolution: {MOBILE_WIDTH}x{MOBILE_HEIGHT} (Mobile 4K)")
    print(f"🖼️  Format: JPEG (95% quality)")

def create_enhanced_variations():
    """Create additional enhanced variations of existing images"""
    output_dir = "pixabay_mobile_4k_wallpapers"
    enhanced_dir = f"{output_dir}/enhanced"
    os.makedirs(enhanced_dir, exist_ok=True)
    
    # Check if we have any existing wallpapers to enhance
    existing_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
    
    if not existing_files:
        print("No existing wallpapers found to enhance")
        return
    
    print(f"\nCreating enhanced variations...")
    
    # Select a few random images to create variations
    selected_files = random.sample(existing_files, min(3, len(existing_files)))
    
    for i, filename in enumerate(selected_files):
        try:
            img_path = os.path.join(output_dir, filename)
            img = Image.open(img_path)
            
            # Create different enhancement styles
            variations = {
                'vibrant': {'color': 1.3, 'contrast': 1.15, 'brightness': 1.05},
                'dramatic': {'color': 1.1, 'contrast': 1.25, 'brightness': 0.95},
                'soft': {'color': 0.9, 'contrast': 0.9, 'brightness': 1.1}
            }
            
            base_name = filename.replace('.jpg', '')
            
            for style, settings in variations.items():
                enhanced_img = img.copy()
                
                # Apply enhancements
                enhancer = ImageEnhance.Color(enhanced_img)
                enhanced_img = enhancer.enhance(settings['color'])
                
                enhancer = ImageEnhance.Contrast(enhanced_img)
                enhanced_img = enhancer.enhance(settings['contrast'])
                
                enhancer = ImageEnhance.Brightness(enhanced_img)
                enhanced_img = enhancer.enhance(settings['brightness'])
                
                # Save enhanced version
                enhanced_filename = f"{enhanced_dir}/{base_name}_{style}.jpg"
                enhanced_img.save(enhanced_filename, "JPEG", quality=95, optimize=True)
                print(f"✅ Created enhanced version: {enhanced_filename}")
                
        except Exception as e:
            print(f"Error creating enhanced version of {filename}: {e}")

def main():
    """Main function to generate wallpapers"""
    print("🖼️  Pixabay Mobile 4K Wallpaper Generator")
    print("=" * 50)
    
    # Check if requests library is available
    try:
        import requests
    except ImportError:
        print("❌ Error: 'requests' library not found.")
        print("Please install it using: pip install requests")
        return
    
    # Generate main wallpaper collection
    create_wallpaper_collection()
    
    # Create enhanced variations
    create_enhanced_variations()
    
    print("\n🎨 Wallpaper generation complete!")
    print("\n📋 Tips for best results:")
    print("• Images are optimized for mobile displays")
    print("• All wallpapers are in 4K resolution (2160x3840)")
    print("• JPEG format provides good quality with manageable file sizes")
    print("• Enhanced versions offer different visual styles")

if __name__ == "__main__":
    main()
