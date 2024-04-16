import argparse
from PIL import Image

def make_opaque(image_path, output_path):
    # Open an image file
    with Image.open(image_path) as img:
        # Ensure the image has an alpha channel
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Load the pixel data
        pixels = img.load()
        
        # Dimensions
        width, height = img.size
        treshold = 10
        # Iterate over all pixels
        for x in range(width):
            for y in range(height):
                r, g, b, a = pixels[x, y]
                # If the pixel is not fully transparent
                if a >240:
                    print("alpha val", a)
                    # Set the alpha value to 255 (fully opaque)
                    pixels[x, y] = (r, g, b, 255)

        # Save the modified image
        img.save(output_path)


def copy_non_transparent_pixels(image_path, output_path):
    # Open the source image
    with Image.open(image_path) as img:
        # Ensure the image has an alpha channel
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Get the size of the image
        width, height = img.size
        
        # Create a new fully transparent image
        new_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        # Load pixel data from both images
        src_pixels = img.load()
        new_pixels = new_image.load()
        
        # Iterate over all pixels
        for x in range(width):
            for y in range(height):
                r, g, b, a = src_pixels[x, y]
                # Copy pixel if it's not fully transparent
                if a > 10:
                    new_pixels[x, y] = (r, g, b, a)
        
        # Save the new image
        new_image.save(output_path)

def main():
    parser = argparse.ArgumentParser(description='Apply a transparency mask and threshold to an image.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input image file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output image file')
    
    args = parser.parse_args()

    copy_non_transparent_pixels(args.input_path, args.output_path)
    #make_opaque(args.input_path, args.output_path)
    #apply_mask_from_transparency(args.input_path, args.output_path)

if __name__ == "__main__":
    main()
