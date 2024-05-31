import cv2
import numpy as np
import argparse
from PIL import Image, ImageFilter

def sharpen_image(image):
    # Apply a sharpening filter
    sharpened_image = image.filter(ImageFilter.SHARPEN)
    return sharpened_image

def apply_clahe(image):
    # Convert image to LAB color space
    lab_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
    
    # Split the LAB image into channels
    l, a, b = cv2.split(lab_image)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge the CLAHE-enhanced L channel with the original A and B channels
    lab_clahe = cv2.merge((l_clahe, a, b))
    
    # Convert LAB image back to RGB color space
    clahe_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    return clahe_image

def main(input_path, output_path):
    # Load the input image
    image = Image.open(input_path)

    # Apply sharpening filter if needed
    # sharpened_image = sharpen_image(image)

    # Apply CLAHE
    clahe_image = apply_clahe(image)

    # Save the CLAHE-enhanced image
    cv2.imwrite(output_path, cv2.cvtColor(clahe_image, cv2.COLOR_RGB2BGR))
    print(f"CLAHE-enhanced image saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RGB Image CLAHE Enhancement')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output image path')
    args = parser.parse_args()
    main(args.input, args.output)
