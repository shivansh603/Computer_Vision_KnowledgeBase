import cv2
import argparse
from image_enhancement import image_enhancement


class ImageEnhancer:
    """
    Class to enhance images using various methods.
    """

    def __init__(self, image):
        """
        Initializes the ImageEnhancer object with the input image.
        
        Args:
            image (numpy.ndarray): Input image.
        """
        self.image = image

    def enhance_image(self, method='BBHE'):
        """
        Enhances the input image using the specified method.
        
        Args:
            method (str): Name of the enhancement method. Default is 'BBHE'.
        
        Returns:
            numpy.ndarray: Enhanced image.
        """
        ie = image_enhancement.IE(self.image, 'RGB')
        if method == 'BBHE':
            return ie.BBHE()
        # elif method == 'CLAHE':
        #     return ie.CLAHE()
        else:
            raise ValueError("Invalid enhancement method. Supported methods: 'BBHE'")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Enhancement")
    parser.add_argument("input_image", type=str, help="Path to the input image")
    parser.add_argument("output_image", type=str, help="Path to save the enhanced image")
    parser.add_argument("--method", type=str, default="BBHE", choices=["BBHE", "CLAHE"],
                        help="Enhancement method to use (default: BBHE)")
    args = parser.parse_args()

    input_image = cv2.imread(args.input_image)

    image_enhancer = ImageEnhancer(input_image)
    enhanced_image = image_enhancer.enhance_image(method=args.method)

    cv2.imwrite(args.output_image, enhanced_image)


