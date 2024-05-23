import cv2
import numpy as np

class ImageSharpener:
    def __init__(self):
        """
        Initialize the ImageSharpener class with a sharpening kernel.
        """
        self.kernel = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]])

    def sharpen_image(self, image):
        """
        Sharpens the input image for better OCR performance.

        Args:
            image (numpy.ndarray): The input image to be sharpened.

        Returns:
            numpy.ndarray: The sharpened image.
        """
        # Apply the sharpening kernel to the input image twice
        sharpened_image = cv2.filter2D(sharpened_image, -1, self.kernel)
        sharpened_image = cv2.filter2D(sharpened_image, -1, self.kernel)


        return sharpened_image

# Example usage
# Load the input image
input_image = cv2.imread('/Users/nisargdoshi/Downloads/work/accelerator_cv/image.png')

# Create an instance of the ImageSharpener class
image_sharpener = ImageSharpener()

# Sharpen the image
sharpened_image = image_sharpener.sharpen_image(input_image)

# Save the sharpened image
cv2.imwrite('sharpened_image.jpg', sharpened_image)