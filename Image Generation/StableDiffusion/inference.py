import argparse
import tensorflow as tf
import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt


class TextToImageConverter:
    """
    Class to convert text to images using the StableDiffusion model.
    """

    def __init__(self, img_width=512, img_height=512):
        """
        Initializes the TextToImageConverter object with the specified image dimensions.
        
        Args:
            img_width (int): Width of the output image.
            img_height (int): Height of the output image.
        """
        self.model = keras_cv.models.StableDiffusion(img_width=img_width, img_height=img_height)

    def text_to_image(self, text, batch_size=1):
        """
        Converts the given text to images using the StableDiffusion model.
        
        Args:
            text (str): Text to convert to images.
            batch_size (int): Batch size for inference.
        
        Returns:
            list: List of generated images.
        """
        return self.model.text_to_image(text, batch_size=batch_size)


def plot_images(images):
    """
    Plots the given list of images.
    
    Args:
        images (list): List of images to plot.
    """
    plt.figure(figsize=(8, 8))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text to Image Conversion using StableDiffusion model")
    parser.add_argument("text", type=str, help="Text to convert to images")
    parser.add_argument("--img_width", type=int, default=512, help="Width of the output image")
    parser.add_argument("--img_height", type=int, default=512, help="Height of the output image")
    args = parser.parse_args()

    converter = TextToImageConverter(img_width = args.img_width, img_height=args.img_height)
    images = converter.text_to_image(args.text, batch_size=1)
    plot_images(images)
    plt.show()