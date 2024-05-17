import numpy as np 
from PIL import Image
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity
import argparse


class ImageSimilarity:
    """
    Class to compute similarity between images using VGG16 embeddings.
    """

    def __init__(self):
        """
        Initialize VGG16 model and set layers to non-trainable.
        """
        self.vgg16 = VGG16(weights='imagenet', include_top=False, pooling='max', input_shape=(224, 224, 3))
        for model_layer in self.vgg16.layers:
            model_layer.trainable = False

    def load_image(self, image_path):
        """
        Process the image provided.
        - Resize the image
        
        Args:
            image_path (str): Path to the input image.
        
        Returns:
            PIL.Image.Image: Resized image.
        """
        input_image = Image.open(image_path)
        resized_image = input_image.resize((224, 224))
        return resized_image

    def get_image_embeddings(self, object_image):
        """
        Convert image into 3D array and add an additional dimension for model input.
        
        Args:
            object_image (PIL.Image.Image): Input image.
        
        Returns:
            numpy.ndarray: Embeddings of the given image.
        """
        image_array = np.expand_dims(image.img_to_array(object_image), axis=0)
        image_embedding = self.vgg16.predict(image_array)
        return image_embedding

    def get_similarity_score(self, first_image, second_image):
        """
        Takes image array and computes its embedding using VGG16 model.
        
        Args:
            first_image (str): Path to the first image.
            second_image (str): Path to the second image.
        
        Returns:
            numpy.ndarray: Similarity score between the images.
        """
        first_image = self.load_image(first_image)
        second_image = self.load_image(second_image)
        first_image_vector = self.get_image_embeddings(first_image)
        second_image_vector = self.get_image_embeddings(second_image)
        similarity_score = cosine_similarity(first_image_vector, second_image_vector).reshape(1,)
        return similarity_score

    def show_image(self, image_path):
        """
        Display the image.
        
        Args:
            image_path (str): Path to the image.
        """
        image = Image.open(image_path)
        plt.imshow(image)
        plt.show()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Similarity using VGG16 embeddings")
    parser.add_argument("first_image", type=str, help="Path to the first image")
    parser.add_argument("second_image", type=str, help="Path to the second image")
    args = parser.parse_args()


    image_similarity = ImageSimilarity()

    image_similarity.show_image(args.first_image)
    image_similarity.show_image(args.second_image)

    similarity_score = image_similarity.get_similarity_score(args.first_image, args.second_image)
    print("Similarity Score between the images:", similarity_score)
