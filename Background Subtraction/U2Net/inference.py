import argparse
import time
import cv2
import matplotlib.pyplot as plt
import paddlehub as hub


class BackgroundFilter:
    """
    Class to perform background filtering using the U2Net model.
    """

    def __init__(self, model_name='U2Net'):
        """
        Initializes the BackgroundFilter object with the specified model.
        
        Args:
            model_name (str): Name of the PaddleHub segmentation model to use.
        """
        self.model = hub.Module(name=model_name)

    def filter_background(self, img_path):
        """
        Filters the background of an image using the U2Net model.
        
        Args:
            img_path (str): Path to the input image.
        
        Returns:
            tuple: A tuple containing the filtered image and the mask.
        """
        st = time.time()
        img = cv2.imread(img_path)
        fin = cv2.hconcat([img[:, :50] for _ in range(round(abs(img.shape[0] - img.shape[1]) / 50) - 5)])
        fin = cv2.hconcat([fin, img])
        result = self.model.Segmentation(
            images=[fin],
            paths=None,
            batch_size=1,
            input_size=640,
            visualization=False)
        print("Background Subtraction Time: ", time.time() - st)
        return result[0]['front'][:, :, ::-1][:, (round(abs(img.shape[0] - img.shape[1]) / 50) - 5) * 50:], result[0]['mask']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Background Filtering using U2Net model")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    background_filter = BackgroundFilter()
    filtered_img, mask = background_filter.filter_background(args.image_path)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
    plt.title('Filtered Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')

    plt.show()


