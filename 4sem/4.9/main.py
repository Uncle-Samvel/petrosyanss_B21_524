import dataclasses
import os
import numpy as np
from PIL import Image

from tqdm import tqdm

Gradient_x = np.array([[3, 0, -3],
                [10, 0, -10],
                [3, 0, -3]])
Gradient_y = np.array([[3, 10, 3],
                [0, 0, 0],
                [-3, -10, -3]])

White, Black = 255, 0


@dataclasses.dataclass
class ImageOperations:
    def __init__(self, input_image: Image.Image, output_dir: str):
        self.output_dir = output_dir
        self.image_array = np.array(input_image.convert('L')).astype(np.uint8)
        self.new_image_gradient_x = np.zeros_like(self.image_array, dtype=np.float64)
        self.new_image_gradient_y = np.zeros_like(self.image_array, dtype=np.float64)
        self.new_image_gradient = np.zeros_like(self.image_array, dtype=np.float64)
        self.new_image_gradient_binary = np.zeros_like(self.image_array, dtype=np.float64)

    def save_results(self):
        os.makedirs(self.output_dir, exist_ok=True)
        Image.fromarray(self.new_image_gradient_x.astype(np.uint8), 'L').save(os.path.join(self.output_dir, "gradient_x.png"))
        Image.fromarray(self.new_image_gradient_y.astype(np.uint8), 'L').save(os.path.join(self.output_dir, "gradient_y.png"))
        Image.fromarray(self.new_image_gradient.astype(np.uint8), 'L').save(os.path.join(self.output_dir, "gradient.png"))
        self.new_image_gradient_binary = np.where(self.new_image_gradient > BINARIZATION_THRESHOLD, White, Black)
        Image.fromarray(self.new_image_gradient_binary.astype(np.uint8), 'L').save(os.path.join(self.output_dir, "binary.png"))


def sharr_processing(image: ImageOperations) -> None:
    rows, columns = image.image_array.shape
    for row in tqdm(range(1, rows - 1), desc="Processing lines"):
        for col in range(1, columns - 1):
            frame = image.image_array[row - 1: row + 2, col - 1:col + 2]

            gradient_x = np.sum(Gradient_x * frame.astype(np.int32))
            image.new_image_gradient_x[row, col] = gradient_x

            gradient_y = np.sum(Gradient_y * frame.astype(np.int32))
            image.new_image_gradient_y[row, col] = gradient_y

            image.new_image_gradient[row, col] = np.sqrt(gradient_x  2 + gradient_y  2)

    image.new_image_gradient_x *= White / np.max(image.new_image_gradient_x)
    image.new_image_gradient_y *= White / np.max(image.new_image_gradient_y)
    image.new_image_gradient *= White / np.max(image.new_image_gradient)

    image.save_results()


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(current_dir, 'output')
    input_folder = os.path.join(current_dir, 'input')

    os.makedirs(output_folder, exist_ok=True)
    for image_file in os.scandir(input_folder):
        if not image_file.name.lower().endswith(('.png', '.bmp')):
            print("Error", image_file.name)
            continue
        with Image.open(image_file.path) as input_image:
            print(f"Processing {image_file.name}.")
            output_dir = os.path.join(output_folder, image_file.name)
            sharr_processing(ImageOperations(input_image, output_dir))


if __name__ == "__main__":
    BINARIZATION_THRESHOLD = 30
    main()