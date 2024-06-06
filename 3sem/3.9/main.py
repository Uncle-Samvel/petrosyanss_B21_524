import os
import numpy as np
from PIL import Image
from tqdm import tqdm

BLACK, WHITE = 0, 255
OMEGA_OF_WHITE = 8 * WHITE


def morphological_filtering(image_array):
    result = np.zeros_like(image_array, dtype=np.uint8)
    rows, columns = image_array.shape
    result[0] = image_array[0]
    result[-1] = image_array[-1]
    for row in range(rows):
        result[row, 0] = image_array[row, 0]
        result[row, -1] = image_array[row, -1]

    for row in tqdm(range(1, rows - 1), desc="Processing rows"):
        for column in range(1, columns - 1):
            neighborhood = image_array[row - 1: row + 2, column - 1: column + 2].flatten()
            if sum(neighborhood) == BLACK and image_array[row, column] == WHITE:
                result[row, column] = BLACK
            elif sum(neighborhood) == OMEGA_OF_WHITE and image_array[row, column] == BLACK:
                result[row, column] = WHITE
            else:
                result[row, column] = image_array[row, column]
    return result


def perform_filtering():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    output_directory = os.path.join(current_directory, 'output')
    input_directory = os.path.join(current_directory, 'input')

    os.makedirs(output_directory, exist_ok=True)
    for input_file in os.scandir(input_directory):
        if not input_file.name.lower().endswith(('.png', '.bmp')):
            print(f"File {input_file.name} is not an image.")
            continue
        with Image.open(input_file.path) as input_image:
            print(f"Processing image {input_file.name}.")
            input_array = np.array(input_image, np.uint8)
            filtered_array = morphological_filtering(input_array)
            filtered_image = Image.fromarray(filtered_array, 'L')
            filtered_image.save(os.path.join(output_directory, f"filtered_{input_file.name}"))
            difference_array = input_array ^ filtered_array
            difference_image = Image.fromarray(difference_array, 'L')
            difference_image.save(os.path.join(output_directory, f"xor_{input_file.name}"))


if __name__ == "__main__":
    perform_filtering()