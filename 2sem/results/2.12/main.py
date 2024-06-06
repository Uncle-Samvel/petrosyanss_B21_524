from PIL import Image
import numpy as np
import os
import math
from tqdm import tqdm

MIN_DIFF_THRESHOLDS = [5, 10, 15]
SMALL_WINDOW_SIZE = 5
BIG_WINDOW_SIZE = 15
BLACK_COLOR = 0
WHITE_COLOR = 255


def grayscale(img):
    return (0.3 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.11 * img[:, :, 2]).astype(np.uint8)


def local_otsu_threshold(window):
    bins = np.arange(np.min(window) - 1, np.max(window) + 1)
    hist, base = np.histogram(window, bins=bins, density=True)
    base = base[1:].astype(np.uint8)
    w_0 = np.cumsum(hist)
    t_rank = 0
    i_max = 0
    i = -1
    for w0 in w_0:
        i += 1
        m_0 = np.sum(base[:i] * hist[:i] / w0)
        m_1 = np.sum(base[i + 1:] * hist[i + 1:] / (1 - w0))
        d_0 = np.sum(hist[:i] * (base[:i] - m_0) ** 2)
        d_1 = np.sum(hist[i + 1:] * (base[i + 1:] - m_1) ** 2)
        d_all = w0 * d_0 + (1 - w0) * d_1
        d_class = w0 * (1 - w0) * (m_0 - m_1) ** 2
        if d_all == 0:
            i_max = i
            break
        if d_class / d_all > t_rank:
            t_rank = d_class / d_all
            i_max = i
    return base[i_max]


def get_means(matrix, threshold):
    values = matrix.flatten()
    mean_func = lambda x: x.mean() if x.size else 0
    return mean_func(values[values >= threshold]), mean_func(values[values < threshold])


def pixel_transformation(small_window, big_window, min_diff):
    threshold = local_otsu_threshold(big_window)
    up_mean, less_mean = get_means(big_window, threshold)
    if math.fabs(less_mean - up_mean) >= min_diff:
        new_image_small_window = np.zeros(small_window.shape)
        new_image_small_window[small_window > threshold] = WHITE_COLOR
        return new_image_small_window
    small_window_mean = small_window.mean()
    if math.fabs(less_mean - small_window_mean) < math.fabs(up_mean - small_window_mean):
        return np.full(small_window.shape, WHITE_COLOR)


def get_big_window(image_array, w, h):
    up_row = max(h - BIG_WINDOW_SIZE // 2 + 1, 0)
    down_row = min(h + BIG_WINDOW_SIZE // 2 + SMALL_WINDOW_SIZE - 1, image_array.shape[0])
    left = max(w - BIG_WINDOW_SIZE // 2 + 1, 0)
    right = min(w + BIG_WINDOW_SIZE // 2 + SMALL_WINDOW_SIZE - 1, image_array.shape[1])
    return image_array[up_row:down_row, left:right]


def eikvel_binarization(image, min_diff):
    image_array = grayscale(np.array(image))
    new_image_array = np.zeros(shape=image_array.shape)
    np.full((2, 2), WHITE_COLOR).astype(np.uint8)
    for h in tqdm(range(0, image_array.shape[0], SMALL_WINDOW_SIZE)):
        for w in range(0, image_array.shape[1], SMALL_WINDOW_SIZE):
            big_window = get_big_window(image_array, w, h)
            small_window = image_array[h:h + SMALL_WINDOW_SIZE, w:w + SMALL_WINDOW_SIZE]
            new_small_window = pixel_transformation(small_window, big_window, min_diff)
            new_image_array[h:h + SMALL_WINDOW_SIZE, w:w + SMALL_WINDOW_SIZE] = new_small_window

    return Image.fromarray(new_image_array.astype(np.uint8), 'L')


def main():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    output_directory = os.path.join(current_directory, 'output')
    input_directory = os.path.join(current_directory, 'input')

    os.makedirs(output_directory, exist_ok=True)
    for min_diff in MIN_DIFF_THRESHOLDS:
        min_diff_directory = os.path.join(output_directory, f'min_diff={min_diff}')
        os.makedirs(min_diff_directory, exist_ok=True)
        for image in os.scandir(input_directory):
            if not image.name.lower().endswith(('.png', '.bmp')):
                print("Ошибка", image.name)
                continue
            with Image.open(image.path) as read_image:
                print(f"Работаем с {image.name} with min_diff={min_diff}.")
                binarized_image = eikvel_binarization(read_image, min_diff)
                binarized_image.save(os.path.join(min_diff_directory, image.name))


if __name__ == "__main__":
    main()