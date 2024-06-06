import os
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

WHITE = 255
Pixel_type = typing.Tuple[int, int]
DISTANCE = 2
ANGLES = [angle / 180 * np.pi for angle in [0, 90, 180, 270]]


def analyze_pixel_counts(img: Image, pix, pos: Pixel_type):
    res = np.zeros(WHITE + 1)
    base_x, base_y = pos
    for angle in ANGLES:
        x = base_x + np.around(np.cos(angle)) * DISTANCE
        y = base_y + np.around(np.sin(angle)) * DISTANCE
        if 0 <= x < img.size[0] and 0 <= y < img.size[1]:
            val = pix[x, y]
            res[val] += 1
    return pix[base_x, base_y], res


def pixel_scan(img: Image, func=lambda img, pix, pos: Pixel_type: pix[pos]):
    pix = img.load()
    for row in range(img.size[1]):
        for col in range(img.size[0]):
            pos = (col, row)
            yield pos, func(img, pix, pos)


def compute_haralic_matrix_and_stats(name: str, mono_img: Image):
    res = np.zeros((WHITE + 1, WHITE + 1))
    hist = np.zeros(WHITE + 1)
    max_val = 0
    total_pixels = mono_img.size[0] * mono_img.size[1]
    for pos, (val, row) in tqdm(pixel_scan(mono_img, func=analyze_pixel_counts), total=total_pixels):
        res[val] += row
        max_val = max(max_val, max(row))
        hist[val] += 1
    res_img = Image.fromarray(np.uint8(res * WHITE / max_val))
    res_img.save(f"{name}_matrix.jpg", "JPEG")

    create_graph(hist, name)
    compute_parameters(res_img, name)


def compute_parameters(h_img: Image, filename):
    res_s = pd.Series({"contrast": 0, "luminance": 0})
    for (i, j), p in tqdm(pixel_scan(h_img), total=h_img.size[0] * h_img.size[1]):
        diff = (i - j) ** 2
        res_s["contrast"] += diff * p
        res_s["luminance"] += p / (1 + diff)
    res_s.to_csv(f"{filename}.csv")


def power_transform(img: Image, c=1, f0=0, gamma=0.5):
    res_img = img.copy()
    d = ImageDraw.Draw(res_img)
    for pos, pixel in pixel_scan(img):
        p = min(int(WHITE * c * (pixel / WHITE + f0) ** gamma), WHITE)
        d.point(pos, p)
    return res_img


def create_graph(hist, img_name):
    f = plt.figure()
    plt.bar(np.arange(hist.size), hist)
    plt.savefig(f"{img_name}_bar.png")
    plt.close(f)


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, 'input')
    for image in os.scandir(input_path):
        print(f"Обработка изображения {image.name}.")
        output_path = os.path.join(current_dir, 'output', image.name.split('.')[0])
        os.makedirs(output_path, exist_ok=True)
        mono_name = os.path.join(output_path, "mono")
        pow_name = os.path.join(output_path, "pow")

        mono_img = Image.open(image.path).convert('L')
        mono_img.save(f"{mono_name}.jpg", "JPEG")

        compute_haralic_matrix_and_stats(mono_name, mono_img)

        pow_img = power_transform(mono_img)
        pow_img.save(f"{pow_name}.jpg", "JPEG")
        compute_haralic_matrix_and_stats(pow_name, pow_img)


if __name__ == "__main__":
    main()