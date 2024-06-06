import csv
import os
from math import ceil

import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps
from matplotlib import pyplot as plt

PERSIAN_LETTERS_UNICODE = ["0627", "0628", "067E", "062A", "062B", "062C", "0686", "062D", "062E", "062F", "0630",
                           "0631", "0632", "0698", "0633", "0634", "0635", "0636", "0637", "0638", "0639", "063A",
                           "0641", "0642", "06A9", "06AF", "0644", "0645", "0646", "0648", "0647", "06CC"]
PERSIAN_LETTERS = [chr(int(letter, 16)) for letter in PERSIAN_LETTERS_UNICODE]

IMAGE_SIZE = 52
THRESHOLD = 75
FONT_FILE = "input/Unicode.ttf"

WHITE = 255


def simple_binarization(image, threshold=THRESHOLD):
    grayscale_image = (0.3 * image[:, :, 0] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 2]).astype(np.uint8)
    binary_image = np.zeros(shape=grayscale_image.shape)
    binary_image[grayscale_image > threshold] = WHITE
    return binary_image.astype(np.uint8)


def generate_letter_images(letters):
    font = ImageFont.truetype(FONT_FILE, IMAGE_SIZE)
    os.makedirs("output/letters", exist_ok=True)
    os.makedirs("output/inverse_letters", exist_ok=True)

    for index, letter in enumerate(letters):

        width, height = font.getsize(letter)
        image = Image.new(mode="RGB", size=(ceil(width), ceil(height)), color="white")
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), letter, "black", font=font)

        image = Image.fromarray(simple_binarization(np.array(image), THRESHOLD), 'L')
        image.save(f"output/letters/{index + 1}.png")

        ImageOps.invert(image).save(f"output/inverse_letters/{index + 1}.png")


def calculate_features(image):
    binary_image = np.zeros(image.shape, dtype=int)
    binary_image[image != WHITE] = 1
    (h, w) = binary_image.shape
    h_half, w_half = h // 2, w // 2
    quadrants = {
        'top_left': binary_image[:h_half, :w_half],
        'top_right': binary_image[:h_half, w_half:],
        'bottom_left': binary_image[h_half:, :w_half],
        'bottom_right': binary_image[h_half:, w_half:]
    }
    weights = {k: np.sum(v) for k, v in quadrants.items()}
    relative_weights = {k: v / (h_half * w_half) for k, v in weights.items()}

    total_pixels = np.sum(binary_image)
    y_indices, x_indices = np.indices(binary_image.shape)
    y_center_of_mass = np.sum(y_indices * binary_image) / total_pixels
    x_center_of_mass = np.sum(x_indices * binary_image) / total_pixels
    center_of_mass = (x_center_of_mass, y_center_of_mass)
    normalized_center_of_mass = (x_center_of_mass / (w - 1), y_center_of_mass / (h - 1))
    inertia_x = np.sum((y_indices - y_center_of_mass) * 2 * binary_image) / total_pixels
    normalized_inertia_x = inertia_x / h * 2
    inertia_y = np.sum((x_indices - x_center_of_mass) * 2 * binary_image) / total_pixels
    normalized_inertia_y = inertia_y / w * 2

    return {
        'total_weight': total_pixels,
        'weights': weights,
        'relative_weights': relative_weights,
        'center_of_mass': center_of_mass,
        'normalized_center_of_mass': normalized_center_of_mass,
        'inertia': (inertia_x, inertia_y),
        'normalized_inertia': (normalized_inertia_x, normalized_inertia_y)
    }


def create_feature_dataset(letters):
    with open('output/data.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['total_weight', 'weights', 'relative_weights', 'center_of_mass', 'normalized_center_of_mass',
                      'inertia', 'normalized_inertia']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for index, letter in enumerate(letters):
            image = np.array(Image.open(f'output/letters/{index + 1}.png').convert('L'))
            features = calculate_features(image)
            writer.writerow(features)


def create_profiles(letters):
    os.makedirs("output/profiles/x", exist_ok=True)
    os.makedirs("output/profiles/y", exist_ok=True)

    for index, letter in enumerate(letters):
        image = np.array(Image.open(f'output/letters/{index + 1}.png').convert('L'))
        binary_image = np.zeros(image.shape, dtype=int)
        binary_image[image != WHITE] = 1


if __name__ == "__main__":
    generate_letter_images(PERSIAN_LETTERS)
    create_feature_dataset(PERSIAN_LETTERS)
    create_profiles(PERSIAN_LETTERS)
