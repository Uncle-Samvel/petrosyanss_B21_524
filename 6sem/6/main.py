import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageOps

TEXT = "سوسک ها می توانند تا چند هفته بدون سر زندگی کنند و از زباله تغذیه کنند"
WHITE = 255

FONT_FILE = "input/Unicode.ttf"
FONT_SIZE = 52
THRESHOLD = 75


def create_text_profiles(image: np.array):
    os.makedirs("output/text_profile", exist_ok=True)
    binary_image = np.zeros(image.shape, dtype=int)
    binary_image[image != WHITE] = 1

    plt.bar(
        x=np.arange(start=1, stop=binary_image.shape[1] + 1).astype(int),
        height=np.sum(binary_image, axis=0),
        width=0.9
    )
    plt.savefig(f'output/text_profile/x.png')
    plt.clf()

    plt.barh(
        y=np.arange(start=1, stop=binary_image.shape[0] + 1).astype(int),
        width=np.sum(binary_image, axis=1),
        height=0.9
    )
    plt.savefig(f'output/text_profile/y.png')
    plt.clf()


def binarize_image(image, threshold=THRESHOLD):
    new_image = np.zeros(shape=image.shape)
    new_image[image > threshold] = WHITE
    return new_image.astype(np.uint8)


def generate_text_image():
    space_width = 5
    text_width = sum(FONT.getsize(char)[0] for char in TEXT) + space_width * (len(TEXT) - 1)

    height = max(FONT.getsize(char)[1] for char in TEXT)

    image = Image.new("L", (text_width, height), color="white")
    draw = ImageDraw.Draw(image)

    current_x = 0
    for letter in TEXT:
        width, letter_height = FONT.getsize(letter)
        draw.text((current_x, height - letter_height), letter, "black", font=FONT)
        current_x += width + space_width

    image = Image.fromarray(binarize_image(np.array(image)))
    image.save("output/original_text.bmp")

    np_image = np.array(image)
    create_text_profiles(np_image)
    ImageOps.invert(image).save("output/inverted_text.bmp")
    return np_image


def segment_letters(image):
    profile = np.sum(image == 0, axis=0)

    in_letter = False
    letter_boundaries = []

    for i in range(len(profile)):
        if profile[i] > 0:
            if not in_letter:
                in_letter = True
                start = i
        else:
            if in_letter:
                in_letter = False
                end = i
                letter_boundaries.append((start - 1, end))

    if in_letter:
        letter_boundaries.append((start, len(profile)))

    return letter_boundaries


def draw_letter_boxes(image, boundaries):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    for start, end in boundaries:
        left, right = start, end
        top, bottom = 0, image.shape[0]
        draw.rectangle([left, top, right, bottom], outline="red")

    image.save("output/segmented_text.bmp")


if __name__ == "__main__":
    image = generate_text_image()
    boundaries = segment_letters(image)
    draw_letter_boxes(image, boundaries)