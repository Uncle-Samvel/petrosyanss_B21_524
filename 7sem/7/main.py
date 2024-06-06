import csv
import math
import numpy as np
from PIL import Image

PERSIAN_LETTERS_UNICODE = ["0627", "0628", "067E", "062A", "062B", "062C", "0686", "062D", "062E", "062F", "0630",
                           "0631", "0632", "0698", "0633", "0634", "0635", "0636", "0637", "0638", "0639", "063A",
                           "0641", "0642", "06A9", "06AF", "0644", "0645", "0646", "0648", "0647", "06CC"]
PERSIAN_LETTERS = [chr(int(letter, 16)) for letter in PERSIAN_LETTERS_UNICODE]

WHITE = 255
TARGET_PHRASE = "سوسک ها می توانند تا چند هفته بدون سر زندگی کنند و از زباله تغذیه کنند".replace(" ", "")


def calculate_features(image: np.array):
    black_pixels = np.where(image != WHITE, 1, 0)
    total_black_pixels = np.sum(black_pixels)
    y_coordinates, x_coordinates = np.indices(black_pixels.shape)
    y_center_of_mass = np.sum(y_coordinates * black_pixels) / total_black_pixels
    x_center_of_mass = np.sum(x_coordinates * black_pixels) / total_black_pixels
    inertia_x = np.sum((y_coordinates - y_center_of_mass) * 2 * black_pixels) / total_black_pixels
    inertia_y = np.sum((x_coordinates - x_center_of_mass) * 2 * black_pixels) / total_black_pixels

    return total_black_pixels, x_center_of_mass, y_center_of_mass, inertia_x, inertia_y


def segment_letters(image):
    horizontal_projection = np.sum(image == 0, axis=0)

    in_letter = False
    letter_boundaries = []

    for i in range(len(horizontal_projection)):
        if horizontal_projection[i] > 0:
            if not in_letter:
                in_letter = True
                start = i
        else:
            if in_letter:
                in_letter = False
                end = i
                letter_boundaries.append((start - 1, end))

    if in_letter:
        letter_boundaries.append((start, len(horizontal_projection)))

    return letter_boundaries


def load_letter_features() -> dict[chr, tuple]:
    def parse_tuple(string):
        return tuple(map(float, string.strip('()').split(',')))

    letter_features = {}
    with open('input/data.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            weight = int(row['weight'])
            center_of_mass = parse_tuple(row['center_of_mass'])
            inertia = parse_tuple(row['inertia'])
            letter_features[PERSIAN_LETTERS[int(row['index'])]] = weight, *center_of_mass, *inertia
    return letter_features


def create_hypothesis(letter_features: dict[chr, tuple], target_features):
    def euclidean_distance(feature1, feature2):
        return math.sqrt(sum((a - b) * 2 for a, b in zip(feature1, feature2)))

    distances = {}
    for letter, features in letter_features.items():
        distance = euclidean_distance(target_features, features)
        distances[letter] = distance

    max_distance = max(distances.values())

    similarities = [(letter, round(1 - distance / max_distance, 2)) for letter, distance in distances.items()]

    return sorted(similarities, key=lambda x: x[1])


def recognize_phrase(image: np.array, letter_boundaries) -> str:
    letter_features = load_letter_features()
    recognized_phrase = []
    for start, end in letter_boundaries:
        letter_image = image[:, start: end]
        letter_features = calculate_features(letter_image)
        hypothesis = create_hypothesis(letter_features, target_features)
        most_likely_letter = hypothesis[-1][0]
        recognized_phrase.append(most_likely_letter)
    return "".join(recognized_phrase)


def write_result(recognized_phrase: str):
    max_length = max(len(TARGET_PHRASE), len(recognized_phrase))
    original_phrase = TARGET_PHRASE.ljust(max_length)
    detected_phrase = recognized_phrase.ljust(max_length)

    with open("output/result.txt", 'w') as f:
        correct_letters = 0
        letter_comparison = ["has | got | correct"]
        for i in range(max_length):
            is_correct = original_phrase[i] == detected_phrase[i]
            letter_comparison.append(f"{original_phrase[i]}\t{detected_phrase[i]}\t{is_correct}")
            correct_letters += int(is_correct)
        f.write("\n".join([
            f"phrase:      {original_phrase}",
            f"detected:    {detected_phrase}",
            f"correct:     {math.ceil(correct_letters / max_length * 100)}%\n\n"
        ]))
        f.write("\n".join(letter_comparison))

if __name__ == "__main__":
    image = np.array(Image.open(f'input/original_phrase.bmp').convert('L'))
    letter_boundaries = segment_letters(image)
    recognized_phrase = recognize_phrase(image, letter_boundaries)
    write_result(recognized_phrase)