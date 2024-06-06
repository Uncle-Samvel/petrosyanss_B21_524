from PIL import Image
import numpy as np
import os

def grayscale(photo: Image.Image) -> Image.Image:
    photo_arr = np.array(photo.convert('RGB'))
    grayscale_arr = np.mean(photo_arr, axis=2)
    new_photo = Image.fromarray(grayscale_arr.astype(np.uint8), 'L')
    return new_photo

def primary():
    cwd = os.path.dirname(os.path.abspath(__file__))
    input_directory = os.path.join(cwd, 'input')
    output_directory = os.path.join(cwd, 'output')
    os.makedirs(output_directory, exist_ok=True)
    
    for image in os.scandir(input_directory):
        if not image.name.lower().endswith(('.png', '.bmp')):
            print("Ошибка:", image.name)
            continue
        print(f"Действия с {image.name}...")

        output_path = os.path.join(output_directory, image.name)
        with Image.open(image.path) as output_image:
            grayscale(output_image).save(output_path)

if __name__ == "__main__":
    primary()
