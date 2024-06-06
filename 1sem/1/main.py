import os
from PIL import Image
from typing import Callable, Tuple, Any
import numpy as np

def two_step_resampling(image: np.array) -> np.array:
    upscale_factor = int(input('Введите целое число для апскейла:\n> '))
    downscale_factor = int(input('Введите целое число для даунскейла:\n> '))
    tmp = one_step_resampling(image, upscale_factor,
                              dimension_function=lambda a, b: a * b,
                              value_function=lambda a, b: round(a / b))
    raw_result = one_step_resampling(
        image=tmp,
        factor=downscale_factor,
        dimension_function=lambda a, b: round(a / b),
        value_function=lambda a, b: a * b
    )

    return Image.fromarray(raw_result.astype(np.uint8), 'RGB')


def one_step_resampling(image: np.array, factor: Any, dimension_function: Callable[[Any, Any], Any], value_function: Callable[[Any, Any], Any]) -> np.array:
    dimensions = image.shape[0:2]
    new_dimensions = tuple(dimension_function(dimension, factor) for dimension in dimensions)
    new_shape: Tuple = (*new_dimensions, image.shape[2])
    new_image = np.empty(new_shape)

    for x in range(new_dimensions[0]):
        for y in range(new_dimensions[1]):
            new_image[x, y] = image[
                min(value_function(x, factor), dimensions[0] - 1),
                min(value_function(y, factor), dimensions[1] - 1)
            ]
    return new_image


def one_step_wrapper(image: np.array, var_type: Any, f1: Callable[[Any, Any], Any], f2: Callable[[Any, Any], Any]) -> Image.Image:
    factor = var_type((input(f'Введите {"целое число" if var_type == int else "дробное число"}:\n> ')))
    result = Image.fromarray(one_step_resampling(image, factor, f1, f2).astype(np.uint8), 'RGB')
    return result


def main():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    output_directory = os.path.join(current_directory, 'output')
    input_directory = os.path.join(current_directory, 'input')
    print("""Операции:
    1) Апскейл (интерполяция) изображения в M раз;
    2) Даунскейл(децимация) изображения в N раз;
    3) Передискретизация изображения в K=M/N раз путём апскейла и
    последующего даунскейла (в два прохода);
    4) Передискретизация изображения в K раз за один проход.
    """)
    for image in os.scandir(input_directory):
        print(f"Работаем с {image.name}.")
        image_np = np.array(Image.open(image.path).convert('RGB'))

        operation: Callable[[np.ndarray], Image.Image] = {
            1: lambda img: one_step_wrapper(img, int,
                                            lambda a, b: a * b,
                                            lambda a, b: round(a / b)), 
            2: lambda img: one_step_wrapper(img, int,
                                            lambda a, b: round(a / b),
                                            lambda a, b: a * b),
            3: lambda img: two_step_resampling(img),
            4: lambda img: one_step_wrapper(img, float,
                                            lambda a, b: round(a * b),
                                            lambda a, b: round(a / b))
        }.get(int(input('Выберите операцию:\n> ')))
        if operation:
            operation(image_np).save(os.path.join(output_directory, image.name))
            print('Сохранили изображение!\n\n')
        else:
            print('Ошибка')
            exit()


if __name__ == "__main__":
    main()