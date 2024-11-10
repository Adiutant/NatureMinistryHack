import os
import json
import cv2
import yaml


def load_yaml_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_coco_annotations(coco_annotation_file):
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)
    return coco_data


def create_cropped_dataset(config):
    annotations_path = config['annotations_path']
    images_dir = config['images_dir']
    output_dir = config['output_dir']

    os.makedirs(output_dir, exist_ok=True)

    coco_data = load_coco_annotations(annotations_path)

    # Создаем словарь для быстрого доступа к информации об изображениях
    images_info = {img['id']: img for img in coco_data['images']}

    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        bbox = annotation['bbox']  # [x, y, width, height]

        image_info = images_info[image_id]
        image_path = os.path.join(images_dir, image_info['file_name'])

        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Image {image_path} not found.")
            continue

        # Обрезаем изображение по координатам бокса
        x, y, w, h = map(int, bbox)
        cropped_image = image[y:y + h, x:x + w]

        # Создаем имя файла для обрезанного изображения
        cropped_filename = f"{image_id}_{category_id}.jpg"
        cropped_path = os.path.join(output_dir, cropped_filename)

        # Сохраняем обрезанное изображение
        cv2.imwrite(cropped_path, cropped_image)


# Example usage
yaml_config_file = 'animals/data.yaml'
config = load_yaml_config(yaml_config_file)
create_cropped_dataset(config)

print(f"Cropped dataset created in {config['output_dir']}")