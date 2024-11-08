import json
import os

import cv2
import pandas as pd

from pt_models.model import QualityModel


def process_folders(folders, model, output_csv):
    data = []

    for folder in folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    file_path = os.path.join(root, file)
                    image = cv2.imread(file_path)

                    if image is not None:
                        output_dict = model(image)

                        for i, info in output_dict.items():
                            bbox_str = ','.join(map(str, info['bbox']))
                            data.append({'Name': file, 'Bbox': bbox_str, 'Class': info['class']})

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    yolo_model_path = 'models/v8_augmented.pt'
    model = QualityModel(yolo_model_path)
    folders_to_process = ['/home/roman/train_dataset_train_data_minprirodi/train_data_minprirodi/images',
                          '/home/roman/train_dataset_train_data_minprirodi/train_data_minprirodi/images_empty']
    output_csv_path = 'submissions/submission.csv'

    process_folders(folders_to_process, model, output_csv_path)