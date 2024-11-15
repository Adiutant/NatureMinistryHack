import json
import os

import cv2
import pandas as pd
from dotenv import load_dotenv

import sub_test
from pt_models.model import QualityModel

load_dotenv()

YOLO_MODEL = os.getenv("YOLO_DETECTOR")
RESNET_MODEL=os.getenv("RESNET_CLASSIFICATOR")
YOLO_CLASS = os.getenv("YOLO_CLASSIFICATOR")
CATBOOST_CLASS = os.getenv("CATBOOST_CLASSIFICATOR")

model_config = {
    "detector" : "yolo",
    "detector_path" : YOLO_MODEL,
    "classificator": "disabled",
    "classificator_path" : CATBOOST_CLASS
}

def process_folders(folders, model, output_csv):
    data = []

    for folder in folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    file_path = os.path.join(root, file)
                    image = cv2.imread(file_path)

                    if image is not None:
                        output_dict = model(image, file)

                        for info in output_dict:
                            bbox_str = ','.join(map(str, info['bbox']))
                            data.append({'Name': file, 'Bbox': bbox_str, 'Class': info['class']})

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    folder_models = "models/detector"
    best_score = {"score":0, "model":""}
    for root, _, files in os.walk(folder_models):
        for file in files:
            if file.lower().endswith('.pt'):
                yolo_model_path = os.path.join(root, file)
                model_config["detector_path"] = yolo_model_path
                model = QualityModel(model_config)
                folders_to_process = ['/home/roman/train_dataset_train_data_minprirodi/train_data_minprirodi/images',
                          '/home/roman/train_dataset_train_data_minprirodi/train_data_minprirodi/images_empty']
                output_csv_path = 'submissions/submission.csv'

                process_folders(folders_to_process, model, output_csv_path)
                names = ["Name", "Bbox", "Class"]
                detected_objects = pd.read_csv('submissions/submission.csv', names=names, header=0)
                ground_truth = pd.read_csv('submissions/annotation.csv', names=names, header=0)
                score = sub_test.evaluate_metrics(detected_objects, ground_truth)
                if score > best_score.get("score"):
                    best_score["model"] = file
                    best_score["score"] = score
                print(f"Model: {file} with score {score}")
    b_model = best_score.get('model')
    b_score = best_score.get('score')
    print(f"Best model: {b_model} with score {b_score}")