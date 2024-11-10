import tempfile
import os
import threading
import zipfile

import cv2
import requests
from dotenv import load_dotenv
from flask import jsonify
from flask import request
from tqdm import tqdm

from app import app
from pt_models.model import QualityModel

load_dotenv()

YOLO_MODEL = os.getenv("YOLO_DETECTOR")
RESNET_MODEL=os.getenv("RESNET_CLASSIFICATOR")
YOLO_CLASS = os.getenv("YOLO_CLASSIFICATOR")

model_config = {
    "detector" : "yolo",
    "detector_path" : YOLO_MODEL,
    "classificator": "disabled",
    "classificator_path" : RESNET_MODEL
}

class MainApplication:
    def __init__(self):
        self.model = QualityModel(model_config)

    def compute_labels(self, image_path):
        image = cv2.imread(image_path)
        return self.model(image, image_path.split("/")[-1])


def run_application():
    if __name__ == '__main__':
        threading.Thread(target=lambda: app.run(debug=False)).start()


application = MainApplication()

def transform_result(result):
    transformed_result = []
    for item in result:
        xc, yc, w, h = item['bbox']
        class_id = item['class']
        file_name = item['file_name']
        transformed_item = {
            'xc': xc,
            'yc': yc,
            'w': w,
            'h': h,
            'class': class_id,
            "file_name":file_name
        }
        transformed_result.append(transformed_item)
    return transformed_result

@app.route('/compute_photo', methods=['POST'])
def compute_photo():
    temp_dir = tempfile.TemporaryDirectory()
    download_url = request.json.get("download_url")
    filename = request.json.get("filename")
    if download_url == "":
        return jsonify({"error": "download url cant be empty"}), 422

    if filename == "":
        return jsonify({"error": "filename cant be empty"}), 422

    response = requests.head(download_url)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1 МБ
    filename = download_url.split("/")[-1]
    response = requests.get(download_url, stream=True)
    with open(os.path.join(temp_dir.name, filename), "wb") as handle:
        with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True) as progress_bar:
            for data in response.iter_content(block_size):
                handle.write(data)
                progress_bar.update(len(data))
    response.close()
    path = os.path.join(temp_dir.name, filename)
    result = application.compute_labels(path)
    transformed_result = transform_result(result)
    return jsonify({"annotations" : transformed_result, "images":[filename]}), 200


@app.route('/compute_zip', methods=['POST'])
def compute_zip():
    temp_dir = tempfile.TemporaryDirectory()
    download_url = request.json.get("download_url")
    filename = request.json.get("filename")
    if download_url == "":
        return jsonify({"error": "download url cant be empty"}), 422

    if filename == "":
        return jsonify({"error": "filename cant be empty"}), 422

    response = requests.head(download_url)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1 МБ
    filename = download_url.split("/")[-1]
    response = requests.get(download_url, stream=True)
    with open(os.path.join(temp_dir.name, filename), "wb") as handle:
        with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True) as progress_bar:
            for data in response.iter_content(block_size):
                handle.write(data)
                progress_bar.update(len(data))
    response.close()
    path = os.path.join(temp_dir.name, filename)
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir.name)

    result = []
    files_list =[]
    for root, dirs, files in os.walk(temp_dir.name):
        for name in files:
            image_result = application.compute_labels(name)
            result.append(transform_result(image_result))
            files_list.append(name)
    return jsonify({"annotations": result, "images":files_list}), 200


run_application()