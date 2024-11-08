import json

import cv2
from pt_models.model import QualityModel


if __name__ == "__main__":
    image = cv2.imread("example")
    print(json.dumps(QualityModel(image)))