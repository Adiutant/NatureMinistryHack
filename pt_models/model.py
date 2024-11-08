import torch
from ultralytics import YOLO

class_map = {'good': 1, 'bad': 0}

class QualityModel(torch.nn.Module):
    def __init__(self, yolo_model_path, min_size=150):
        super(QualityModel, self).__init__()
        self.yolo = YOLO(yolo_model_path)
        self.min_size = min_size

    def forward(self, image):
        results = self.yolo(image)
        output_dict = {}
        height, width, _ = image.shape
        for result in results:
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls)
                class_name = self.yolo.names[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                xc = ((x1 + x2) / 2) / width
                yc = ((y1 + y2) / 2) / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                class_id = 0
                if width >= self.min_size and height >= self.min_size:
                    class_id = class_map[class_name]

                output_dict[i] = {'bbox': (xc, yc, w, h), 'class': class_id}

        return output_dict