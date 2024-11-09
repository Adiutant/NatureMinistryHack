import cv2
import torch
from torchvision import transforms
from torchvision.models import resnet101
from ultralytics import YOLO


def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

class_map = {'good': 1, 'bad': 0}
available_detector = ["yolo"]
available_classificator = ["yolo", "resnet", "disable"]
class QualityModel(torch.nn.Module):
    def __init__(self, config, min_size=128):
        super(QualityModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config_detector = config.get("detector")
        config_detector_path = config.get("detector_path")
        if config_detector not in available_detector:
            exit(1)
        if config_detector == "yolo":
            self.detector = YOLO(config_detector_path).to(self.device)

        config_classificator = config.get("classificator")
        config_classificator_path = config.get("classificator_path")
        if config_classificator not in available_classificator:
            exit(1)
        if config_classificator == "yolo":
            self.classificator = YOLO(config_classificator_path).to(self.device)
        elif config_classificator == "resnet":
            self.classificator = torch.load(config_classificator_path).to(self.device)
        elif config_classificator == "disable":
            self.classificator = None
        self.config = config
        self.min_size = min_size

    def forward(self, image, file_name):
        results = self.detector(image, verbose=False)
        output_list = []
        height, width, _ = image.shape
        for result in results:
            if not result.boxes:
                continue
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls)
                class_name = self.detector.names[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                xc = ((x1 + x2) / 2) / width
                yc = ((y1 + y2) / 2) / height
                if (x2 - x1) > self.min_size and (y2 - y1) > self.min_size:
                    class_id = class_map[class_name]
                    cropped_image = image[y1:y2, x1:x2]
                    if self.config.get("classificator") == "resnet":
                        input_tensor = preprocess_image(cropped_image).to(self.device)
                        with torch.no_grad():
                            output = self.classificator(input_tensor)
                            predicted_class = (torch.sigmoid(output) > 0.5).float()
                            class_id = predicted_class.item()
                    elif self.config.get("classificator") == "yolo":
                        cropped_results = self.classificator(cropped_image, verbose=False)
                        if not cropped_results or not cropped_results[0].probs:
                            continue
                        class_id = torch.argmax(cropped_results[0].probs.data).item()
                else :
                    class_id = 0
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                output_list.append({'bbox': (xc, yc, w, h), 'class': class_id, 'file_name': file_name })

        return output_list