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

class QualityModel(torch.nn.Module):
    def __init__(self, yolo_model_path, min_size=128):
        super(QualityModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo = YOLO(yolo_model_path).to(self.device)
        self.resnet = torch.load("models/resnet50.pth").to(self.device)
        self.min_size = min_size

    def forward(self, image, file_name):
        results = self.yolo(image, verbose=False)
        output_list = []
        height, width, _ = image.shape
        for result in results:
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls)
                class_name = self.yolo.names[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                xc = ((x1 + x2) / 2) / width
                yc = ((y1 + y2) / 2) / height
                class_id = 0
                if (x2 - x1) > self.min_size and (y2 - y1) > self.min_size:
                    #class_id = class_map[class_name]
                    cropped_image = image[y1:y2, x1:x2]
                    input_tensor = preprocess_image(cropped_image).to(self.device)
                    with torch.no_grad():
                        output = self.resnet(input_tensor)
                        predicted_class = (torch.sigmoid(output) > 0.5).float()
                        class_id = predicted_class.item()
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                output_list.append({'bbox': (xc, yc, w, h), 'class': class_id, 'file_name': file_name })

        return output_list