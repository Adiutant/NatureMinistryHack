import cv2
import numpy as np
import torch
from torch.xpu import device
from torchvision import transforms
from torchvision.models import resnet101
from ultralytics import YOLO

def is_largest_contour_broken(image, thresh=100, break_threshold=0.1):
    my_photo = image
    filtered_image = cv2.medianBlur(my_photo, 7)
    img_grey = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_perimeter = 0
    sel_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, False)
        if perimeter > max_perimeter:
            sel_contour = contour
            max_perimeter = perimeter
    if sel_contour is None:
        return False
    border_points_count = 0
    is_closed = np.array_equal(sel_contour[0][0], sel_contour[-1][0])
    for point in sel_contour:
        x, y = point[0]
        height, width = my_photo.shape[:2]
        if x == 0 or x == width - 1 or y == 0 or y == height - 1:
            border_points_count += 1
    if not is_closed:
        start_point = sel_contour[0][0]
        end_point = sel_contour[-1][0]
        height, width = my_photo.shape[:2]
        start_on_border = start_point[0] == 0 or start_point[0] == width - 1 or start_point[1] == 0 or start_point[1] == height - 1
        end_on_border = end_point[0] == 0 or end_point[0] == width - 1 or end_point[1] == 0 or end_point[1] == height - 1
        if start_on_border and end_on_border:
            distance = np.linalg.norm(start_point - end_point) + border_points_count
            #print(f"Distance between endpoints on border: {distance}")
            if distance > break_threshold * max_perimeter:
                return True

    contour_image = my_photo.copy()
    cv2.drawContours(contour_image, [sel_contour], -1, (0, 255, 0), 2)
    return False


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
available_classificator = ["yolo", "resnet","catboost", "disabled"]
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
        elif config_classificator == "disabled":
            self.classificator = None
        self.config = config
        self.min_size = min_size

    def forward(self, image, file_name):
        results = self.detector(image, iou=0.7, conf =0.25, verbose=False)
        output_list = []
        height, width, _ = image.shape
        for result in results:
            if not result.boxes:
                continue
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls)

                class_name = self.detector.names[class_id]
                class_id = class_map[class_name]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # xc = ((x1 + x2) / 2) / width
                # yc = ((y1 + y2) / 2) / height
                (xc, yc, w, h) = map(float, box.xywhn[0])
                cropped_image = image[y1:y2, x1:x2]

                ### heuristic МОЖНО ОТКЛЮЧИТЬ!!!
                is_near_border = 0 if (
                    (((xc - w / 2 < 0.01 or
                        xc + w / 2 > 0.99) and
                        h > w)
                        and ((x2 - x1) < 200 or (y2 - y1) < 200))
                    or is_largest_contour_broken(cropped_image)

                ) else 1
                ### heuristic


                if (x2 - x1) > self.min_size and (y2 - y1) > self.min_size:
                    class_id = class_id  & is_near_border
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
                output_list.append({'bbox': (xc, yc, w, h), 'class': class_id, 'file_name': file_name })

        return output_list