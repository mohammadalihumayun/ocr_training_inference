#! git clone https://github.com/mohammadalihumayun/urdu-text-detection.git
#! pip install torch==2.0.1 ultralytics==8.1.8
#
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw
source_path=input('enter source images path')
roi_path=input('enter target roi path')
#os.makedirs(roi_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
for img_path in os.listdir(output_path):
    detection_model = YOLO("/content/urdu-text-detection/yolov8m_UrduDoc.pt")
    input = Image.open(output_path+img_path)
    detection_results = detection_model.predict(source=input, conf=0.2, imgsz=1280, save=False, nms=True, device=device)
    bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
    bounding_boxes.sort(key=lambda x: x[1])
    for i, box in enumerate(bounding_boxes):
      # Crop the ROI from the input image
      roi = input.crop(box)
      # Save each cropped ROI as a separate image
      roi.save(f"{roi_path}{img_path}_roi_{i+1}.jpg")
