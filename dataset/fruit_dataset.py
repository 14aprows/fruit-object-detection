import os
import cv2 
import torch 
import numpy as np
import xml.etree.ElementTree as et
from torch.utils.data import Dataset

class FruitImageDataset(Dataset):
    def __init__(self, files_dir, transform=None):
        self.files_dir = files_dir
        self.transform = transform
        self.imgs = [
            image for image in sorted(os.listdir(files_dir))
            if image.endswith(".jpg")
        ]

        self.classes = [
            "background",
            "apple",
            "banana",
            "orange"
        ]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.files_dir, img_name)

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        annot_filename = img_name.replace(".jpg", ".xml")
        annot_file_path  = os.path.join(self.files_dir, annot_filename)

        tree = et.parse(annot_file_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for member in root.findall("object"):
            class_name = member.find("name").text
            labels.append(self.classes.index(class_name)) 

            xmin = int(member.find("bndbox").find("xmin").text)
            ymin = int(member.find("bndbox").find("ymin").text)
            xmax = int(member.find("bndbox").find("xmax").text)
            ymax = int(member.find("bndbox").find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
    
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if boxes.shape[0] == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        if self.transform:
            sample = self.transform(
                image = img_rgb,
                bboxes = boxes.tolist(),
                labels = labels.tolist()
            )

            img_rgb = sample["image"]
            boxes = torch.tensor(sample["bboxes"], dtype=torch.float32)
            labels = torch.tensor(sample["labels"], dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([idx])
        }

        return img_rgb, target