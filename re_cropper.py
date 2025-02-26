import os
import cv2
import matplotlib.pyplot as plt

image_path = "D:/src/YOLO_v3_tutorial_from_scratch/data/feb12/images"

for folder in os.listdir(image_path):
    folder_path = os.path.join(image_path, folder)
    for file in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_UNCHANGED)
        dimensions = img.shape

        # if dimensions == (213, 193):
        print(file)
        image_path_single = os.path.join(folder_path, file)
        image = cv2.imread(image_path_single)
        stretch_near = cv2.resize(image, (192, 192), interpolation=cv2.INTER_LINEAR)
        gray_image = cv2.cvtColor(stretch_near, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(image_path_single, gray_image)
