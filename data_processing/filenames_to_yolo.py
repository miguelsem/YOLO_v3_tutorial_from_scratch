import os
import re

folder_base = "D:/src/YOLO_v3_tutorial_from_scratch/data/feb12/"

folder_image = [folder_base + r"images\test", folder_base + r"images\train", folder_base + r"images\val"]
folder_label = [folder_base + r"labels\test", folder_base + r"labels\train", folder_base + r"labels\val"]

# image_folder = r"C:\dev\NetworkTutorialRealtime\data"
# label_folder = "dataset/labels"
for image_folder, label_folder in zip(folder_image, folder_label):
    # Paths
    os.makedirs(label_folder, exist_ok=True)

    # Regex to extract coordinates from filenames
    pattern = re.compile(r'\[(\d+),(\d+)\](?:\[(\d+),(\d+)\])?(?:\[(\d+),(\d+)\])?(?:\[(\d+),(\d+)\])?(?:\[(\d+),(\d+)\])?(?:\[(\d+),(\d+)\])?(?:\[(\d+),(\d+)\])?')

    for filename in os.listdir(image_folder):
        # if ".txt" in filename:
        #     os.remove(image_folder + "\\" + filename)
        #
        # elif False:
            match = pattern.search(filename)
            if match:
                coords = [int(c) if c else None for c in match.groups()]
                coords = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2) if coords[i] is not None]

                # Normalize coordinates (assuming 640x640 images)
                width, height = 192, 192  # Adjust if images are different sizes
                keypoints = []
                for x, y in coords:
                    keypoints.extend([x / width, y / height, 1])  # Visibility = 1

                # Pad missing keypoints with (0,0,0)
                while len(keypoints) < 6 * 3:
                    keypoints.extend([0, 0, 0])

                # Create label file
                label_path = os.path.join(label_folder, filename.replace(".png", ".txt"))
                with open(label_path, "w") as f:
                    f.write(" ".join(map(str, keypoints)) + "\n")

    print("Conversion complete! Labels saved in:", label_folder)
