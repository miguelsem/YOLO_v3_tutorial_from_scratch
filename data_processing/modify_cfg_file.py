cfg_file = "D:\src\YOLO_v3_tutorial_from_scratch\cfg\yolov3_pose_custom.cfg"

# Read the file
with open(cfg_file, "r") as file:
    lines = file.readlines()

# Modify lines
new_lines = [("classes=0\n" if line.strip().startswith("classes=") else line) for line in lines]

# Write back
with open(cfg_file, "w") as file:
    file.writelines(new_lines)