import os
import pandas as pd

data_path = r"D:\src\YOLO_v3_tutorial_from_scratch\data\feb12\images"
csv_path = "\\".join(data_path.split("\\")[:-1])

for folder in os.listdir(data_path):
    data_folder = os.path.join(data_path, folder)
    paths = []
    for filename in os.listdir(data_folder):
        filename_padded = f'{"[".join(filename.split("[")[1:])}'
        filename_padded = filename_padded.split(".")[0].split("][")
        filename_padded[-1] = filename_padded[-1][:-1]
        if filename_padded == [""]:
            filename_padded = []
        for i in range(6-len(filename_padded)):
            filename_padded.append("0,0")
        filename_padded = f"[{']['.join(filename_padded)}]"
        paths.append([os.path.join(data_path, folder, filename), filename_padded])

    df = pd.DataFrame(paths, columns=["path", "label"])
    df.to_csv(os.path.join(csv_path, f"{folder}.csv"))
