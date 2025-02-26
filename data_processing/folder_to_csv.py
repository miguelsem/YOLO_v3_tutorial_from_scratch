import os
import pandas as pd

data_path = r"D:\src\YOLO_v3_tutorial_from_scratch\data\feb12\images"
csv_path = "\\".join(data_path.split("\\")[:-1])

for folder in os.listdir(data_path):
    data_folder = os.path.join(data_path, folder)
    paths = []
    for filename in os.listdir(data_folder):
        # List sstring
        filename_padded = f'{"[".join(filename.split("[")[1:])}'
        filename_padded = filename_padded.split(".")[0].split("][")
        filename_padded[-1] = filename_padded[-1][:-1]
        if filename_padded == [""]:
            filename_padded = []
        for i in range(6-len(filename_padded)):
            filename_padded.append("0,0,0")
        filename_padded = f"[{']['.join(filename_padded)}]"
        filename_padded.split("][")

        # List of floats
        filename_split = f'{"[".join(filename.split("[")[1:])}'.replace("].png", "").split("][")
        filename_split = [val+",1" for val in filename_split]
        if filename_split == [",1"]:
            filename_split = []
        while len(filename_split) < 6:
            filename_split.append("0,0,0")
        # fileout = [int(val) for val in sum([val.split(",") for val in filename_split], [])]
        fileout = ",".join(filename_split)
        # print(fileout)

        paths.append([os.path.join(data_path, folder, filename), fileout])

    df = pd.DataFrame(paths, columns=["path", "label"])
    df.to_csv(os.path.join(csv_path, f"{folder}.csv"))
