import h5py
import os
from PIL import Image
from tqdm import tqdm
from glob import glob
path = "../train"

file_paths = glob(os.path.join(path, "*"))
with h5py.File("data_256.h5", "w") as f:
    for img_path in tqdm(file_paths):
        group_name = os.path.split(img_path)[-1]

        # g = f.create_group(group_name)
        imgs = glob(os.path.join(img_path, "*jpg"))

        data = f.create_dataset(group_name, (len(imgs), 256, 256, 3))
        for i, img in enumerate(imgs):
            data[i] = Image.open(img).resize((256, 256))

