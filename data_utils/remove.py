from os import path as op, remove

import json
from glob import glob


def remove_non_important_images(cfg_file):
    with open(cfg_file) as cfd:
        d = json.load(cfd)
    for img_name in glob(d['imgs_path']+"/*.png"):
        txt_name = img_name[:-4] + ".txt"
        if not op.exists(txt_name):
            remove(img_name)
