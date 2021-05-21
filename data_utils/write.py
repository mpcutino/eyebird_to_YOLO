from os import path as op, remove

import json
import pandas as pd

from data_utils.split import get_full_dataframe, split_df


def write_txt_per_image(image_folder_path, df, img_w=346, img_h=260):
    # erasing old txt files
    for _, r in df.iterrows():
        if r.x == -1: continue

        txt_name = op.join(image_folder_path, r.timestamp + ".txt")
        if op.exists(txt_name):
            remove(txt_name)

    class_dict = {c: i for i, c in enumerate(df.class_name.unique())}
    # adding new ones
    for _, r in df.iterrows():
        img_name = op.join(image_folder_path, r.timestamp + ".png")
        if op.exists(img_name):
            txt_name = img_name[:-4] + ".txt"
            # open in append mode to add new objects
            with open(txt_name, 'a') as fd:
                class_index = class_dict[r.class_name]
                if r.x != -1:
                    # data to write: center of the image and with and height of the object, relative to the full image
                    center_x = (r.x + r.w)/2
                    center_y = (r.y + r.w)/2
                    # person is class 0
                    fd.write(" ".join([str(class_index),
                                       str(center_x/img_w),
                                       str(center_y/img_h),
                                       str(r.w/img_w),
                                       str(r.h/img_h)]) + "\n")
                else:
                    # non_person is class 1
                    fd.write(" ".join([str(class_index),
                                       "0",
                                       "0",
                                       "0",
                                       "0"]) + "\n")
    print(class_dict)


def main_write(cfg_file):
    with open(cfg_file) as cfd:
        d = json.load(cfd)
    df_ = get_full_dataframe(d['txt_file'])
    write_txt_per_image(d['imgs_path'], df_)

    vc = df_[~df_.duplicated('timestamp')].class_name.value_counts()
    print(vc)
    print(len(df_[~df_.duplicated('timestamp')]))
    print(len(df_[~df_.duplicated('timestamp')]) + vc.person)


def main_train_test_txt():
    train_list, test_list = [], []
    for cfg_file in ["../Soccer_People_2.json", "../Soccer_People_1.json"]:
        with open(cfg_file) as cfd:
            d = json.load(cfd)
        df_ = get_full_dataframe(d['txt_file'])
        df_['full_img_path'] = d['imgs_path'] + "/" + df_['timestamp'] + ".png"

        train_df_, test_df_ = split_df(df_)
        train_list.append(train_df_)
        test_list.append(test_df_)

    train_df = pd.concat(train_list, axis=0).sample(frac=1).reset_index(drop=True)
    test_df = pd.concat(test_list, axis=0).sample(frac=1).reset_index(drop=True)

    txt_paths = d['save_train_path'] if 'save_train_path' in d else ""
    with open(op.join(txt_paths, "train.txt"), 'w') as fd:
        fd.write("\n".join(train_df.full_img_path))
    with open(op.join(txt_paths, "test.txt"), 'w') as fd:
        fd.write("\n".join(test_df.full_img_path))
