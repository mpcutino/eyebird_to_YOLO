from data_utils.write import main_write, main_train_test_txt
from data_utils.remove import remove_non_important_images


if __name__ == "__main__":
    cfg_ = "Soccer_People_1.json"
    # main_write(cfg_)
    # main_train_test_txt()

    remove_non_important_images(cfg_)
