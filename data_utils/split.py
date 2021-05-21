import pandas as pd
from sklearn.model_selection import train_test_split


def get_full_dataframe(txt_file):
    df = pd.read_csv(txt_file, converters={'timestamp': str})
    df["class_name"] = df.apply(lambda x: "person" if x.x >= 0 else "non_person", axis=1)

    return df


def split_df(df, test_size=.3):
    """
    split but the size can be reduced because we are only interested on images
    (dataframe may contain several inputs for the same image if there is more than one object)
    :param df:
    :param test_size:
    :return:
    """
    # remove duplicated images and get rid of data that we don't need
    df = df[~df.duplicated('timestamp')].drop(['x', 'y', 'w', 'h'], axis=1)
    train_df, val_df = train_test_split(df,
                                        test_size=test_size, random_state=42, shuffle=True, stratify=df["class_name"])
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    return train_df, val_df
