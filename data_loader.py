from pathlib import Path
import pandas as pd

DATA_PATH = Path.cwd().parent / 'data' / 'hw2_094295' / 'data'


def load_balanced_class_check():
    # checking how many labels are in each class
    train_path = DATA_PATH / 'train'

    counter_dict = dict()
    for label_path in train_path.iterdir():
        number_of_images_in_class = len(list(label_path.iterdir()))
        label_name = label_path.parts[-1]
        counter_dict[label_name] = number_of_images_in_class

    return counter_dict


def print_counter(count_labels_dict, export=False):
    data = []
    total_labels = sum(count_labels_dict.values())
    for key, val in count_labels_dict.items():
        tup = (key, val, f"{round(100 * val / total_labels, 2)}%")
        data.append(tup)
    data.sort(key=lambda x: x[1], reverse=True)

    print_df = pd.DataFrame(data, columns=["class", "count", "%"])
    print("----------  label balance ----------")
    print(print_df)
    print(f"Size of train dataset: {sum(print_df['count']):,}")
    print("------------------------------------")
    print()
    if export:
        print_df.to_csv("class_distribution.csv", index=False)


def print_counter_post_manual_clean(count_labels_dict, export=False):
    pre_manual_clean = []
    post_manual_clean = []  # called the new folder example: 1_out
    total_labels = sum(count_labels_dict.values())
    for key, val in count_labels_dict.items():
        tup = (key, val, f"{round(100 * val / total_labels, 2)}%")
        if "_out" in key:
            post_manual_clean.append(tup)
        else:
            pre_manual_clean.append(tup)

    pre_manual_clean.sort(key=lambda x: x[1], reverse=True)
    post_manual_clean.sort(key=lambda x: x[1], reverse=True)

    print_df_pre = pd.DataFrame(pre_manual_clean, columns=["class", "count", "%"])
    print_df_post = pd.DataFrame(post_manual_clean, columns=["class", "count", "%"])
    print("---------- manual clean ----------")
    print(print_df_pre)
    print(f"Size of clean train dataset: {sum(print_df_pre['count']):,}")
    print("--------------------------------------")
    print()

    print("---------- removed manual clean ----------")
    print(print_df_post)
    print(f"Size of removed: {sum(print_df_post['count']):,}")
    print("--------------------------------------")

    if export:
        print_df_pre.to_csv("manual_clean_class_distribution.csv", index=False)
        print_df_post.to_csv("removed_manual_clean_class_distribution.csv", index=False)




if __name__ == '__main__':
    initial_balance_check = False
    post_manual_clean_check = True

    if initial_balance_check:
        count_labels_dict = load_balanced_class_check()
        print_counter(count_labels_dict)

    if post_manual_clean_check:
        count_labels_dict = load_balanced_class_check()
        print_counter_post_manual_clean(count_labels_dict)
