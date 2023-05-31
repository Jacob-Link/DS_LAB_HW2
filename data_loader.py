from pathlib import Path
import pandas as pd

DATA_PATH = Path.cwd().parent / 'data' / 'hw2_094295' / 'data'


def load_balanced_class_check(path=DATA_PATH):
    # checking how many labels are in each class
    train_path = path / 'train'

    counter_dict = dict()
    for label_path in train_path.iterdir():
        number_of_images_in_class = len(list(label_path.iterdir()))
        label_name = label_path.parts[-1]
        counter_dict[label_name] = number_of_images_in_class

    return counter_dict


def print_counter(count_labels_dict, title='', export=False):
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
        if title != '':
            print_df.to_csv(f"{title}.csv", index=False)
        else:
            print_df.to_csv("class_distribution.csv", index=False)


def print_counter_post_manual_clean(count_labels_dict, export=False):
    pre_manual_clean = []
    post_manual_clean = []  # called the new folder example: 1_out

    for key, val in count_labels_dict.items():
        tup = (key, val)
        if "_out" in key:
            post_manual_clean.append(tup)
        else:
            pre_manual_clean.append(tup)

    cleaned = []
    total = sum(x[1] for x in pre_manual_clean)
    for x in pre_manual_clean:
        tup = (x[0], x[1], f"{round(100 * x[1] / total, 2)}%")
        cleaned.append(tup)

    removed = []
    total = sum(x[1] for x in post_manual_clean)
    for x in post_manual_clean:
        tup = (x[0], x[1], f"{round(100 * x[1] / total, 2)}%")
        removed.append(tup)

    cleaned.sort(key=lambda x: x[1], reverse=True)
    removed.sort(key=lambda x: x[1], reverse=True)

    print_df_clean = pd.DataFrame(pre_manual_clean, columns=["class", "count", "%"])
    print_df_removed = pd.DataFrame(post_manual_clean, columns=["class", "count", "%"])
    print("---------- manual clean ----------")
    print(print_df_clean)
    print(f"Size of clean train dataset: {sum(print_df_clean['count']):,}")
    print("--------------------------------------")
    print()

    print("---------- removed manual clean ----------")
    print(print_df_removed)
    print(f"Size of removed: {sum(print_df_removed['count']):,}")
    print("--------------------------------------")

    if export:
        print_df_clean.to_csv("manually_cleaned_class_distribution.csv", index=False)
        print_df_removed.to_csv("removed_data_class_distribution.csv", index=False)


def load_pre_post_clean():
    df_pre = pd.read_csv("class_distribution.csv")
    df_post = pd.read_csv("after_manually_cleaned_distribution.csv")

    df_pre = df_pre[["class", "count"]].rename(columns={"count": "pre_count"})
    df = df_post.merge(df_pre, on="class", how="left")
    df["items removed/added"] = df["count"] - df["pre_count"]
    df = df.drop(columns=["pre_count"])
    return df


if __name__ == '__main__':
    initial_balance_check = False
    post_manual_clean_check = False
    print_diff = True

    if initial_balance_check:
        count_labels_dict = load_balanced_class_check()
        print_counter(count_labels_dict)

    if post_manual_clean_check:
        count_labels_dict = load_balanced_class_check()
        print_counter(count_labels_dict, title="after_manually_cleaned_distribution")

    if print_diff:
        df = load_pre_post_clean()
        print(df)
        export = False
        if export:
            df.to_csv('post_manual_clean_info.csv', index=False)
            print(">>> successfully exported info about data set post clean")


