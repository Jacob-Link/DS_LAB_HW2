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


def print_counter(count_labels_dict):
    data = []
    total_labels = sum(count_labels_dict.values())
    for key, val in count_labels_dict.items():
        tup = (key, val, f"{round(100 * val / total_labels, 2)}%")
        data.append(tup)
    data.sort(key=lambda x: x[1], reverse=True)

    print_df = pd.DataFrame(data, columns=["class", "count", "%"])
    print(print_df)


if __name__ == '__main__':
    count_labels_dict = load_balanced_class_check()
    print_counter(count_labels_dict)
