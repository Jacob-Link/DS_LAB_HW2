"""
Idea: move all wrongfully predicted images into the
train folder. Cool test, dropped overfit gap. but didnt get to 100 val acc, randomness in init, train data causes it
to learn different representation function
"""
import numpy as np
import os
import secrets
from torchvision.utils import save_image
from pathlib import Path
from error_analysis import extract_data_from_missed, update_labels_to_numerals
from model_loader import get_correct_and_missed_labels


def extract_images_from_prediction_dict(pred_dict):
    res_list = []
    for k, v in pred_dict.items():
        if type(v) == type(dict()):
            if len(v["gt"]) != 1:
                for i in range(len(v["gt"])):
                    tempt_dict = {"img": v["img"][i], "gt": v["gt"][i].item()}
                    res_list.append(tempt_dict)
            else:
                tempt_dict = {"img": v["img"][0], "gt": v["gt"][0]}
                res_list.append(tempt_dict)
    return res_list


def export_images(folder_path, images):
    os.makedirs(folder_path, exist_ok=True)

    for img_dict in images:
        class_label = img_dict["gt"]
        class_folder = folder_path / class_label
        os.makedirs(class_folder, exist_ok=True)

        # Save the transformed image
        save_image(img_dict["img"], f"{class_folder}/{secrets.token_hex(8)}.png")

    print(f">>> successfully saved {len(images)} images to: {folder_path}")


def labels_to_numerals(res_list, mapping):
    for dict in res_list:
        try:
            dict["gt"] = mapping[dict["gt"]]
        except KeyError:
            dict["gt"] = mapping[dict["gt"].item()]


if __name__ == '__main__':
    path_to_data_folder = Path.cwd().parent / 'data' / 'all_train_val_folder' / '11. attempt_3_train_data_with_wrong_pred'

    path_to_model = Path.cwd().parent.parent / "Run Results" / "3. no weird flips 8850 0p95 split" / "trained_model.pt"

    correct_predictions, missed_predictions, index_to_class_map = get_correct_and_missed_labels(
        path_to_data_folder / 'val', path_to_model)
    val_missed = extract_images_from_prediction_dict(missed_predictions)
    val_correct = extract_images_from_prediction_dict(correct_predictions)

    labels_to_numerals(val_missed, index_to_class_map)
    labels_to_numerals(val_correct, index_to_class_map)

    # export_path = Path.cwd().parent / 'new_train_folders' / 'moving_images_to'

    export_images(path_to_data_folder / 'train', val_missed)
    export_images(path_to_data_folder / 'new_val', val_correct)

    print(">>> created new_val folder and moved all incorrectly predicted images from val to the train folder")
