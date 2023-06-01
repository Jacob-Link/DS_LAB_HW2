import random
import pandas as pd
import torch
import torchvision
import numpy as np
import os
from pathlib import Path
from matplotlib import pyplot as plt
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
from torch.utils import data
import secrets
import random
import copy


def imshow(inp, title=None):
    """Imshow for Tensors."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(15, 15))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def load_images(img_path):
    """Loads and transforms the datasets."""
    data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])
    images = datasets.ImageFolder(img_path, data_transforms)
    print(">>> loaded images from train folder...")
    return images


def load_images_with_erasing(img_path):
    """load images with erasing a random portion, used to achieve the same result as the dropout technique"""
    data_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor(),
                                          transforms.RandomErasing(p=1, scale=(0.08, 0.12), value=0),
                                          ])  # with prob p erase a portion == scale, and fill with black (0) white(1)

    images = datasets.ImageFolder(img_path, data_transforms)
    print(">>> loaded images with random erasing...")
    return images


def load_images_with_rotation_right(img_path, rotation):
    """load images with rotation"""
    data_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor(),
                                          transforms.RandomRotation(degrees=rotation, fill=1),
                                          ])  # randomly chooses a number in the range given, fills the edges white
    images = datasets.ImageFolder(img_path, data_transforms)
    print(">>> loaded images with random rotation to the right...")
    return images


def load_images_with_rotation_left(img_path, rotation):
    """load images with rotation"""
    data_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor(),
                                          transforms.RandomRotation(degrees=rotation, fill=1),
                                          ])  # randomly chooses a number in the range given, fills the edges white
    images = datasets.ImageFolder(img_path, data_transforms)
    print(">>> loaded images with random rotation to the left...")
    return images


def load_images_gaussian_blur(img_path):
    """load images with blur"""
    data_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor(),
                                          transforms.GaussianBlur(3)
                                          ])
    images = datasets.ImageFolder(img_path, data_transforms)
    print(">>> loaded images with gaussian blur...")
    return images


def load_images_horizontal_flip(img_path):
    """load images flipped horizontally"""
    data_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor(),
                                          transforms.RandomHorizontalFlip(p=1)
                                          ])
    images = datasets.ImageFolder(img_path, data_transforms)
    print(">>> loaded images with horizontal flip...")
    return images


def load_images_vertical_flip(img_path):
    """load images flipped vertically"""
    data_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor(),
                                          transforms.RandomVerticalFlip(p=1)
                                          ])
    images = datasets.ImageFolder(img_path, data_transforms)
    print(">>> loaded images with vertical flip...")
    return images


def filter_images_dataset(images, classes_to_keep, change_class_map={}):
    classes_to_keep = [images.class_to_idx[keep] for keep in classes_to_keep]
    change_class_map = {images.class_to_idx[k]: images.class_to_idx[v] for k, v in change_class_map.items()}

    indices_to_keep = []
    for i, (img, class_index) in enumerate(images.samples):
        if class_index in classes_to_keep:
            indices_to_keep.append(i)
            if class_index in change_class_map:  # overwrite the tuple label if in the change label mapping
                images.samples[i] = (img, change_class_map[class_index])

    filtered_dataset = torch.utils.data.Subset(images, indices_to_keep)

    return filtered_dataset



def save_images(save_path, dataset):
    os.makedirs(save_path, exist_ok=True)

    for img, target in dataset:
        class_label = dataset.classes[target]
        class_folder = os.path.join(save_path, class_label)
        os.makedirs(class_folder, exist_ok=True)

        # Save the transformed image
        save_image(img, f"{class_folder}/{secrets.token_hex(8)}.png")


def sample_data(images, per_class=1000, train_val_split=0.8):
    # use the filter code to create a count per class to know how much i can take max per label and the randomly sample data

    index_dict = dict()
    for i, (_, class_index) in enumerate(images):
        if class_index in index_dict:
            index_dict[class_index].append(i)
        else:
            index_dict[class_index] = [i]

    sampled_train_indices = []
    sampled_val_indices = []

    split_index = int(per_class * train_val_split)

    for label in index_dict:
        index_label_list = index_dict[label]
        per_class_indices = random.sample(index_label_list, k=per_class)
        random.shuffle(per_class_indices)

        sampled_train_indices += per_class_indices[:split_index]
        sampled_val_indices += per_class_indices[split_index:]

    # print([(k, len(v)) for k, v in index_dict.items()])
    sampled_train = torch.utils.data.Subset(images, sampled_train_indices)
    sampled_val = torch.utils.data.Subset(images, sampled_val_indices)
    print(f">>> sampled {per_class} images per class, split train val with {train_val_split} ratio...")
    return sampled_train, sampled_val


def create_image_folder_type(img_folder_type, subset_tensor_target_tup):
    img_folder_type.samples = [x for x in subset_tensor_target_tup]
    return img_folder_type


def save_images_from_subset(folder_path, dataset, class_list):
    os.makedirs(folder_path, exist_ok=True)

    for img, target in dataset:
        class_label = class_list[target]
        class_folder = os.path.join(folder_path, class_label)
        os.makedirs(class_folder, exist_ok=True)

        # Save the transformed image
        save_image(img, f"{class_folder}/{secrets.token_hex(8)}.png")

    print(f">>> successfully saved {len(dataset)} images to: {folder_path}")


def print_dataset_distribution(dataset, title, index_to_class_map, export=False):
    count = {k: 0 for k in index_to_class_map.values()}
    for _, label_index in dataset:
        count[index_to_class_map[label_index]] += 1

    data = []
    total_labels = sum(count.values())
    for key, val in count.items():
        tup = (key, val, f"{round(100 * val / total_labels, 2)}%")
        data.append(tup)
    data.sort(key=lambda x: x[1], reverse=True)

    print_df = pd.DataFrame(data, columns=["class", "count", "%"])
    print("----------  label balance ----------")
    print(print_df)
    print(f"Size of all dataset: {sum(print_df['count']):,}")

    if export:
        print_df.to_csv(f"{title}.csv", index=False)
        print(f">>> successfully exported {title}")

    print("------------------------------------")
    print()


if __name__ == '__main__':
    img_path_to_folders = Path.cwd().parent / 'data' / 'all_train_val_folder' / '1. all manually clean - for augmentation creation' / 'manually_cleaned_data'

    original_images = load_images(img_path_to_folders)
    random_erasing_images = load_images_with_erasing(img_path_to_folders)
    random_rotation_right_images = load_images_with_rotation_right(img_path_to_folders, rotation=(15, 165))
    random_rotation_left_images = load_images_with_rotation_left(img_path_to_folders, rotation=(-165, -15))
    random_gaussian_blur_images = load_images_gaussian_blur(img_path_to_folders)
    random_vertical_flip_images = load_images_vertical_flip(img_path_to_folders)
    random_horizontal_flip_images = load_images_horizontal_flip(img_path_to_folders)

    # imshow(random_horizontal_flip_images[0][0])
    # imshow(random_vertical_flip_images[0][0])
    # imshow(random_gaussian_blur_images[0][0])
    # imshow(random_rotation_left_images[0][0])
    # imshow(random_rotation_right_images[0][0])
    # imshow(random_erasing_images[0][0])
    # imshow(original_images[0][0])

    # filter the labels which can be flipped vertically and horizontally
    random_vertical_flip_images = filter_images_dataset(random_vertical_flip_images,
                                                        classes_to_keep=["i", "ii", "iii", "iv", "v", "vi", "vii",
                                                                         "viii", "ix", "x"])
    random_horizontal_flip_images = filter_images_dataset(random_horizontal_flip_images,
                                                          classes_to_keep=["i", "ii", "iii", "iv", "v", "vi", "vii",
                                                                           "viii", "ix", "x"],
                                                          change_class_map={"iv": "vi", "vi": "iv"})

    all_data = original_images + random_erasing_images + random_rotation_right_images + random_rotation_left_images \
               + random_gaussian_blur_images + random_horizontal_flip_images + random_vertical_flip_images

    index_to_class_map = {original_images.class_to_idx[label]: label for label in original_images.classes}

    print_dataset_distribution(all_data, "all_data_augmented", index_to_class_map, False)

    # split
    train_imgs, val_imgs = sample_data(all_data, per_class=1000, train_val_split=0.9)
    print_dataset_distribution(train_imgs, "train_9000_data_augmented", index_to_class_map, False)
    print_dataset_distribution(val_imgs, "test_1000_data_augmented", index_to_class_map, False)

    # save all images -- in the train and val folders
    save_images_from_subset(
        Path.cwd().parent / 'data' / 'all_train_val_folder' / '5. augmented_split_creation_1000_0p9' / "train",
        train_imgs,
        original_images.classes)

    save_images_from_subset(
        Path.cwd().parent / 'data' / 'all_train_val_folder' / '5. augmented_split_creation_1000_0p9' / "val", val_imgs,
        original_images.classes)
