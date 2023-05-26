import random

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

"""
rotate right
rotate left 
flip horizontal 
flip vertical
random erase
gaussian blur
"""


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
    # Resize the samples and transform them into tensors
    data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])

    # Create a pytorch dataset from a directory of images
    images = datasets.ImageFolder(img_path, data_transforms)
    return images


def load_images_with_erasing(img_path):
    """load images with erasing a random portion, used to achieve the same result as the dropout technique"""
    data_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor(),
                                          transforms.RandomErasing(p=1, scale=(0.08, 0.12), value=0),
                                          ])  # with prob p erase a portion == scale, and fill with black (0) white(1)

    images = datasets.ImageFolder(img_path, data_transforms)
    return images


def load_images_with_rotation_right(img_path):
    """load images with rotation"""
    data_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor(),
                                          transforms.RandomRotation(degrees=(15, 45), fill=1),
                                          ])  # randomly chooses a number in the range given, fills the edges white
    images = datasets.ImageFolder(img_path, data_transforms)
    return images


def load_images_with_rotation_left(img_path):
    """load images with rotation"""
    data_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor(),
                                          transforms.RandomRotation(degrees=(-45, -15), fill=1),
                                          ])  # randomly chooses a number in the range given, fills the edges white
    images = datasets.ImageFolder(img_path, data_transforms)
    return images


def load_images_gaussian_blur(img_path):
    """load images with blur"""
    data_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor(),
                                          transforms.GaussianBlur(3)
                                          ])
    images = datasets.ImageFolder(img_path, data_transforms)
    return images


def load_images_horizontal_flip(img_path):
    """load images flipped horizontally"""
    data_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor(),
                                          transforms.RandomHorizontalFlip(p=1)
                                          ])
    images = datasets.ImageFolder(img_path, data_transforms)
    return images


def load_images_vertical_flip(img_path):
    """load images flipped vertically"""
    data_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor(),
                                          transforms.RandomVerticalFlip(p=1)
                                          ])
    images = datasets.ImageFolder(img_path, data_transforms)
    return images


def filter_images_dataset(images, classes_to_keep):
    classes_to_keep = [images.class_to_idx[keep] for keep in classes_to_keep]

    indices_to_keep = []
    for i, (_, class_index) in enumerate(images.samples):
        if class_index in classes_to_keep:
            indices_to_keep.append(i)
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

    return sampled_train, sampled_val


if __name__ == '__main__':
    # img_path_to_folders = Path.cwd().parent / 'augmenting_test'
    img_path_to_folders = Path.cwd().parent / 'new_train_folders' / '1. manual_clean_remove_noise_fix_incorrectly_labelled' / 'train'

    original_images = load_images(img_path_to_folders)
    random_erasing_images = load_images_with_erasing(img_path_to_folders)
    random_rotation_right_images = load_images_with_rotation_right(img_path_to_folders)
    random_rotation_left_images = load_images_with_rotation_left(img_path_to_folders)
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
                                                        classes_to_keep=["i", "ii", "iii", "x"])
    random_horizontal_flip_images = filter_images_dataset(random_horizontal_flip_images,
                                                          classes_to_keep=["i", "ii", "iii", "v", "x"])

    all_data = original_images + random_erasing_images + random_rotation_right_images + random_rotation_left_images \
               + random_gaussian_blur_images + random_horizontal_flip_images + random_vertical_flip_images

    train_imgs, val_imgs = sample_data(all_data, per_class=10, train_val_split=0.9)

    print(len(train_imgs))
    print(len(val_imgs))

    # TODO: start from here!!!!
    print([x for x in val_imgs]) # each item in list is a tup of tensor and class index in the original dataset from folder

    # save all images -- in the train and val folders -- create new func, which can work with subset and concat dataset

    # save_images(Path.cwd().parent / "new_train_folders" / "augmented_results" / "train", train_imgs)
    # save_images(Path.cwd().parent / "new_train_folders" / "augmented_results" / "val", val_imgs)

    # TODO: add prints at each step!!!
