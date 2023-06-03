from pathlib import Path
from augmentation import load_images, load_images_gaussian_blur, load_images_with_rotation_left, \
    load_images_with_rotation_right, save_images_from_subset

PATH_TO_TRAIN_TARGET = Path.cwd().parent / "data" / "all_train_val_folder" / "10. final_data_create" / "train"
PATH_TO_HAND_WRITTEN = Path.cwd().parent / "data" / "handwritten"

if __name__ == '__main__':
    # augments the images in the hand drawn and saves 5x the amount with simple augmentations
    hand_drawn = load_images(PATH_TO_HAND_WRITTEN)
    hand_drawn_right_rot_1 = load_images_with_rotation_right(PATH_TO_HAND_WRITTEN, rotation=(15, 35))
    hand_drawn_right_rot_2 = load_images_with_rotation_right(PATH_TO_HAND_WRITTEN, rotation=(35, 60))
    load_images_with_rotation_left = load_images_with_rotation_left(PATH_TO_HAND_WRITTEN, rotation=(-165, -15))
    random_gaussian_blur_images = load_images_gaussian_blur(PATH_TO_HAND_WRITTEN)

    new_hand_drawn = hand_drawn + hand_drawn_right_rot_2 + hand_drawn_right_rot_1 + load_images_with_rotation_left + random_gaussian_blur_images

    index_to_class_map = {hand_drawn.class_to_idx[label]: label for label in hand_drawn.classes}

    save_imgs = True
    if save_imgs:
        save_images_from_subset(PATH_TO_TRAIN_TARGET, new_hand_drawn,
                                hand_drawn.classes)
