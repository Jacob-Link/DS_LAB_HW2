from pathlib import Path
from data_loader import load_dataset
import torch.nn as nn
import torch
from torchvision import models
import numpy as np

BATCH_SIZE = 16
DATA_PATH = Path.cwd().parent / 'online data' / 'test-data-challenge' / 'val'
# DATA_PATH = Path.cwd()/ 'val_error_analysis'/ 'val'
MODEL_PATH = Path.cwd().parent / 'models' / "trained_model_8850_0p9_limit_rotation.pt"


def init_loader(data):
    test_dataloader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    return test_dataloader


def load_state_dict(model, model_path):
    model.load_state_dict(torch.load(model_path))
    print(f">>> successfully loaded model from {model_path}")
    return model


def print_stats(correct, data_size):
    print()
    print("*******************")
    print(f"Accuracy on the test set supplied: {100 * correct.double() / data_size:.4f}%")
    print("*******************")


def predict_data(model, data_loader, device, criterion):
    print("predicting test set...")
    running_corrects = 0
    running_incorrects = 0
    model.eval()
    with torch.no_grad():
        missed = dict()  # log all missed classified images for error analysis
        for i, (inputs, classes) in enumerate(data_loader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == classes)

            single_losses = np.array([criterion(outputs[i], classes[i]) for i in range(len(preds))])

            diff = torch.argwhere(preds != classes)
            running_incorrects += len(diff)

            if len(diff) != 0:
                missed[f'batch_{i}'] = {"img": inputs[diff], "gt": classes[diff], "pred": preds[diff],
                                        "losses": single_losses[diff]}
            else:
                missed[f'batch_{i}'] = "predicted all correctly"


    print(f">>> finished predicting on test set! (correct: {running_corrects}, incorrect: {running_incorrects})")
    return running_corrects, missed


def get_missed_labels():
    test_data = load_dataset(DATA_PATH)
    print(f"Dataset size: {len(test_data)} images")
    test_dataloader = init_loader(test_data)
    NUM_CLASSES = len(test_data.classes)
    model_ft = models.resnet50(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = load_state_dict(model_ft, MODEL_PATH)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_function = nn.CrossEntropyLoss()
    correct, missed = predict_data(model, test_dataloader, device, loss_function)
    print_stats(correct, len(test_data))
    index_to_class_map = {test_data.class_to_idx[label]: label for label in test_data.classes}
    return missed, index_to_class_map


if __name__ == '__main__':
    # data_path = Path.cwd().parent / 'online data' / 'test-data-challenge' / 'val'
    # model_path = Path.cwd().parent / 'models' / 'trained_model_1.pt'
    # model_path = Path.cwd().parent / 'models' / 'trained_model_2.pt'
    # model_path = Path.cwd().parent / 'models' / "trained_model_10k_black.pt"
    # model_path = Path.cwd().parent / 'models' / "trained_model_10k_white.pt"
    # model_path = Path.cwd().parent / 'models' / "trained_model_8850_no_weird_flips.pt"

    test_data = load_dataset(DATA_PATH)
    test_dataloader = init_loader(test_data)

    class_names = test_data.classes
    print("The classes are: ", class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_function = nn.CrossEntropyLoss()
    NUM_CLASSES = len(class_names)

    # used from code given in assignment
    model_ft = models.resnet50(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    # load model, predict, print_results
    model = load_state_dict(model_ft, MODEL_PATH)
    correct, _ = predict_data(model, test_dataloader, device, loss_function)
    print_stats(correct, len(test_data))
