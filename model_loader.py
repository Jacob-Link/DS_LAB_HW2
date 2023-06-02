from pathlib import Path
from data_loader import load_dataset
import torch.nn as nn
import torch
from torchvision import models

BATCH_SIZE = 16


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
    print(f"Accuracy on the test set supplied: {100*correct.double()/data_size:.4f}%")
    print("*******************")


def predict_data(model, data_loader):
    print("predicting test set...")
    running_corrects = 0
    model.eval()
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(data_loader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == classes)
            # print(preds)
            # print(classes)
    print(">>> finished predicting on test set!")
    return running_corrects


if __name__ == '__main__':
    data_path = Path.cwd().parent / 'online data' / 'test-data-challenge' / 'val'
    # model_path = Path.cwd().parent / 'models' / 'trained_model_1.pt'
    # model_path = Path.cwd().parent / 'models' / 'trained_model_2.pt'
    # model_path = Path.cwd().parent / 'models' / "trained_model_10k_black.pt"
    model_path = Path.cwd().parent / 'models' / "trained_model_10k_white.pt"

    test_data = load_dataset(data_path)
    test_dataloader = init_loader(test_data)

    class_names = test_data.classes
    print("The classes are: ", class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = len(class_names)

    # used from code given in assignment
    model_ft = models.resnet50(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    # load model, predict, print_results
    model = load_state_dict(model_ft, model_path)
    correct = predict_data(model_ft, test_dataloader)
    print_stats(correct, len(test_data))
