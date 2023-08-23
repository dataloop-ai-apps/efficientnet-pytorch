import dtlpy as dl
import os
import re
import json
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import efficientnet_model
import matplotlib.pyplot as plt
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import logging
import copy

dl.setenv('prod')
if dl.token_expired():
    dl.login()


def custom_dataloaders(path: str, batch_size=64):
    def get_image_filepaths(directory):
        image_filepaths = list()
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_filepaths.append(os.path.join(root, file))

        return image_filepaths

    def get_json_filepaths(directory):
        json_filepaths = list()
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.json'):
                    json_filepaths.append(os.path.join(root, file))

        return json_filepaths

    def convert_image_filepath_to_array(image_filepath):
        # image_list = list()
        # for image_filepath in image_filepaths:
        image = Image.open(fp=image_filepath)
        image_array = np.array(image)
        image_array = image_array.astype(float)
        # image_list.append(image_array)
        image.close()

        return image_array

    def convert_json_filepath_to_label(json_filepath):
        # label_list = list()
        # for json_filepath in json_filepaths:
        json_file = open(file=json_filepath, mode="r")
        json_data = json.load(fp=json_file)
        label = int(json_data["annotations"][0]["label"])
        # label_list.append(label)

        return label

    items = os.path.join(path, "items")
    j_son = os.path.join(path, "json")
    # Get data directories
    train_directory = os.path.join(items, "training")
    valid_directory = os.path.join(items, "validation")
    train_json_directory = os.path.join(j_son, "training")
    valid_json_directory = os.path.join(j_son, "validation")

    # Get all filepaths
    train_image_filepaths = get_image_filepaths(train_directory)
    train_json_files = get_json_filepaths(train_json_directory)
    valid_image_filepaths = get_image_filepaths(valid_directory)
    valid_json_files = get_json_filepaths(valid_json_directory)

    # Sorting files
    def sort_regex(x):
        ext = x.split('.')[-1]
        return int(re.search(r'img_(\w+).{}'.format(ext), x).group(1))

    train_image_filepaths.sort(key=lambda x: sort_regex(x))
    train_json_files.sort(key=lambda x: sort_regex(x))
    valid_image_filepaths.sort(key=lambda x: sort_regex(x))
    valid_json_files.sort(key=lambda x: sort_regex(x))

    # Extracting data
    # train_image_data = convert_image_filepaths_to_arrays(image_filepaths=train_image_filepaths)
    # train_image_labels = convert_json_filepaths_to_labels(json_filepaths=train_json_files)
    # valid_image_data = convert_image_filepaths_to_arrays(image_filepaths=valid_image_filepaths)
    # valid_image_labels = convert_json_filepaths_to_labels(json_filepaths=valid_json_files)

    # Custom Dataset Creation
    class CustomDataset(Dataset):
        def __init__(self, data_list, labels_list, transforms_list):
            self.dataset = [(data, label) for data, label in zip(data_list, labels_list)]
            self.image_filepaths = data_list
            self.json_filepaths = labels_list
            self.length = len(self.dataset)
            self.transforms = transforms_list

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            image = convert_image_filepath_to_array(self.image_filepaths[idx])
            label = convert_json_filepath_to_label(self.json_filepaths[idx])
            # image, label = self.dataset[idx]
            for transform in self.transforms:
                image = transform(image)

            return image, label

    data_transforms = efficientnet_model.get_data_transforms()
    train_dataset = CustomDataset(
        data_list=train_image_filepaths,
        labels_list=train_json_files,
        transforms_list=data_transforms["train"]
    )
    valid_dataset = CustomDataset(
        data_list=valid_image_filepaths,
        labels_list=valid_json_files,
        transforms_list=data_transforms["valid"]
    )

    dataloaders = {
        "train": torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size),
        "valid": torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size)
    }
    return dataloaders


def train_model(model, device: torch.device, hyper_parameters: dict, dataloaders: dict, output_path: str,
                dataloader_option: str = 'custom'):
    save_path = "{}/model.pth".format(output_path)

    #########################
    # Load Hyper Parameters #
    #########################
    num_epochs = hyper_parameters.get("num_epochs", 50)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hyper_parameters.get("optimizer_lr", 0.001))

    ######################################
    # Data Structures for Saving Results #
    ######################################
    optimal_val_epoch = 0
    optimal_val_accuracy = 0

    x_axis = list(range(num_epochs))
    model_graph_data = dict()
    model_graph_data["epochs"] = x_axis
    model_graph_data["optimal_val_epoch"] = optimal_val_epoch
    model_graph_data["optimal_val_accuracy"] = optimal_val_accuracy

    model_graph_data["train"] = dict()
    model_graph_data["valid"] = dict()
    model_graph_data["train"]["loss"] = list()
    model_graph_data["valid"]["loss"] = list()
    model_graph_data["train"]["accuracy"] = list()
    model_graph_data["valid"]["accuracy"] = list()

    #########
    # Train #
    #########
    for epoch in tqdm(range(num_epochs)):
        # Each epoch has a training and validation phase
        for phase in ["train", "valid"]:
            if phase == "train":
                # Set model to training mode
                model.train()
                dataloader = dataloaders["train"]
                dataset_size = len(dataloader.batch_sampler.sampler)
            else:
                # Set model to evaluate mode
                model.eval()
                dataloader = dataloaders["valid"]
                dataset_size = len(dataloader.batch_sampler.sampler)

            running_loss = 0.0
            running_corrects = 0.0

            # Looping over all the dataloader images
            for i, data in enumerate(dataloader, start=0):
                # TODO: Make sure to use the correct dataloader_option
                if dataloader_option == "custom":
                    inputs, labels = data
                    inputs = inputs.to(device).float()
                    labels = labels.to(device)
                elif dataloader_option == "dataloop":
                    inputs = data["image"].to(device)
                    labels = data["annotations"].squeeze().to(device)
                else:
                    raise Exception("Invalid dataloader_option selected")

                # zero the gradient
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, predicts = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Calculating backward and optimize only during the training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Total loss of the mini batch
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predicts == labels.data).cpu().item()

            # Saving epoch loss and accuracy
            epoch_loss = running_loss / dataset_size
            epoch_accuracy = (running_corrects / dataset_size) * 100
            model_graph_data[phase]["loss"].append(epoch_loss)
            model_graph_data[phase]["accuracy"].append(epoch_accuracy)

            # Saving the weights of the best epoch
            if epoch_accuracy > optimal_val_accuracy:
                optimal_val_epoch = epoch
                optimal_val_accuracy = epoch_accuracy
                model_graph_data["optimal_val_epoch"] = optimal_val_epoch
                model_graph_data["optimal_val_accuracy"] = optimal_val_accuracy

                torch.save(copy.deepcopy(model.state_dict()), save_path)

    info = "Saving weights in path: {}".format(save_path)
    # print(info)
    logging.info(info)
    results = "Optimal hyper parameters were found at:\n" \
              "Epoch: {}\n" \
              "The Validation Accuracy: {}".format(model_graph_data["optimal_val_epoch"],
                                                   model_graph_data["optimal_val_accuracy"])
    # print(results)
    logging.info(results)
    plot_graph(model_graph_data)
    return model_graph_data


def plot_graph(model_graph_data: dict):
    plt.title("Model Loss:")
    plt.plot(model_graph_data["epochs"], model_graph_data["train"]["loss"], label="Train")
    plt.plot(model_graph_data["epochs"], model_graph_data["valid"]["loss"], label="Validation")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.png")

    plt.clf()
    plt.title("Model Accuracy:")
    plt.plot(model_graph_data["epochs"], model_graph_data["train"]["accuracy"], label="Train")
    plt.plot(model_graph_data["epochs"], model_graph_data["valid"]["accuracy"], label="Validation")
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy.png")


def local_training(model, device, hyper_parameters, dataloaders, output_path):
    model_graph_data = train_model(
        model=model,
        device=device,
        hyper_parameters=hyper_parameters,
        dataloaders=dataloaders,
        output_path=output_path
    )
    plot_graph(model_graph_data=model_graph_data)


def local_testing(model, device, dataloader):
    correct_count = 0
    all_count = 0
    y_true = list()
    y_predict = list()

    weights_path = "model.pth"
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    ########
    # Test #
    ########
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            predicts_prob = nn.functional.softmax(input=outputs, dim=1)
            _, predicts = torch.max(input=predicts_prob, dim=1)
            # eval labels here numeric (not one-hot)
            correct_predicts = torch.eq(labels, predicts).cpu()
            correct_count += correct_predicts.numpy().sum()
            all_count += len(labels)

            y_true.extend(labels.tolist())
            y_predict.extend(predicts.tolist())

    print("Number Of Images Tested =", all_count)
    print("Model Accuracy =", (correct_count / all_count) * 100)


def main():
    project = dl.projects.get(project_name="Abeer N Ofir Project")  # Enter your Project Name

    dataset = project.datasets.get(dataset_name="MNIST_Dataset")  # Enter Dataset Name

    list = dataset.download(local_path='/Users/saarahabdulla/Documents/efficientnet-pytorch',
                            # Specify local path to download dataset (same path as code)
                            annotation_options=[dl.ViewAnnotationOptions.JSON])
    data_path = os.getcwd()
    hyperparameters = {
        "num_epochs": 20,
        "optimizer_lr": 0.001,
        "output_size": 10,
    }
    dataloaders = custom_dataloaders(path=data_path)  # Default batch size is 64
    output_path = "."
    model = EfficientNet.from_pretrained(model_name="efficientnet-b0", num_classes=10, in_channels=1, image_size=28)
    local_training(model, device='cpu', hyper_parameters=hyperparameters,
                   dataloaders=dataloaders, output_path=output_path)


if __name__ == "__main__":
    main()
