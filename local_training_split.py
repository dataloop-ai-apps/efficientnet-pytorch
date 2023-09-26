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
import numpy as np
import glob2
import cv2

dl.setenv('prod')
if dl.token_expired():
    dl.login()



def custom_dataloaders(path: str, split_path: str, labels_json: str, batch_size=64, image_size=224, train_augmentations=None):
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

    def convert_image_filepath_to_array(image_filepath, image_size):
        # image_list = list()
        # for image_filepath in image_filepaths:
        image = Image.open(fp=image_filepath)
        image_array = np.array(image)
        #image_array = image_array.transpose(2, 0, 1)
        # image_list.append(image_array)
        image.close()

        return image_array

    def convert_json_filepath_to_label(json_filepath):
        # label_list = list()
        # for json_filepath in json_filepaths:
        json_file = open(file=json_filepath, mode="r")
        json_data = json.load(fp=json_file)
        label = json_data["annotations"][0]["label"]
        # label_list.append(label)
        return label

    def split(train_ratio, images_path):
        image_train_filepaths = list()
        image_val_filepaths = list()
        all_files = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp']:
            all_files += glob2.glob(f'{path}/**/*.{ext}')
        for file in all_files:
            filename_without_prefix = file[len(images_path)+1:]
            if np.random.rand()<train_ratio:
                image_train_filepaths.append(filename_without_prefix)
            else:
                image_val_filepaths.append(filename_without_prefix)

        return image_train_filepaths, image_val_filepaths

    with open(labels_json) as f:
        labels_dict = json.load(f)

    print('Preparing split')
    items = os.path.join(path, "items")
    j_son = os.path.join(path, "json")
    if os.path.isfile(split_path):
        with open(split_path) as f:
            split_items = json.load(f)
            image_train_filepaths = split_items['train']
            image_val_filepaths = split_items['val']
    else:
        image_train_filepaths, image_val_filepaths = split(0.85, items)
        with open(split_path, 'w') as f:
            json.dump({'train': image_train_filepaths, 'val': image_val_filepaths}, f)

    train_image_filepaths = [os.path.join(items, fn) for fn in image_train_filepaths]
    train_json_files = [os.path.join(j_son, '.'.join(fn.split('.')[:-1])+'.json') for fn in image_train_filepaths]
    valid_image_filepaths = [os.path.join(items, fn) for fn in image_val_filepaths]
    valid_json_files = [os.path.join(j_son, '.'.join(fn.split('.')[:-1])+'.json') for fn in image_val_filepaths]

    # Extracting data
    #train_image_data = convert_image_filepaths_to_arrays(image_filepaths=train_image_filepaths)
    #train_image_labels = convert_json_filepaths_to_labels(json_filepaths=train_json_files)
    #valid_image_data = convert_image_filepaths_to_arrays(image_filepaths=valid_image_filepaths)
    #valid_image_labels = convert_json_filepaths_to_labels(json_filepaths=valid_json_files)

    # Custom Dataset Creation
    class CustomDataset(Dataset):
        def __init__(self, data_list, labels_list, transforms_list, labels_dict, image_size):
            self.dataset = [(data, label) for data, label in zip(data_list, labels_list)]
            self.image_filepaths = data_list
            self.json_filepaths = labels_list
            self.length = len(self.dataset)
            self.transforms = transforms_list
            self.labels_dict = labels_dict
            self.image_size = image_size

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            image = convert_image_filepath_to_array(self.image_filepaths[idx], self.image_size)
            label = labels_dict[convert_json_filepath_to_label(self.json_filepaths[idx])]
            # image, label = self.dataset[idx]
            for transform in self.transforms:
                image = transform(image)
            return image, label, self.image_filepaths[idx]

    data_transforms = efficientnet_model.get_data_transforms(img_size=image_size, train_augmentations=train_augmentations)
    train_dataset = CustomDataset(
        data_list=train_image_filepaths,
        labels_list=train_json_files,
        transforms_list=data_transforms["train"],
        labels_dict = labels_dict,
        image_size = image_size
    )
    train_display_dataset = CustomDataset(
        data_list=train_image_filepaths,
        labels_list=train_json_files,
        transforms_list=data_transforms["train_display"],
        labels_dict = labels_dict,
        image_size = image_size
    )

    valid_dataset = CustomDataset(
        data_list=valid_image_filepaths,
        labels_list=valid_json_files,
        transforms_list=data_transforms["valid"],
        labels_dict = labels_dict,
        image_size = image_size
    )

    dataloaders = {
        "train": torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=10),
        "valid": torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=10),
        "train_display": torch.utils.data.DataLoader(dataset=train_display_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    }
    return dataloaders


def train_model(model, device: torch.device, hyper_parameters: dict, dataloaders: dict, output_path: str,
                dataloader_option: str = 'custom'):
    model.to(device)
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
                    inputs, labels, _ = data
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
                if i % 50==0:
                    print(f'{i}/{int(dataset_size/dataloader.batch_size)} Accuracy: {running_corrects/(i+1)/dataloader.batch_size} Loss: {running_loss/(i+1)/dataloader.batch_size}')

            # Saving epoch loss and accuracy
            epoch_loss = running_loss / dataset_size
            epoch_accuracy = (running_corrects / dataset_size) * 100
            model_graph_data[phase]["loss"].append(epoch_loss)
            model_graph_data[phase]["accuracy"].append(epoch_accuracy)
            print(f'Epoch loss {epoch_loss} Epoch Accuracy {epoch_accuracy}%')

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
    os.makedirs(output_path, exist_ok=True)
    model_graph_data = train_model(
        model=model,
        device=device,
        hyper_parameters=hyper_parameters,
        dataloaders=dataloaders,
        output_path=output_path
    )
    plot_graph(model_graph_data=model_graph_data)


def local_testing(model, device, dataloader, model_path, show_mistakes=False):
    correct_count = 0
    all_count = 0
    y_true = list()
    y_predict = list()

    weights_path = f"{model_path}/model.pth"
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()

    ########
    # Test #
    ########
    with torch.no_grad():
        for inputs, labels, filenames in dataloader:
            print(inputs.shape)
            print(inputs[0])
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            # print(outputs.cpu().numpy())
            predicts_prob = nn.functional.softmax(input=outputs, dim=1)
            _, predicts = torch.max(input=predicts_prob, dim=1)
            # eval labels here numeric (not one-hot)
            correct_predicts = torch.eq(labels, predicts).cpu()
            if show_mistakes:
                predicts_prob_cpu = predicts_prob.cpu()
                for i, corr in enumerate(correct_predicts):
                    if corr==False:
                        print(f"{predicts[i]}!={labels[i]} {predicts_prob[i]}")
                        img = cv2.imread(filenames[i])
                        cv2.imshow('tmp', img)
                        cv2.waitKey(-1)
            correct_count += correct_predicts.numpy().sum()
            all_count += len(labels)
            #print(predicts.cpu().numpy())

            y_true.extend(labels.tolist())
            y_predict.extend(predicts.tolist())

    print("Number Of Images Tested =", all_count)
    print("Model Accuracy =", (correct_count / all_count) * 100)

def display_data(dataloader):
    for inputs, labels, filenames in dataloader:
        for i in range(inputs.shape[0]):
            cv2.destroyAllWindows()
            m = inputs[i].cpu().numpy()
            m = m.transpose(1,2,0)
            m = m[..., ::-1]
            cv2.imshow(str(labels[i].cpu()), m)
            cv2.waitKey(-1)


def main():
    project = dl.projects.get(project_name="Orpak") # Enter your Project Name
    dataset = project.datasets.get(dataset_name="NozzleImagesClassification") # Enter Dataset Name
    dataset_local_path = '/media/lab/orpak_nozzle_classifier/dataset3'
    split_local_path = '/media/lab/orpak_nozzle_classifier/split3.json'
    labels_json = 'labels.json'
    model_name = 'efficientnet-b2'
    finetune_weights = '/media/lab/orpak_nozzle_classifier/'+model_name+'_5/model.pth'
    output_path = '/media/lab/orpak_nozzle_classifier/'+model_name+'_6'
    device='cuda:3'

    display_train = False
    test_on_valid = False
    test_on_train = False
    show_test_mistakes = False

    batch_size = 64
    if model_name=='efficientnet-b0':
        image_size = 224
    elif model_name=='efficientnet-b1':
        image_size = 240
    elif model_name=='efficientnet-b2':
        image_size = 260
    elif model_name=='efficientnet-b4':
        image_size = 380
        batch_size = 16

    train_augmentations = [
            {'prob': 1, 'type': 'resize', 'params': {'size': 500}},
            {'prob': 0.5, 'type': 'horiz_flip'},
            {'prob': 0.5, 'type': 'rotation', 'params': {'degrees': 20}},
            {'prob': 1, 'type': 'resized_crop', 'params': {'size': image_size, 'scale': (0.3, 1), 'ratio': (0.75, 1.33)}}
    ]
    if os.path.exists(dataset_local_path)==False:
        dataset.download(local_path=dataset_local_path, # Specify local path to download dataset (same path as code)
                                annotation_options=[dl.ViewAnnotationOptions.JSON])
    hyperparameters = {
        "num_epochs": 20,
        "optimizer_lr": 0.001,
        "output_size": 10,
    }
    dataloaders = custom_dataloaders(path=dataset_local_path, split_path=split_local_path, labels_json=labels_json, image_size=image_size, train_augmentations=train_augmentations, batch_size=batch_size) # Default batch size is 64
    print('Get model')
    model = EfficientNet.from_pretrained(model_name=model_name, num_classes=3,in_channels=3,image_size=image_size, weights_path=finetune_weights) 
    if display_train:
        display_data(dataloaders['display_train'])
    if test_on_train:
        print('Testing on train')
        local_testing(model, device=device, dataloader=dataloaders['train'], model_path=output_path, show_mistakes=show_test_mistakes)
    if test_on_valid:
        print('Testing on validation')
        local_testing(model, device=device, dataloader=dataloaders['valid'], model_path=output_path, show_mistakes=show_test_mistakes)
    print('Training')
    local_training(model, device=device, hyper_parameters=hyperparameters,
                   dataloaders=dataloaders, output_path=output_path)
    print('Testing')
    local_testing(model, device=device, dataloader=dataloaders['valid'], model_path=output_path)


if __name__ == "__main__":
    main()
