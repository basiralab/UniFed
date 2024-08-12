# In this code, the first step wchich is spliting data into client data and server data (80% for client and 20% for server)has been stratified.
# Second step: spliting clients data into 8 sub-clients (ordered split, it is not stratified)
# Third step: spliting each sub-clients into train/valid/test using Stratifed sampling.

import medmnist
from medmnist import INFO
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import math
import pandas as pd
from collections import defaultdict


class IIDData:
    # Constructor
    def __init__(self, data_flags: list, split_ratio=0.2, test_ratio=0.2, validation_ratio=0.1, split_method="random", amount=8000):
        # List of datasets that we are going to use in this experiment
        # data_flags = ['octmnist', 'organamnist', 'tissuemnist']
        self.data_flags = data_flags
        self.dataflag_to_dataset = dict()
        self.server_batch_size = 64
        self.split_ratio = split_ratio

        self.test_ratio = test_ratio
        self.valid_ratio = validation_ratio
        self.split_method = split_method
        self.amount = amount

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.server_data = list() 
        self.server_labels  = list() 
        self.clients_data  = list() 
        self.clients_labels = list()

        self.get_dataset_dict()

    def get_transform(self):
        '''
        preprocessing
        '''
        data_transform = transforms.Compose([
            # Usually, 'transforms.ToTensor()' is used to turn the input data in the range of [0,255] to a 3-dimensional Tensor. 
            # This function automatically scales the input data to the range of [0,1]. (This is equivalent to scaling the data down to 0,1)
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        return data_transform

    def get_dataset(self, dataset_name='pathmnist', split='train'):
        '''
        load the data
        '''
        info = INFO[dataset_name]
        DataClass = getattr(medmnist, info['python_class']) # To load selected MedMNIST dataset

        data_transform = self.get_transform()
        dataset = DataClass(split=split, transform=data_transform, download=True)

        return dataset, info

    def get_dataset_dict(self):
        for data_flag in self.data_flags:
            dataset, info = self.get_dataset(dataset_name = data_flag, split='train')
            self.dataflag_to_dataset[data_flag] = (dataset, info)
    
    def get_label_indices(self, labels, label, samples_per_label):
        # Get indices of n occurrences of value x in the 'Column_Name'
        labels_df = pd.DataFrame(labels, columns=['label'])
        indices = np.where(labels_df['label'] == label)[0][:samples_per_label]
        return indices

    def get_server_data(self):
        # Splitting server and cline dataset while shifting the labels
        # For example: labels 0-3 -> 0-3, 0-4 -> 4-8
        # amount = 8000
        shift = 0

        for data_flag, (data, info) in self.dataflag_to_dataset.items():
            # print(data_flag)
            # print(len(data.imgs))
            # dataset = data.imgs[0: self.amount]
            # print(len(dataset))
            # select the amount of labels and make it falt
            # labels = list(np.ravel(data.labels[0: self.amount]))

            data_set = []
            label_set = []

            # getting the unique labels
            unique_labels = set(data.labels.flatten())
            samples_per_label = math.floor(self.amount / len(unique_labels))
            for eachlabel in unique_labels:
                label_indices = self.get_label_indices(data.labels, eachlabel, samples_per_label)
                label_set.append(np.take(data.labels, label_indices))
                data_set.append(np.take(data.imgs, label_indices, axis=0))

            dataset = np.concatenate(data_set, axis=0)
            dataset = np.array(dataset)

            labels = np.concatenate(label_set, axis=0)
            labels = np.array(labels)

            labels = [label + shift for label in labels]
            shift += len(self.dataflag_to_dataset[data_flag][1]['label'].keys())

            if self.split_method == "stratify":
                # splitting each data into 80% for clients and 20% for server
                X_clients, X_server, Y_clients, Y_server = train_test_split(dataset, labels, test_size=self.split_ratio, random_state=42, stratify=labels)
            else:
                X_clients, X_server, Y_clients, Y_server = train_test_split(dataset, labels, test_size=self.split_ratio, random_state=42)  

            self.server_data.append(X_server)
            self.server_labels.append(Y_server)

            self.clients_data.append(X_clients)
            self.clients_labels.append(Y_clients)

        # Concatanating the data that comes from clients to the server to create shared data (20 percent data of each client)
        main_server_data = np.concatenate(self.server_data, axis=0)
        main_server_label = np.concatenate(self.server_labels, axis=0)

        # Shuffeling the server data and labels simultaneously
        idx = np.random.permutation(len(main_server_label))
        main_server_data, main_server_label = main_server_data[idx], main_server_label[idx]

        # Spliting the server data
        server_trainvalid_data, server_test_data, server_trainvalid_labels, server_test_labels = train_test_split(main_server_data, main_server_label, test_size=self.test_ratio, random_state=42, stratify=main_server_label)
        server_train_data, server_valid_data, server_train_labels, server_valid_labels = train_test_split(server_trainvalid_data, server_trainvalid_labels, test_size=self.valid_ratio / (1 - self.test_ratio), random_state=42, stratify=server_trainvalid_labels)

        # dimension/format modification and correction
        # train data modification
        server_train_data = server_train_data.astype('float32')
        server_train_data = server_train_data[:,None,:,:]
        server_train_labels = server_train_labels[:, None]

        # valid data modification
        server_valid_data = server_valid_data.astype('float32')
        server_valid_data = server_valid_data[:,None,:,:]
        server_valid_labels = server_valid_labels[:, None]

        # test data modification
        server_test_data = server_test_data.astype('float32')
        server_test_data = server_test_data[:,None,:,:]
        server_test_labels = server_test_labels[:, None]

        server_train_data, server_train_labels = map(torch.tensor, (server_train_data, server_train_labels))
        server_valid_data, server_valid_labels = map(torch.tensor, (server_valid_data, server_valid_labels))
        server_test_data, server_test_labels = map(torch.tensor, (server_test_data, server_test_labels))

        # Transfer server data to GPU
        server_train_data = server_train_data.to(self.device)
        server_train_labels = server_train_labels.to(self.device)
        server_valid_data = server_valid_data.to(self.device)
        server_valid_labels = server_valid_labels.to(self.device)
        server_test_data = server_test_data.to(self.device)
        server_test_labels = server_test_labels.to(self.device)

        # Apply the custom transformation to the entire dataset
        resized_images_train = [self.resize_transform(image, 32, 32) for image in server_train_data]
        train_ds_server = TensorDataset(torch.stack(resized_images_train), server_train_labels)

        resized_images_valid = [self.resize_transform(image, 32, 32) for image in server_valid_data]
        valid_ds_server = TensorDataset(torch.stack(resized_images_valid), server_valid_labels)

        resized_images_test = [self.resize_transform(image, 32, 32) for image in server_test_data]
        test_ds_server = TensorDataset(torch.stack(resized_images_test), server_test_labels)

        train_dl_server = DataLoader(train_ds_server, batch_size=self.server_batch_size, shuffle=True)
        valid_dl_server = DataLoader(valid_ds_server, batch_size=self.server_batch_size, shuffle=True)
        test_dl_server = DataLoader(test_ds_server, batch_size=self.server_batch_size)
        
        return train_dl_server, valid_dl_server, test_dl_server

    # Define a custom transformation function to resize the images
    def resize_transform(self, image_tensor, new_height, new_width):
        image_pil = transforms.ToPILImage()(image_tensor)
        resized_image = transforms.Resize((new_height, new_width))(image_pil)
        return transforms.ToTensor()(resized_image)

    def get_ordered_client_split(self, client_data, client_label, num_clients):
        # # Shuffeling the client data and labels simultaneously
        idxc = np.random.permutation(len(client_label))
        client_label = np.array(client_label)[idxc]
        client_data = client_data[idxc]

        splited_client_data = np.array_split(client_data, num_clients)
        splited_client_label = np.array_split(client_label, num_clients)

        return splited_client_data, splited_client_label
    
    def get_stratified_client_split(self, clinet_data, client_label, num_clinets):
        pass

    def get_clients_data(self, num_clients=8, client_splitting_method=None):

        if client_splitting_method == None:
            print("The 'client_splitting_method' parameter should be either random, stratified, or equal not a None! Please indicate it!")

        # Here we are going to split all the client's dataset(3*8 clients) into test and train sets, change the type of data and labels, reshape the client's data, map the client's data to torch tensor
        x_train_dict, y_train_dict = {}, {}
        x_valid_dict, y_valid_dict  = {}, {}
        x_test_dict, y_test_dict  = {}, {}   

        client_id = 0
        # print("Yaaaay ---- s")

        for (each_client_data, each_client_label) in zip(self.clients_data, self.clients_labels):

            splited_client_data = []
            splited_client_label = []

            if client_splitting_method == "random":
                splited_client_data, splited_client_label = self.get_ordered_client_split(each_client_data, each_client_label, num_clients)
            elif client_splitting_method == "stratified":
                # getting the unique labels
                unique_labels = set(each_client_label)
                sample_per_client = len(each_client_data) / num_clients
                samples_per_label = math.floor(sample_per_client / len(unique_labels))
                # print(sample_per_client, samples_per_label)
                for eachclient in range(num_clients):
                    dataset = []
                    labels = []
                    for eachlabel in unique_labels:
                        label_indices = self.get_label_indices(each_client_label, eachlabel, samples_per_label)
                        dataset.append(np.take(each_client_data, label_indices, axis=0))
                        labels.append(np.take(each_client_label, label_indices))

                        each_client_data = np.delete(each_client_data, label_indices, axis=0)
                        each_client_label = np.delete(each_client_label, label_indices)
                    # print(len(splited_client_data), len(splited_client_label))

                    dataset = np.concatenate(dataset, axis=0)
                    dataset = np.array(dataset)

                    labels = np.concatenate(labels, axis=0)
                    labels = np.array(labels)

                    splited_client_data.append(dataset)
                    splited_client_label.append(labels)
            elif client_splitting_method == "equal":
                # getting the unique labels
                unique_labels = set(each_client_label)
                sample_per_client = len(each_client_data) / num_clients
                samples_per_label = math.floor(sample_per_client / len(unique_labels))
                # print(sample_per_client, samples_per_label)
                for eachclient in range(num_clients):
                    dataset = []
                    labels = []
                    for eachlabel in unique_labels:
                        label_indices = self.get_label_indices(each_client_label, eachlabel, samples_per_label)
                        dataset.append(np.take(each_client_data, label_indices, axis=0))
                        labels.append(np.take(each_client_label, label_indices))

                        each_client_data = np.delete(each_client_data, label_indices, axis=0)
                        each_client_label = np.delete(each_client_label, label_indices)
                    # print(len(splited_client_data), len(splited_client_label))

                    dataset = np.concatenate(dataset, axis=0)
                    dataset = np.array(dataset)

                    labels = np.concatenate(labels, axis=0)
                    labels = np.array(labels)

                    splited_client_data.append(dataset)
                    splited_client_label.append(labels)


            for (chunk_client_data, chunk_client_label) in zip(splited_client_data, splited_client_label):

                # Spliting the server data
                client_trainvalid_data, client_test_data, client_trainvalid_labels, client_test_labels = train_test_split(chunk_client_data, chunk_client_label, test_size=self.test_ratio, random_state=42, stratify=chunk_client_label)
                client_train_data, client_valid_data, client_train_labels, client_valid_labels = train_test_split(client_trainvalid_data, client_trainvalid_labels, test_size=self.valid_ratio / (1 - self.test_ratio), random_state=42, stratify=client_trainvalid_labels)

                chunk_client_data_train = client_train_data.astype('float32')
                chunk_client_data_valid = client_valid_data.astype('float32')
                chunk_client_data_test = client_test_data.astype('float32')

                chunk_client_data_train = chunk_client_data_train[:,None,:,:]
                chunk_client_label_train = client_train_labels[:, None]

                chunk_client_data_valid = chunk_client_data_valid[:,None,:,:]
                chunk_client_label_valid = client_valid_labels[:, None]

                chunk_client_data_test = chunk_client_data_test[:,None,:,:]
                chunk_client_label_test = client_test_labels[:, None]

                chunk_client_data_train, chunk_client_label_train, chunk_client_data_valid, chunk_client_label_valid,  chunk_client_data_test, chunk_client_label_test = map(torch.tensor, (chunk_client_data_train, chunk_client_label_train, chunk_client_data_valid, chunk_client_label_valid, chunk_client_data_test, chunk_client_label_test))

                chunk_client_data_train_resized = [self.resize_transform(image, 32, 32) for image in chunk_client_data_train]
                chunk_client_data_valid_resized = [self.resize_transform(image, 32, 32) for image in chunk_client_data_valid]
                chunk_client_data_test_resized = [self.resize_transform(image, 32, 32) for image in chunk_client_data_test]

                chunk_client_data_train, chunk_client_data_valid, chunk_client_data_test = torch.stack(chunk_client_data_train_resized), torch.stack(chunk_client_data_valid_resized), torch.stack(chunk_client_data_test_resized)

                x_train_dict[f'x_train{client_id}'] = chunk_client_data_train.to(self.device)
                y_train_dict[f'y_train{client_id}'] = chunk_client_label_train.to(self.device)

                x_valid_dict[f'x_valid{client_id}'] = chunk_client_data_valid.to(self.device)
                y_valid_dict[f'y_valid{client_id}'] = chunk_client_label_valid.to(self.device)

                x_test_dict[f'x_test{client_id}'] = chunk_client_data_test.to(self.device)
                y_test_dict[f'y_test{client_id}'] = chunk_client_label_test.to(self.device)

                client_id += 1

        return x_train_dict, y_train_dict, x_valid_dict, y_valid_dict, x_test_dict, y_test_dict 