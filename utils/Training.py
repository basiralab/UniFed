from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import torch
import numpy as np
from copy import deepcopy

class Train:
    def __init__(self, n_classes) -> None:
     self.n_classes = n_classes
     self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, model, train_loader, criterion, optimizer):

        model.train()

        train_loss = 0.0
        correct = 0
        conf_matrix = torch.zeros(self.n_classes, self.n_classes)
        # metrics variables
        accuracy, precision, recall, f1 = [], [], [], []

        for data, target in train_loader:
            data = data.to(self.device)
            target = target.to(self.device)

            optimizer.zero_grad()
            output = model(data)

            target = target.squeeze().long()
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

            accuracy.append(accuracy_score(prediction.cpu().numpy(), target.cpu().numpy()))

            # Calculate Sensitivity, Specificity, and F1-score for each class
            metrics = precision_recall_fscore_support(prediction.cpu().numpy(), target.cpu().numpy(), average='macro', zero_division=0.0)
            precision_e, recall_e, f1_e, _ = metrics

            precision.append(precision_e)
            recall.append(recall_e)
            f1.append(f1_e)

            for t, p in zip(target.cpu().numpy(), prediction.cpu().numpy()):
              conf_matrix[t, p] += 1

        # TP = conf_matrix.diag()
        specificity = []
        for c in range(self.n_classes):
            idx = torch.ones(self.n_classes).long()
            # print(idx)
            idx[c] = 0
            # all non-class samples classified as non-class
            TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum() #conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
            # all non-class samples classified as class
            # print(idx.dtype, type(c))
            FP = conf_matrix[idx, c].sum()
            # all class samples not classified as class
            # FN = conf_matrix[c, idx].sum()
            
            # print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(c, TP[c], TN, FP, FN))

            specificity_c = TN / (TN + FP)
            specificity.append(specificity_c)

        # Print the results
        # perfromance metrics per epoch 
        # print(f'Accuracy: {np.mean(accuracy):.4f}')
        # print(f'Sensitivity (Recall): {np.mean(recall):.4f}')
        # print(f'Specificity: {np.mean(specificity):.4f}')
        # print(f'F1-score: {np.mean(f1):.4f}')
        

        return train_loss/len(train_loader), correct/len(train_loader.dataset), np.mean(accuracy), np.mean(recall), np.mean(specificity), np.mean(f1)
    
    def train_fedprox(self, model, train_loader, criterion, optimizer):

        model_0=deepcopy(model)

        model.train()

        train_loss = 0.0
        correct = 0
        conf_matrix = torch.zeros(self.n_classes, self.n_classes)
        # metrics variables
        accuracy, precision, recall, f1 = [], [], [], []

        for data, target in train_loader:
            data = data.to(self.device)
            target = target.to(self.device)

            optimizer.zero_grad()
            output = model(data)

            target = target.squeeze().long()
            loss = criterion(output, target)
            mu = 0.3
            loss += mu/2*self.difference_models_norm_2(model, model_0)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

            accuracy.append(accuracy_score(prediction.cpu().numpy(), target.cpu().numpy()))

            # Calculate Sensitivity, Specificity, and F1-score for each class
            metrics = precision_recall_fscore_support(prediction.cpu().numpy(), target.cpu().numpy(), average='macro', zero_division=0.0)
            precision_e, recall_e, f1_e, _ = metrics

            precision.append(precision_e)
            recall.append(recall_e)
            f1.append(f1_e)

            for t, p in zip(target.cpu().numpy(), prediction.cpu().numpy()):
              conf_matrix[t, p] += 1

        # TP = conf_matrix.diag()
        specificity = []
        for c in range(self.n_classes):
            idx = torch.ones(self.n_classes).long()
            # print(idx)
            idx[c] = 0
            # all non-class samples classified as non-class
            TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum() #conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
            # all non-class samples classified as class
            # print(idx.dtype, type(c))
            FP = conf_matrix[idx, c].sum()
            # all class samples not classified as class

            specificity_c = TN / (TN + FP)
            specificity.append(specificity_c)

        return train_loss/len(train_loader), correct/len(train_loader.dataset), np.mean(accuracy), np.mean(recall), np.mean(specificity), np.mean(f1)
    
    def difference_models_norm_2(self, model_1, model_2):
        """Return the norm 2 difference between the two model parameters
        """
        
        tensor_1=list(model_1.parameters())
        tensor_2=list(model_2.parameters())
        
        norm=sum([torch.sum((tensor_1[i]-tensor_2[i])**2) 
            for i in range(len(tensor_1))])
        
        return norm

    def train_with_batch_loss(self, model, train_loader, criterion, optimizer):

        model.train()
        train_loss = []
        correct = 0
        conf_matrix = torch.zeros(self.n_classes, self.n_classes)
        # metrics variables
        accuracy, precision, recall, f1 = [], [], [], []

        for data, target in train_loader:

            optimizer.zero_grad()
            output = model(data)

            target = target.squeeze().long()
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

            accuracy.append(accuracy_score(prediction.cpu().numpy(), target.cpu().numpy()))

            # Calculate Sensitivity, Specificity, and F1-score for each class
            metrics = precision_recall_fscore_support(prediction.cpu().numpy(), target.cpu().numpy(), average='macro', zero_division=0.0)
            precision_e, recall_e, f1_e, _ = metrics

            precision.append(precision_e)
            recall.append(recall_e)
            f1.append(f1_e)

            for t, p in zip(target.cpu().numpy(), prediction.cpu().numpy()):
                conf_matrix[t, p] += 1

        # TP = conf_matrix.diag()
        specificity = []
        for c in range(self.n_classes):
            idx = torch.ones(self.n_classes).long()
            idx[c] = 0
            # all non-class samples classified as non-class
            TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum() #conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
            # all non-class samples classified as class
            FP = conf_matrix[idx, c].sum()
            # all class samples not classified as class
            # FN = conf_matrix[c, idx].sum()
            
            # print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(c, TP[c], TN, FP, FN))

            specificity_c = TN / (TN + FP)
            specificity.append(specificity_c)

        # Print the results
        # perfromance metrics per epoch 
        # print(f'Accuracy: {np.mean(accuracy):.4f}')
        # print(f'Sensitivity (Recall): {np.mean(recall):.4f}')
        # print(f'Specificity: {np.mean(specificity):.4f}')
        # print(f'F1-score: {np.mean(f1):.4f}')

        return train_loss, correct/len(train_loader.dataset), np.mean(accuracy), np.mean(recall), np.mean(specificity), np.mean(f1)