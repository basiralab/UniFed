from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import torch
import numpy as np
import torch

class Evaluator:
    def __init__(self, n_classes) -> None:
     self.n_classes = n_classes 
     self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def validation(self, model, test_loader, criterion):
        model.eval()
        test_loss = 0
        correct = 0
        conf_matrix = torch.zeros(self.n_classes, self.n_classes)
        # metrics variables
        accuracy, precision, recall, f1 = [], [], [], []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data)
                # output.to(self.device)
                target = target.squeeze().long()
                # target.to(self.device)
                test_loss += criterion(output, target).item()
                prediction = output.argmax(dim=1, keepdim=True)
                correct += prediction.eq(target.view_as(prediction)).sum().item()
                
                # accuracy per batch
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

        # test_loss = 0 if len(test_loader) == 0 else test_loss /= len(test_loader)
        if len(test_loader) == 0:
            # print("Can't perform division: the y variable is 0")
            test_loss = 0
        else:
            test_loss /= len(test_loader)
            
        if len(test_loader.dataset) == 0:
            # print("Can't perform division: the yy variable is 0")
            correct = 0
        else:
            correct /= len(test_loader.dataset)

        return (test_loss, correct, np.mean(accuracy), np.mean(recall), np.mean(specificity), np.mean(f1))



    def validation_with_batch_loss(self, model, test_loader, criterion):
        model.eval()
        test_loss = []
        correct = 0
        conf_matrix = torch.zeros(self.n_classes, self.n_classes)
        # metrics variables
        accuracy, precision, recall, f1 = [], [], [], []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data)
                # output.to(self.device)
                target = target.squeeze().long()
                # target.to(self.device)
                test_loss.append(criterion(output, target).item()) 

                prediction = output.argmax(dim=1, keepdim=True)
                correct += prediction.eq(target.view_as(prediction)).sum().item()
                
                # accuracy per batch
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

        # test_loss = 0 if len(test_loader) == 0 else test_loss /= len(test_loader)
        # if len(test_loader) == 0:
        #     # print("Can't perform division: the y variable is 0")
        #     test_loss = 0
        # else:
        #     test_loss /= len(test_loader)
            
        # if len(test_loader.dataset) == 0:
        #     # print("Can't perform division: the yy variable is 0")
        #     correct = 0
        # else:
        #     correct /= len(test_loader.dataset)

        return (test_loss, correct, np.mean(accuracy), np.mean(recall), np.mean(specificity), np.mean(f1))

