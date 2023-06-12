import torch
import numpy as np
import math
from .utilities import plot_results
from torchmetrics import SpearmanCorrCoef

class Trainer():
    def __init__(self):
        pass
    
    def train_LLMRegresor(self,train_iterator, val_iterator, model, device, criterion, optimizer, epoch_num):
        
        val_loss = []
        
        model.to(device)

        for epoch in range(epoch_num):
            model.train() 
            for i, (input, label) in enumerate(train_iterator):
                optimizer.zero_grad()
                
                input = input.to(dtype=torch.float32, device=device)
                label = label.to(dtype=torch.float32, device=device) 
                out = model(input)
                loss = criterion(out, label.unsqueeze(-1))
                loss.backward() 
                optimizer.step()

            with torch.no_grad(): # evaluate validation loss here 

                model.eval()
                val_loss_epochs = []

                for (inputval, labelval) in val_iterator:
                    
                    inputval = inputval.to(device)
                    labelval = labelval.to(device)
                    outval = model(inputval)
                    lossval = criterion(outval, labelval.unsqueeze(-1)) # Calculate validation loss 
                    val_loss_epochs.append(lossval.item())

                val_loss_epoch = np.mean(val_loss_epochs)
                val_loss.append(round(val_loss_epoch, 3))

            print('epoch: %d loss: %.3f val loss: %.3f' % (epoch + 1, loss.item(), val_loss_epoch))

        return model, val_loss
    
    def test_model(self, model, test_set, test_labels, loss, device):
        test_set = test_set.to(dtype=torch.float32, device=device)
        test_labels = test_labels.to(dtype=torch.float32, device=device) 
        outcome = model(test_set)

        loss_test = loss(outcome, test_labels.unsqueeze(-1))
        mae = torch.nn.L1Loss()(outcome.flatten(), test_labels)
        
        spearman = SpearmanCorrCoef()
        spear_corr = spearman(outcome.flatten(), test_labels)
        print('MSE: ' + str(loss_test))
        print('RMSE: ' + str(torch.sqrt(loss_test)))
        print('MAE: ' + str(mae))
        print('Spearman Corr: ' + str(spear_corr))
        
        return loss_test, outcome
