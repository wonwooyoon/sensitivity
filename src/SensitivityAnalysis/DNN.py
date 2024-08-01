# This code read the data from ./input directory, and split the data into two parts: training data and testing data.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance  
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
        

class DNN_learning:

    def __init__(self):
        self.train_x = []
        self.train_Y = []
        self.test_x = []
        self.test_Y = []
        self.output_path = './src/SensitivityAnalysis/output'
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def data_split(self, file_path):
        self.input = pd.read_csv(f'{file_path}/input.csv')
        self.output = pd.read_csv(f'{file_path}/output.csv')
        self.output = self.output.iloc[:, 2]
        
        self.input_scaler = MinMaxScaler()
        self.output_scaler = MinMaxScaler()
        self.input = pd.DataFrame(self.input_scaler.fit_transform(self.input), columns=self.input.columns)
        self.output = pd.DataFrame(self.output_scaler.fit_transform(self.output.values.reshape(-1, 1)), columns=['Y'])
        
        self.train_x, self.test_x, self.train_Y, self.test_Y = train_test_split(self.input, self.output, test_size=0.15, random_state=42)

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        self.train_x = torch.tensor(self.train_x.values, device=device, dtype=torch.float)
        self.train_Y = torch.tensor(self.train_Y.values, device=device, dtype=torch.float)
        self.test_x = torch.tensor(self.test_x.values, device=device, dtype=torch.float)
        self.test_Y = torch.tensor(self.test_Y.values, device=device, dtype=torch.float)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.train_x.size(1), 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1024, 1),
        ).to(device)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.epoch = 10000
        # model parameters initialization
        for param in self.model.parameters():
            if len(param.size()) == 2:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=300, factor=0.5)
        lowest_loss = 1000000

        for t in tqdm(range(self.epoch)):
            
            y_pred = self.model(self.train_x)
            loss = self.loss_fn(y_pred, self.train_Y)
            print(t, loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            test_loss = self.loss_fn(self.model(self.test_x), self.test_Y)
            scheduler.step(test_loss)

            if t % 100 == 0:
                tqdm.write(f'epoch {t}, loss {loss.item() / len(self.train_x)}, test_loss {test_loss.item() / len(self.test_x)}')
            
            with torch.no_grad():
               # save loss and test loss per epoch to draw a loss evolution graph in self.less_evo and self.test_loss_evo
                if t == 0:
                    self.loss_evo = [loss.item() / len(self.train_x)]
                    self.test_loss_evo = [test_loss.item() / len(self.test_x)]
                else:
                    self.loss_evo.append(loss.item() / len(self.train_x))
                    self.test_loss_evo.append(test_loss.item() / len(self.test_x))

                if test_loss.item()/len(self.test_x) < lowest_loss:
                    lowest_loss = test_loss.item()/len(self.test_x)
                    torch.save(self.model.state_dict(), f'{self.output_path}/model.pth')

        # draw a loss evolution graph
        plt.plot(self.loss_evo, label='train')
        plt.plot(self.test_loss_evo, label='test')
        plt.legend()
        plt.savefig(f'{self.output_path}/loss_evo.png')

        # after the learning, unload the data and model from GPU to CPU
        self.train_x = self.train_x.cpu()
        self.train_Y = self.train_Y.cpu()
        self.test_x = self.test_x.cpu()
        self.test_Y = self.test_Y.cpu()
        self.model = self.model.cpu()
        
        # load the best model saved
        self.model.load_state_dict(torch.load(f'{self.output_path}/model.pth'))
        
    def test(self):
        y_pred = self.model(self.test_x)
        test_loss = self.loss_fn(y_pred, self.test_Y)
        print(test_loss.item())
        
        plt.clf()
        # Inverse transform the normalized data
        test_Y_inverse = self.output_scaler.inverse_transform(self.test_Y)
        y_pred_inverse = self.output_scaler.inverse_transform(y_pred.detach().numpy())

        # Calculate the R^2 value of the model
        y_pred = self.model(self.test_x)
        y_pred_inverse = self.output_scaler.inverse_transform(y_pred.detach().numpy())
        test_Y_inverse = self.output_scaler.inverse_transform(self.test_Y)
        y_pred_inverse = np.array(y_pred_inverse).reshape(-1)
        test_Y_inverse = np.array(test_Y_inverse).reshape(-1)
        ss_res = np.sum((test_Y_inverse - y_pred_inverse) ** 2)
        ss_tot = np.sum((test_Y_inverse - np.mean(test_Y_inverse)) ** 2)
        r2 = 1 - ss_res / ss_tot
       
        plt.plot(test_Y_inverse, y_pred_inverse, 'o')
        plt.plot([0, 70], [0, 70], 'r--')  # Add the x=y line
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.xlim(0, 70)  # Set x-axis range
        plt.ylim(0, 70)  # Set y-axis range
        plt.text(10, 60, f'R^2={r2:.3f}')
        plt.savefig(f'{self.output_path}/output.png')

    def sensitivity(self):
        # calculate the sensitivity of the model by permutation importance fator

        base_loss = self.loss_fn(self.model(self.test_x), self.test_Y)

        pif = []
        for i in range(self.test_x.size(1)):
            test_x = self.test_x.clone()
            sum_permute_loss = 0
            for j in range(100):
                permute = torch.randperm(test_x.size(0))
                test_x[:, i] = test_x[permute, i]
                permute_loss = self.loss_fn(self.model(test_x), self.test_Y)
                sum_permute_loss += permute_loss
                j += 1
            permute_loss = sum_permute_loss / 100
            
            pif.append((permute_loss / base_loss).detach().numpy())

        pif = np.array(pif)
        print(pif)
        plt.clf()
        plt.barh(self.input.columns, pif)
        plt.savefig(f'{self.output_path}/sensitivity.png')
        

if __name__ == '__main__':
    dnn = DNN_learning()
    dnn.data_split('./src/SensitivityAnalysis/input')
    dnn.train()
    dnn.test()
    dnn.sensitivity()

