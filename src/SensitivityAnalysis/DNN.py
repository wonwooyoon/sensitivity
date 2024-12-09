# This code read the data from ./input directory, and split the data into two parts: training data and testing data.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
        
        self.inout = pd.read_csv(f'{file_path}/inout.csv')

        self.input = self.inout.iloc[:, 0:5] # 0, 1, 2, 3, 4
        self.output = self.inout.iloc[:, -4] # -4, -3, -2, -1
        
        self.input_scaler = StandardScaler()
        self.output_scaler = MinMaxScaler()
        self.input = pd.DataFrame(self.input_scaler.fit_transform(self.input), columns=self.input.columns)
        self.output = pd.DataFrame(self.output_scaler.fit_transform(self.output.values.reshape(-1, 1)), columns=['Y'])
        
        self.train_x, self.test_x, self.train_Y, self.test_Y = train_test_split(self.input, self.output, test_size=0.2, random_state=83)

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        self.train_x = torch.tensor(self.train_x.values, device=device, dtype=torch.float)
        self.train_Y = torch.tensor(self.train_Y.values, device=device, dtype=torch.float)
        self.test_x = torch.tensor(self.test_x.values, device=device, dtype=torch.float)
        self.test_Y = torch.tensor(self.test_Y.values, device=device, dtype=torch.float)

        ##################################
        node_num = 256
        dropout_rate = 0.2
        layer_num = 4
        initial_lr = 0.001 
        l2_reg = 1e-4
        batch_size = 8
        self.epoch = 1000
        ##################################

        layers = []
        input_size = self.train_x.size(1)
        layers.append(torch.nn.Linear(input_size, node_num))
        layers.append(torch.nn.GELU())
        layers.append(torch.nn.Dropout(dropout_rate))

        for _ in range(layer_num-1):
            layers.append(torch.nn.Linear(node_num, node_num))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(dropout_rate))
        
        layers.append(torch.nn.Linear(node_num, 1))
        self.model = torch.nn.Sequential(*layers).to(device)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=initial_lr, weight_decay=l2_reg)

        # model parameters initialization
        for param in self.model.parameters():
            if len(param.size()) == 2:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)

        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=150, gamma=0.9)
        lowest_loss = 1000000
    
        train_dataset = torch.utils.data.TensorDataset(self.train_x, self.train_Y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for t in tqdm(range(self.epoch)):
            self.model.train()
            
            for batch_x, batch_Y in train_loader:
                y_pred = self.model(batch_x)
                loss = self.loss_fn(y_pred, batch_Y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            train_loss = self.loss_fn(self.model(self.train_x), self.train_Y)
            test_loss = self.loss_fn(self.model(self.test_x), self.test_Y)
            scheduler.step()

            if t % 25 == 0:
                tqdm.write(f'epoch {t}: train_loss {train_loss.item() / len(self.train_x)}, test_loss {test_loss.item() / len(self.test_x)}, lr = {self.optimizer.param_groups[0]["lr"]}')

            with torch.no_grad():
                if t == 0:
                    self.loss_evo = [train_loss.item() / len(self.train_x)]
                    self.test_loss_evo = [test_loss.item() / len(self.test_x)]
                else:
                    self.loss_evo.append(train_loss.item() / len(self.train_x))
                    self.test_loss_evo.append(test_loss.item() / len(self.test_x))

            if test_loss.item() / len(self.test_x) < lowest_loss:
                lowest_loss = test_loss.item() / len(self.test_x)
                torch.save(self.model.state_dict(), f'{self.output_path}/model.pth')
        
        print(f'lowest test loss: {lowest_loss}')

        # draw a loss evolution graph
        plt.plot(self.loss_evo, label='train')
        plt.plot(self.test_loss_evo, label='test')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'{self.output_path}/loss_evo.png')

        # after the learning, unload the data and model from GPU to CPU
        self.train_x = self.train_x.cpu()
        self.train_Y = self.train_Y.cpu()
        self.test_x = self.test_x.cpu()
        self.test_Y = self.test_Y.cpu()
        self.model = self.model.cpu()
        
        # load the best model saved
        self.model.load_state_dict(torch.load(f'{self.output_path}/model.pth', weights_only=True))
        
    def test(self):

        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(self.train_x).cpu().numpy()
            test_pred = self.model(self.test_x).cpu().numpy()
            train_Y = self.train_Y.cpu().numpy()
            test_Y = self.test_Y.cpu().numpy()

        plt.figure(figsize=(5, 5))

        plt.scatter(train_Y, train_pred, alpha=0.2, color='orange')
        plt.scatter(test_Y, test_pred, alpha=0.2, color='blue')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.plot([0, 0], [1, 1], 'r--')

        plt.tight_layout()
        plt.savefig(f'{self.output_path}/prediction_vs_actual.png')


    def sensitivity(self):
        # calculate the sensitivity of the model by permutation importance fator

        base_loss = self.loss_fn(self.model(self.test_x), self.test_Y)

        pif = []
        for i in range(self.test_x.size(1)):
            test_x = self.test_x.clone()
            sum_permute_loss = 0
            for _ in range(100):
                permute = torch.randperm(test_x.size(0))
                test_x[:, i] = test_x[permute, i]
                permute_loss = self.loss_fn(self.model(test_x), self.test_Y)
                sum_permute_loss += permute_loss
            permute_loss = sum_permute_loss / 100
            
            pif.append((permute_loss / base_loss).detach().numpy())

        pif = np.array(pif)
        print(pif)
        plt.clf()
        plt.barh(self.input.columns, pif)
        plt.savefig(f'{self.output_path}/sensitivity.png')
        

if __name__ == '__main__':
    dnn = DNN_learning()
    dnn.data_split('./src/TargetValueAnalysis/output')
    dnn.train()
    dnn.test()
    # dnn.sensitivity()

