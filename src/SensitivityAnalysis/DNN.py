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
        self.input = pd.read_csv(f'{file_path}/input_wwy.csv')
        self.output = pd.read_csv(f'{file_path}/output_wwy.csv')
        self.output = self.output.iloc[:, 0]
        
        self.input_scaler = StandardScaler()
        self.output_scaler = MinMaxScaler()
        self.input = pd.DataFrame(self.input_scaler.fit_transform(self.input), columns=self.input.columns)
        self.output = pd.DataFrame(self.output_scaler.fit_transform(self.output.values.reshape(-1, 1)), columns=['Y'])
        
        self.train_x, self.test_x, self.train_Y, self.test_Y = train_test_split(self.input, self.output, test_size=0.2, random_state=128378)

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        self.train_x = torch.tensor(self.train_x.values, device=device, dtype=torch.float)
        self.train_Y = torch.tensor(self.train_Y.values, device=device, dtype=torch.float)
        self.test_x = torch.tensor(self.test_x.values, device=device, dtype=torch.float)
        self.test_Y = torch.tensor(self.test_Y.values, device=device, dtype=torch.float)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.train_x.size(1), 16),
            torch.nn.GELU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(16, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(256, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(256, 16),
            torch.nn.GELU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(16, 1),
        ).to(device)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        self.epoch = 1000
        # model parameters initialization
        for param in self.model.parameters():
            if len(param.size()) == 2:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=1000, factor=0.8)
        lowest_loss = 1000000

    
        batch_size = 4
        train_dataset = torch.utils.data.TensorDataset(self.train_x, self.train_Y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for t in tqdm(range(self.epoch)):
            self.model.train()
            epoch_loss = 0
            for batch_x, batch_Y in train_loader:
                y_pred = self.model(batch_x)
                loss = self.loss_fn(y_pred, batch_Y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            test_loss = self.loss_fn(self.model(self.test_x), self.test_Y)
            scheduler.step(test_loss)

            if t % 100 == 0:
                tqdm.write(f'epoch {t}, loss {epoch_loss}, test_loss {test_loss.item() / len(self.test_x)}, lr = {self.optimizer.param_groups[0]["lr"]}')

            with torch.no_grad():
                if t == 0:
                    self.loss_evo = [epoch_loss]
                    self.test_loss_evo = [test_loss.item() / len(self.test_x)]
                else:
                    self.loss_evo.append(epoch_loss)
                    self.test_loss_evo.append(test_loss.item() / len(self.test_x))

            if test_loss.item() / len(self.test_x) < lowest_loss:
                lowest_loss = test_loss.item() / len(self.test_x)
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

        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(self.train_x).cpu().numpy()
            test_pred = self.model(self.test_x).cpu().numpy()
            train_Y = self.train_Y.cpu().numpy()
            test_Y = self.test_Y.cpu().numpy()

        plt.figure(figsize=(10, 5))

        # Plot for training data
        plt.subplot(1, 2, 1)
        plt.scatter(train_Y, train_pred, alpha=0.5)
        plt.plot([train_Y.min(), train_Y.max()], [train_Y.min(), train_Y.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Training Data')

        # Plot for testing data
        plt.subplot(1, 2, 2)
        plt.scatter(test_Y, test_pred, alpha=0.5)
        plt.plot([test_Y.min(), test_Y.max()], [test_Y.min(), test_Y.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Testing Data')

        plt.tight_layout()
        plt.savefig(f'{self.output_path}/prediction_vs_actual.png')
        plt.show()

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
    #dnn.sensitivity()

