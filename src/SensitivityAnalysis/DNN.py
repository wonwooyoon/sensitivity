# This code read the data from ./input directory, and split the data into two parts: training data and testing data.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from itertools import product
import joblib
        

class DNN_learning:

    def __init__(self):
        self.train_x = []
        self.train_Y = []
        self.test_x = []
        self.test_Y = []
        self.output_path = './src/SensitivityAnalysis/output'
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def data_split(self, file_path, column_num):
        
        self.column_num = column_num
        self.inout = pd.read_csv(f'{file_path}/inout.csv')

        self.input = self.inout.iloc[:, 0:5] # 0, 1, 2, 3, 4
        self.output = self.inout.iloc[:, self.column_num] # -4, -3, -2, -1
        
        self.input_scaler = MinMaxScaler()
        self.output_scaler = MinMaxScaler()
        self.input = pd.DataFrame(self.input_scaler.fit_transform(self.input), columns=self.input.columns)
        self.output = pd.DataFrame(self.output_scaler.fit_transform(self.output.values.reshape(-1, 1)), columns=['Y'])

        # Save the scalers
        joblib.dump(self.input_scaler, f'{self.output_path}/input_scaler_{self.column_num+5}.joblib')
        joblib.dump(self.output_scaler, f'{self.output_path}/output_scaler_{self.column_num+5}.joblib')
        
        self.train_x, self.test_x, self.train_Y, self.test_Y = train_test_split(self.input, self.output, test_size=0.2, random_state=42)

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        self.train_x = torch.tensor(self.train_x.values, device=device, dtype=torch.float)
        self.train_Y = torch.tensor(self.train_Y.values, device=device, dtype=torch.float)
        self.test_x = torch.tensor(self.test_x.values, device=device, dtype=torch.float)
        self.test_Y = torch.tensor(self.test_Y.values, device=device, dtype=torch.float)

        ##################################
        node_num = [512]
        dropout_rate = [0.1]
        layer_num = [4, 5, 6]
        initial_lr = 0.001 
        l2_reg = [1e-3]
        batch_size = [32]
        self.epoch = 2000
        ##################################

        hyperparameters = {
            'node_num': node_num,
            'dropout_rate': dropout_rate,
            'layer_num': layer_num,
            'initial_lr': [initial_lr],
            'l2_reg': l2_reg,
            'batch_size': batch_size
        }

        hyperparameter_combinations = list(product(*hyperparameters.values()))

        best_loss = 1000000
        best_hyperparameters = []

        for i, (node_num, dropout_rate, layer_num, initial_lr, l2_reg, batch_size) in enumerate(hyperparameter_combinations):

            print(f'Hyperparameter combination {i+1}/{len(hyperparameter_combinations)}')
            print(f'node_num: {node_num}, dropout_rate: {dropout_rate}, layer_num: {layer_num}, initial_lr: {initial_lr}, l2_reg: {l2_reg}, batch_size: {batch_size}')

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

            #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epoch/10, eta_min=initial_lr/1000)
            #scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=200)
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
                scheduler.step(test_loss)

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
            self.model.load_state_dict(torch.load(f'{self.output_path}/model.pth'))

            if lowest_loss < best_loss:
                
                best_loss = lowest_loss
                best_hyperparameters = [node_num, dropout_rate, layer_num, initial_lr, l2_reg, batch_size]
                
                plt.plot(self.loss_evo, label='train')
                plt.plot(self.test_loss_evo, label='test')
                plt.yscale('log')
                plt.legend()
                plt.savefig(f'{self.output_path}/loss_evo_{self.column_num+5}.png')
                plt.clf()
                
                with open(f'{self.output_path}/best_hyperparameters_{self.column_num+5}.txt', 'w') as f:
                    f.write('\n'.join([f'{k}: {v}' for k, v in zip(['node_num', 'dropout_rate', 'layer_num', 'initial_lr', 'l2_reg', 'batch_size'], best_hyperparameters)]))
                f.close()

                torch.save(self.model.state_dict(), f'{self.output_path}/best_model_{self.column_num+5}.pth')

        print(f'best loss: {best_loss}')
        print(f'best NRMSE: {np.sqrt(best_loss)}')
        print(f'best hyperparameters: {best_hyperparameters}')

        # after the learning, unload the data and model from GPU to CPU
        self.train_x = self.train_x.cpu()
        self.train_Y = self.train_Y.cpu()
        self.test_x = self.test_x.cpu()
        self.test_Y = self.test_Y.cpu()
        self.model = self.model.cpu()
        
    def test(self):

        self.model.eval()
        with torch.no_grad():
            train_pred = self.output_scaler.inverse_transform(self.model(self.train_x).cpu().numpy())
            test_pred = self.output_scaler.inverse_transform(self.model(self.test_x).cpu().numpy())
            train_Y = self.output_scaler.inverse_transform(self.train_Y.cpu().numpy())
            test_Y = self.output_scaler.inverse_transform(self.test_Y.cpu().numpy())

        plt.figure(figsize=(5, 5))

        plt.scatter(train_Y, train_pred, alpha=0.2, color='orange')
        plt.scatter(test_Y, test_pred, alpha=0.2, color='blue')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')

        plt.tight_layout()
        plt.savefig(f'{self.output_path}/prediction_vs_actual_{self.column_num+5}.png')


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
        plt.savefig(f'{self.output_path}/sensitivity_{self.column_num+5}.png')
        np.savetxt(f'{self.output_path}/sensitivity_{self.column_num+5}.txt', pif)
        

if __name__ == '__main__':
    
    for i in range(4):
        dnn = DNN_learning()
        dnn.data_split('./src/TargetValueAnalysis/output', column_num=i-4)
        dnn.train()
        dnn.test()
        dnn.sensitivity()
