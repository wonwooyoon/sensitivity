import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
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
        self.alpha = 0.3
        self.validation_loss_pre = None
        self.output_path = './src/SensitivityAnalysis/output'
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def ema_loss(self, validation_loss):

        if self.validation_loss_pre is None:
            self.validation_loss_pre = validation_loss
        
        validation_loss = validation_loss * self.alpha + (1-self.alpha) * self.validation_loss_pre
        self.validation_loss_pre = validation_loss
        return validation_loss

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
        
        self.train_x, self.test_x, self.train_Y, self.test_Y = train_test_split(self.input, self.output, test_size=0.3, random_state=91)
        self.test_x, self.validation_x, self.test_Y, self.validation_Y = train_test_split(self.test_x, self.test_Y, test_size=0.5, random_state=91)

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        self.train_x = torch.tensor(self.train_x.values, device=device, dtype=torch.float)
        self.train_Y = torch.tensor(self.train_Y.values, device=device, dtype=torch.float)
        self.validation_x = torch.tensor(self.validation_x.values, device=device, dtype=torch.float)
        self.validation_Y = torch.tensor(self.validation_Y.values, device=device, dtype=torch.float)
        self.test_x = torch.tensor(self.test_x.values, device=device, dtype=torch.float)
        self.test_Y = torch.tensor(self.test_Y.values, device=device, dtype=torch.float)

        ##################################
        node_num = [1024, 2048]
        dropout_rate = [0.1]
        layer_num = [4, 5, 6]
        initial_lr = [0.0001] 
        l2_reg = [0, 1e-5]
        batch_size = [16, 32]
        self.epoch = 3000
        ##################################

        hyperparameters = {
            'node_num': node_num,
            'dropout_rate': dropout_rate,
            'layer_num': layer_num,
            'initial_lr': initial_lr,
            'l2_reg': l2_reg,
            'batch_size': batch_size
        }

        hyperparameter_combinations = list(product(*hyperparameters.values()))

        best_loss = 1000000
        self.best_hyperparameters = []
        self.loss_fn = torch.nn.MSELoss(reduction='sum')

        # for i, (node_num, dropout_rate, layer_num, initial_lr, l2_reg, batch_size) in enumerate(hyperparameter_combinations):

        #     print(f'Hyperparameter combination {i+1}/{len(hyperparameter_combinations)}')
        #     print(f'node_num: {node_num}, dropout_rate: {dropout_rate}, layer_num: {layer_num}, initial_lr: {initial_lr}, l2_reg: {l2_reg}, batch_size: {batch_size}')

        #     layers = []
        #     input_size = self.train_x.size(1)
        #     layers.append(torch.nn.Linear(input_size, node_num))
        #     layers.append(torch.nn.GELU())
        #     layers.append(torch.nn.Dropout(dropout_rate))

        #     for _ in range(layer_num-1):
        #         layers.append(torch.nn.Linear(node_num, node_num))
        #         layers.append(torch.nn.GELU())
        #         layers.append(torch.nn.Dropout(dropout_rate))
            
        #     layers.append(torch.nn.Linear(node_num, 1))
        #     #layers.append(torch.nn.ReLU())
        #     layers.append(torch.nn.Softplus())
            
        #     self.model = torch.nn.Sequential(*layers).to(device)

        #     self.loss_fn = torch.nn.MSELoss(reduction='sum')
        #     self.optimizer = torch.optim.Adam(self.model.parameters(), lr=initial_lr, weight_decay=l2_reg)

        #     for param in self.model.parameters():
        #         if len(param.size()) == 2:
        #             torch.nn.init.xavier_uniform_(param)
        #         else:
        #             torch.nn.init.zeros_(param)

        #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.95, patience=50)
        #     lowest_loss = 1000000
        
        #     train_dataset = torch.utils.data.TensorDataset(self.train_x, self.train_Y)
        #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        #     for t in tqdm(range(self.epoch)):
                
        #         self.model.train()
        #         for batch_x, batch_Y in train_loader:
        #             y_pred = self.model(batch_x)
        #             loss = self.loss_fn(y_pred, batch_Y)
        #             loss.backward()
        #             self.optimizer.step()
        #             self.optimizer.zero_grad()

        #         self.model.eval()
        #         train_loss = self.loss_fn(self.model(self.train_x), self.train_Y)
        #         validation_loss = self.loss_fn(self.model(self.validation_x), self.validation_Y)
        #         scheduler.step(validation_loss)

        #         if t % 25 == 0:
        #             tqdm.write(f'epoch {t}: train_loss {train_loss.item() / len(self.train_x)}, val_loss {validation_loss.item() / len(self.validation_x)}, lr = {self.optimizer.param_groups[0]["lr"]}')

        #         with torch.no_grad():
        #             if t == 0:
        #                 self.loss_evo = [train_loss.item() / len(self.train_x)]
        #                 self.val_loss_evo = [validation_loss.item() / len(self.validation_x)]
        #             else:
        #                 self.loss_evo.append(train_loss.item() / len(self.train_x))
        #                 self.val_loss_evo.append(validation_loss.item() / len(self.validation_x))

        #         if validation_loss.item() / len(self.validation_x) < lowest_loss:
        #             lowest_loss = validation_loss.item() / len(self.validation_x)
        #             torch.save(self.model.state_dict(), f'{self.output_path}/model.pth')
            
        #     print(f'lowest validation loss: {lowest_loss}')
        #     self.model.load_state_dict(torch.load(f'{self.output_path}/model.pth', weights_only=True))

        #     if lowest_loss < best_loss:
                
        #         best_loss = lowest_loss
        #         self.best_hyperparameters = [node_num, dropout_rate, layer_num, initial_lr, l2_reg, batch_size]
                
        #         plt.plot(self.loss_evo, label='train')
        #         plt.plot(self.val_loss_evo, label='validation')
        #         plt.legend()
        #         plt.savefig(f'{self.output_path}/loss_evo_{self.column_num+5}.png')
        #         plt.clf()
                
        #         with open(f'{self.output_path}/best_hyperparameters_{self.column_num+5}.txt', 'w') as f:
        #             f.write('\n'.join([f'{k}: {v}' for k, v in zip(['node_num', 'dropout_rate', 'layer_num', 'initial_lr', 'l2_reg', 'batch_size'], self.best_hyperparameters)]))
        #         f.close()

        #         torch.save(self.model.state_dict(), f'{self.output_path}/best_model_{self.column_num+5}.pth')

        # after the learning, unload the data and model from GPU to CPU
        self.train_x = self.train_x.cpu()
        self.train_Y = self.train_Y.cpu()
        self.validation_x = self.validation_x.cpu()
        self.validation_Y = self.validation_Y.cpu()
        self.test_x = self.test_x.cpu()
        self.test_Y = self.test_Y.cpu()
        
    def test(self):

        with open(f'{self.output_path}/best_hyperparameters_{self.column_num+5}.txt', 'r') as f:
            best_hyperparameters = f.readlines()
        best_hyperparameters = [float(param.split(': ')[1].strip()) for param in best_hyperparameters]

        node_num, dropout_rate, layer_num, initial_lr, l2_reg, batch_size = best_hyperparameters

        layers = []
        input_size = self.train_x.size(1)
        layers.append(torch.nn.Linear(input_size, int(node_num)))
        layers.append(torch.nn.GELU())
        layers.append(torch.nn.Dropout(dropout_rate))

        for _ in range(int(layer_num)-1):
            layers.append(torch.nn.Linear(int(node_num), int(node_num)))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(dropout_rate))

        layers.append(torch.nn.Linear(int(node_num), 1))
        #layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Softplus())

        self.model = torch.nn.Sequential(*layers)
        
        self.model.load_state_dict(torch.load(f'{self.output_path}/best_model_{self.column_num+5}.pth', weights_only=True))
        self.model = self.model.cpu()

        self.model.eval()

        train_loss = self.loss_fn(self.model(self.train_x), self.train_Y).item() / len(self.train_x)
        validation_loss = self.loss_fn(self.model(self.validation_x), self.validation_Y).item() / len(self.validation_x)
        test_loss = self.loss_fn(self.model(self.test_x), self.test_Y).item() / len(self.test_x)

        print(f'best train loss: {train_loss} train NRMSE: {np.sqrt(train_loss)}')
        print(f'best val loss: {validation_loss} val NRMSE: {np.sqrt(validation_loss)}')
        print(f'best test loss: {test_loss} test NRMSE: {np.sqrt(test_loss)}')
        print(f'best hyperparameters: {self.best_hyperparameters}')

        with torch.no_grad():
            train_pred = self.output_scaler.inverse_transform(self.model(self.train_x).cpu().numpy())
            test_pred = self.output_scaler.inverse_transform(self.model(self.test_x).cpu().numpy())
            val_pred = self.output_scaler.inverse_transform(self.model(self.validation_x).cpu().numpy())
            train_Y = self.output_scaler.inverse_transform(self.train_Y.cpu().numpy())
            test_Y = self.output_scaler.inverse_transform(self.test_Y.cpu().numpy())
            val_Y = self.output_scaler.inverse_transform(self.validation_Y.cpu().numpy())
        
        # save the ground truth with corresponding prediction as csv
        pd.DataFrame(np.concatenate([train_Y, train_pred], axis=1)).to_csv(f'{self.output_path}/train_{self.column_num+5}.csv', index=False)
        pd.DataFrame(np.concatenate([test_Y, test_pred], axis=1)).to_csv(f'{self.output_path}/test_{self.column_num+5}.csv', index=False)
        pd.DataFrame(np.concatenate([val_Y, val_pred], axis=1)).to_csv(f'{self.output_path}/validation_{self.column_num+5}.csv', index=False)

        all_Y = np.concatenate([train_Y, test_Y, val_Y], axis=0)

        residuals = test_Y.flatten() - test_pred.flatten()
        sigma = np.var(residuals)
        z = 1.96
        lower = all_Y - z*np.sqrt(sigma)
        upper = all_Y + z*np.sqrt(sigma)

        plt.figure(figsize=(6, 6))
        plt.scatter(train_Y, train_pred, alpha=0.6, color='blue', s=40, label = 'train', marker='o')
        plt.scatter(test_Y, test_pred, alpha=0.6, color='orange', s=40, label = 'test', marker='^')
        plt.scatter(val_Y, val_pred, alpha=0.6, color='green', s=40, label = 'validation', marker='s')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')

        # calculate the r^2 value for train, test, and validation
        r2_train = r2_score(train_Y.flatten(), train_pred.flatten())
        r2_test = r2_score(test_Y.flatten(), test_pred.flatten())
        r2_val = r2_score(val_Y.flatten(), val_pred.flatten())
        
        plt.text(0.1, 0.95, f'Test R^2: {r2_test:.3f}', ha='left', va='center', transform=plt.gca().transAxes)
        plt.text(0.1, 0.9, f'Test NRMSE: {np.sqrt(test_loss):.3f}', ha='left', va='center', transform=plt.gca().transAxes)
        plt.text(0.1, 0.85, f'Validation R^2: {r2_val:.3f}', ha='left', va='center', transform=plt.gca().transAxes)
        plt.text(0.1, 0.8, f'Validation NRMSE: {np.sqrt(validation_loss):.3f}', ha='left', va='center', transform=plt.gca().transAxes)
        plt.text(0.1, 0.75, f'Train R^2: {r2_train:.3f}', ha='left', va='center', transform=plt.gca().transAxes)
        plt.text(0.1, 0.7, f'Train NRMSE: {np.sqrt(train_loss):.3f}', ha='left', va='center', transform=plt.gca().transAxes)

        sorted_order = np.argsort(all_Y.T)
        all_Y = all_Y[sorted_order].flatten()
        lower = lower[sorted_order].flatten()  
        upper = upper[sorted_order].flatten()

        plt.plot([[0], [np.max(all_Y)]], [[0], [np.max(all_Y)]], color='black', linestyle='-', linewidth=1.5)
        plt.plot(all_Y, lower, color='red', linestyle='--', linewidth=1.0, alpha = 0.7)
        plt.plot(all_Y, upper, color='red', linestyle='--', linewidth=1.0, alpha = 0.7)

        plt.xlim(0, np.max(all_Y))
        plt.ylim(0, np.max(all_Y))
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{self.output_path}/prediction_vs_actual_{self.column_num+5}.png')
        plt.clf()

    def sensitivity(self):
        
        all_x = torch.cat([self.train_x, self.validation_x , self.test_x], dim=0)
        all_Y = torch.cat([self.train_Y, self.validation_Y, self.test_Y], dim=0)
        
        base_loss = self.loss_fn(self.model(all_x), all_Y).item() / len(all_Y)

        pif = []
        for i in range(all_x.size(1)):
            perm_x = all_x.clone()
            importance = []

            for _ in range(1000):
                permute = torch.randperm(perm_x.size(0))
                perm_x[:, i] = perm_x[permute, i]
                permute_loss = self.loss_fn(self.model(perm_x), all_Y).item() / len(all_Y)
                importance = np.append(importance, (permute_loss - base_loss)/base_loss)

            pif.append(np.mean(importance))

        pif = np.array(pif)
        print(pif)
        plt.boxplot(pif.T, vert=False)
        plt.savefig(f'{self.output_path}/sensitivity_{self.column_num+5}.png')
        plt.clf()
        np.savetxt(f'{self.output_path}/sensitivity_{self.column_num+5}.txt', pif)
        

if __name__ == '__main__':
    
    for i in [0, 1, 2, 3]:
        dnn = DNN_learning()
        dnn.data_split('./src/TargetValueAnalysis/output', column_num=i-4)
        dnn.train()
        dnn.test()
        dnn.sensitivity()
