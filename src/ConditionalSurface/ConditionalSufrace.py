import torch
import numpy as np
from joblib import load
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def load_model(model_path, hyperparameters_path):
    # Load the model from the specified path
    
    with open(hyperparameters_path, 'r') as file:
        lines = file.readlines()
        node_num = int(lines[0].strip().split()[1])
        dropout_rate = float(lines[1].strip().split()[1])
        layer_num = int(lines[2].strip().split()[1])
    
    layers = []
    layers.append(torch.nn.Linear(5, node_num))
    layers.append(torch.nn.GELU())
    layers.append(torch.nn.Dropout(dropout_rate))

    for _ in range(layer_num-1):
        layers.append(torch.nn.Linear(node_num, node_num))
        layers.append(torch.nn.GELU())
        layers.append(torch.nn.Dropout(dropout_rate))
                
    layers.append(torch.nn.Linear(node_num, 1))
    layers.append(torch.nn.Softplus())
    
    model = torch.nn.Sequential(*layers)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    return model

def find_range_single(model, x1, x3, x5, num):

    x2_range = np.linspace(1300, 1900, 101)
    x4_range = np.linspace(0.00, 1.00, 101)
    value_mesh = pd.DataFrame(index=x4_range, columns=x2_range)
    value_row = pd.DataFrame(index=range(len(x2_range)*len(x4_range)), columns=['x2', 'x4', 'y'])
    
    input_scaler = load(f'./src/SensitivityAnalysis/output/input_scaler_{num}.joblib')
    output_scaler = load(f'./src/SensitivityAnalysis/output/output_scaler_{num}.joblib')
    
    with torch.no_grad():

        for i in range(len(x2_range)):
            for j in range(len(x4_range)):
                input_tensor = torch.tensor(input_scaler.transform(torch.tensor([[x1, x2_range[i], x3, x4_range[j], x5]], dtype=torch.float32)), dtype=torch.float32)
                output = output_scaler.inverse_transform(model(input_tensor).numpy()).flatten()[0]
                value_mesh.iloc[j, i] = output
                value_row.iloc[j*len(x2_range)+i] = [x2_range[i], x4_range[j], output]
                # print(f'x2: {x2_range[i]}, x4: {x4_range[j]}, y: {output}')

    return value_mesh, value_row

    
if __name__ == "__main__":

    model_path_1 = './src/SensitivityAnalysis/output/best_model_1.pth'
    hyperparameters_path_1 = './src/SensitivityAnalysis/output/best_hyperparameters_1.txt'
    model_path_2 = './src/SensitivityAnalysis/output/best_model_2.pth'
    hyperparameters_path_2 = './src/SensitivityAnalysis/output/best_hyperparameters_2.txt'
    model_path_3 = './src/SensitivityAnalysis/output/best_model_3.pth'
    hyperparameters_path_3 = './src/SensitivityAnalysis/output/best_hyperparameters_3.txt'
    model_path_4 = './src/SensitivityAnalysis/output/best_model_4.pth'
    hyperparameters_path_4 = './src/SensitivityAnalysis/output/best_hyperparameters_4.txt'

    model_1 = load_model(model_path_1, hyperparameters_path_1)
    model_2 = load_model(model_path_2, hyperparameters_path_2)
    model_3 = load_model(model_path_3, hyperparameters_path_3)
    model_4 = load_model(model_path_4, hyperparameters_path_4)

    print("Model loaded and ready to use.")

    x1 = 7.0e-15
    x3 = 505325
    x5 = 0.7

    value_y1_mesh, value_y1_row = find_range_single(model_1, x1, x3, x5, 1)
    #value_y2_mesh, value_y2_row = find_range_single(model_2, x1, x3, x5, 2)
    #value_y3_mesh, value_y3_row = find_range_single(model_3, x1, x3, x5, 3)
    value_y4_mesh, value_y4_row = find_range_single(model_4, x1, x3, x5, 4)

    value_mesh = value_y1_mesh + value_y4_mesh
    value_row = value_y1_row + value_y4_row

    value_y1_row.to_csv('./src/ConditionalSurface/output/value_y1.csv')
    #value_y2_row.to_csv('./src/ConditionalSurface/output/value_y2.csv')
    #value_y3_row.to_csv('./src/ConditionalSurface/output/value_y3.csv')
    value_y4_row.to_csv('./src/ConditionalSurface/output/value_y4.csv')

    x2_range = value_mesh.columns.to_numpy().astype(float)
    x4_range = value_mesh.index.to_numpy().astype(float)

    x2, x4 = np.meshgrid(x2_range, x4_range)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x2, x4, value_mesh, cmap='viridis')

    ax.set_xlabel('x2')
    ax.set_ylabel('x4')
    ax.set_zlabel('Output Value')
    ax.set_title('3D Surface Plot of Output Value for x2 and x4')

    ax.invert_yaxis()  # Invert the y-axis to reverse the direction of x4

    plt.savefig('./src/ConditionalSurface/output/value_3d.png')
    plt.clf()



    
