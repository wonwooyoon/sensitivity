import torch
import numpy as np
from joblib import load
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def load_model(model_path):
    # Load the model from the specified path
    
    node_num = 256
    dropout_rate = 0.1
    layer_num = 4
    
    layers = []
    layers.append(torch.nn.Linear(5, node_num))
    layers.append(torch.nn.GELU())
    layers.append(torch.nn.Dropout(dropout_rate))

    for _ in range(layer_num-1):
        layers.append(torch.nn.Linear(node_num, node_num))
        layers.append(torch.nn.GELU())
        layers.append(torch.nn.Dropout(dropout_rate))
                
    layers.append(torch.nn.Linear(node_num, 1))
    
    model = torch.nn.Sequential(*layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model

def find_range_single(model, x1, x3, x5):

    x2_range = np.linspace(1350, 1850, 10)
    x4_range = np.linspace(0.01, 0.99, 10)
    value = np.zeros((len(x2_range), len(x4_range)))

    input_scaler = load('./src/SensitivityAnalysis/output/input_scaler_1.joblib')
    output_scaler = load('./src/SensitivityAnalysis/output/output_scaler_1.joblib')
    
    with torch.no_grad():

        for i in range(len(x2_range)):
            for j in range(len(x4_range)):
                input_tensor = torch.tensor(input_scaler.transform(torch.tensor([[x1, x2_range[i], x3, x4_range[j], x5]], dtype=torch.float32)), dtype=torch.float32)
                output = output_scaler.inverse_transform(model(input_tensor).numpy())
                value[j, i] = output
                print(f'x2: {x2_range[i]}, x4: {x4_range[j]}, y: {output}')

    return value

def find_range_double(model_1, model_2, x1, x3, x5):

    x2_range = np.linspace(1350, 1850, 10)
    x4_range = np.linspace(0.01, 0.99, 10)
    value = np.zeros((len(x2_range), len(x4_range)))

    input_scaler = load('./src/SensitivityAnalysis/output/input_scaler_1.joblib')
    output_scaler_1 = load('./src/SensitivityAnalysis/output/output_scaler_1.joblib')
    output_scaler_2 = load('./src/SensitivityAnalysis/output/output_scaler_4.joblib')

    with torch.no_grad():

        for i in range(len(x2_range)):
            for j in range(len(x4_range)):
                input_tensor = torch.tensor(input_scaler.transform(torch.tensor([[x1, x2_range[i], x3, x4_range[j], x5]], dtype=torch.float32)), dtype=torch.float32)
                output_1 = output_scaler_1.inverse_transform(model_1(input_tensor).numpy())
                output_2 = output_scaler_2.inverse_transform(model_2(input_tensor).numpy())
                    
                value[j, i] = output_1 + output_2
                print(f'x2: {x2_range[i]}, x4: {x4_range[j]}, y: {value[j, i]}')

    return value

    
if __name__ == "__main__":
    model_path_1 = './src/SensitivityAnalysis/output/best_model_1.pth'
    model_path_2 = './src/SensitivityAnalysis/output/best_model_4.pth'

    model_1 = load_model(model_path_1)
    model_2 = load_model(model_path_2)

    print("Model loaded and ready to use.")

    x1 = 6e-15
    x3 = 507025
    x5 = 0.3
    value = find_range_single(model_1, x1, x3, x5)
    # value = find_range_double(model_1, model_2, x1, x3, x5)

    x2_range = np.linspace(1350, 1850, 10)
    x4_range = np.linspace(0.01, 0.99, 10)
    x2, x4 = np.meshgrid(x2_range, x4_range)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x2, x4, value, cmap='viridis')

    ax.set_xlabel('x2')
    ax.set_ylabel('x4')
    ax.set_zlabel('Output Value')
    ax.set_title('3D Surface Plot of Output Value for x2 and x4')

    ax.invert_yaxis()  # Invert the y-axis to reverse the direction of x4

    plt.savefig('value_3d.png')
    plt.clf()



    
