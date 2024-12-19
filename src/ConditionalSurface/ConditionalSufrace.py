import torch
import numpy as np
from joblib import load
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def load_model(model_path):
    # Load the model from the specified path
    
    node_num = 512
    dropout_rate = 0.1
    layer_num = 6
    
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

def find_range(model, x1, x3, x5, y_condition):

    x2_range = np.linspace(1350, 1850, 10)
    x4_range = np.linspace(0.01, 0.99, 10)
    valid = np.zeros((len(x2_range), len(x4_range)))
    value = np.zeros((len(x2_range), len(x4_range)))

    input_scaler = load('./src/SensitivityAnalysis/output/input_scaler_1.joblib')
    output_scaler = load('./src/SensitivityAnalysis/output/output_scaler_1.joblib')
    
    with torch.no_grad():

        for i in range(len(x2_range)):
            for j in range(len(x4_range)):
                input_tensor = torch.tensor(input_scaler.transform(torch.tensor([[x1, x2_range[i], x3, x4_range[j], x5]], dtype=torch.float32)), dtype=torch.float32)
                output = output_scaler.inverse_transform(model(input_tensor).numpy())
                if y_condition(output):
                    valid[j, i] = 1
                value[j, i] = output
                print(f'x2: {x2_range[i]}, x4: {x4_range[j]}, y: {output}')

    return valid, value

def y_condition(output):
    return output > 0.0005

    
if __name__ == "__main__":
    model_path = './src/SensitivityAnalysis/output/best_model_1.pth'
    model = load_model(model_path)
    print("Model loaded and ready to use.")

    x1 = 6e-15
    x3 = 507025
    x5 = 0.5
    valid, value = find_range(model, x1, x3, x5, y_condition)

    # Plotting the valid matrix
    plt.imshow(valid, cmap='gray_r', interpolation='nearest', extent=[0.01, 0.99, 0.01, 0.99])
    plt.colorbar(label='Validity (1=Valid, 0=Invalid)')
    plt.xlabel('x5')
    plt.ylabel('x4')
    plt.title('Validity Matrix for x4 and x5')
    
    # ax = plt.gca()
    # ax.set_aspect(600, adjustable='box')

    plt.savefig('valid.png')
    plt.clf()

    # Optional: Plot the value matrix
    plt.imshow(value, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Output Value')
    plt.xlabel('x2')
    plt.ylabel('x4')
    plt.title('Output Value Matrix for x2 and x4')
    # ax = plt.gca()
    # ax.set_aspect(600, adjustable='box')
    plt.savefig('value.png')
    plt.clf()



    
