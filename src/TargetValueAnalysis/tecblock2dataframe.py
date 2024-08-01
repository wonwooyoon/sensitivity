import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

class TECBLOCK2EXCEL:

    def __init__(self, inputfilepath):
        self.inputfilepath = inputfilepath


        with open(self.inputfilepath, 'r') as file:
            self.inputfile = file.readlines()


    def ReadVariables(self):
        variables_line = self.inputfile[1]
        if 'VARIABLES=' in variables_line:
            variables = variables_line.split('=')[1].strip().strip('"')
            self.variables = [var.strip().strip('"') for var in variables.split('","')]

    def ReadNodeNum(self):
        node_line = self.inputfile[2]
        self.vertice_num = int(node_line.split(' N=')[1].split(', ')[0])
        self.mesh_num = int(node_line.split(' E=')[1].split(', ')[0])


    def ReadData(self):
        i = 0
        line_index = 4

        line_index, self.x_coord = self.ReadNumbers(line_index, self.vertice_num)
        line_index, self.y_coord = self.ReadNumbers(line_index+1, self.vertice_num)
        line_index, self.z_coord = self.ReadNumbers(line_index+1, self.vertice_num)

        self.variables_data = np.zeros([self.mesh_num, len(self.variables)-3])

        for i in range(len(self.variables)-3):
            line_index, data = self.ReadNumbers(line_index, self.mesh_num)
            self.variables_data[:, i] = data[:, 0]

        self.connection_data = np.zeros([self.mesh_num, 8])

        for i in range(self.mesh_num):
            line_index, data = self.ReadNumbers(line_index, 8)
            self.connection_data[i, :] = data.reshape(1, -1)

        self.connection_data = self.connection_data.astype(int)

    def ReadNumbers(self, init_line, num):
        vector = []
        line_index = init_line

        while len(vector) < num:
            vector.extend([var.strip() for var in self.inputfile[line_index].split()])
            line_index += 1

        processed_vector = []
        for value in vector:
            if '-' in value:
                exponent = float(value.split('-')[-1])
                if exponent >= 30:
                    processed_vector.append(0.0)
                else:
                    processed_vector.append(float(value))
            else:
                processed_vector.append(float(value))

        processed_vector = np.array(processed_vector).reshape(-1, 1)
        return line_index, processed_vector

    def CalculateMesh(self):

        self.mesh_coord = np.zeros([self.mesh_num, 3])

        for i in range(self.mesh_num):

            x_avg = 0
            y_avg = 0
            z_avg = 0

            for j in range(8):
                x_avg += self.x_coord[self.connection_data[i, j]-1][0]
                y_avg += self.y_coord[self.connection_data[i, j]-1][0]
                z_avg += self.z_coord[self.connection_data[i, j]-1][0]

            self.mesh_coord[i, 0] = x_avg/8
            self.mesh_coord[i, 1] = y_avg/8
            self.mesh_coord[i, 2] = z_avg/8

    def SaveAsDataFrame(self, outputfilepath):

        combined_data = np.hstack((self.mesh_coord, self.variables_data))
        headers = self.variables
        self.df = pd.DataFrame(combined_data, columns=headers)

        print(self.df)

        self.df.to_csv(outputfilepath, index=False)

    def plot_spatial_data_2d(self, data_column_index, fig_outputfilepath):
        x = self.df.iloc[:, 0]
        y = self.df.iloc[:, 1]
        data = self.df.iloc[:, data_column_index]
        
        grid_x, grid_y = np.mgrid[x.min():x.max():1000j, y.min():y.max():1000j]
        
        grid_data = griddata((x, y), data, (grid_x, grid_y), method='linear')
        
        plt.figure(figsize=(10, 8))
        plt.imshow(grid_data.T, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis')
        plt.colorbar(label='Data Value')
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        plt.title('Spatial Data Distribution')
        
        plt.savefig(fig_outputfilepath)
        print(f'Plot saved to {fig_outputfilepath}')
        plt.close()
        


if __name__ == '__main__':

    for i in range(200):

        for j in range(100):

            inputfilepath = f'./src/RunPFLOTRAN/output/sample_{i}/sample_{i}-{j:03d}.tec'
            output_dir = f'./src/TargetValueAnalysis/input/sample_{i}'
            os.makedirs(output_dir, exist_ok=True)
            outputfilepath = f'{output_dir}/mesh_centered_data_{j}.csv'
            
            converter = TECBLOCK2EXCEL(inputfilepath)
            converter.ReadVariables()
            converter.ReadNodeNum()
            converter.ReadData()
            converter.CalculateMesh()
            converter.SaveAsDataFrame(outputfilepath)
            

