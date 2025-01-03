import numpy as np
import csv
import subprocess
import h5py
import glob
import pandas as pd


class RatioEquilibrium:

    def __init__(self, ratio_dir, default_script_dir, ratio_result_dir):
        self.ratio_dir = ratio_dir
        self.default_script_dir = default_script_dir
        self.ratio_result_dir = ratio_result_dir
        self.ratios = None
        self.dicData = {}
        self.concentration = None

    def write_script(self):

        with open(f'{self.default_script_dir}', 'r') as file:
            lines = file.readlines()

        ratio_target_1 = None
        ratio_target_2 = None

        for i, line in enumerate(lines):

            if 'MATERIAL_PROPERTY seawater' in line:
                ratio_target_1 = i+2

            if 'MATERIAL_PROPERTY granite_fracture' in line:
                ratio_target_2 = i+2

        for i in range(np.shape(self.ratios)[0]):

            if ratio_target_1 is not None:
                lines[ratio_target_1] = f'POROSITY {self.ratios[i, 1]}\n'

            if ratio_target_2 is not None:
                lines[ratio_target_2] = f'POROSITY {self.ratios[i, 0]}\n' 

            with open(f'{self.ratio_result_dir}/ratio_calculation_{i}.in', 'w') as file:
                file.writelines(lines)

    def read_ratio(self):

        with open(f'{self.ratio_dir}', 'r') as csvfile:
            data = csv.reader(csvfile)
            data = np.array([row for row in data], dtype=float)

        self.ratios = np.zeros([data.shape[0], 2], float)

        self.ratios[:, 1] = data[:, -1]
        self.ratios[:, 0] = 1 - data[:, -1]

    def run_pflotran_ratio(self):

        bash_code = """
#!/bin/bash
mkdir -p ./src/RatioCalculation/output
base_dir="$(pwd)"

for i in {406..409}; do
  infile="${base_dir}/src/RatioCalculation/output/ratio_calculation_${i}.in"
  echo "Running pflotran on $infile..."
  mpirun -n 1 /home/geofluids/pflotran/src/pflotran/pflotran -input_prefix "${infile%.*}"
done

echo "All simulations completed and results moved to ./src/RatioCalculation/output/"
"""

        subprocess.run(['bash', '-c', bash_code], check=True)

    def read_pflotran_result(self, components):

        ratio_results_dir = f'{self.ratio_result_dir}/ratio_calculation_*.h5'
        file_num = len(glob.glob(ratio_results_dir))
        self.concentration = np.zeros([file_num, len(components)])
        
        for i in range(file_num):
        

            with h5py.File(f'{self.ratio_result_dir}/ratio_calculation_{i}.h5', 'r') as file:
            
                group = file['Time:  5.00000E+02 y/']
                self.keys = list(group.keys())
                for key in self.keys:
                    self.dicData[key] = file['Time:  5.00000E+02 y/'+key][:].reshape(-1)
            
                j = 0
                for component in components:
                    for key in self.dicData.keys():
                        if component in key:
                            self.concentration[i][j] = self.dicData[key][-1]
                            j += 1
                            break
            
        with open('./src/RatioCalculation/output/mixed_components.csv', 'w', newline='') as file:
            
            df = pd.DataFrame(self.concentration, columns=components)
            df.to_csv(f'{self.ratio_result_dir}/mixed_components.csv', index=False, header=False)

            
if __name__ == '__main__':

    ratio_dir = './src/RandomSampling/output/lhs_sampled_data.csv'
    default_script_dir = './src/RatioCalculation/input/PFLOTRAN_mixing.in'
    ratio_results_dir = './src/RatioCalculation/output'
    components = ['pH', 'pe', 'Al+++', 'CO3--', 'Ca++', 'Cl-', 'Fe++', 'H4(SiO4)', 'K+', 'Mg++', 'Na+', 'SO4--', 'UO2++']
    
    ratio_calculation = RatioEquilibrium(ratio_dir, default_script_dir, ratio_results_dir)

    ratio_calculation.read_ratio()
    ratio_calculation.write_script()
    ratio_calculation.run_pflotran_ratio()
    ratio_calculation.read_pflotran_result(components)

