import pandas as pd
import os

class TargetValueAnalysis:

    def __init__(self):
        pass
        

    def read_path(self, file_path):
        self.data = pd.read_csv(file_path)
    

    def calculate_aqueous(self):
        material_1 = self.data[self.data['Material ID'] == 1]
        self.aqueous_granite = (material_1['Total UO2++ [m]'] * material_1['Volume [m^3]'] * material_1['Porosity']).sum()
        
        material_2 = self.data[self.data['Material ID'] == 2]
        self.aqueous_bentonite = (material_2['Total UO2++ [m]'] * material_2['Volume [m^3]'] * material_2['Porosity']).sum()
    
    def calculate_adsorbed(self):
        material_2 = self.data[self.data['Material ID'] == 2]
        self.adsorbed = (material_2['Total Sorbed UO2++ [mol/m^3]'] * material_2['Volume [m^3]']).sum()
    

    def calculate_efflux(self, efflux_path):
        with open(efflux_path, 'r') as f:
            lines = f.readlines()
            data = lines[0].split(',')
            self.efflux_total_uo2_index = []
            self.efflux_qlx_index = []
            for i, item in enumerate(data):
                if 'Total UO2++' in item:
                    self.efflux_total_uo2_index.append(i)
                elif 'qlx' in item:
                    self.efflux_qlx_index.append(i)
            self.efflux = []
            for line in lines[1:]:
                line_data = line.split()
                efflux_total_uo2 = [float(line_data[i]) for i in self.efflux_total_uo2_index]
                efflux_qlx = [float(line_data[i]) for i in self.efflux_qlx_index]
                efflux = [efflux_total_uo2[i] * efflux_qlx[i] * 0.74074 / 1000 for i in range(len(efflux_total_uo2))]
                efflux_sum = sum(efflux)            
                self.efflux.append(efflux_sum) 
            
        self.efflux = pd.DataFrame({'Efflux': self.efflux})
            
    def save_target_values(self):
        target_values = pd.DataFrame({'Aqueous UO2++ in Granite': [self.aqueous_granite], 'Aqueous UO2++ in Bentonite': [self.aqueous_bentonite], 'Adsorbed UO2++ in Bentonite': [self.adsorbed]})
        if not hasattr(self, 'target_values'):
            self.target_values = target_values
        else:
            self.target_values = pd.concat([self.target_values, target_values], ignore_index=True)

    def save_csv(self, target_path, efflux_path):
        self.efflux.to_csv(efflux_path, index=False)
        self.target_values.to_csv(target_path, index=False)

    
if __name__ == '__main__':


    for i in range (200):
    
        efflux_path = f'/home/wwy/pflotran_sensitivity_analysis/SensitivityAnalysis/src/RunPFLOTRAN/output/sample_{i}/sample_{i}-obs-3.pft'

        os.makedirs(f'/home/wwy/pflotran_sensitivity_analysis/SensitivityAnalysis/src/TargetValueAnalysis/output/sample_{i}', exist_ok=True)
        
        target_csv_path = f'/home/wwy/pflotran_sensitivity_analysis/SensitivityAnalysis/src/TargetValueAnalysis/output/sample_{i}/target_values.csv'
        efflux_csv_path = f'/home/wwy/pflotran_sensitivity_analysis/SensitivityAnalysis/src/TargetValueAnalysis/output/sample_{i}/efflux.csv'
        
        tva = TargetValueAnalysis()

        for j in range(100):
            file_path = f'/home/wwy/pflotran_sensitivity_analysis/SensitivityAnalysis/src/TargetValueAnalysis/input/sample_{i}/mesh_centered_data_{j}.csv'
            tva.read_path(file_path)
            tva.calculate_aqueous()
            tva.calculate_adsorbed()
            tva.save_target_values()
    
        tva.calculate_efflux(efflux_path)
        tva.save_csv(target_csv_path, efflux_csv_path)
