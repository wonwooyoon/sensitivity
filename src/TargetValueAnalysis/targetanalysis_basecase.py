import pandas as pd
import os

class TargetValueAnalysis:

    def __init__(self):
        pass
        

    def read_path(self, file_path):
        try:
            self.data = pd.read_csv(file_path)
            return 1
        except FileNotFoundError:
            return 0
            

    def calculate_aqueous(self):
        material_1 = self.data[self.data['Material ID'] == 1]
        self.aqueous_granite = (material_1['Total UO2++ [m]'] * material_1['Volume [m^3]'] * material_1['Porosity']).sum()
        
        material_2 = self.data[self.data['Material ID'] == 2]
        self.aqueous_bentonite = (material_2['Total UO2++ [m]'] * material_2['Volume [m^3]'] * material_2['Porosity']).sum()
    
    def calculate_adsorbed(self):
        material_2 = self.data[self.data['Material ID'] == 2]
        self.adsorbed = (material_2['Total Sorbed UO2++ [mol_m^3]'] * material_2['Volume [m^3]']).sum()
    
    def calculate_efflux_aux(self, efflux_path):
        with open(efflux_path, 'r') as f:
            lines = f.readlines()
            data = lines[0].split(',')
            efflux_total_uo2_index = []
            efflux_qlx_index = []
            for i, item in enumerate(data):
                if 'Total UO2++' in item:
                    efflux_total_uo2_index.append(i)
                elif 'qlx' in item:
                    efflux_qlx_index.append(i)
            result = []
            for line in lines[1:]:
                line_data = line.split()
                efflux_total_uo2 = [float(line_data[i]) for i in efflux_total_uo2_index]
                efflux_qlx = [float(line_data[i]) for i in efflux_qlx_index]
                efflux = [efflux_total_uo2[i] * efflux_qlx[i] * 1 / 1000 for i in range(len(efflux_total_uo2))]
                efflux_sum = sum(efflux)            
                result.append(efflux_sum)
            return result 

    def calculate_efflux(self, efflux_path, efflux_csv_path):
        
        efflux_path_1 = efflux_path + '7.pft'

        efflux_1 = self.calculate_efflux_aux(efflux_path_1)

        for i in range(len(efflux_1)):
            if i == 0:
                self.efflux = efflux_1[i] 
                self.efflux_seq = [self.efflux]
            else:
                self.efflux = self.efflux + efflux_1[i]
                self.efflux_seq.append(self.efflux)
                
        self.efflux_seq = pd.DataFrame({'Efflux': self.efflux_seq})
        self.efflux_seq.to_csv(efflux_csv_path, index=False)
            
    def save_target_values(self):
        target_values = pd.DataFrame({'Aqueous UO2++ in Granite': [self.aqueous_granite], 'Aqueous UO2++ in Bentonite': [self.aqueous_bentonite], 'Adsorbed UO2++ in Bentonite': [self.adsorbed]})
        if not hasattr(self, 'target_values'):
            self.target_values = target_values
        else:
            self.target_values = pd.concat([self.target_values, target_values], ignore_index=True)

    def save_csv(self, target_path):
        
        self.target_values.to_csv(target_path, index=False)

    
if __name__ == '__main__':

    tva = TargetValueAnalysis()
    target_csv_path = f'/home/wwy/research/sensitivity/src/TargetValueAnalysis/output/target_values.csv'

    efflux_path = f'/home/wwy/research/sensitivity/src/RunPFLOTRAN/output/sample_201/sample_201-obs-'
    efflux_csv_path = f'/home/wwy/research/sensitivity/src/TargetValueAnalysis/output/sample_201/efflux.csv'
    tva.calculate_efflux(efflux_path, efflux_csv_path)
        
    for i in range(0, 101):
    
        file_path = f'/home/wwy/research/sensitivity/src/TargetValueAnalysis/output/sample_201/sample_201_time_{i*100:.1f}.csv'
        
        check = tva.read_path(file_path)
        if check == 0:
            continue
        tva.calculate_aqueous()
        tva.calculate_adsorbed()
        tva.save_target_values()
    
    tva.save_csv(target_csv_path)
