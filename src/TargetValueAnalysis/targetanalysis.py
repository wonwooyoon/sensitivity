import pandas as pd
import os
import glob

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

        self.efflux_seq = None

        for file in glob.glob(efflux_path + '*.pft'):
            efflux_single_file = self.calculate_efflux_aux(file)

            if self.efflux_seq is None:
                self.efflux_seq = efflux_single_file
            else: 
                self.efflux_seq = [a + b for a, b in zip(self.efflux_seq, efflux_single_file)]

        self.efflux = self.efflux_seq[-1]                
        self.efflux_seq_df = pd.DataFrame({'Efflux': self.efflux_seq})
        self.efflux_seq_df.to_csv(efflux_csv_path, index=False)
        

    def save_target_values(self, sample_num):
        
        target_values = pd.DataFrame({'sample number': [sample_num], 'Aqueous UO2++ in Granite': [self.aqueous_granite], 'Aqueous UO2++ in Bentonite': [self.aqueous_bentonite], 'Adsorbed UO2++ in Bentonite': [self.adsorbed], 'Efflux UO2++': [self.efflux]})
        
        if not hasattr(self, 'target_values'):
            self.target_values = target_values
        else:
            self.target_values = pd.concat([self.target_values, target_values], ignore_index=True)

    def save_csv(self, target_path):
        self.target_values.to_csv(target_path, index=False)


    def save_input_output(self, input_path, output_path):
        
        input_raw = pd.read_csv(input_path, header=None)
        output_raw = pd.read_csv(output_path, header=None)
        input_matched = pd.DataFrame()

        for i in range(len(output_raw)):
            sample_num = int(output_raw.iloc[i][0])
            input_matched = pd.concat([input_matched, input_raw.iloc[[sample_num - 1]]], axis=0)

        input_matched = pd.DataFrame(input_matched)
        input_matched.reset_index(drop=True, inplace=True)
        output_raw.reset_index(drop=True, inplace=True)
        input_output = pd.concat([input_matched, output_raw.iloc[:, 1:]], ignore_index=True, axis=1)
        input_output.to_csv('/home/wwy/research/sensitivity/src/TargetValueAnalysis/output/input_output.csv', index=False, header=False)




if __name__ == '__main__':

    tva = TargetValueAnalysis()
    target_csv_path = f'/home/wwy/research/sensitivity/src/TargetValueAnalysis/output/target_values.csv'

    for i in range(1, 202):

        efflux_path = f'/mnt/d/wwy/Personal/0. Paperwork/3. ML_sensitivity_analysis/Model/output_export/sample_{i}/sample_{i}-obs-'
        efflux_csv_path = f'/home/wwy/research/sensitivity/src/TargetValueAnalysis/output/sample_{i}/efflux.csv'
        file_path = f'/home/wwy/research/sensitivity/src/TargetValueAnalysis/output/sample_{i}/sample_{i}.csv'
        
        check = tva.read_path(file_path)
        if check == 0:
            continue
        tva.calculate_aqueous()
        tva.calculate_adsorbed()
        tva.calculate_efflux(efflux_path, efflux_csv_path)
        tva.save_target_values(i)
    
    tva.save_csv(target_csv_path)

    # tva.save_input_output('/home/wwy/research/sensitivity/src/RandomSampling/output/lhs_sampled_data.csv', '/home/wwy/research/sensitivity/src/TargetValueAnalysis/output/target_values.csv')
