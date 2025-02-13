import pandas as pd
import glob
import os
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/Arial.ttf')
font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf')
plt.rcParams['font.family'] = 'Arial'

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
        self.aqueous_granite = (material_1['Total UO2++ [M]'] * material_1['Volume [m^3]'] * material_1['Porosity']).sum()
        
        material_2 = self.data[self.data['Material ID'] == 2]
        self.aqueous_bentonite = (material_2['Total UO2++ [M]'] * material_2['Volume [m^3]'] * material_2['Porosity']).sum()
    
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
            result = [0]
            for line in lines[1:]:
                line_data = line.split()
                efflux_total_uo2 = [float(line_data[i]) for i in efflux_total_uo2_index]
                efflux_qlx = [float(line_data[i]) for i in efflux_qlx_index]
                efflux = [efflux_total_uo2[i] * efflux_qlx[i] for i in range(len(efflux_total_uo2))]
                efflux_sum = sum(efflux)            
                result.append(efflux_sum+result[-1])
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
            
    def save_target_values(self):
        target_values = pd.DataFrame({'Aqueous UO2++ in Granite': [self.aqueous_granite], 'Aqueous UO2++ in Bentonite': [self.aqueous_bentonite], 'Adsorbed UO2++ in Bentonite': [self.adsorbed]})
        if not hasattr(self, 'target_values'):
            self.target_values = target_values
        else:
            self.target_values = pd.concat([self.target_values, target_values], ignore_index=True)

    def save_csv(self, target_path):
        
        self.target_values.to_csv(target_path, index=False)

    
if __name__ == '__main__':

    for j in range(420):
        if os.path.exists(f'/home/geofluids/research/sensitivity/src/TargetValueAnalysis/output/sample_{j}/sample_{j}_time_10000.0.csv'):
            if not os.path.exists(f'/home/geofluids/research/sensitivity/src/TargetValueAnalysis/output/sample_{j}/target_values.csv'):
            
                tva = TargetValueAnalysis()
                
                target_csv_path = f'/home/geofluids/research/sensitivity/src/TargetValueAnalysis/output/sample_{j}/target_values.csv'
                efflux_path = f'/mnt/d/WWY/Personal/0. Paperwork/3. ML_sensitivity_analysis/Model/output_export/sample_{j}/sample_{j}-obs-'
                efflux_csv_path = f'/home/geofluids/research/sensitivity/src/TargetValueAnalysis/output/sample_{j}/efflux.csv'

                tva.calculate_efflux(efflux_path, efflux_csv_path)
                print(f'Efflux calculated for sample {j}')
        
                for i in range(0, 101):
    
                    file_path = f'/home/geofluids/research/sensitivity/src/TargetValueAnalysis/output/sample_{j}/sample_{j}_time_{i*100:.1f}.csv'
        
                    check = tva.read_path(file_path)
                    if check == 0:
                        continue
                    tva.calculate_aqueous()
                    tva.calculate_adsorbed()
                    tva.save_target_values()
    
                tva.save_csv(target_csv_path)
    
    num = 0    
    target_df = pd.DataFrame()
    input_csv_path = f'/home/geofluids/research/sensitivity/src/RandomSampling/output/lhs_sampled_data.csv'
    input_csv = pd.read_csv(input_csv_path, names=['x1', 'x2', 'x3', 'x4', 'x5'], header=None)

    # plot x1, x2, and x3
    for k in range(3):

        for j in range(420):

            target_csv_path = f'/home/geofluids/research/sensitivity/src/TargetValueAnalysis/output/sample_{j+1}/target_values.csv'
    
            if os.path.exists(target_csv_path):
                df = pd.read_csv(target_csv_path)
                        
                if j == 197:
                    plt.plot(df.index * 100, df.iloc[:, k], label=f'Sample {j+1}', color='red', zorder=10)
                else:
                    plt.plot(df.index * 100, df.iloc[:, k], label=f'Sample {j+1}', color=(0.86, 0.86, 1.0), linewidth=0.5)
                
                num += 1
                
                if k == 0:
                    df_last_row = df.iloc[-1, :].to_frame().T
                    df_last_row.index = [j]
                    input_row = input_csv.iloc[j, :].to_frame().T
                    input_row.index = [j]
                    df_last_row = pd.concat([input_csv.iloc[j, :].to_frame().T, df_last_row], axis=1)
                    target_df = pd.concat([target_df, df_last_row], axis=0)

        print(f'Case loaded: {num}')
        num = 0
    
        plt.xlabel('Time [yr]', fontfamily='Arial', fontweight='bold', fontsize=25)
        
        if k == 0:
            plt.ylabel(f'U$_{{frac}}$ [mol]', fontfamily='Arial', fontweight='bold', fontsize=22)
            plt.ylim(0, 5e-3)
            plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1e-3))
        elif k == 1:
            plt.ylabel(f'U$_{{bent}}$ [mol]', fontfamily='Arial', fontweight='bold', fontsize=22)
            plt.ylim(0, 1.2e-4)
            plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(3e-5))
        elif k == 2:
            plt.ylabel(f'U$_{{sorb}}$ [mol]', fontfamily='Arial', fontweight='bold', fontsize=22)
            plt.ylim(0, 10)
            plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(2))

        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:,.0f}'))
        plt.xticks(fontfamily='Arial', fontweight='bold', fontsize=20)
        plt.yticks(fontfamily='Arial', fontweight='bold', fontsize=20)

        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().tick_params(axis='both', which='major', width=2, length=6, direction='in')

        plt.gca().set_axisbelow(False)
        
        formatter = ticker.ScalarFormatter(useMathText=False)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))    
        formatter.format = lambda x, pos: f'{x:.1f}'
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.gca().yaxis.get_offset_text().set_fontsize(18)
        plt.gca().yaxis.get_offset_text().set_fontfamily('Arial')
        plt.gca().yaxis.get_offset_text().set_fontweight('bold')
        
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2500))
        plt.xlim(0, 10000)    
        
        plt.gcf().set_size_inches(6, 6)
        plt.gcf().set_dpi(1200)
        plt.savefig(f'./src/TargetValueAnalysis/output/target_value_{k+1}.png')
        plt.close()

    efflux_df = pd.DataFrame()

    # plot x4
    
    for j in range(420):

        efflux_csv_path = f'/home/geofluids/research/sensitivity/src/TargetValueAnalysis/output/sample_{j+1}/efflux.csv'

        if os.path.exists(efflux_csv_path):
            df = pd.read_csv(efflux_csv_path)
                
            if j == 197:
                plt.plot(df.index, df.iloc[:, 0], label=f'Sample {j+1}', color='red', zorder=10)
            else:
                plt.plot(df.index, df.iloc[:, 0], label=f'Sample {j+1}', color=(0.86, 0.86, 1.0))
                
            num += 1

            df_last_row = df.iloc[-1, :].to_frame().T
            df_last_row.index = [j]
            efflux_df = pd.concat([efflux_df, df_last_row], axis=0)

    target_df = pd.concat([target_df, efflux_df], axis=1)

    print(f'Case loaded: {num}')

    plt.xlabel('Time [yr]', fontfamily='Arial', fontweight='bold', fontsize=25)
    plt.ylabel(f'U$_{{out}}$ [mol]', fontfamily='Arial', fontweight='bold', fontsize=22)
    plt.ylim(0, 3e-2)
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1e-2))
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:,.0f}'))
    plt.xticks(fontfamily='Arial', fontweight='bold', fontsize=20)
    plt.yticks(fontfamily='Arial', fontweight='bold', fontsize=20)

    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().tick_params(axis='both', which='major', width=2, length=6, direction='in')

    plt.gca().set_axisbelow(False)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    formatter.format = lambda x, pos: f'{x:.1f}'
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().yaxis.get_offset_text().set_fontsize(18)
    plt.gca().yaxis.get_offset_text().set_fontfamily('Arial')
    plt.gca().yaxis.get_offset_text().set_fontweight('bold')

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2500))
    plt.xlim(0, 10000)

    plt.gcf().set_size_inches(6, 6)
    plt.gcf().set_dpi(1200)

    plt.savefig('./src/TargetValueAnalysis/output/target_value_4.png')
    plt.close()

    # make input-output pairs
    target_df.to_csv('./src/TargetValueAnalysis/output/inout.csv', index=False)

    # plot pdf
    target_path = './src/TargetValueAnalysis/output/inout.csv'

    # Read data from the csv file
    df = pd.read_csv(target_path)
    df = df.iloc[:, -4:]

    # Normalize the data by dividing by the maximum value of each column
    for i in range(4):
        df.iloc[:, i] = df.iloc[:, i] / df.iloc[:, i].max()

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(4, 1, figsize=(5, 20), constrained_layout=True)

    # Set the x-axis label of each subplot
    axs[3].set_xlabel('Probability Density', fontfamily='Arial', fontweight='bold', fontsize=25)

    # Set the y-axis label of each subplot
    axs[0].set_ylabel('y$_{1}$/y$_{1,max}$', fontfamily='Arial', fontweight='bold', fontsize=22)
    axs[1].set_ylabel('y$_{2}$/y$_{2,max}$', fontfamily='Arial', fontweight='bold', fontsize=22)
    axs[2].set_ylabel('y$_{3}$/y$_{3,max}$', fontfamily='Arial', fontweight='bold', fontsize=22)
    axs[3].set_ylabel('y$_{4}$/y$_{4,max}$', fontfamily='Arial', fontweight='bold', fontsize=22)
    
    # Plot the pdf of the target values, do not draw a histogram but only the pdf
    axs[0].hist(df.iloc[:, 0], bins=10, edgecolor='black', color='grey', density=True, orientation='horizontal')
    axs[1].hist(df.iloc[:, 1], bins=10, edgecolor='black', color='grey', density=True, orientation='horizontal')
    axs[2].hist(df.iloc[:, 2], bins=10, edgecolor='black', color='grey', density=True, orientation='horizontal')
    axs[3].hist(df.iloc[:, 3], bins=10, edgecolor='black', color='grey', density=True, orientation='horizontal')
        
    for ax in axs:
        ax.tick_params(axis='x', which='major', width=2, length=6, direction='in', labelsize=20)
        ax.tick_params(axis='y', which='major', width=2, length=6, direction='in', labelsize=20)
        for label in ax.get_xticklabels():
            label.set_fontsize(20)
            label.set_fontfamily('Arial')
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontsize(20)
            label.set_fontfamily('Arial')
            label.set_fontweight('bold')
        ax.set_xlim(0, 8)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.1f}'))
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.set_axisbelow(False)

    #plt.tight_layout()
    plt.gcf().set_dpi(1200)
    plt.savefig(target_path.replace('.csv', '_pdf.png'))
    plt.close()

