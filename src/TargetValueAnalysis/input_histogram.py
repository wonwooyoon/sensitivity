import pandas as pd
import glob
import os
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/Arial.ttf')
font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf')
plt.rcParams['font.family'] = 'Arial'

def plot_histogram(file_path, output_path):

    df = pd.read_csv(f'{file_path}/inout.csv')
    df = df.iloc[:, 0:5]
    df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

    fig, ax = plt.subplots(5, 1, figsize=(5, 20))
    for i in range(5):
        ax[i].hist(df.iloc[:, i], bins=10, edgecolor='black', color='grey', density=True, orientation='horizontal')
        ax[i].set_ylabel(f'x$_{{{i+1}}}$/x$_{{{i+1}}}$$_{{,max}}$', fontfamily='Arial', fontweight='bold', fontsize=18)
        ax[i].tick_params(axis='x', direction='in', labelsize=15)
        ax[i].tick_params(axis='y', direction='out', labelsize=15)
        ax[i].set_xlim(0, 1.2)
    
    ax[-1].set_xlabel('Probability Density', fontfamily='Arial', fontweight='bold', fontsize=18)

    plt.tight_layout()
    plt.savefig(f'{output_path}/input_histogram.png')
    plt.close()

    print('Input histogram is saved.')


if __name__ == '__main__':

    file_path = './src/TargetValueAnalysis/output'
    output_path = './src/TargetValueAnalysis/output'
    plot_histogram(file_path, output_path)