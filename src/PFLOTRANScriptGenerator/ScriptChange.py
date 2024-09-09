import csv
import numpy as np
import pandas as pd


def change_input(default_filename, output_filename, updated_data, components):

    with open(default_filename, 'r') as file:
        lines = file.readlines()

    fracture_perm_target = None
    bentonite_poro_target = None
    bentonite_mineral_target = None
    granite_mineral_target = None
    pressure_grad_target = None
    mixing_ratio_target = None
    tolerance_traget = None
    timestep_target = None

    for i, line in enumerate(lines):

        if 'MATERIAL_PROPERTY granite_fracture' in line:
            fracture_perm_target = i + 6
        if 'MATERIAL_PROPERTY bentonite' in line:
            bentonite_poro_target = i + 2
        if 'CONSTRAINT bentonite_conc' in line:
            bentonite_mineral_target = i + 17
        if 'CONSTRAINT granite_conc' in line:
            granite_mineral_target = i + 21
        if 'LIQUID_PRESSURE 561325.d0' in line:
            pressure_grad_target = i
        if 'CONSTRAINT seawater_conc' in line:
            mixing_ratio_target = i + 2
        if 'ATOL 1e-13' in line:
            tolerance_traget = i
        if 'MAXIMUM_TIMESTEP_SIZE 2.d-2 yr at 10 yr' in line:
            timestep_target = i

    density_based_porosity = 1 - (updated_data[1]/2750)
    density_based_smectite = 0.806 * (1-density_based_porosity)
    density_based_inert = 0.13702 * (1-density_based_porosity)
    density_based_quartz = 0.0488 * (1-density_based_porosity)
    density_based_gypsum = 0.00782 * (1-density_based_porosity)
    density_based_pyrite = 0.00036 * (1-density_based_porosity) * updated_data[3]

    if updated_data[0] <= 5e-17:
        lines[tolerance_traget] = '    ATOL 1e-15\n'
    elif updated_data[0] >= 5e-16:
        lines[timestep_target] = '    MAXIMUM_TIMESTEP_SIZE 5.d-3 yr at 10 yr\n'
    elif updated_data[0] >= 1e-16:
        lines[timestep_target] = '    MAXIMUM_TIMESTEP_SIZE 1.d-2 yr at 10 yr\n'

    if fracture_perm_target is not None:
        lines[fracture_perm_target] = f'    PERM_ISO {updated_data[0]} ! unit: m^2\n'
    if bentonite_poro_target is not None:
        lines[bentonite_poro_target] = f'   POROSITY {density_based_porosity}\n'
    if pressure_grad_target is not None:
        lines[pressure_grad_target] = f'    LIQUID_PRESSURE {updated_data[2]} ! unit: Pa\n'
    if bentonite_mineral_target is not None:

        lines[bentonite_mineral_target] = f'    Smectite_MX80 	{density_based_smectite}	8.5 	m^2/g\n'
        lines[bentonite_mineral_target+1] = f'    InertMineral		{density_based_inert}	0.0 	m^2/m^3\n'
        lines[bentonite_mineral_target+2] = f'    Quartz		{density_based_quartz}	0.05 	m^2/g\n'
        lines[bentonite_mineral_target+3] = f'    Gypsum		{density_based_gypsum}	0.05 	m^2/g\n'
        lines[bentonite_mineral_target+4] = f'    Pyrite		{density_based_pyrite}	0.05 	m^2/g\n'

    if granite_mineral_target is not None:

        lines[granite_mineral_target] = f'    Pyrite		{0.0 * updated_data[4]}	1e2 	m^2/m^3\n'

    if mixing_ratio_target is not None:

        lines[mixing_ratio_target] = f'H+ {components[0]} pH\n'
        lines[mixing_ratio_target+1] = f'O2(aq) {components[1]} PE\n'
        lines[mixing_ratio_target+2] = f'Al+++ {components[2]} T\n'
        lines[mixing_ratio_target+3] = f'CO3-- {components[3]} T\n'
        lines[mixing_ratio_target+4] = f'Ca++ {components[4]} T\n'
        lines[mixing_ratio_target+5] = f'Cl- {components[5]} Z\n'
        lines[mixing_ratio_target+6] = f'Fe++ {components[6]} T\n'
        lines[mixing_ratio_target+7] = f'H4(SiO4) {components[7]} T\n'
        lines[mixing_ratio_target+8] = f'K+ {components[8]} T\n'
        lines[mixing_ratio_target+9] = f'Mg++ {components[9]} T\n'
        lines[mixing_ratio_target+10] = f'Na+ {components[10]} T\n'
        lines[mixing_ratio_target+11] = f'SO4-- {components[11]} T\n'
        lines[mixing_ratio_target+12] = f'UO2++ {components[12]} T\n'

    with open(output_filename, 'w') as file:
        file.writelines(lines)
    

if __name__ == "__main__":

    default_file_path = './src/PFLOTRANScriptGenerator/input/base_code.txt'
    output_filename_base = './src/RunPFLOTRAN/input/sample'
    components_path = './src/RatioCalculation/output/mixed_components.csv'
    updated_data_path = './src/RandomSampling/output/lhs_sampled_data.csv'

    with open(updated_data_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = np.array([row for row in reader], dtype = float)

    ratios = data[:, -1]

    components = pd.read_csv(components_path, header=None).to_numpy()
    
    for idx, row in enumerate(data):
        updated_data = row
        output_filename = f'{output_filename_base}_{idx + 1}.in'
        change_input(default_file_path, output_filename, updated_data, components[idx, :])
