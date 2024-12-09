import pandas as pd
import h5py
import os

def find_max_specific_material(input_path, target, target_material):
    max_value = 0
    df = pd.DataFrame()
    with h5py.File(input_path, 'r') as f:

        target_data = f[' 100 Time  1.00000E+04 y'][target][()]
        material_data = f[' 100 Time  1.00000E+04 y']['Material ID'][()]
        df['Material'] = material_data
        df[target] = target_data
            
        # find maximum value of target where material == target_material
        max_value = df[df['Material'] == target_material][target].max()
    return max_value


def avg_velocity_specific_material(input_path, target_material):
    df = pd.DataFrame()
    with h5py.File(input_path, 'r') as f:

        x_velocity = f[' 100 Time  1.00000E+04 y']['Liquid X-Velocity [m_per_yr]'][()]
        y_velocity = f[' 100 Time  1.00000E+04 y']['Liquid Y-Velocity [m_per_yr]'][()]
        velocity = (x_velocity**2 + y_velocity**2)**0.5
        material_data = f[' 100 Time  1.00000E+04 y']['Material ID'][()]
        volume = f[' 100 Time  1.00000E+04 y']['Volume [m^3]'][()]
        df['Material'] = material_data
        df['Velocity'] = velocity
        df['Volume'] = volume

        df['Velocity * Volume'] = df['Velocity'] * df['Volume']
        avg_velocity = df[df['Material'] == target_material]['Velocity * Volume'].sum() / df[df['Material'] == target_material]['Volume'].sum()
    
    return avg_velocity


    
def read_hdf5_file(input_path, output_path):
    df = pd.DataFrame()
    with h5py.File(input_path, 'r') as f:
        domain = f['Domain']
        xc = domain['XC'][()]
        yc = domain['YC'][()]
        zc = domain['ZC'][()]
        df['XC'] = xc
        df['YC'] = yc
        df['ZC'] = zc

        for key in f.keys():
            if 'Time' in key and key.endswith(' y'):
                time_value = float(key.split()[2])
                time_data = f[key]
                for subkey in time_data:
                    df[subkey] = time_data[subkey][()]
                df.to_csv(f"{output_path}_time_{time_value}.csv", index=False)
           

if __name__ == '__main__':

###########################################################################################
    # read every sample file in ./src/RunPFLOTRAN/output/sample_*/sample_*.h5
    for i in range(200):
        
        if os.path.exists(f'/mnt/d/WWY/Personal/0. Paperwork/3. ML_sensitivity_analysis/Model/output_export/sample_{i}'):
            if not os.path.exists(f'/home/geofluids/research/sensitivity/src/TargetValueAnalysis/output/sample_{i}'):
                print(f'Reading sample_{i}\n')
                os.makedirs(f'/home/geofluids/research/sensitivity/src/TargetValueAnalysis/output/sample_{i}', exist_ok=True)
                read_hdf5_file(f'/mnt/d/WWY/Personal/0. Paperwork/3. ML_sensitivity_analysis/Model/output_export/sample_{i}/sample_{i}.h5', f'./src/TargetValueAnalysis/output/sample_{i}/sample_{i}')
###########################################################################################


###########################################################################################
    # # find maximum value of target where material == target_material
    # target = 'Total Sorbed UO2++ [mol_m^3]'
    # target_material = 2 # 1 = fracture, 2 = bentonite, 3 = source
    
    # max_value = find_max_specific_material(f'/mnt/d/WWY/Personal/0. Paperwork/3. ML_sensitivity_analysis/Model/output_export/sample_198/sample_198.h5', target, target_material)
    # print(f'Maximum value of {target} where material == {target_material} is {max_value}')
###########################################################################################


###########################################################################################
    # # find average velocity of specific material
    # target_material = 2 # 1 = fracture, 2 = bentonite, 3 = source
    # avg_velocity = avg_velocity_specific_material(f'/mnt/d/WWY/Personal/0. Paperwork/3. ML_sensitivity_analysis/Model/output_export/sample_198/sample_198.h5', target_material)
    # print(f'Average velocity of {target_material} is {avg_velocity} m/yr, {avg_velocity / 3600 / 24 / 365} m/s')

###########################################################################################