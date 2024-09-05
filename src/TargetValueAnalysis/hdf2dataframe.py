import pandas as pd
import h5py
import os

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

        if ' 100 Time  1.00000E+04 y' in f:
            time_data = f[' 100 Time  1.00000E+04 y']
            for subkey in time_data:
                df[subkey] = time_data[subkey][()]
            df.to_csv(output_path + '.csv', index=False)


if __name__ == '__main__':

    
    # read every sample file in ./src/RunPFLOTRAN/output/sample_*/sample_*.h5
    for i in range(101, 113):
        os.makedirs(f'/home/wwy/research/sensitivity/src/TargetValueAnalysis/output/sample_{i}', exist_ok=True)
        read_hdf5_file(f'./src/RunPFLOTRAN/output/sample_{i}/sample_{i}.h5', f'./src/TargetValueAnalysis/output/sample_{i}/sample_{i}')

    