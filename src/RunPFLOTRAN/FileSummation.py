import os
import shutil
import glob

def export_files():
    input_dir = './src/RunPFLOTRAN/output/'
    output_dir = './src/RunPFLOTRAN/output_export/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sample_dirs = glob.glob(os.path.join(input_dir, 'sample_*'))

    for sample_dir in sample_dirs:
        sample_name = os.path.basename(sample_dir)
        new_sample_dir = os.path.join(output_dir, sample_name)

        if not os.path.exists(new_sample_dir):
            os.makedirs(new_sample_dir)

        for file_extension in ['*.h5', '*.xmf', '*.pft']:
            if file_extension == '*.h5':
                for file_path in glob.glob(os.path.join(sample_dir, f'{sample_name}.h5')):
                    shutil.copy(file_path, new_sample_dir)
            else:
                for file_path in glob.glob(os.path.join(sample_dir, file_extension)):
                    shutil.copy(file_path, new_sample_dir)

        shutil.copy(os.path.join(sample_dir, sample_name + '.in'), new_sample_dir)
    
    print('Files exported successfully!')


if __name__ == "__main__":
    export_files()