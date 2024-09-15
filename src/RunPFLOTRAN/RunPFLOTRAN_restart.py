
import subprocess
import os


def run_pflotran_restart():
    
    check_range = range(101, 201)

    for i in check_range:
        if os.path.exists(f'../output/sample_{i}/restart.h5'):
            with open (f'../output/sample_{i}/sample_{i}.in', 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if 'CHECKPOINT' in line:
                        new_line = f'RESTART\n   FILENAME sample_{i}.restart\n'
                        lines.insert(lines.index(line) + 4, new_line)
                    if 'NUMERICAL_METHODS transport' in line:
                        pass

                with open (f'../output/sample_{i}/sample_{i}_restart.in', 'w') as restart_file: 
                    restart_file.writelines(lines)
    
            bash_code = f"""
            #!/bin/bash
            mpirun -n 36 /home/wwy/pflotran/src/pflotran/pflotran -input_prefix sample_{i}_restart.in
            """

            subprocess.run(['bash', '-c', bash_code], check=True)

                    
    
if __name__ == '__main__':

    run_pflotran_restart()
