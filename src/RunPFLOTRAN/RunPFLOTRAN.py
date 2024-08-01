
import subprocess


def run_pflotran_main():
    bash_code = """
#!/bin/bash
base_dir="$(pwd)"
mkdir -p "${base_dir}/src/RunPFLOTRAN/output"

for infile in ${base_dir}/src/RunPFLOTRAN/input/sample_*.in; do
  mpirun -n 6 /home/wwy/pflotran/src/pflotran/pflotran -input_prefix "${infile%.*}"
  output_subdir="${base_dir}/src/RunPFLOTRAN/output/$(basename ${infile%.*})"
  mkdir -p "${output_subdir}"
  mv ${base_dir}/src/RunPFLOTRAN/input/*.tec "${output_subdir}"
  mv ${base_dir}/src/RunPFLOTRAN/input/*.pft "${output_subdir}"
  mv ${base_dir}/src/RunPFLOTRAN/input/*.out "${output_subdir}"
done
"""
    subprocess.run(['bash', '-c', bash_code], check=True)


if __name__ == '__main__':

    run_pflotran_main()
