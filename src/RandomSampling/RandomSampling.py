import numpy as np
import pandas as pd
from scipy.stats import qmc


def lhs_sampling(num_samples, ranges, log_scale_vars, output_csv):

    num_vars = len(ranges)

    sampler = qmc.LatinHypercube(d=num_vars)
    sample = sampler.random(n=num_samples)

    scaled_sample = np.zeros_like(sample)

    for i in range(num_vars):

        if log_scale_vars[i]:
            scaled_sample[:, i] = np.power(10, ranges[i][0] + sample[:, i] * (ranges[i][1] - ranges[i][0]))
        else:
            scaled_sample[:, i] = ranges[i][0] + sample[:, i] * (ranges[i][1] - ranges[i][0])

    columns = ['fracture_perm', 'bentonite_dry_density', 'pressure_grad', 'bentonite_pyrite_factor', 'mixing_ratio']
    df = pd.DataFrame(scaled_sample, columns=columns)
    df.to_csv(output_csv, index=False, header=False)


if __name__ == "__main__":
    num_samples = 80
    ranges = [
        [1e-16, 1e-14],
        [1300, 1900],
        [501925, 507325],
        [0.0, 1.0],
        [0.0, 0.33]
    ]

    # perm, density, pressure gradient, pyrite factor, mixing ratio
    log_scale_vars = [False, False, False, False, False]

    output_csv = "./src/RandomSampling/output/lhs_sampled_data.csv"
    lhs_sampling(num_samples, ranges, log_scale_vars, output_csv)

