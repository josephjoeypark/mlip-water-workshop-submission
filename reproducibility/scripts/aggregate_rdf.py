"""Rebuild the 295 K RDF mean and uncertainty from all 42 NPT replicas."""
import numpy as np
import pandas as pd
from config import DATA


def aggregate_rdf(frame):
    expected = {'ZS': 3, 'T12': 9, 'D00': 9}
    rows, seeds = [], []
    for (backbone, condition), group in frame.groupby(['backbone','condition']):
        if group.lane_id.nunique() != expected[condition]:
            raise ValueError('Incomplete RDF replica coverage')
        grid = None
        seed_means, all_curves = [], []
        for seed, seed_group in group.groupby('training_seed', dropna=False):
            curves = []
            if seed_group.md_replica.nunique() != 3:
                raise ValueError('Each RDF training seed must contain three MD replicas')
            for lane, lane_group in seed_group.groupby('lane_id'):
                lane_group = lane_group.sort_values('r_A')
                if grid is None:
                    grid = lane_group.r_A.to_numpy(float)
                np.testing.assert_allclose(lane_group.r_A, grid, rtol=0, atol=1e-12)
                if len(grid) != 300 or not lane_group.r_A.is_unique:
                    raise ValueError('RDF radius grid mismatch')
                curves.append(lane_group.g_OO.to_numpy(float))
            mean = np.mean(curves, axis=0)
            seed_means.append(mean)
            all_curves.extend(curves)
            for radius, value in zip(grid, mean):
                seeds.append(dict(backbone=backbone,condition=condition,training_seed=seed,
                                  temperature_K=295,r_A=radius,g_OO=value))
        mean = np.mean(seed_means,axis=0)
        sd = np.std(all_curves if condition=='ZS' else seed_means,axis=0,ddof=1)
        for radius,value,uncertainty in zip(grid,mean,sd):
            rows.append(dict(backbone=backbone,condition=condition,temperature_K=295,
                             r_A=radius,g_OO=value,sample_sd=uncertainty))
    return pd.DataFrame(rows), pd.DataFrame(seeds)


def main():
    mean, seeds = aggregate_rdf(pd.read_csv(DATA/'rdf_295K_per_replica.csv'))
    mean.to_csv(DATA/'rdf_295K.csv',index=False)
    seeds.to_csv(DATA/'rdf_295K_per_seed.csv',index=False)
    print('Rebuilt RDF from 42 replicas and 14 model/seed groups.')


if __name__=='__main__':
    main()
