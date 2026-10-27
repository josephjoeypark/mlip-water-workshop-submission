"""Descriptive experimental errors for all 24 model-temperature conditions."""
from config import *
import pandas as pd
import numpy as np

def main():
    model=pd.read_csv(DATA/'model_summary_2ns.csv');reference=pd.read_csv(DATA/'thermophysical_reference.csv')
    merged=model.merge(reference,on='temperature_K',validate='many_to_one');assert len(merged)==24
    metrics={'density':('density_g_per_cm3','iapws95_density_g_cm3'),
        'diffusion_raw':('diffusion_vacf_pbc_nm2_ns','oxygen_diffusion_1e-9_m2_s'),
        'diffusion_corrected':('diffusion_vacf_corrected_nm2_ns','oxygen_diffusion_1e-9_m2_s'),
        'viscosity':('viscosity_mPa_s','viscosity_mPa_s_y')}
    # Pandas suffixes both viscosity columns because the reference uses the same unit label.
    assert 'viscosity_mPa_s_x' in merged and 'viscosity_mPa_s_y' in merged
    metrics['viscosity']=('viscosity_mPa_s_x','viscosity_mPa_s_y')
    out=merged[['backbone','condition','temperature_K','n_trajectories','n_training_seeds','uncertainty']].copy()
    for name,(prediction,target) in metrics.items():
        out[name+'_prediction']=merged[prediction];out[name+'_reference']=merged[target]
        out[name+'_signed_error_percent']=100*(merged[prediction]/merged[target]-1)
        out[name+'_absolute_error_percent']=abs(out[name+'_signed_error_percent'])
        assert np.isfinite(out[name+'_signed_error_percent']).all()
    out.to_csv(DATA/'transport_reference_errors.csv',index=False)
    comparisons=[]
    for (backbone,temperature),group in out.groupby(['backbone','temperature_K']):
        indexed=group.set_index('condition')
        for origin in ['ZS','T12']:
            for metric in metrics:
                comparisons.append(dict(backbone=backbone,temperature_K=int(temperature),comparison='D00 versus '+origin,metric=metric,
                    starting_absolute_error_percent=float(indexed.loc[origin,metric+'_absolute_error_percent']),
                    broad_absolute_error_percent=float(indexed.loc['D00',metric+'_absolute_error_percent']),
                    broad_minus_starting_error_percentage_points=float(indexed.loc['D00',metric+'_absolute_error_percent']-indexed.loc[origin,metric+'_absolute_error_percent'])))
    pd.DataFrame(comparisons).to_csv(DATA/'transport_error_changes.csv',index=False)
    lookup=lambda b,c,t:out[(out.backbone==b)&(out.condition==c)&(out.temperature_K==t)].iloc[0]
    assert abs(lookup('LES','D00',275).viscosity_signed_error_percent-60.174441626472856)<1e-9
    assert abs(lookup('OFF','D00',336).diffusion_corrected_signed_error_percent-23.831328365689796)<1e-9
    save(OUT/'validation/transport_reference_comparison.json',dict(passed=True,utc=utc(),states=24,contrasts=len(comparisons),
        source_sha256=sha(DATA/'model_summary_2ns.csv'),reference_sha256=sha(DATA/'thermophysical_reference.csv'),
        interpretation='All conditions are reported; selected manuscript examples are descriptive, not hypothesis tests. Experimental agreement does not establish reference-level fidelity.'))
    print('PASS: all 24 state-wise experimental error comparisons',flush=True)

if __name__=='__main__':main()
