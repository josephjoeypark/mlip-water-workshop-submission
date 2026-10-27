"""Rebuild compact statistics and paper plots in the archived submission layout."""
from config import *
import importlib.util, shutil, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D='diffusion_vacf_pbc_nm2_ns'
DC='diffusion_vacf_corrected_nm2_ns'
ETA='viscosity_mPa_s'
METRICS=[D,DC,'diffusion_msd_pbc_nm2_ns','diffusion_msd_corrected_nm2_ns',ETA,
         'viscosity_10_20ps_mPa_s','yeh_hummer_nm2_ns','density_g_per_cm3']
TEMPS=[275,295,315,336]
CONDS=['ZS','T12','D00']
COLORS={'ZS':'#E14F62','T12':'#E8A23B','D00':'#95CA42'}
LABELS={'ZS':'Baseline','T12':'G1–G2','D00':'G1–G6'}
MARKERS={'ZS':'o','T12':'s','D00':'^'}
BACKBONES={'OFF':'MACE-OFF23(M)','LES':'MACELES-OFF'}
MODELS=[(b,c) for b in BACKBONES for c in CONDS]

def units(g,metric):
    if g.condition.iloc[0]=='ZS':
        assert len(g)==3
        return g[metric].to_numpy(float)
    assert len(g)==9 and set(g.training_seed)=={17,29,43}
    return g.groupby('training_seed')[metric].mean().to_numpy(float)

def aggregate(frame,metrics=METRICS):
    records=[]
    for key,g in frame.groupby(['backbone','condition','temperature_K']):
        r=dict(zip(['backbone','condition','temperature_K'],key))
        r.update(n_trajectories=len(g),n_training_seeds=0 if key[1]=='ZS' else 3,
                 uncertainty='MD-replica sample SD' if key[1]=='ZS' else 'sample SD across training-seed means')
        for metric in metrics:
            u=units(g,metric);assert np.isfinite(u).all()
            r[metric]=float(u.mean());r[metric+'_sd']=float(u.std(ddof=1))
        records.append(r)
    return pd.DataFrame(records)

def sym(a,b):
    return 200*np.abs(np.asarray(b)-np.asarray(a))/(np.abs(a)+np.abs(b))

def load_style(number):
    p=Path(__file__).with_name(f'archived_build_figure{number}.py')
    spec=importlib.util.spec_from_file_location(f'archived{number}',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def savefig(fig,name):
    fig.savefig(FIG/(name+'.pdf'),facecolor='white',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(FIG/(name+'.png'),dpi=300,facecolor='white')
    plt.close(fig)

def legend(fig,style,reference='EXP (bulk)'):
    handles=[Line2D([0],[0],color=COLORS[c],marker=MARKERS[c],label=LABELS[c]) for c in CONDS]
    handles.append(Line2D([0],[0],label=reference,**style.EXPERIMENT_STYLE))
    fig.legend(handles=handles,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.55,.985),
               handlelength=1.6,columnspacing=1.25)

def setup(style):
    fig,axes=plt.subplots(2,2,figsize=style.FIGSIZE,sharey='row')
    fig.subplots_adjust(**style.LAYOUT)
    for i,ax in enumerate(axes.flat):style.panel_label(ax,'abcd'[i]);style.style_axis(ax)
    for j,b in enumerate(BACKBONES):style.backbone_heading(axes[0,j],'MACE-OFF' if b=='OFF' else 'MACELES-OFF')
    return fig,axes

def structure():
    style=load_style(2);fig,axes=setup(style)
    density=pd.read_csv(DATA/'density_summary.csv');rdf=pd.read_csv(DATA/'rdf_295K.csv')
    reference=pd.read_csv(DATA/'thermophysical_reference.csv');rr=pd.read_csv(DATA/'oo_rdf_reference.csv')
    for j,b in enumerate(BACKBONES):
        for c in CONDS:
            g=rdf[(rdf.backbone==b)&(rdf.condition==c)].sort_values('r_A')
            axes[0,j].plot(g.r_A,g.g_OO,color=COLORS[c],marker=MARKERS[c],markersize=2.2,markevery=8,markeredgewidth=0,zorder=3)
            axes[0,j].fill_between(g.r_A,g.g_OO-g.sample_sd,g.g_OO+g.sample_sd,color=COLORS[c],alpha=.24,edgecolor=COLORS[c],lw=.4,zorder=2)
            q=density[(density.backbone==b)&(density.condition==c)].sort_values('temperature_K')
            axes[1,j].errorbar(q.temperature_K,q['mean'],yerr=q.sd,color=COLORS[c],marker=MARKERS[c],**style.ERRORBAR_STYLE)
        axes[0,j].plot(rr.r_A,rr.g_OO,markevery=8,**style.EXPERIMENT_STYLE)
        axes[0,j].fill_between(rr.r_A,rr.g_OO-rr.g_OO_uncertainty,rr.g_OO+rr.g_OO_uncertainty,color='black',alpha=.16,edgecolor='black',lw=.4,zorder=1)
        axes[1,j].plot(reference.temperature_K,reference.iapws95_density_g_cm3,**style.EXPERIMENT_STYLE)
        axes[0,j].set(xlim=(2.2,5.5),ylim=(-.05,3.25),yticks=[0,.6,1.2,1.8,2.4,3.0],xlabel=r'$r$ ($\mathrm{\AA}$)')
        axes[1,j].set(xlim=(272,339),xticks=TEMPS,ylim=(min(.89,float((density['mean']-density.sd).min())-.008),1.23),yticks=[.90,.96,1.02,1.08,1.14,1.20],xlabel='Temperature (K)')
    axes[0,0].set_ylabel(r'$g_{\mathrm{OO}}(r)$')
    axes[1,0].set_ylabel(r'$\rho$ (g cm$^{-3}$)')
    legend(fig,style,reference='EXP')
    savefig(fig,'figure1_structure')

def transport(summary,metric=DC,name='figure2_transport'):
    style=load_style(3);fig,axes=setup(style)
    refs=pd.read_csv(DATA/'thermophysical_reference.csv')
    for j,b in enumerate(BACKBONES):
        for i,m in enumerate([metric,ETA]):
            for c in CONDS:
                g=summary[(summary.backbone==b)&(summary.condition==c)].sort_values('temperature_K')
                axes[i,j].errorbar(g.temperature_K,g[m],yerr=g[m+'_sd'],color=COLORS[c],marker=MARKERS[c],**style.ERRORBAR_STYLE)
            axes[i,j].plot(refs.temperature_K,refs['oxygen_diffusion_1e-9_m2_s' if i==0 else 'viscosity_mPa_s'],**style.EXPERIMENT_STYLE)
            axes[i,j].set(xlim=(272,339),xticks=TEMPS)
            axes[i,j].set_ylim(bottom=0)
            if i==0:axes[i,j].tick_params(labelbottom=False)
            else:axes[i,j].set_xlabel('Temperature (K)')
    for i,m,original_top in [(0,metric,5.45),(1,ETA,4.35)]:
        top=max(original_top,float((summary[m]+summary[m+'_sd']).max())*1.07)
        axes[i,0].set_ylim(0,top)
        assert (summary[m]-summary[m+'_sd']).min()>=0
        assert (summary[m]+summary[m+'_sd']).max()<top
    axes[0,0].set_ylabel((r'$D_{\mathrm{YH}}$' if metric==DC else r'$D_{L}$')+r' ($10^{-9}$ m$^2$ s$^{-1}$)')
    axes[1,0].set_ylabel(r'$\eta$ (mPa s)')
    legend(fig,style)
    savefig(fig,name)

def main():
    assert json.loads((OUT/'validation/input_audit.json').read_text())['passed']
    frames={n:pd.read_csv(DATA/n/'lanes.csv') for n in ['prefix1000ps','prefix1500ps','prefix2000ps','window1000_2000ps']}
    full=frames['prefix2000ps'];summary=aggregate(full)
    summary.to_csv(DATA/'model_summary_2ns.csv',index=False)
    summary[summary.temperature_K==295].to_csv(DATA/'transport_295K.csv',index=False)
    comparisons={}
    for n,old in frames.items():
        if n=='prefix2000ps':continue
        merged=full.merge(old,on='lane_id',suffixes=('_2ns','_other'),validate='one_to_one')
        rec=[]
        for m in [D,DC,ETA]:
            p=sym(merged[m+'_other'],merged[m+'_2ns'])
            merged[m+'_symmetric_change_percent']=p
            rec.append(dict(metric=m,median_absolute_symmetric_change_percent=float(np.median(p)),above10_percent=int((p>10).sum())))
        merged.to_csv(DATA/f'comparison_{n}_to_2ns.csv',index=False)
        comparisons[n]=rec
    disjoint=frames['prefix1000ps'].merge(frames['window1000_2000ps'],on='lane_id',suffixes=('_first','_second'),validate='one_to_one')
    dis={}
    for m in [D,DC,ETA]:
        p=sym(disjoint[m+'_first'],disjoint[m+'_second'])
        disjoint[m+'_symmetric_change_percent']=p
        dis[m]=dict(median_absolute_symmetric_change_percent=float(np.median(p)),above10_percent=int((p>10).sum()),above20_percent=int((p>20).sum()))
    disjoint.to_csv(DATA/'disjoint_1ns_windows.csv',index=False)
    # Lane-wise correction precedes averaging; covariance is retained exactly.
    cov=[]
    for key,g in full.groupby(['backbone','condition','temperature_K']):
        a,b,z=units(g,D),units(g,'yeh_hummer_nm2_ns'),units(g,DC)
        covariance=np.cov(a,b,ddof=1)[0,1]
        rhs=a.var(ddof=1)+b.var(ddof=1)+2*covariance
        assert np.allclose(z,a+b) and np.isclose(z.var(ddof=1),rhs,atol=1e-12)
        cov.append(dict(zip(['backbone','condition','temperature_K'],key))|dict(
            raw_variance=a.var(ddof=1),correction_variance=b.var(ddof=1),raw_correction_covariance=covariance,
            corrected_variance=z.var(ddof=1),variance_identity_rhs=rhs,
            correction_percent_of_raw_mean=100*b.mean()/a.mean()))
    pd.DataFrame(cov).to_csv(DATA/'finite_size_covariance.csv',index=False)
    # Recheck NPT summaries against the completed final lane tables.
    dens=pd.read_csv(DATA/'density_summary.csv')
    merged=dens.merge(summary,on=['backbone','condition','temperature_K'],validate='one_to_one')
    assert np.allclose(merged['mean'],merged.density_g_per_cm3,atol=1e-12)
    assert np.allclose(merged.sd,merged.density_g_per_cm3_sd,atol=1e-12)
    refs=pd.read_csv(DATA/'thermophysical_reference.csv')
    dd=dens.merge(refs,on='temperature_K');dd['signed_error_percent']=100*(dd['mean']/dd.iapws95_density_g_cm3-1)
    dd['absolute_error_percent']=abs(dd.signed_error_percent)
    dd.to_csv(DATA/'density_reference_comparison.csv',index=False)
    errors=dd.groupby(['backbone','condition'])[['signed_error_percent','absolute_error_percent']].mean().reset_index()
    errors.to_csv(DATA/'density_four_temperature_errors.csv',index=False)
    # Explicit seed means with within-seed MD variability.
    seeds=full.groupby(['backbone','condition','training_seed','temperature_K'],dropna=False)[METRICS].agg(['mean','std'])
    seeds.columns=['_'.join(x) for x in seeds.columns];seeds.reset_index().to_csv(DATA/'seed_means_and_md_sd.csv',index=False)
    molecular=pd.read_csv(DATA/'cluster_rmse_all_seeds.csv')
    molecular=molecular[molecular.cluster_group.isin(['G3','G4','G5','G6']) & molecular.training_seed.notna()]
    assert (molecular.groupby(['model','condition','training_seed']).size()==4).all()
    molecular.groupby(['model','condition','training_seed'])[['energy_rmse_meV_per_atom','force_component_rmse_meV_per_A']].mean().reset_index().to_csv(DATA/'molecular_multineighbor_per_seed_macro_rmse.csv',index=False)
    decomposition=[]
    for r in full.itertuples():
        with np.load(DATA/'stress_diagnostics'/f'lane_{r.lane_id:03d}.npz') as z:
            modulus=r.volume_A3*1e-30/(1.380649e-23*r.mean_temperature_K)*float(z['stress_diagonal_Pa2'][0])*1e-9
        decomposition.append(dict(lane_id=r.lane_id,backbone=r.backbone,condition=r.condition,
            temperature_K=r.temperature_K,training_seed=r.training_seed,md_replica=r.md_replica,
            G_inf_GPa=modulus,tau_effective_ps=r.viscosity_mPa_s/modulus,viscosity_mPa_s=r.viscosity_mPa_s))
    pd.DataFrame(decomposition).to_csv(DATA/'stress_decomposition_lanes.csv',index=False)
    structure();transport(summary);transport(summary,D,'figureS1_raw_transport')
    m=load_style(1);m.SOURCE=DATA/'cluster_rmse_all_seeds.csv';m.OUTPUT=FIG;m.main()
    for ext in ['pdf','png']:(FIG/f'figure1.{ext}').rename(FIG/f'figureA1_molecular.{ext}')
    save(OUT/'validation/core_results.json',dict(passed=True,utc=utc(),model_temperature_states=24,
        seed_temperature_states=56,comparison_statistics=comparisons,disjoint_window_statistics=dis,
        lane_wise_finite_size_covariance_preserved=True,density_reproduces_all168=True,
        plot_sequence=['Figure 1: RDF top / density bottom; OFF left / LES right',
                       'Figure 2: diffusion top / viscosity bottom; OFF left / LES right',
                       'Figure A1: molecular errors; original archived layout'],
        plotting_basis='Archived final supplement builders; Figure 1/2/A1 mapping from user-provided OpenReview review text. Direct PDF download returned HTTP 403.',
        primary_diffusion_figure='Yeh-Hummer corrected; raw version Figure S1'))
    print('Core tables and paper-layout figures complete.',flush=True)

if __name__=='__main__':main()
