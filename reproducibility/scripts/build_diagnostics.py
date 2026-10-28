"""Reviewer diagnostics with the submission's color, marker and backbone conventions."""
from build_results import *
from matplotlib.backends.backend_pdf import PdfPages

def curve_units(g,curves,key):
    if g.condition.iloc[0]=='ZS':return np.stack([curves[int(i)][key] for i in g.lane_id])
    return np.stack([np.mean([curves[int(i)][key] for i in q.lane_id],axis=0) for _,q in g.groupby('training_seed')])

def line(ax,g,curves,xkey,ykey,band=True,individual=False):
    c=g.condition.iloc[0];x=curves[int(g.lane_id.iloc[0])][xkey]
    u=curve_units(g,curves,ykey);m=u.mean(axis=0);sd=u.std(axis=0,ddof=1)
    if individual:
        for v in u:ax.plot(x,v,color=COLORS[c],lw=.65,alpha=.45)
    ax.plot(x,m,color=COLORS[c],lw=1.4)
    if band:ax.fill_between(x,m-sd,m+sd,color=COLORS[c],alpha=.16,lw=0)

def grid(title):
    fig,axes=plt.subplots(4,2,figsize=(7,8.4),sharex=True,sharey='row')
    fig.subplots_adjust(left=.10,right=.985,bottom=.07,top=.89,hspace=.33,wspace=.12)
    fig.suptitle(title,y=.985,fontsize=10)
    handles=[Line2D([0],[0],color=COLORS[c],marker=MARKERS[c],label=LABELS[c]) for c in CONDS]
    fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.55,.959),frameon=False)
    for i,t in enumerate(TEMPS):
        for j,b in enumerate(BACKBONES):
            axes[i,j].set_title(f'{BACKBONES[b]} · {t} K',fontsize=8)
            axes[i,j].grid(axis='y',color='.8',alpha=.45,lw=.55)
            axes[i,j].text(0,1.02,'abcdefgh'[i*2+j],transform=axes[i,j].transAxes,fontweight='bold',fontsize=8)
    return fig,axes

def main():
    assert json.loads((OUT/'validation/stress_diagnostics.json').read_text())['passed']
    load_style(3)
    lanes=pd.read_csv(DATA/'prefix2000ps/lanes.csv')
    extra=pd.read_csv(DATA/'stress_window_values.csv')
    lanes=lanes.merge(extra[['lane_id','viscosity_20_30ps_mPa_s','viscosity_30_50ps_mPa_s']],on='lane_id',validate='one_to_one')
    curves={i:dict(np.load(DATA/'curves'/f'lane_{i:03d}.npz')) for i in range(168)}
    stress={i:dict(np.load(DATA/'stress_diagnostics'/f'lane_{i:03d}.npz')) for i in range(168)}
    summary=aggregate(lanes,METRICS+['viscosity_20_30ps_mPa_s','viscosity_30_50ps_mPa_s'])
    summary.to_csv(DATA/'model_summary_with_lag_sensitivity.csv',index=False)
    sensitivity={}
    for m in ['viscosity_10_20ps_mPa_s','viscosity_20_30ps_mPa_s','viscosity_30_50ps_mPa_s']:
        p=sym(lanes[ETA],lanes[m]);q=sym(summary[ETA],summary[m])
        lanes[m+'_change_percent']=p;summary[m+'_change_percent']=q
        sensitivity[m]=dict(lanes_above10=int((p>10).sum()),states_above10=int((q>10).sum()),
                            median_lane_change=float(np.median(p)),max_state_change=float(np.max(q)))
    lanes.to_csv(DATA/'lane_lag_sensitivity.csv',index=False)
    summary.to_csv(DATA/'model_lag_sensitivity.csv',index=False)
    for name,title,key,xlim,ylabel in [
        ('figureS2_stress_short','Stress autocorrelation · 2 ns trajectories','stress_normalized',(0,1),r'$C_\sigma(t)/C_\sigma(0)$'),
        ('figureS3_stress_tail','Signed stress tail · 2 ns trajectories','stress_normalized',(2,50),r'$C_\sigma(t)/C_\sigma(0)$'),
        ('figureS4_green_kubo_50ps','Green–Kubo running viscosity · 2 ns trajectories','running_viscosity_mPa_s',(0,50),r'$\eta(t)$ (mPa s)'),
        ('figureS5_green_kubo_20ps','Green–Kubo running viscosity · primary and sensitivity windows','running_viscosity_mPa_s',(0,20),r'$\eta(t)$ (mPa s)')]:
        fig,axes=grid(title)
        for i,t in enumerate(TEMPS):
            for j,b in enumerate(BACKBONES):
                ax=axes[i,j]
                for c in CONDS:
                    g=lanes[(lanes.temperature_K==t)&(lanes.backbone==b)&(lanes.condition==c)]
                    # Restrict displayed samples before plotting so tail limits reflect the tail.
                    selected={int(k):{kk:(vv[(stress[int(k)]['stress_time_ps']>=xlim[0])&(stress[int(k)]['stress_time_ps']<=xlim[1])] if kk in ['stress_time_ps',key] else vv) for kk,vv in stress[int(k)].items()} for k in g.lane_id}
                    line(ax,g,selected,'stress_time_ps',key)
                ax.axhline(0,color='.45',lw=.5)
                ax.set_xlim(*xlim)
                if j==0:ax.set_ylabel(ylabel)
                if i==3:ax.set_xlabel('Lag time (ps)')
        savefig(fig,name)
    for name,title,key,xkey,xlim,ylabel in [
        ('figureS6_vacf','Oxygen velocity autocorrelation · 2 ns trajectories','vacf_normalized','vacf_time_ps',(0,1),r'$C_v(t)/C_v(0)$'),
        ('figureS7_running_diffusion','Raw finite-cell VACF integral · 2 ns trajectories','running_diffusion_pbc_nm2_ns','vacf_time_ps',(0,10),r'$D_L(t)$ (nm$^2$/ns)')]:
        fig,axes=grid(title)
        for i,t in enumerate(TEMPS):
            for j,b in enumerate(BACKBONES):
                ax=axes[i,j]
                for c in CONDS:line(ax,lanes[(lanes.temperature_K==t)&(lanes.backbone==b)&(lanes.condition==c)],curves,xkey,key)
                ax.set_xlim(*xlim);ax.axhline(0,color='.5',lw=.5)
                if j==0:ax.set_ylabel(ylabel)
                if i==3:ax.set_xlabel('Lag time (ps)')
        savefig(fig,name)
    first=pd.read_csv(DATA/'prefix1000ps/lanes.csv');second=pd.read_csv(DATA/'window1000_2000ps/lanes.csv')
    for metric,name,title in [(ETA,'figureS8_disjoint_viscosity','Disjoint 1 ns windows · viscosity'),(DC,'figureS9_disjoint_diffusion','Disjoint 1 ns windows · Yeh–Hummer diffusion')]:
        fig,axes=grid(title)
        for i,t in enumerate(TEMPS):
            for j,b in enumerate(BACKBONES):
                ax=axes[i,j]
                for k,c in enumerate(CONDS):
                    a=first[(first.temperature_K==t)&(first.backbone==b)&(first.condition==c)]
                    z=second[(second.temperature_K==t)&(second.backbone==b)&(second.condition==c)]
                    u,v=units(a,metric),units(z,metric)
                    for n in range(3):ax.plot([k-.16,k+.16],[u[n],v[n]],color=COLORS[c],lw=.6,alpha=.5)
                    for x,values,fill in [(k-.16,u,'white'),(k+.16,v,COLORS[c])]:
                        ax.errorbar([x],[values.mean()],yerr=[values.std(ddof=1)],color=COLORS[c],marker=MARKERS[c],mfc=fill,capsize=2)
                        ax.scatter(np.repeat(x,3),values,color=COLORS[c],s=6,alpha=.7)
                ax.set_xticks(range(3),[LABELS[c] for c in CONDS]);ax.set_xlim(-.5,2.5)
                if j==0:ax.set_ylabel(r'$\eta$ (mPa s)' if metric==ETA else r'$D_{\mathrm{YH}}$ (nm$^2$/ns)')
        fig.text(.54,.018,'Left/open: 0–1 ns   ·   Right/filled: 1–2 ns   ·   Thin lines: paired seed means (baseline: replicas)',ha='center',fontsize=7)
        savefig(fig,name)
    blocks=pd.read_csv(DATA/'block_viscosity.csv')
    bmean=blocks.groupby(['lane_id','block_length_ps'])[ETA].agg(['mean','std','count']).reset_index()
    bmean['diagnostic_sem']=bmean['std']/np.sqrt(bmean['count'])
    bmean=bmean.merge(lanes[['lane_id','backbone','condition','training_seed','temperature_K',ETA]],on='lane_id',validate='many_to_one')
    bmean['diagnostic_sem_percent_of_full']=100*bmean.diagnostic_sem/bmean[ETA]
    bmean['mean_change_percent']=sym(bmean['mean'],bmean[ETA])
    bmean.to_csv(DATA/'block_length_statistics.csv',index=False)
    fig,axes=grid('Block-length sensitivity · primary 5–10 ps viscosity window')
    for i,t in enumerate(TEMPS):
        for j,b in enumerate(BACKBONES):
            ax=axes[i,j]
            for c in CONDS:
                q=bmean[(bmean.temperature_K==t)&(bmean.backbone==b)&(bmean.condition==c)]
                values=[];sd=[]
                for length in [100,200,500]:
                    u=units(q[q.block_length_ps==length],'mean');values.append(u.mean());sd.append(u.std(ddof=1))
                ax.errorbar([100,200,500],values,yerr=sd,color=COLORS[c],marker=MARKERS[c],capsize=2)
                full=summary[(summary.temperature_K==t)&(summary.backbone==b)&(summary.condition==c)][ETA].iloc[0]
                ax.axhline(full,color=COLORS[c],ls=':',lw=.8)
            ax.set_xticks([100,200,500])
            if j==0:ax.set_ylabel(r'$\eta$ (mPa s)')
            if i==3:ax.set_xlabel('Block length (ps)')
    fig.text(.54,.018,'Dots/lines: mean of disjoint block estimates   ·   Horizontal dotted: full 2 ns estimate',ha='center',fontsize=7)
    savefig(fig,'figureS10_block_length')
    # True whole-prefix estimates, not cumulative averages of 100 ps block estimates.
    fig,axes=setup(load_style(3))
    prefixes=[1000,1500,2000]
    summaries={p:aggregate(pd.read_csv(DATA/f'prefix{p}ps/lanes.csv')) for p in prefixes}
    for j,b in enumerate(BACKBONES):
        for i,metric in enumerate([DC,ETA]):
            for c in CONDS:
                q=[s[(s.temperature_K==295)&(s.backbone==b)&(s.condition==c)].iloc[0] for s in summaries.values()]
                axes[i,j].errorbar(np.array(prefixes)/1000,[s[metric] for s in q],yerr=[s[metric+'_sd'] for s in q],color=COLORS[c],marker=MARKERS[c],capsize=2)
            axes[i,j].set_xticks([1,1.5,2]);axes[i,j].set_xlim(.95,2.05)
            if i==1:axes[i,j].set_xlabel('Analyzed prefix (ns)')
    axes[0,0].set_ylabel(r'$D_{\mathrm{YH}}$ (nm$^2$/ns)');axes[1,0].set_ylabel(r'$\eta$ (mPa s)')
    handles=[Line2D([0],[0],color=COLORS[c],marker=MARKERS[c],label=LABELS[c]) for c in CONDS]
    fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.55,.985))
    fig.text(.54,.015,'295 K · whole-prefix reanalyses share trajectory data',ha='center',fontsize=7)
    savefig(fig,'figureS11_prefix_length_295K')
    # Per-state supplement makes all MD replicas and all fitted seeds visible.
    with PdfPages(FIG/'all_24_states.pdf') as pdf:
        for (b,c,t),g in lanes.groupby(['backbone','condition','temperature_K']):
            fig,axes=plt.subplots(2,2,figsize=(7,5));fig.subplots_adjust(left=.11,right=.98,bottom=.12,top=.85,hspace=.42,wspace=.30)
            fig.suptitle(f'{BACKBONES[b]} · {LABELS[c]} · {t} K',fontsize=11)
            for ax,key,xmax in [(axes[0,0],'stress_normalized',1),(axes[0,1],'stress_normalized',50),(axes[1,0],'running_viscosity_mPa_s',50)]:
                for lane in g.lane_id:
                    z=stress[int(lane)];ax.plot(z['stress_time_ps'],z[key],color=COLORS[c],lw=.4,alpha=.2)
                line(ax,g,stress,'stress_time_ps',key,individual=True);ax.set_xlim(0,xmax);ax.axhline(0,color='.5',lw=.5);ax.set_xlabel('Lag time (ps)')
            axes[0,0].set_ylabel('Normalized stress ACF');axes[0,1].set_ylim(-.02,.02);axes[0,1].set_xlim(2,50)
            # The zoom is explicit; the untruncated signed tail is in Figure S3.
            axes[0,1].set_title('Tail zoom (±0.02)',fontsize=8)
            axes[1,0].set_ylabel(r'$\eta(t)$ (mPa s)')
            ax=axes[1,1]
            for k,f in enumerate([first,second]):
                q=f[f.lane_id.isin(g.lane_id)]
                ax.scatter(np.repeat(k,len(q)),q[ETA],s=9,color=COLORS[c],alpha=.45)
                u=units(q,ETA);ax.errorbar(k,u.mean(),yerr=u.std(ddof=1),color='black',marker='o',capsize=3)
            ax.set_xticks([0,1],['0–1 ns','1–2 ns']);ax.set_xlim(-.5,1.5);ax.set_ylabel(r'$\eta$ (mPa s)')
            fig.text(.5,.025,'Faint: MD replicas · thin: seed means · bold/band: mean ± sample SD (baseline: MD SD)',ha='center',fontsize=7)
            pdf.savefig(fig);plt.close(fig)
    save(OUT/'validation/diagnostic_report.json',dict(passed=True,utc=utc(),window_sensitivity=sensitivity,
        block_mean_change_median_percent={str(n):float(g.mean_change_percent.median()) for n,g in bmean.groupby('block_length_ps')},
        block_sem_is_diagnostic_not_validated_ci=True,appendix_pages=24))
    print('Reviewer diagnostics and 24-state appendix complete.',flush=True)

if __name__=='__main__':main()
