#!/usr/bin/env python3
"""Rebuild report Figures 4-8 from explicit protocols and saved aggregate inputs.

No model calls or new statistical estimation. Use --verify-sources with the two
evidence package paths to authenticate and compare the source JSON fields.
"""
from pathlib import Path
import argparse, hashlib, json, os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/projectxlm-figure-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
D = json.loads((ROOT/'inputs.json').read_text())
BLACK='#202020'; GRAY='#737373'; LIGHT='#dddddd'
BLUE, ORANGE = sns.color_palette('colorblind')[:2]
sns.set_theme(context='paper', style='whitegrid', palette='colorblind',
 font='DejaVu Sans', font_scale=1,
 rc={'font.size':10,'axes.labelsize':9.5,'axes.titlesize':10.5,
 'xtick.labelsize':9,'ytick.labelsize':9,'axes.spines.top':False,
 'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'refusal-seaborn-v1',
 'text.color':BLACK,'axes.labelcolor':BLACK,'axes.edgecolor':GRAY,
 'xtick.color':BLACK,'ytick.color':BLACK,'grid.color':'#e4e4e4',
 'grid.linewidth':.6,'axes.linewidth':.7,'savefig.facecolor':'white'})
OUTPUTS=[]

def save(fig, name, out):
    for ext in ['png','svg']:
        p=out/(name+'.'+ext)
        fig.savefig(p,dpi=240,metadata={'Date':None} if ext=='svg' else {})
        OUTPUTS.append({'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    plt.close(fig)

def canvas(height, ylim):
    fig=plt.figure(figsize=(6.5,height))
    ax=fig.add_axes([.015,.015,.97,.97]);ax.set_xlim(0,100);ax.set_ylim(0,ylim);ax.axis('off')
    return fig,ax

def txt(ax,x,y,s,size=10,ha='center',weight='normal',**kw):
    return ax.text(x,y,s,fontsize=size,ha=ha,va='center',weight=weight,linespacing=1.25,**kw)

def box(ax,x,y,w,h,s,size=10,edge=GRAY):
    ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor=edge,lw=.85,zorder=2))
    txt(ax,x+w/2,y+h/2,s,size,zorder=3)

def arrow(ax,a,b,style='-',**kw):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,lw=.9,
                                linestyle=style,color=GRAY,zorder=1,**kw))

def figure4(out):
    fig,ax=canvas(2.85,45)
    box(ax,1,25,27,13,'One specified edit\n'+r'$\Delta h=\Delta a\,d_s$'+'\nfinal prompt token',9.6)
    box(ax,39,32,30,11,'AtP\n'+r'$\nabla_h z\,\cdot\,\Delta h$',10)
    box(ax,39,18,30,11,'Graph prediction\nsigned edge × '+r'$\Delta a/a_{\rm ref}$',9.5)
    box(ax,79,25,20,13,'Freeze both\npredictions',9.5)
    arrow(ax,(28,34),(39,37));arrow(ax,(28,29),(39,23))
    arrow(ax,(69,37),(79,34));arrow(ax,(69,23),(79,29))
    box(ax,1,1,33,12,'Apply the edit in the\noriginal model',10)
    box(ax,43,1,26,12,'Measure receiver '+r'$\Delta z$'+'\nbefore ReLU',9.5)
    box(ax,79,1,20,12,'Compare\nprediction\nwith effect',9.5)
    arrow(ax,(14.5,25),(14.5,13));arrow(ax,(34,7),(43,7));arrow(ax,(69,7),(79,7))
    arrow(ax,(89,25),(89,13))
    save(fig,'figure_4_shared_intervention',out)

def toy_values():
    toy=D['toy'];paths=toy['paths'];names=toy['candidate_order']
    def cov(nodes):return sum(p['probability'] for p in paths if set(nodes)&set(p['nodes']))
    influence={n:cov([n]) for n in names}
    ranked=sorted(names,key=lambda n:(-influence[n],names.index(n)))[:toy['budget']]
    greedy=[]
    for _ in range(toy['budget']):
        greedy.append(min((n for n in names if n not in greedy),
                          key=lambda n:(-(cov(greedy+[n])-cov(greedy)),names.index(n))))
    assert ranked==['U','V'] and greedy==['U','W']
    return influence,ranked,greedy,cov(ranked),cov(greedy)

def figure5(out):
    inf,ranked,greedy,rank_cov,greedy_cov=toy_values()
    fig=plt.figure(figsize=(6.5,2.45))
    ax=fig.add_axes([.015,.06,.47,.90]);ax.set_xlim(0,50);ax.set_ylim(0,39);ax.axis('off')
    txt(ax,24,37,'A  Illustrative backward walk',10,weight='bold')
    points={'target':(6,19),'U':(26,28),'V':(43,28),'W':(26,10)}
    arrow(ax,(10,21),(22,26));arrow(ax,(30,28),(39,28));arrow(ax,(10,17),(22,12))
    for name,(x,y) in points.items():
        ax.add_patch(Circle((x,y),3.6,fc='white',ec=GRAY,lw=.9,zorder=2))
        txt(ax,x,y,'T' if name=='target' else name,10,zorder=3)
    txt(ax,15,28,'0.6',9);txt(ax,35,33,'1',9);txt(ax,15,10,'0.4',9)
    txt(ax,6,11,'target',9)
    txt(ax,25,1.8,'Arrows follow the walk toward sources',8.6)
    right=fig.add_axes([.69,.27,.275,.52])
    names=['Individual ranking\nU + V','Greedy coverage\nU + W']
    data=pd.DataFrame({'Method':names,'Coverage':[rank_cov,greedy_cov]})
    sns.barplot(data=data,x='Coverage',y='Method',hue='Method',order=names,
                hue_order=names,palette=[ORANGE,BLUE],dodge=False,legend=False,
                saturation=1,errorbar=None,width=.48,ax=right)
    for b,v in zip(right.patches,[rank_cov,greedy_cov]):
        right.text(v+.03,b.get_y()+b.get_height()/2,f'{v:.1f}',va='center',fontsize=9.5)
    right.set(xlim=(0,1.2),xticks=[0,.5,1],ylabel='',xlabel='Path coverage')
    right.grid(axis='y',visible=False);right.tick_params(axis='y',length=0)
    sns.despine(ax=right,left=True,bottom=True)
    fig.text(.54,.91,'B  Pick two features',fontsize=10,weight='bold')
    save(fig,'figure_5_path_overlap',out)

def figure6(out):
    fig,ax=canvas(3.95,72)
    txt(ax,0,69,'A  Four runs at the final prompt token',10.5,ha='left',weight='bold')
    txt(ax,32,62,'MLP output at layer 31',9.4)
    txt(ax,62,62,'Whole MLP output at layer 32',9.1)
    txt(ax,91,62,'Readout',9.4)
    rows=[('Clean',54,'Unedited','Compute and save '+r'$R_0$',r'$y_0$'),
          ('Sender edit',44,'Add '+r'$\Delta a\,d_C$','Recompute and save '+r'$R_s$',r'$y_s$'),
          ('Restore receiver',34,'Add '+r'$\Delta a\,d_C$','Replace with clean '+r'$R_0$',r'$y_r$'),
          ('Receiver rescue',24,'Unedited','Transplant edited '+r'$R_s$',r'$y_q$')]
    for label,y,s,r,o in rows:
        txt(ax,0,y,label,8.9,ha='left')
        box(ax,21,y-3.5,22,7,s,9.2)
        box(ax,48,y-3.5,32,7,r,9.2)
        arrow(ax,(43,y),(48,y));arrow(ax,(80,y),(87,y));txt(ax,91,y,o,11)
    for x,label,formula in [(10,'Total',r'$T=y_s-y_0$'),(36,'Dependence',r'$D=y_s-y_r$'),
                            (63,'Remainder',r'$B=y_r-y_0$'),(90,'Rescue',r'$Q=y_q-y_0$')]:
        txt(ax,x,16,label,9.3);txt(ax,x,12,formula,10)
    ax.plot([0,100],[8.7,8.7],color=LIGHT,lw=.7)
    txt(ax,0,5.3,'B  Where the dose comes from',10,ha='left',weight='bold')
    txt(ax,0,.8,'Natural activation: encoder + ReLU → dose may be zero',9.1,ha='left')
    txt(ax,100,.8,'Fixed dose: +18.625 in both groups',9.1,ha='right')
    save(fig,'figure_6_receiver_interventions',out)

def figure7(out):
    fig=plt.figure(figsize=(6.5,3.55))
    ax1=fig.add_axes([.19,.60,.23,.22]);ax2=fig.add_axes([.19,.22,.23,.22])
    ax3=fig.add_axes([.67,.24,.29,.58])
    fig.text(.20,.94,'A  Internal readouts',ha='center',weight='bold',fontsize=10.5)
    fig.text(.20,.885,'84 addition prompts',ha='center',fontsize=9.4)
    fig.text(.79,.94,'B  Generated answers',ha='center',weight='bold',fontsize=10.5)
    fig.text(.79,.885,'31 fictional compliances',ha='center',fontsize=9.4)
    arms=['a2_natural','rd_a2_natural'];names=['Add C','Refusal\ndirection']
    for ax,field,label,xmax in [(ax1,'median_delta_primary_score','Median next-token-score change',.245),
                               (ax2,'median_delta_probe','Median probe change',5.4)]:
        vals=[D['behavior']['internal'][k][field] for k in arms]
        assert all(D['behavior']['internal'][k]['n']==84 for k in arms)
        data=pd.DataFrame({'Arm':names,'Change':vals})
        sns.barplot(data=data,x='Change',y='Arm',hue='Arm',order=names,hue_order=names,
                    palette=[BLUE,ORANGE],dodge=False,legend=False,errorbar=None,
                    saturation=1,width=.46,ax=ax)
        ax.set(xlim=(0,xmax),ylabel='')
        ax.set_xlabel(label,fontsize=8.3,labelpad=6);ax.tick_params(axis='y',length=0)
        for b,v in zip(ax.patches,vals):
            ax.text(v+xmax*.035,b.get_y()+b.get_height()/2,f'+{v:.3f}',va='center',fontsize=9)
        ax.set_xticks([0,.1,.2] if xmax<1 else [0,2,4]);ax.grid(axis='y',visible=False)
        sns.despine(ax=ax,left=True,bottom=True)
    keys=['sufficiency_A2_fictional_complying','sufficiency_RB_A2_fictional_complying',
          'sufficiency_RD_A2_fictional_complying','sufficiency_RND_A2_fictional_complying']
    labels=['Add C','Receiver\nblocked','Refusal\ndirection','Random\ndirection']
    colors=[BLUE,GRAY,ORANGE,GRAY]
    rows=[]
    for y,k,color in zip([3,2,1,0],keys,colors):
        s=D['behavior']['primary'][k];v=s['estimate']*100;lo,hi=[a*100 for a in s['ci95']]
        assert s['n']==31
        # Frozen intervals are drawn explicitly; no Seaborn bootstrapping.
        ax3.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='none',color=color,capsize=3,lw=1.3,zorder=2)
        rows.append({'estimate':v,'row':y,'arm':k})
        ax3.text(v,y+.23,f'{s["count"]}/{s["n"]}',ha='center',va='bottom',fontsize=9.1)
    sns.scatterplot(data=pd.DataFrame(rows),x='estimate',y='row',hue='arm',
                    palette=dict(zip(keys,colors)),legend=False,s=30,edgecolor='white',
                    linewidth=.45,ax=ax3,zorder=3)
    ax3.set_yticks([3,2,1,0],labels);ax3.set_xlim(-1,35);ax3.set_ylim(-.45,3.6)
    ax3.set_xticks([0,10,20,30]);ax3.set_xlabel('Changed to refusal-like (%)',fontsize=9)
    ax3.set_ylabel('');ax3.tick_params(axis='y',length=0)
    ax3.grid(axis='y',visible=False);sns.despine(ax=ax3,left=True,bottom=True)
    fig.text(.5,.025,'Different eligible populations and units; no within-prompt score–behavior correlation is estimated.',ha='center',fontsize=8.7)
    save(fig,'figure_7_internal_and_behavioral',out)

def figure8(out):
    f=D['fidelity'];n=f['prompt_count'];assert n==8
    keys=['hf_vs_circuit_tracer_exact','circuit_tracer_exact_vs_strict_reconstruction']
    fig=plt.figure(figsize=(6.5,2.20))
    left=fig.add_axes([.35,.26,.31,.54]);right=fig.add_axes([.77,.26,.20,.54])
    labels=['Standard model vs\nexact Circuit Tracer','Exact Circuit Tracer vs\nstrict replacement']
    colors=[BLUE,ORANGE]
    rows=[]
    for i,(k,color) in enumerate(zip(keys,colors)):
        v=f[k]['mean_kl'];matches=round(f[k]['top1_agreement_rate']*n)
        rows.append({'comparison':k,'row':i,'kl':v,'matches':matches})
        left.text(v,i-.17,f'{v:.5f}' if v<1 else f'{v:.2f}',ha='center',fontsize=9.5)
        right.text(matches+.3,i,f'{matches}/{n}',va='center',fontsize=9.5)
        if matches==0:right.plot(0,i,'o',ms=4,color=color)
    data=pd.DataFrame(rows)
    sns.scatterplot(data=data,x='kl',y='row',hue='comparison',palette=dict(zip(keys,colors)),
                    legend=False,s=33,edgecolor='white',linewidth=.4,ax=left)
    sns.barplot(data=data,x='matches',y='comparison',hue='comparison',
                order=keys,hue_order=keys,palette=colors,dodge=False,legend=False,
                errorbar=None,saturation=1,width=.34,ax=right)
    left.set_xscale('log');left.set_xlim(.001,100);left.set_xticks([.001,.01,.1,1,10,100])
    left.xaxis.set_major_formatter(LogFormatterMathtext());left.minorticks_off()
    left.set_yticks([0,1],labels);left.set_xlabel('Mean KL (log scale)');left.set_ylabel('')
    right.set_xlim(-.3,10);right.set_xticks([0,4,8]);right.set_yticks([])
    right.set_xlabel('Top-1 matches');right.set_ylabel('')
    for ax in [left,right]:
        ax.set_ylim(1.45,-.55);ax.tick_params(axis='y',length=0)
        ax.grid(axis='y',visible=False);sns.despine(ax=ax,left=True,bottom=True)
    fig.text(.02,.94,'Fidelity on eight prompts',fontsize=10.5,weight='bold')
    save(fig,'figure_8_fidelity',out)

def verify_sources(earlier,behavioral):
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    for key,base in [('behavior',behavioral),('fidelity',earlier)]:
        item=manifest[key];p=base/item['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],str(p)
        raw=json.loads(p.read_text())
        if key=='behavior':
            for k,v in D['behavior']['primary'].items():assert v==raw['primary_agent_labels'][k]
            for k,v in D['behavior']['internal'].items():assert v==raw['numeric_continuity_median_deltas'][k]
            assert D['behavior']['payload_sha256']==raw['payload_sha256']
        else:
            for k,v in D['fidelity'].items():assert v==raw[k]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'rendered')
    p.add_argument('--earlier-bundle',type=Path);p.add_argument('--behavioral-addendum',type=Path)
    args=p.parse_args()
    if args.earlier_bundle or args.behavioral_addendum:
        assert args.earlier_bundle and args.behavioral_addendum
        verify_sources(args.earlier_bundle,args.behavioral_addendum)
    args.output.mkdir(parents=True,exist_ok=True)
    for fn in [figure4,figure5,figure6,figure7,figure8]:fn(args.output)
    manifest={'matplotlib':matplotlib.__version__,'numpy':np.__version__,
              'seaborn':sns.__version__,'pandas':pd.__version__,
              'plotting_context':'paper / whitegrid / colorblind; frozen values and intervals',
              'source_inputs_sha256':hashlib.sha256((ROOT/'inputs.json').read_bytes()).hexdigest(),
              'original_sources_verified':bool(args.earlier_bundle),
              'toy_computation':toy_values(),'outputs':OUTPUTS}
    (args.output/'new_figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
