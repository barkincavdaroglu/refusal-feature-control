#!/usr/bin/env python3
"""Rebuild Figures 1-3 from frozen results; Seaborn performs no estimation."""
from pathlib import Path
import json, os, hashlib
os.environ.setdefault('MPLCONFIGDIR', '/tmp/refusal-seaborn-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'submission/figures'
BLUE, ORANGE = sns.color_palette('colorblind')[:2]
GRAY = '#8c8c8c'
PALETTE = {'AtP': BLUE, 'Graph': ORANGE, 'Tie': GRAY}

def read(p):
    return json.loads((ROOT/p).read_text())

def theme():
    sns.set_theme(context='paper', style='whitegrid', palette='colorblind',
                  font='DejaVu Sans', font_scale=1,
                  rc={'font.size':9, 'axes.labelsize':9, 'axes.titlesize':9.5,
                      'xtick.labelsize':8.5, 'ytick.labelsize':8.5,
                      'grid.color':'#e4e4e4', 'grid.linewidth':.6,
                      'axes.linewidth':.7, 'svg.fonttype':'none',
                      'svg.hashsalt':'refusal-seaborn-v1'})

def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(OUT/(name+'.'+ext), dpi=240, facecolor='white',
                    metadata={'Date':None} if ext=='svg' else {})
    plt.close(fig)

def bars(ax, names, values, palette):
    data = pd.DataFrame({'method':names, 'value':values})
    sns.barplot(data=data, x='method', y='value', hue='method', order=names,
                hue_order=names, palette=palette, dodge=False, legend=False,
                errorbar=None, saturation=1, width=.62, ax=ax)
    ax.set_xlabel(''); ax.set_ylabel(''); ax.grid(axis='x', visible=False)
    sns.despine(ax=ax, left=True, bottom=True)
    return [p for p in ax.patches if p.get_width() > 0]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    theme()
    rep = read('ALL WORK/steps/09-intervention-method-exploration/outputs/s9_e8b_analysis.json')['hierarchical_decision']
    fixed = read('ALL WORK/steps/09-intervention-method-exploration/outputs/s9_e9c_analysis.json')['primary']
    wins = [rep['receiver_dependence']['compound_wins'],rep['receiver_rescue']['compound_joint_wins'],
            fixed['harmful_dependence']['wins'],fixed['harmful_rescue']['wins']]
    den = [10,10,12,12]
    labels = ['Replication\ndependence','Replication\nrescue','Fixed dose\ndependence','Fixed dose\nrescue']
    fig = plt.figure(figsize=(6.5,3.1))
    left = fig.add_axes([.18,.22,.27,.57]);right = fig.add_axes([.72,.22,.255,.57])
    data = pd.DataFrame({'Test':labels,'Fraction':np.array(wins)/den})
    sns.barplot(data=data,x='Fraction',y='Test',order=labels,color=BLUE,
                errorbar=None,saturation=1,width=.55,ax=left)
    for b,a,n in zip(left.patches,wins,den):
        left.text(b.get_width()+.035,b.get_y()+b.get_height()/2,f'{a}/{n}',va='center',fontsize=9)
    left.set(xlim=(0,1.25),xticks=[0,.5,1],ylabel='',xlabel='Harmful units passing')
    left.set_xticklabels(['0','½','1'])
    vals = [fixed['risk_differential']['paired_D_positive'],fixed['risk_differential']['paired_Q_positive'],fixed['benign_relative_smallness']['wins']]
    labels2 = ['Harmful > benign\ndependence','Harmful > benign\nrescue','Benign ≤ ¼\nof harmful']
    data = pd.DataFrame({'Test':labels2,'Pairs':vals})
    sns.barplot(data=data,x='Pairs',y='Test',order=labels2,color=ORANGE,
                errorbar=None,saturation=1,width=.45,ax=right)
    for b,v in zip(right.patches,vals):
        right.text(v+.25,b.get_y()+b.get_height()/2,f'{v}/12',va='center',fontsize=9)
    right.axvline(10,color='#666666',ls='--',lw=1)
    right.set(xlim=(0,12.4),xticks=[0,4,8,12],ylabel='',xlabel='Topic pairs passing')
    right.text(.99,1.06,'Dashed line: required 10/12',ha='right',va='bottom',
               fontsize=8.3,transform=right.transAxes)
    right.set_ylim(2.65,-.5)
    for ax in (left,right):
        ax.grid(axis='y',visible=False);ax.tick_params(axis='y',length=0,pad=4)
        sns.despine(ax=ax,left=True,bottom=True)
    fig.text(.035,.93,'A  Dependence and rescue repeat',fontsize=9.5,weight='bold')
    fig.text(.555,.93,'B  Selectivity gates fail',fontsize=9.5,weight='bold')
    save(fig,'figure_3_route_and_selectivity')

if __name__ == '__main__': main()
