"""PNG overview for verified cross-provider AI Race panels."""
import csv, json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'figures'; OUT.mkdir(exist_ok=True)
PANELS={'Luna vs Haiku':next((ROOT/'openai_claude').rglob('turns.jsonl')),
        'Gemini vs Haiku':next((ROOT/'gemini_claude').rglob('turns.jsonl')),
        'Gemini vs Luna':next((ROOT/'gemini_openai').rglob('turns.jsonl'))}
R=(.1,.6,.9)
COL=['#3977af','#d0643c','#4e9b6e']

def load(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def grouped_bars(metric,title,ylabel,name):
    fig,axs=plt.subplots(1,3,figsize=(14,4.4),sharey=True,layout='constrained')
    for ax,(label,path) in zip(axs,PANELS.items()):
        rows=load(path); players=sorted({x['player'] for x in rows}); x=np.arange(3)
        for i,player in enumerate(players):
            vals=[]
            for risk in R:
                q=[a for a in rows if a['player']==player and a['max_private_risk']==risk]
                raw=np.array([metric(a) for a in q],dtype=float)
                vals.append(np.nanmean(raw) if np.isfinite(raw).any() else np.nan)
            bars=ax.bar(x+(i-.5)*.34,vals,.34,label=player,color=COL[i])
            for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+.025,f'{v:.0%}',ha='center',fontsize=8)
        ax.set(title=label,xticks=x,xticklabels=R,ylim=(0,1),xlabel='Maximum private risk'); ax.grid(axis='y',alpha=.25); ax.legend(frameon=False,fontsize=8)
    axs[0].set_ylabel(ylabel); fig.suptitle(title,y=1.03,fontsize=14); fig.savefig(OUT/name,dpi=180,bbox_inches='tight'); plt.close(fig)

def n3_composition():
    p=next((ROOT/'gemini_openai_claude_n3').rglob('turns.jsonl')); rows=load(p); g=defaultdict(dict)
    for a in rows:g[(a['max_private_risk'],a['game_id'],a['round'])][a['player']]=a['action']
    fig,ax=plt.subplots(figsize=(7.6,4.5)); bottom=np.zeros(3)
    for k,(lab,col) in enumerate(zip(['3 SAFE','2 SAFE','1 SAFE','0 SAFE'],['#68a57a','#b8bd50','#e3a342','#c9514b'])):
        vals=[]
        for risk in R:
            sets=[v for (rr,_,_),v in g.items() if rr==risk]; vals.append(np.mean([sum(a=='safe' for a in s.values())==3-k for s in sets]))
        ax.bar(range(3),vals,bottom=bottom,label=lab,color=col); bottom+=vals
    ax.set(title='Three-provider joint-action composition',xticks=range(3),xticklabels=R,ylim=(0,1),xlabel='Maximum private risk',ylabel='Share of simultaneous decisions'); ax.legend(frameon=False,ncol=4); ax.grid(axis='y',alpha=.25); fig.savefig(OUT/'n3_joint_action_composition.png',dpi=180,bbox_inches='tight');plt.close(fig)

def n3_outcomes():
    p=next((ROOT/'gemini_openai_claude_n3').rglob('players.csv'))
    with p.open(encoding='utf-8') as h: rows=list(csv.DictReader(h))
    names=sorted({x['player'] for x in rows}); fig,axs=plt.subplots(1,3,figsize=(14,4.5),sharey=True,layout='constrained')
    for ax,risk in zip(axs,R):
        q=[[x for x in rows if x['player']==n and float(x['max_private_risk'])==risk] for n in names]; x=np.arange(3)
        ax.bar(x-.18,[np.mean([a['outcome'] in ('winner','tie') for a in z]) for z in q],.36,label='Prize recipient',color='#4e9b6e')
        ax.bar(x+.18,[np.mean([int(a['setback']) for a in z]) for z in q],.36,label='Setback',color='#c9514b')
        ax.set(title=f'Risk cap {risk}',xticks=x,xticklabels=names,ylim=(0,1));ax.tick_params(axis='x',rotation=18);ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.25)
    axs[0].set_ylabel('Share of games');fig.suptitle('Three-provider terminal outcomes',y=1.03,fontsize=14);fig.savefig(OUT/'n3_terminal_outcomes.png',dpi=180,bbox_inches='tight');plt.close(fig)

def pair_payoff():
    fig,axs=plt.subplots(1,3,figsize=(14,4.4),sharey=True,layout='constrained')
    for ax,(label,path) in zip(axs,PANELS.items()):
        rows=load(path); names=sorted({a['player'] for a in rows}); x=np.arange(3)
        for i,n in enumerate(names):
            vals=[np.mean([a['round_payoff'] for a in rows if a['player']==n and a['max_private_risk']==r]) for r in R]
            ax.plot(x,vals,marker=['o','s'][i],color=COL[i],label=n)
        ax.set(title=label,xticks=x,xticklabels=R,xlabel='Maximum private risk',ylim=(0,2.5));ax.grid(alpha=.25);ax.legend(frameon=False,fontsize=8)
    axs[0].set_ylim(0,2.5); axs[0].set_ylabel('Mean stage payoff');fig.suptitle('Immediate payoff under escalating private risk',y=1.03,fontsize=14);fig.savefig(OUT/'pair_stage_payoff.png',dpi=180,bbox_inches='tight');plt.close(fig)

def pair_trajectory():
    fig,axs=plt.subplots(1,3,figsize=(14,4.4),sharey=True,layout='constrained')
    styles={.1:'-',.6:'--',.9:':' }
    for ax,(label,path) in zip(axs,PANELS.items()):
        rows=load(path); names=sorted({a['player'] for a in rows})
        for i,n in enumerate(names):
            for risk in R:
                by=defaultdict(list)
                for a in rows:
                    if a['player']==n and a['max_private_risk']==risk:by[a['round']].append(a['unsafe'])
                x=sorted(by);ax.plot(x,[np.mean(by[j]) for j in x],color=COL[i],linestyle=styles[risk],marker='o',ms=2.5,label=f'{n} · {risk}')
        ax.set(title=label,xlabel='Round',ylim=(0,1));ax.grid(alpha=.25);ax.legend(frameon=False,fontsize=6)
    axs[0].set_ylabel('Unsafe choice rate');fig.suptitle('Within-race aggression trajectory (solid .1, dash .6, dot .9)',y=1.03,fontsize=14);fig.savefig(OUT/'pair_unsafe_trajectory.png',dpi=180,bbox_inches='tight');plt.close(fig)

def n3_individual_behavior():
    p=next((ROOT/'gemini_openai_claude_n3').rglob('turns.jsonl')); rows=load(p); names=sorted({a['player'] for a in rows});x=np.arange(3)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for i,n in enumerate(names):
        vals=[np.mean([a['unsafe'] for a in rows if a['player']==n and a['max_private_risk']==r]) for r in R]
        ax.plot(x,vals,marker=['o','s','^'][i],label=n,color=COL[i])
    ax.set(title='Three-provider risk response',xticks=x,xticklabels=R,ylim=(0,1),xlabel='Maximum private risk',ylabel='Unsafe choice rate');ax.grid(alpha=.25);ax.legend(frameon=False);fig.savefig(OUT/'n3_individual_unsafe_rate.png',dpi=180,bbox_inches='tight');plt.close(fig)

def n3_group_response():
    p=next((ROOT/'gemini_openai_claude_n3').rglob('turns.jsonl')); rows=load(p); names=sorted({a['player'] for a in rows});fig,axs=plt.subplots(1,3,figsize=(14,4.4),sharey=True,layout='constrained')
    for ax,risk in zip(axs,R):
        for i,n in enumerate(names):
            vals=[]
            for k in range(3):
                q=[a['unsafe'] for a in rows if a['player']==n and a['max_private_risk']==risk and a['own_prev_action'] is not None and sum(z=='unsafe' for z in a['others_prev_actions'])==k]
                vals.append(np.mean(q) if q else np.nan)
            ax.plot(range(3),vals,marker=['o','s','^'][i],color=COL[i],label=n)
        ax.set(title=f'Risk cap {risk}',xticks=range(3),xticklabels=['0','1','2'],ylim=(0,1),xlabel='Unsafe opponents in previous round');ax.grid(alpha=.25);ax.legend(frameon=False,fontsize=8)
    axs[0].set_ylabel('Next Unsafe probability');fig.suptitle('Three-provider response to group escalation',y=1.03,fontsize=14);fig.savefig(OUT/'n3_group_escalation_response.png',dpi=180,bbox_inches='tight');plt.close(fig)

def ending_rounds():
    sources={**PANELS, 'Luna vs Haiku vs Gemini':next((ROOT/'gemini_openai_claude_n3').rglob('turns.jsonl'))}
    fig,axs=plt.subplots(2,2,figsize=(11,7),sharey=True,layout='constrained')
    for ax,(label,turns) in zip(axs.flat,sources.items()):
        races=turns.with_name('races.csv')
        with races.open(encoding='utf-8') as h: rows=list(csv.DictReader(h))
        vals=[[int(a['n_rounds']) for a in rows if float(a['max_private_risk'])==risk] for risk in R]
        ax.boxplot(vals,positions=range(3),widths=.55,showmeans=True,meanprops={'marker':'D','markerfacecolor':'black','markersize':4})
        means=[np.mean(v) for v in vals]
        ax.plot(range(3),means,color='#3977af',marker='o',label='Observed mean')
        ax.axhline(9,color='#c9514b',linestyle='--',label='Theoretical expectation (9)')
        for i,v in enumerate(means): ax.text(i,v+.35,f'{v:.1f}',ha='center',fontsize=8)
        ax.set(title=label,xticks=range(3),xticklabels=R,xlabel='Maximum private risk',ylim=(4,24));ax.grid(axis='y',alpha=.25);ax.legend(frameon=False,fontsize=8)
    axs[0,0].set_ylabel('End round');axs[1,0].set_ylabel('End round')
    fig.suptitle('Observed end rounds versus the geometric stopping-rule expectation',y=1.02,fontsize=14)
    fig.savefig(OUT/'expected_end_rounds.png',dpi=180,bbox_inches='tight');plt.close(fig)

grouped_bars(lambda a:a['unsafe'],'Unsafe choice rate by matchup and risk','Unsafe choice rate','pair_unsafe_rate.png')
grouped_bars(lambda a:a['unsafe'] if a['opponent_prev_action']=='unsafe' else np.nan,'Reciprocity after opponent chose Unsafe','Unsafe choice rate','pair_reciprocity.png')
n3_composition();n3_outcomes();pair_payoff();pair_trajectory();n3_individual_behavior();n3_group_response();ending_rounds()
