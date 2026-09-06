"""Build five visually distinct, approval-only story figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# Type 3 fonts are rejected by publishers and are matplotlib's default.
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
OUT=HERE/"top5_story_figures_20260803"; OUT.mkdir(parents=True,exist_ok=True); D=HERE/"data"
INK="#142038"; MUTED="#5B6A82"; GRID="#DFE6EF"; BLUE="#315EF4"; TEAL="#0F9D8B"; GOLD="#E9A23B"; PINK="#D34A91"; PURPLE="#7C3AED"
SEQ=LinearSegmentedColormap.from_list("unsafe",["#EEF8F6","#BCE5DD","#F3C96F","#E75A55"])
N={"gpt-5-nano":"GPT-5 nano","gpt-5.4-nano":"GPT-5.4 nano","google/gemini-3-flash-preview":"Gemini 3 Flash","google/gemini-3.1-flash-lite-preview":"Gemini 3.1 Lite","google/gemini-3.5-flash-lite":"Gemini 3.5 Lite","gpt-5.6-luna":"GPT-5.6 Luna","gpt-5.6-terra":"GPT-5.6 Terra","claude-opus-5":"Claude Opus 5","claude-sonnet-5":"Claude Sonnet 5"}

def setup():
 plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"text.color":INK,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,"axes.edgecolor":GRID,"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white","savefig.bbox":"tight"})
def save(fig,name):
 fig.savefig(OUT/f"{name}.png",dpi=240,pad_inches=.13); fig.savefig(OUT/f"{name}.pdf",pad_inches=.13); plt.close(fig)
def tile(ax,a,rows,cols,fs=8):
 a=np.asarray(a,float); ax.imshow(a,aspect="auto",cmap=SEQ,vmin=0,vmax=1)
 for i in range(a.shape[0]):
  for j in range(a.shape[1]):
   ax.text(j,i,f"{a[i,j]:.0%}",ha="center",va="center",weight="bold",fontsize=fs,color="white" if a[i,j]>.76 else INK)
 ax.set_yticks(range(len(rows)),rows); ax.set_xticks(range(len(cols)),cols); ax.tick_params(length=0,pad=5)
 ax.set_xticks(np.arange(-.5,len(cols),1),minor=True); ax.set_yticks(np.arange(-.5,len(rows),1),minor=True); ax.grid(which="minor",color="white",lw=3); ax.tick_params(which="minor",length=0); ax.spines[:].set_visible(False)

def story1(risk):
 human=json.load(open(D/"human_vs_llm_validation.json"))["fig2a_mean_by_risk"]
 egt=pd.read_csv(ROOT/"results/open_source/egt_reproduction/egt_stationary_summary.csv"); ev=egt[egt.regime=="main_reference"].sort_values("max_private_risk").unsafe_frequency_mean.to_numpy()
 pts=[("EGT",ev,PURPLE),("Human",np.array([human[str(x)] for x in(.1,.6,.9)]),GOLD)]
 for m,s in risk.groupby("model"): pts.append((N[m],s.sort_values("max_private_risk").mean_unsafe_rate.to_numpy(),TEAL))
 fig=plt.figure(figsize=(12.4,6.8)); gs=fig.add_gridspec(1,2,width_ratios=[1.12,1],left=.08,right=.97,top=.80,bottom=.14,wspace=.28); ah=fig.add_subplot(gs[0]); ap=fig.add_subplot(gs[1])
 vals=np.vstack([p[1] for p in pts]); tile(ah,vals,[p[0] for p in pts],["risk .1","risk .6","risk .9"],7.6); ah.axhline(1.5,color=INK,lw=1.2); ah.set_title("Absolute Unsafe rate",loc="left",weight="bold",pad=12)
 key={"EGT","Human","Claude Opus 5","Claude Sonnet 5","GPT-5 nano"}
 for lab,v,c in pts:
  x=v[1]-v[0]; y=v[2]-v[1]; ap.scatter(x,y,s=90,color=c,edgecolor="white",lw=1.2,zorder=3)
  if lab in key: ap.annotate(lab,(x,y),xytext=(5,4),textcoords="offset points",fontsize=7.5,weight="bold" if lab in {"EGT","Claude Opus 5"} else "normal")
 ap.axhline(0,color=INK,lw=1); ap.axvline(0,color=INK,lw=1); ap.grid(color=GRID,lw=.7); ap.set_axisbelow(True); ap.spines[["top","right"]].set_visible(False); ap.set_xlim(-1.05,.12); ap.set_ylim(-1.05,.12); ap.xaxis.set_major_formatter(PercentFormatter(1)); ap.yaxis.set_major_formatter(PercentFormatter(1)); ap.set_xlabel("Early transition · .1→.6"); ap.set_ylabel("Late transition · .6→.9"); ap.set_title("Where the policy changes",loc="left",weight="bold",pad=12)
 fig.suptitle("Story 1 · The same risk signal creates different policy topologies",x=.04,ha="left",fontsize=19,weight="bold",y=.965); fig.text(.04,.89,"EGT changes late, Claude changes early, human behavior plateaus, and GPT-5 nano slightly reverses",color=MUTED,fontsize=10.4); fig.text(.04,.035,"Aligned descriptively on Unsafe rate; EGT, human, and LLM self-play remain different estimands.",color=MUTED,fontsize=8.5); save(fig,"01_risk_topology_story")

def story2(persona):
 p=persona[persona.role!="none"].pivot(index="model",columns="role",values="mean_unsafe_rate")[[f"R{i}" for i in range(1,7)]]; p=p.loc[p.assign(jump=lambda x:x.R3-x.R2).sort_values("jump",ascending=False).index]
 fig=plt.figure(figsize=(11.6,6.4)); gs=fig.add_gridspec(1,2,width_ratios=[2.7,1],left=.19,right=.96,top=.79,bottom=.14,wspace=.12); ax=fig.add_subplot(gs[0]); ar=fig.add_subplot(gs[1],sharey=ax)
 tile(ax,p.to_numpy(),[N[x] for x in p.index],list(p.columns)); ax.axvline(1.5,color=PURPLE,lw=3); ax.text(1.58,len(p)-.1,"R2→R3 boundary",ha="left",va="bottom",rotation=90,color=PURPLE,weight="bold",fontsize=7.5); ax.set_title("Persona control surface",loc="left",weight="bold",pad=12)
 jumps=(p.R3-p.R2).to_numpy(); y=np.arange(len(p)); ar.hlines(y,0,jumps,color="#D9D0FA",lw=7); ar.scatter(jumps,y,s=70,color=PURPLE,edgecolor="white"); ar.axvline(0,color=INK,lw=1); ar.set_xlim(0,.52); ar.xaxis.set_major_formatter(PercentFormatter(1)); ar.grid(axis="x",color=GRID,lw=.7); ar.spines[["top","right","left"]].set_visible(False); ar.tick_params(axis="y",left=False,labelleft=False); ar.set_xlabel("R3 − R2 jump"); ar.set_title("Transition magnitude",loc="left",weight="bold",pad=12)
 fig.suptitle("Story 2 · Persona acts like a policy switch, not cosmetic wording",x=.04,ha="left",fontsize=19,weight="bold",y=.965); fig.text(.04,.89,"Several checkpoints cross a sharp behavioral boundary between R2 and R3",color=MUTED,fontsize=10.4); save(fig,"02_persona_phase_switch")

def story3(pbr):
 models=["claude-opus-5","gpt-5.6-terra","gpt-5.6-luna","claude-sonnet-5"]
 fig,axes=plt.subplots(1,4,figsize=(13,4.9),sharey=True); risks=[.1,.6,.9]
 for ax,m in zip(axes,models):
  s=pbr[(pbr.model==m)&(pbr.role.isin(["R1","R6"]))].pivot(index="max_private_risk",columns="role",values="mean_unsafe_rate").reindex(risks)
  y=np.arange(3); ax.hlines(y,s.R1,s.R6,color="#E7D7E2",lw=8,zorder=1); ax.scatter(s.R1,y,s=78,color=TEAL,label="R1",zorder=2,edgecolor="white"); ax.scatter(s.R6,y,s=78,color=PINK,label="R6",zorder=2,edgecolor="white")
  for i,r in enumerate(risks): ax.text(s.loc[r,"R1"],i-.14,f"{s.loc[r,'R1']:.0%}",ha="center",va="bottom",fontsize=7,color=TEAL,weight="bold"); ax.text(s.loc[r,"R6"],i+.14,f"{s.loc[r,'R6']:.0%}",ha="center",va="top",fontsize=7,color=PINK,weight="bold")
  ax.set_xlim(-.05,1.05); ax.set_ylim(2.42,-.42); ax.set_xticks([0,.5,1]); ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.set_yticks(y,["risk .1","risk .6","risk .9"]); ax.grid(axis="x",color=GRID,lw=.7); ax.spines[["top","right","left"]].set_visible(False); ax.tick_params(axis="y",length=0); ax.set_title(N[m],weight="bold",fontsize=10); ax.set_xlabel("Unsafe rate")
 axes[-1].legend(frameon=False,loc="lower right"); fig.suptitle("Story 3 · A strong persona can erase payoff-risk deterrence",x=.04,ha="left",fontsize=19,weight="bold",y=1.02); fig.text(.04,.94,"At risk .9, Opus moves 1%→99% and Terra 0%→98% when the assigned role changes from R1 to R6",color=MUTED,fontsize=10.2); fig.subplots_adjust(top=.80,bottom=.18,wspace=.18); save(fig,"03_persona_overrides_risk")

def story4(risk,persona,nsrc):
 sets=[]
 sets.append(("Checkpoint\nat fixed risk",risk.groupby("max_private_risk").mean_unsafe_rate.agg(lambda x:x.max()-x.min()).to_numpy(),PURPLE))
 sets.append(("Persona\nR1–R6",persona[persona.role!="none"].groupby("model").mean_unsafe_rate.agg(lambda x:x.max()-x.min()).to_numpy(),PINK))
 sets.append(("Risk\n.1–.9",risk.groupby("model").mean_unsafe_rate.agg(lambda x:x.max()-x.min()).to_numpy(),TEAL))
 sets.append(("Group size\nN=3–5",nsrc.groupby(["model","risk"])["mean"].agg(lambda x:x.max()-x.min()).to_numpy(),GOLD))
 fig,ax=plt.subplots(figsize=(10.4,5.8)); rng=np.random.default_rng(31)
 for i,(lab,v,c) in enumerate(sets):
  y=np.full(len(v),i)+rng.uniform(-.11,.11,len(v)); ax.scatter(v,y,s=58,color=c,alpha=.74,edgecolor="white"); med=np.median(v); ax.plot([med,med],[i-.28,i+.28],color=INK,lw=3); ax.text(med+.025,i-.28,f"median {med:.0%}",weight="bold",fontsize=8.5)
 ax.set_yticks(range(4),[x[0] for x in sets]); ax.set_xlim(-.02,1.05); ax.set_ylim(3.55,-.55); ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis="x",color=GRID,lw=.75); ax.set_axisbelow(True); ax.spines[["top","right","left"]].set_visible(False); ax.tick_params(axis="y",length=0); ax.set_xlabel("Descriptive span in Unsafe rate")
 ax.set_title("Story 4 · Identity and persona dominate the current structural perturbations",loc="left",fontsize=17,weight="bold",pad=18); ax.text(0,1.02,"Each dot is an eligible model or model×risk cell; medians compare descriptive scales, not pooled causal effects",transform=ax.transAxes,color=MUTED); save(fig,"04_control_layer_effect_spans")

def story5(peer,social):
 fig=plt.figure(figsize=(11.8,5.8)); gs=fig.add_gridspec(1,2,width_ratios=[1,1.35],left=.14,right=.97,top=.78,bottom=.15,wspace=.38); ap=fig.add_subplot(gs[0]); ad=fig.add_subplot(gs[1])
 peer=peer.copy(); peer["row"]=peer.model.map(N)+" · "+peer.persona_role; pm=peer.pivot(index="row",columns="peer_adv",values="mean"); tile(ap,pm.to_numpy(),pm.index,["0 peers","1 peer","2 peers"]); ap.set_title("N=3 · change peers",loc="left",weight="bold",pad=11)
 social=social.copy(); social["own"]=social.own_adversarial.map({0:"own cooperative",1:"own adversarial"}); own=social.groupby(["model","own"])["mean"].mean().unstack(); own.index=[N[x] for x in own.index]; own=own.sort_values("own adversarial")
 y=np.arange(len(own)); ad.hlines(y,own["own cooperative"],own["own adversarial"],color="#E7D7E2",lw=7); ad.scatter(own["own cooperative"],y,s=65,color=TEAL,label="own cooperative",edgecolor="white"); ad.scatter(own["own adversarial"],y,s=65,color=PINK,label="own adversarial",edgecolor="white"); ad.set_yticks(y,own.index); ad.set_xlim(-.03,1.03); ad.xaxis.set_major_formatter(PercentFormatter(1)); ad.grid(axis="x",color=GRID,lw=.7); ad.spines[["top","right","left"]].set_visible(False); ad.tick_params(axis="y",length=0); ad.set_xlabel("Unsafe rate"); ad.set_title("N=2 · change own role",loc="left",weight="bold",pad=11); ad.legend(frameon=False,loc="lower right")
 fig.suptitle("Story 5 · Agents follow assigned identity more than group ecology",x=.04,ha="left",fontsize=19,weight="bold",y=.965); fig.text(.04,.89,"Peer composition moves behavior modestly; changing the focal agent’s own role moves it across much of the scale",color=MUTED,fontsize=10.2); fig.text(.04,.035,"Panels use different dyadic and N-player protocols and are linked as a qualitative control hierarchy, not one estimand.",color=MUTED,fontsize=8.4); save(fig,"05_identity_vs_group_ecology")

if __name__=="__main__":
 setup(); risk=pd.read_csv(D/"risk_response_by_model.csv"); persona=pd.read_csv(D/"persona_role_gradient_extended.csv"); pbr=pd.read_csv(D/"persona_by_risk_extended.csv"); nsrc=pd.read_csv(HERE/"figure_candidates_20260802/candidate_fig6_source.csv"); peer=pd.read_csv(D/"nplayer_peer_composition.csv"); social=pd.read_csv(D/"social_persona_cell_means.csv")
 story1(risk); story2(persona); story3(pbr); story4(risk,persona,nsrc); story5(peer,social); print(OUT)
