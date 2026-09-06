"""Generate a diverse, approval-only deep-insight gallery from reviewed summaries."""
from pathlib import Path
import json, textwrap
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# Type 3 fonts are rejected by publishers and are matplotlib's default.
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "deep_insight_gallery_20260802"
OUT.mkdir(parents=True, exist_ok=True)
D = HERE / "data"
INK, MUTED, GRID = "#172033", "#5B6980", "#DEE5EE"
BLUE, TEAL, GOLD, PINK, PURPLE = "#315EF4", "#0F9D8B", "#E9A23B", "#D34A91", "#7C3AED"
SEQ = LinearSegmentedColormap.from_list("seq", ["#EEF8F6", "#BCE5DD", "#F2C66D", "#E65A55"])
DIV = LinearSegmentedColormap.from_list("div", [BLUE, "#F4F6F9", PINK])
NAMES={"gpt-5-nano":"GPT-5 nano","gpt-5.4-nano":"GPT-5.4 nano","google/gemini-3-flash-preview":"Gemini 3 Flash","google/gemini-3.1-flash-lite-preview":"Gemini 3.1 Lite","google/gemini-3.5-flash-lite":"Gemini 3.5 Lite","gpt-5.6-luna":"GPT-5.6 Luna","gpt-5.6-terra":"GPT-5.6 Terra","claude-opus-5":"Claude Opus 5","claude-sonnet-5":"Claude Sonnet 5","human":"Human"}

def setup():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"text.color":INK,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,"axes.edgecolor":GRID,"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white","savefig.bbox":"tight"})

def finish(fig, name):
    fig.savefig(OUT/f"{name}.png", dpi=210, pad_inches=.12); fig.savefig(OUT/f"{name}.pdf", pad_inches=.12); plt.close(fig)

def clean(ax, grid="x"):
    ax.spines[["top","right","left"]].set_visible(False); ax.tick_params(axis="y", length=0)
    if grid: ax.grid(axis=grid, color=GRID, lw=.75); ax.set_axisbelow(True)

def matrix(df, title, name, fmt=".0%", diverge=False, subtitle=""):
    fig, ax=plt.subplots(figsize=(max(6.8,.72*df.shape[1]+3.4), max(3.2,.43*df.shape[0]+1.8)))
    vals=df.to_numpy(float); norm=TwoSlopeNorm(vmin=np.nanmin(vals),vcenter=0,vmax=np.nanmax(vals)) if diverge and np.nanmin(vals)<0<np.nanmax(vals) else None
    ax.imshow(vals, aspect="auto", cmap=DIV if diverge else SEQ, vmin=None if norm else (0 if not diverge else None), vmax=None if norm else (1 if not diverge else None), norm=norm)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v=vals[i,j]
            if np.isfinite(v): ax.text(j,i,format(v,fmt),ha="center",va="center",weight="bold",fontsize=8,color="white" if (not diverge and v>.76) or (diverge and abs(v)>.65*np.nanmax(abs(vals))) else INK)
    ax.set_xticks(range(df.shape[1]),df.columns); ax.set_yticks(range(df.shape[0]),df.index); ax.tick_params(length=0,pad=5)
    ax.set_xticks(np.arange(-.5,df.shape[1],1),minor=True); ax.set_yticks(np.arange(-.5,df.shape[0],1),minor=True); ax.grid(which="minor",color="white",lw=3); ax.tick_params(which="minor",length=0); ax.spines[:].set_visible(False)
    ax.set_title(title,loc="left",fontsize=15,weight="bold",pad=22); ax.text(0,1.02,subtitle,transform=ax.transAxes,color=MUTED)
    finish(fig,name)

def main():
    risk=pd.read_csv(D/"risk_response_by_model.csv")
    rp=risk.pivot(index="model",columns="max_private_risk",values="mean_unsafe_rate")
    rp.index=[NAMES[x] for x in rp.index]; rp.columns=["risk .1","risk .6","risk .9"]
    matrix(rp,"01 · Cross-model risk fingerprints","01_risk_fingerprint_matrix",subtitle="Dyadic neutral-prompt condition means")

    # 02 threshold topology phase portrait
    human=json.load(open(D/"human_vs_llm_validation.json"))["fig2a_mean_by_risk"]
    egt=pd.read_csv(ROOT/"results/open_source/egt_reproduction/egt_stationary_summary.csv"); egt=egt[egt.regime=="main_reference"].sort_values("max_private_risk")
    pts=[]
    for label,vals,kind in [("EGT",egt.unsafe_frequency_mean.to_numpy(),"reference"),("Human",np.array([human[str(x)] for x in (.1,.6,.9)]),"human")]: pts.append((label,vals[1]-vals[0],vals[2]-vals[1],kind))
    for m,s in risk.groupby("model"):
        v=s.sort_values("max_private_risk").mean_unsafe_rate.to_numpy(); pts.append((NAMES[m],v[1]-v[0],v[2]-v[1],"LLM"))
    fig,ax=plt.subplots(figsize=(8.8,6.5)); colors={"reference":PURPLE,"human":GOLD,"LLM":TEAL}
    for lab,x,y,k in pts:
        ax.scatter(x,y,s=95,color=colors[k],edgecolor="white",lw=1.3,zorder=3); ax.annotate(lab,(x,y),xytext=(5,5),textcoords="offset points",fontsize=7.7)
    ax.axhline(0,color=INK,lw=1); ax.axvline(0,color=INK,lw=1); ax.set_xlim(-1.05,.12); ax.set_ylim(-1.05,.12); ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1)); clean(ax,"both")
    ax.set_xlabel("Early transition: Unsafe(.6) − Unsafe(.1)"); ax.set_ylabel("Late transition: Unsafe(.9) − Unsafe(.6)"); ax.set_title("02 · Risk-response topology separates early, late, flat, and reversed policies",loc="left",fontsize=14,weight="bold",pad=15); finish(fig,"02_risk_transition_phase_portrait")

    delta=(rp["risk .9"]-rp["risk .1"]).sort_values(); fig,ax=plt.subplots(figsize=(8,5.4)); c=[TEAL if v<0 else GOLD for v in delta]; ax.barh(delta.index,delta.values,color=c); ax.axvline(0,color=INK,lw=1); ax.xaxis.set_major_formatter(PercentFormatter(1)); clean(ax); ax.set_title("03 · Total disclosed-risk response varies from −100 to +3 pp",loc="left",fontsize=14,weight="bold"); ax.set_xlabel("Unsafe(.9) − Unsafe(.1)"); finish(fig,"03_total_risk_response_rank")

    fig,ax=plt.subplots(figsize=(8,5.5)); means=rp.mean(axis=1); spans=rp.max(axis=1)-rp.min(axis=1); ax.scatter(means,spans,s=95,color=PURPLE,alpha=.85,edgecolor="white");
    for lab,x,y in zip(means.index,means,spans): ax.annotate(lab,(x,y),xytext=(5,4),textcoords="offset points",fontsize=7.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1)); clean(ax,"both"); ax.set_xlabel("Mean Unsafe level"); ax.set_ylabel("Within-model risk span"); ax.set_title("04 · Mean riskiness does not predict risk sensitivity",loc="left",fontsize=14,weight="bold"); finish(fig,"04_level_vs_risk_sensitivity")

    persona=pd.read_csv(D/"persona_role_gradient_extended.csv"); pp=persona[persona.role!="none"].pivot(index="model",columns="role",values="mean_unsafe_rate"); pp=pp[[f"R{i}" for i in range(1,7)]]; pp.index=[NAMES[x] for x in pp.index]
    matrix(pp,"05 · Persona control surface","05_persona_gradient_matrix",subtitle="R1 is least risk-seeking; R6 is most risk-seeking")
    lev=(pp.R6-pp.R1).sort_values(); fig,ax=plt.subplots(figsize=(7.7,4.8)); ax.barh(lev.index,lev,color=PINK); ax.xaxis.set_major_formatter(PercentFormatter(1)); clean(ax); ax.set_title("06 · Persona leverage reaches almost the full policy scale",loc="left",fontsize=14,weight="bold"); ax.set_xlabel("R6 − R1 Unsafe rate"); finish(fig,"06_persona_leverage_rank")
    jump=(pp.R3-pp.R2).sort_values(); fig,ax=plt.subplots(figsize=(7.7,4.8)); ax.barh(jump.index,jump,color=PURPLE); ax.xaxis.set_major_formatter(PercentFormatter(1)); clean(ax); ax.set_title("07 · The largest persona phase transition occurs at R2→R3",loc="left",fontsize=14,weight="bold"); ax.set_xlabel("R3 − R2 Unsafe rate"); finish(fig,"07_persona_r2_r3_transition")
    neutral=persona[persona.role=="none"].set_index("model").mean_unsafe_rate.rename(index=NAMES); common=pp.index.intersection(neutral.index); fig,ax=plt.subplots(figsize=(7.8,5.7)); ax.scatter(neutral[common],pp.loc[common,"R6"],s=105,color=PINK,edgecolor="white"); ax.plot([0,1],[0,1],color=INK,ls="--",lw=1)
    for m in common: ax.annotate(m,(neutral[m],pp.loc[m,"R6"]),xytext=(5,4),textcoords="offset points",fontsize=7.5)
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1)); clean(ax,"both"); ax.set_xlabel("Neutral Unsafe rate"); ax.set_ylabel("R6 Unsafe rate"); ax.set_title("08 · Strong risk persona pushes every checkpoint above neutral",loc="left",fontsize=14,weight="bold"); finish(fig,"08_neutral_vs_r6")

    pbr=pd.read_csv(D/"persona_by_risk_extended.csv")
    for idx,m in enumerate(pbr.model.unique(),start=9):
        q=pbr[pbr.model==m].pivot(index="role",columns="max_private_risk",values="mean_unsafe_rate").reindex([f"R{i}" for i in range(1,7)]); q.columns=["risk .1","risk .6","risk .9"]
        matrix(q,f"{idx:02d} · Persona × risk: {NAMES[m]}",f"{idx:02d}_persona_risk_{m.replace('/','_').replace('.','_')}",subtitle="Persona can attenuate or overwhelm payoff-risk deterrence")

    nsrc=pd.read_csv(OUT.parent/"figure_candidates_20260802/candidate_fig6_source.csv")
    nr=[]
    for m in ("gpt-5-nano","gpt-5.4-nano"):
        for rv in (.1,.6,.9): nr.append(pd.Series(nsrc[(nsrc.model==m)&(nsrc.risk==rv)].sort_values("n_players")["mean"].to_numpy(),index=["N=3","N=4","N=5"],name=f"{NAMES[m]} · r={rv}"))
    nm=pd.DataFrame(nr); matrix(nm,"13 · Harmonized N-player robustness","13_nplayer_matrix",subtitle="Player-race means; N=2 is a separate dyadic protocol")
    nrange=nm.max(axis=1)-nm.min(axis=1); fig,ax=plt.subplots(figsize=(8,4.7)); ax.barh(nrange.sort_values().index,nrange.sort_values(),color=GOLD); ax.xaxis.set_major_formatter(PercentFormatter(1)); clean(ax); ax.set_title("14 · N=3–5 variation remains modest within each checkpoint+risk cell",loc="left",fontsize=13.5,weight="bold"); ax.set_xlabel("Maximum minus minimum Unsafe rate across N"); finish(fig,"14_nplayer_range_rank")
    contrasts=pd.DataFrame({"N4−N3":nm["N=4"]-nm["N=3"],"N5−N3":nm["N=5"]-nm["N=3"]}); matrix(contrasts,"15 · N-player contrasts are non-monotone","15_nplayer_contrast_matrix",fmt="+.0%",diverge=True,subtitle="Signed descriptive differences; no causal group-size claim")

    peer=pd.read_csv(D/"nplayer_peer_composition.csv"); peer["row"]=peer.model.map(NAMES)+" · "+peer.persona_role; peerm=peer.pivot(index="row",columns="peer_adv",values="mean"); peerm.columns=["0 adversarial peers","1 adversarial peer","2 adversarial peers"]; matrix(peerm,"16 · Own assigned role dominates peer composition","16_peer_composition_matrix",subtitle="N=3 composition pilot")
    social=pd.read_csv(D/"social_persona_cell_means.csv"); social["cell"]=social.own_adversarial.map({0:"own coop",1:"own adv"})+" / "+social.opponent_adversarial.map({0:"opp coop",1:"opp adv"}); sm=social.pivot(index="model",columns="cell",values="mean"); sm.index=[NAMES[x] for x in sm.index]; matrix(sm,"17 · Dyadic social framing is asymmetric","17_dyadic_social_persona",subtitle="Own role changes behavior more than opponent label")

    pos=pd.read_csv(D/"nplayer_position_effect_by_persona.csv"); pos=pos[pos.converged].pivot(index="model",columns="persona",values="coef"); pos.index=[NAMES[x] for x in pos.index]; matrix(pos,"18 · Relative-position response changes sign with persona","18_nplayer_position_coefficients",fmt="+.2f",diverge=True,subtitle="Exploratory logit coefficients; non-converged cells omitted")
    pos2=pd.read_csv(D/"2p_position_effect_by_persona.csv"); pos2=pos2[pos2.converged].pivot(index="model",columns="level",values="coef"); pos2.index=[NAMES[x] for x in pos2.index]; matrix(pos2,"19 · Dyadic position effects are checkpoint- and persona-specific","19_dyadic_position_coefficients",fmt="+.2f",diverge=True,subtitle="Exploratory coefficients; different protocol from N-player panel")

    score=pd.read_csv(D/"human_effect_scorecard.csv").set_index("model"); score.index=[NAMES[x] for x in score.index]; code={"not_replicated":0,"inconclusive":.5,"replicated":1}; matrix(score.replace(code).infer_objects(copy=False),"20 · No checkpoint reproduces the full human signature","20_human_effect_scorecard",fmt=".1f",subtitle="0 = not replicated · 0.5 = inconclusive · 1 = replicated")
    fp=pd.read_csv(D/"behavioral_fingerprint.csv").set_index("display_model"); cols=["risk_response_pp","reciprocity_median_pp","persona_swing_pp","realized_payoff_r","canonical_fit_above_chance_pct"]; z=(fp[cols]-fp[cols].mean())/fp[cols].std(); z.columns=["risk response","reciprocity","persona swing","payoff alignment","canonical fit"]; matrix(z,"21 · Multidimensional behavioral fingerprints","21_behavioral_fingerprint_z",fmt="+.1f",diverge=True,subtitle="Column-wise z-scores; blank cells are unavailable, not zero")
    fig,ax=plt.subplots(figsize=(7.8,5.6)); s=fp.dropna(subset=["risk_response_pp","reciprocity_median_pp"]); ax.scatter(s.risk_response_pp,s.reciprocity_median_pp,s=95,color=TEAL,edgecolor="white")
    for m,r in s.iterrows(): ax.annotate(m,(r.risk_response_pp,r.reciprocity_median_pp),xytext=(5,4),textcoords="offset points",fontsize=7.5)
    ax.axhline(0,color=INK,lw=1); ax.axvline(0,color=INK,lw=1); clean(ax,"both"); ax.set_xlabel("Risk response (pp)"); ax.set_ylabel("Median reciprocity response (pp)"); ax.set_title("22 · Risk sensitivity and reciprocity are distinct behavioral axes",loc="left",fontsize=14,weight="bold"); finish(fig,"22_risk_vs_reciprocity")

    traj=pd.read_csv(D/"round_trajectory_all_checkpoints.csv"); tm=traj.pivot(index="model",columns="round",values="unsafe"); tm.index=[NAMES.get(x,x) for x in tm.index]; matrix(tm,"23 · Unsafe behavior evolves across rounds","23_round_trajectory_matrix",subtitle="Decision-weighted trajectory summaries; horizons differ")
    comp=egt.set_index("max_private_risk")[["frequency_AS_mean","frequency_AU_mean","frequency_CS_mean","frequency_CAS_mean"]]; comp.columns=["AS","AU","CS","CAS"]; matrix(comp,"24 · EGT stationary composition changes regime at high risk","24_egt_strategy_composition",subtitle="Main reference β=2, μ=.02")

    canon=pd.DataFrame(json.load(open(D/"canonical_strategy_summary.json"))).set_index("population"); canon.index=[NAMES.get(x,x) for x in canon.index]; fig,ax=plt.subplots(figsize=(8,5.4)); v=canon.pct_beats_chance.sort_values(); ax.barh(v.index,v,color=PURPLE); clean(ax); ax.set_xlabel("Trajectories beating chance (%)"); ax.set_title("25 · Canonical strategy fit is weak across populations",loc="left",fontsize=14,weight="bold"); finish(fig,"25_canonical_strategy_fit")

    # Coverage/readiness map from catalog classifications.
    cat=pd.read_csv(ROOT/"results/catalog.csv"); counts=cat.status.value_counts().rename_axis("status").to_frame("count"); fig,ax=plt.subplots(figsize=(7.4,4.5)); ax.barh(counts.index,counts["count"],color=[TEAL if x=="completed" else GOLD for x in counts.index]); clean(ax); ax.set_title("26 · Result catalog readiness by run status",loc="left",fontsize=14,weight="bold"); ax.set_xlabel("Catalog records"); finish(fig,"26_result_catalog_status")

    print(f"generated={len(list(OUT.glob('*.png')))} folder={OUT}")

if __name__=="__main__": setup(); main()
