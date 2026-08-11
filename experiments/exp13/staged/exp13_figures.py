"""exp13_figures — the exemplar deck, redrawable from real pipeline outputs.

draw_all(df, dfine, dctrl, systems, out_dir, illustrative=True)

`df`, `dfine`, `dctrl` are exactly the three tables exp13_noise_robustness_mockup.py writes.
Point this at REAL run outputs carrying the same columns and the deck redraws unchanged --
pass illustrative=False to drop the on-face ILLUSTRATIVE tags, which should require a real
result to justify.

Required columns
  df     key morph topo sigma_train seed turing kstar_model kstar_fft_rel_err
         sigma_max_model Dratio rob_<sigma> kcv_<sigma>
  dfine  key sigma_train seed sigma_eval rob kcv mode_class
  dctrl  arm key seed turing kstar_fft_rel_err rob_0.2
  baseline_csv (optional)  the exp11 measured answer-key table, for Fig 1a/1b and Fig 6c
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

CMAP   = plt.get_cmap("viridis")
ST_COL = {0.0: "0.25", 0.05: CMAP(0.72), 0.10: CMAP(0.55), 0.20: CMAP(0.34), 0.35: CMAP(0.12)}
ST_LAB = {0.0: "no noise", 0.05: "5%", 0.10: "10%", 0.20: "20%", 0.35: "35%"}
MCOL   = {"spots": "#4c72b0", "labyrinth": "#55a868", "stripes": "#8172b2"}
ALARM  = "#c1272d"
FOCAL  = ST_COL[0.20]
TICA48 = 0.048        # Tica's measured experimental parameter CV
ILLUS  = "ILLUSTRATIVE - surrogate, not a trained RNGRN run"


def apply_style(sizes=(8, 7, 6)):
    base, mid, small = sizes
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": base,
        "axes.titlesize": base, "axes.labelsize": base, "legend.fontsize": mid,
        "xtick.labelsize": small, "ytick.labelsize": small,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "axes.grid": False,
    })


def _stagger(vals, min_gap):
    """Nudge end-of-line label y-positions apart, preserving order (figure-style §6.3)."""
    order = sorted(vals, key=lambda k: vals[k])
    out, prev = {}, None
    for k in order:
        y = vals[k] if prev is None else max(vals[k], prev + min_gap)
        out[k] = y; prev = y
    return out


def panel_letter(ax, letter):
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="right")


def _tag(fig, illustrative, txt=ILLUS):
    if illustrative:
        fig.text(0.995, 0.005, txt, ha="right", va="bottom", fontsize=6,
                 color=ALARM, style="italic", zorder=50)


# ======================================================================================
def fig1_mechanism(df, dfine, baseline, prof, out, illustrative=True):
    """(a,b) measured answer-key headroom | (c) schematic | (d,e) surrogate selection."""
    apply_style()
    arms = sorted(df.sigma_train.unique())
    fig = plt.figure(figsize=(7.4, 5.8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.62, wspace=0.44,
                           left=0.085, right=0.975, top=0.845, bottom=0.095)
    axa = fig.add_subplot(gs[0, :2])
    if baseline is not None:
        bkey = ["ds", "key"] if "ds" in baseline.columns else ["key"]
        pb = baseline.pivot_table(index=bkey, columns="sigma", values="frac_strict", aggfunc="median")
        sg = sorted(baseline.sigma.unique())
        for k in pb.index:
            axa.plot(sg, pb.loc[k, sg].values, "-", color="0.74", lw=0.5, alpha=0.55, zorder=1)
        axa.plot(sg, [pb[s].median() for s in sg], "o-", color="k", lw=2.0, ms=5, zorder=4,
                 label=f"median of {len(pb)} systems")
        axa.plot(sg, [pb[s].min() for s in sg], "v--", color=ALARM, lw=1.2, ms=4, zorder=3,
                 label="least robust system")
        axa.axvline(TICA48, color="0.4", ls=":", lw=1.1, zorder=2)
        axa.annotate("Tica measured\nexperimental CV 4.8%", xy=(TICA48, 0.62), xytext=(0.085, 0.72),
                     fontsize=6, color="0.3", ha="left", va="center",
                     arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7, shrinkA=1, shrinkB=1))
        axa.legend(loc="lower left", fontsize=6)
    axa.set_xlabel("parameter perturbation SD (log-normal, multiplicative)")
    axa.set_ylabel("fraction still Turing")
    axa.set_title("Identical wiring, up to 4x spread in robustness", loc="left")
    axa.set_ylim(0, 1.05); axa.margins(x=0.04)
    panel_letter(axa, "a")

    axb = fig.add_subplot(gs[0, 2])
    if baseline is not None:
        at10 = baseline[baseline.sigma == 0.10]
        order = at10.groupby("topology")["frac_strict"].median().sort_values().index.tolist()
        for i, t in enumerate(order):
            v = at10[at10.topology == t]["frac_strict"].values
            axb.plot(i + np.random.default_rng(i).uniform(-.16, .16, len(v)), v, "o",
                     ms=2.4, color="0.55", alpha=0.75, mew=0)
            axb.plot([i-0.30, i+0.30], [np.median(v)]*2, "-", color="k", lw=1.8)
        axb.set_xticks(range(len(order)))
        axb.set_xticklabels([t.replace("_", "\n") for t in order], fontsize=5.2)
    axb.set_ylabel("fraction still Turing at 10%")
    axb.set_title("Spread is parametric,\nnot topological", loc="left", fontsize=7)
    axb.set_ylim(0, 1.06)
    panel_letter(axb, "b")

    axc = fig.add_subplot(gs[1, 0])
    xx = np.linspace(-1, 1, 400)
    for w, col in ((0.24, "0.25"), (0.66, FOCAL)):
        y = 0.95*np.minimum(1.05, (np.abs(xx)/w)**2)
        axc.plot(xx, np.where(y <= 0.95, y, np.nan), color=col, lw=1.8)
        axc.fill_between(xx, 0, 0.05, where=np.abs(xx) < w, color=col, alpha=0.21, lw=0)
    axc.text(-0.16, 0.62, "sharp\n(no noise)", ha="right", fontsize=6, color="0.25")
    axc.text(0.70, 0.34, "wide\n(weight noise)", ha="center", fontsize=6, color=FOCAL)
    axc.set_xlabel("displacement in parameter space"); axc.set_ylabel("recovery loss")
    axc.set_title("Noise favours wide minima", loc="left")
    axc.set_yticks([0]); axc.set_ylim(-0.03, 1.05); axc.set_xlim(-1.05, 1.05)
    panel_letter(axc, "c")

    axd = fig.add_subplot(gs[1, 1])
    if prof:
        for st in sorted(prof):
            ts, frac = prof[st][0], prof[st][1]
            axd.plot(ts, frac, "-", color=ST_COL.get(st, "0.5"), lw=2.0)
        axd.text(0.44, 1.005, "20% noise", fontsize=6, color=FOCAL, va="bottom")
        axd.text(0.44, 0.80, "no noise", fontsize=6, color="0.25", va="top")
    axd.set_xlabel("distance from recovered point ||dtheta||")
    axd.set_ylabel("directions still Turing")
    axd.set_title("Basin width, one system", loc="left")
    axd.set_ylim(0.60, 1.04); axd.set_xlim(0, 0.88)
    panel_letter(axd, "d")

    axe = fig.add_subplot(gs[1, 2])
    mm = dfine.groupby(["sigma_train", "sigma_eval"])["rob"].median().unstack()
    for st in arms:
        axe.plot(mm.columns, mm.loc[st].values, "-", color=ST_COL.get(st, "0.5"),
                 lw=2.2 if st == 0.20 else 1.3, zorder=5 if st == 0.20 else 3)
    axe.axvline(TICA48, color="0.4", ls=":", lw=1.0)
    xmax = float(mm.columns.max())
    _ends = _stagger({st: mm.loc[st].values[-1] for st in arms}, min_gap=0.022)
    for st in arms:
        axe.text(xmax*1.04, _ends[st], ST_LAB.get(st, str(st)),
                 fontsize=6, color=ST_COL.get(st, "0.5"), va="center")
    axe.set_xlabel("evaluation noise SD"); axe.set_ylabel("fraction still Turing")
    axe.set_title("Trained-in noise widens\nthe survival curve", loc="left", fontsize=7)
    axe.set_xlim(0, xmax*1.45); axe.set_ylim(0.60, 1.03)
    panel_letter(axe, "e")

    fig.suptitle("Weight noise should select wider Turing basins: the headroom is measured, "
                 "the selection is a surrogate", fontsize=8.5, x=0.085, ha="left", y=0.985)
    _tag(fig, illustrative, "(c)-(e) ILLUSTRATIVE surrogate  .  (a)-(b) real measured data")
    p = os.path.join(out, "fig1_mechanism_and_headroom.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig2_headline(df, dfine, out, illustrative=True):
    apply_style()
    arms = sorted(df.sigma_train.unique())
    tur = df[df.turing]
    fig = plt.figure(figsize=(7.4, 3.1))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42, left=0.075, right=0.985,
                           top=0.76, bottom=0.17)
    ax = fig.add_subplot(gs[0])
    for st in arms:
        s = dfine[dfine.sigma_train == st]
        med = s.groupby("sigma_eval")["rob"].median()
        ax.plot(med.index, med.values, "-", color=ST_COL.get(st, "0.5"),
                lw=2.4 if st in (0.0, 0.20) else 1.2, zorder=6 if st in (0.0, 0.20) else 3)
        if st in (0.0, 0.20):
            ax.fill_between(med.index, s.groupby("sigma_eval")["rob"].quantile(.25).values,
                            s.groupby("sigma_eval")["rob"].quantile(.75).values,
                            color=ST_COL[st], alpha=0.16, lw=0, zorder=2)
    ax.axvline(TICA48, color="0.4", ls=":", lw=1.0)
    ax.text(TICA48+0.006, 0.63, "4.8% measured\nexperimental CV", fontsize=5.6, color="0.35", va="bottom")
    xm = float(dfine.sigma_eval.max())
    _e = _stagger({st: dfine[dfine.sigma_train == st].groupby("sigma_eval")["rob"].median().values[-1]
                   for st in arms}, min_gap=0.022)
    for st in arms:
        ax.text(xm*1.05, _e[st], ST_LAB.get(st, str(st)), fontsize=6, color=ST_COL.get(st, "0.5"),
                fontweight="bold" if st == 0.20 else "normal", va="center")
    ax.set_xlabel("evaluation perturbation SD"); ax.set_ylabel("fraction still Turing")
    ax.set_title("Survival under perturbation", loc="left")
    ax.set_xlim(0, xm*1.55); ax.set_ylim(0.60, 1.03)
    panel_letter(ax, "a")

    ax = fig.add_subplot(gs[1])
    piv = tur.pivot_table(index="key", columns="sigma_train", values="rob_0.2", aggfunc="median")
    pp = piv.dropna(subset=[0.0, 0.20])
    for k in pp.index:
        ax.plot([0, 1], [pp.loc[k, 0.0], pp.loc[k, 0.20]], "-", color="0.78", lw=0.6, zorder=1)
    for j, st in enumerate((0.0, 0.20)):
        v = pp[st].values
        ax.plot(np.full(len(v), j)+np.random.default_rng(j).uniform(-.07, .07, len(v)), v,
                "o", ms=3.2, color=ST_COL[st], alpha=0.85, mew=0, zorder=4)
        ax.plot([j-0.20, j+0.20], [np.median(v)]*2, "-", color="k", lw=2.0, zorder=5)
        ax.text(j, 1.055, f"{np.median(v):.3f}", ha="center", fontsize=6.5, fontweight="bold")
    w = stats.wilcoxon(pp[0.20], pp[0.0])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["no noise", "20% train noise"])
    ax.set_ylabel("fraction still Turing at 20%")
    ax.set_title(f"{int((pp[0.20]-pp[0.0] > 0.005).sum())} of {len(pp)} systems improve", loc="left")
    ax.set_xlim(-0.42, 1.42); ax.set_ylim(0.35, 1.10)
    ax.text(0.5, 0.40, f"Wilcoxon signed-rank\n$p = {w.pvalue:.1e}$", ha="center",
            fontsize=6, color="0.3")
    panel_letter(ax, "b")

    ax = fig.add_subplot(gs[2])
    hr = df.groupby("sigma_train")["turing"].agg(["mean", "size"])
    xs = np.arange(len(arms))
    se = np.sqrt(hr["mean"]*(1-hr["mean"])/hr["size"])
    ax.errorbar(xs, hr["mean"].values, yerr=1.96*se.values, fmt="o", ms=5, lw=1.2,
                capsize=2.5, color="0.3", ecolor="0.6", zorder=4)
    if 0.20 in arms:
        i = arms.index(0.20)
        ax.plot(xs[i], hr["mean"].iloc[i], "o", ms=8, color=FOCAL, zorder=5)
    ax.set_xticks(xs); ax.set_xticklabels([ST_LAB.get(s, str(s)) for s in arms], fontsize=6)
    ax.set_xlabel("weight-noise SD during training"); ax.set_ylabel("seeds reaching Turing")
    ax.set_title("More seeds succeed", loc="left")
    ax.margins(x=0.08, y=0.18)
    ax.text(0.02, 0.02, "higher = better", transform=ax.transAxes, fontsize=5.6, color="0.45")
    panel_letter(ax, "c")

    fig.suptitle("Positive result, headline form: training noise enlarges the local Turing "
                 "volume of what recovery finds", fontsize=8.5, x=0.075, ha="left", y=0.985)
    n_sys = df.key.nunique(); n_seed = df.seed.nunique()
    fig.text(0.075, 0.90, f"{n_sys} systems x {n_seed} seeds per arm; bands are seed IQR; "
             "strict Turing criterion; error bars 95% CI", fontsize=6, color="0.35", ha="left")
    _tag(fig, illustrative)
    p = os.path.join(out, "fig2_headline_robustness.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig3_tradeoff(df, matched, out, illustrative=True):
    """(a,b) matched-subset cost curve | (c) per-morphology gain."""
    apply_style()
    arms = sorted(df.sigma_train.unique())
    tur = df[df.turing]
    base = df[(df.sigma_train == 0.0) & df.turing]
    kerr = [base["kstar_fft_rel_err"].median()] + matched.kerr_arm.tolist()
    rob  = [base["rob_0.2"].median()] + matched.rob_arm.tolist()
    xs = np.arange(len(arms))
    fig = plt.figure(figsize=(7.4, 3.2))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.46, left=0.075, right=0.985,
                          top=0.75, bottom=0.18)
    ax = fig.add_subplot(gs[0])
    ax.plot(xs, rob, "o-", color=FOCAL, lw=2.0, ms=5)
    ax.set_ylabel("fraction still Turing at 20%", color=FOCAL)
    ax.tick_params(axis="y", colors=FOCAL); ax.margins(y=0.18)
    ax2 = ax.twinx()
    ax2.plot(xs, 100*np.array(kerr), "s--", color=ALARM, lw=1.6, ms=4)
    ax2.set_ylabel("$k^*$ error vs image FFT (%)", color=ALARM)
    ax2.tick_params(axis="y", colors=ALARM); ax2.margins(y=0.18)
    ax.set_xticks(xs); ax.set_xticklabels([ST_LAB.get(s, str(s)) for s in arms], fontsize=6)
    ax.set_xlabel("weight-noise SD during training")
    ax.set_title("Robustness is bought with $k^*$ accuracy", loc="left", fontsize=7.5)
    ax.text(0.04, 0.94, "robustness up", transform=ax.transAxes, fontsize=6, color=FOCAL)
    ax.text(0.04, 0.86, "$k^*$ error up (worse)", transform=ax.transAxes, fontsize=6, color=ALARM)
    panel_letter(ax, "a")

    ax = fig.add_subplot(gs[1])
    ax.plot(100*np.array(kerr), rob, "-", color="0.6", lw=1.0, zorder=1)
    offs = {0.0: (-4, -11, "right"), 0.05: (8, -4, "left"), 0.10: (8, 2, "left"),
            0.20: (6, 7, "left"), 0.35: (-4, 7, "right")}
    for i, st in enumerate(arms):
        ax.plot(100*kerr[i], rob[i], "o", ms=9 if st == 0.20 else 6,
                color=ST_COL.get(st, "0.5"), mew=1.2 if st == 0.20 else 0, mec="k", zorder=4)
        dx, dy, ha = offs.get(st, (6, 4, "left"))
        ax.annotate(ST_LAB.get(st, str(st)), (100*kerr[i], rob[i]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=6, color=ST_COL.get(st, "0.5"), ha=ha,
                    fontweight="bold" if st == 0.20 else "normal")
    ax.set_xlabel("$k^*$ error vs image FFT (%)  -  lower better")
    ax.set_ylabel("fraction still Turing at 20%")
    ax.set_title("An interior optimum, not a free lunch", loc="left", fontsize=7.5)
    ax.margins(0.14)
    panel_letter(ax, "b")

    ax = fig.add_subplot(gs[2])
    mo = tur.groupby(["morph", "sigma_train"])["rob_0.2"].median().unstack()
    for m_ in [m for m in MCOL if m in mo.index]:
        ax.plot(xs, mo.loc[m_].values, "o-", color=MCOL[m_], lw=1.6, ms=4)
        ax.text(xs[-1]+0.14, mo.loc[m_].values[-1], m_, fontsize=6, color=MCOL[m_], va="center")
    ax.set_xticks(xs); ax.set_xticklabels([ST_LAB.get(s, str(s)) for s in arms], fontsize=6)
    ax.set_xlabel("weight-noise SD during training")
    ax.set_ylabel("fraction still Turing at 20%")
    ax.set_title("Gain holds in every morphology", loc="left", fontsize=7.5)
    ax.set_xlim(-0.3, len(arms)+0.9); ax.margins(y=0.14)
    panel_letter(ax, "c")

    fig.suptitle("The cost side: the same noise that widens the basin blurs the wavelength, "
                 "so there is an operating point to choose", fontsize=8.5, x=0.075, ha="left", y=0.985)
    fig.text(0.075, 0.885, "(a, b) matched subset - only seeds Turing in BOTH arms, so the "
             "$k^*$ comparison is like-for-like", fontsize=6, color="0.35", ha="left")
    _tag(fig, illustrative)
    p = os.path.join(out, "fig3_tradeoff.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig4_controls(df, dctrl, out, illustrative=True):
    apply_style()
    arms_spec = [("no\nnoise", "0.25", df[df.sigma_train == 0.0]),
                 ("20%\nnoise", FOCAL, df[df.sigma_train == 0.20]),
                 ("post-hoc\njitter", "#d98c00", dctrl[dctrl.arm == "posthoc_jitter"]),
                 ("wide\ninit", "#7a6a9b", dctrl[dctrl.arm == "wide_init_nonoise"])]
    fig = plt.figure(figsize=(7.4, 3.35))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.46, left=0.075, right=0.985,
                          top=0.74, bottom=0.255)
    ax = fig.add_subplot(gs[0])
    for i, (lab, col, src) in enumerate(arms_spec):
        v = src.loc[src.turing, "rob_0.2"].dropna().values
        ax.plot(i+np.random.default_rng(i).uniform(-.15, .15, len(v)), v, "o",
                ms=1.9, color=col, alpha=0.45, mew=0, zorder=2)
        ax.plot([i-0.28, i+0.28], [np.median(v)]*2, "-", color=col, lw=2.4, zorder=4)
        ax.text(i, 1.045, f"{np.median(v):.3f}", ha="center", fontsize=6, color=col,
                fontweight="bold" if i == 1 else "normal")
    ax.set_xticks(range(4)); ax.set_xticklabels([a[0] for a in arms_spec], fontsize=5.8)
    ax.set_xlim(-0.55, 3.55); ax.set_ylim(0.30, 1.10)
    ax.set_ylabel("fraction still Turing at 20%")
    ax.set_title("Only trained-in noise gets there", loc="left", fontsize=7.5)
    panel_letter(ax, "a")

    ax = fig.add_subplot(gs[1])
    for i, (lab, col, src) in enumerate(arms_spec):
        v = src.loc[src.turing, "kstar_fft_rel_err"].dropna().values*100
        ax.plot(i+np.random.default_rng(i+9).uniform(-.15, .15, len(v)), v, "o",
                ms=1.9, color=col, alpha=0.45, mew=0, zorder=2)
        ax.plot([i-0.28, i+0.28], [np.median(v)]*2, "-", color=col, lw=2.4, zorder=4)
        ax.text(i, 92, f"{np.median(v):.1f}%", ha="center", fontsize=6, color=col,
                fontweight="bold" if i == 1 else "normal")
    ax.set_yscale("log"); ax.set_yticks([1, 10, 100]); ax.set_yticklabels(["1", "10", "100"])
    ax.set_ylim(0.3, 260)
    ax.set_xticks(range(4)); ax.set_xticklabels([a[0] for a in arms_spec], fontsize=5.8)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylabel("$k^*$ error vs image FFT (%)")
    ax.set_title("Controls pay in wavelength", loc="left", fontsize=7.5)
    ax.text(0.03, 0.05, "lower = better", transform=ax.transAxes, fontsize=5.6, color="0.45")
    panel_letter(ax, "b")

    ax = fig.add_subplot(gs[2])
    arms = sorted(df.sigma_train.unique()); xs = np.arange(len(arms))
    gg = df[df.turing].groupby("sigma_train").agg(smax=("sigma_max_model", "median"),
                                                 dr=("Dratio", "median"))
    ax.plot(xs, gg.smax.values, "o-", color="#2a7f62", lw=1.8, ms=4)
    ax.set_ylabel("growth rate at $k^*$", color="#2a7f62")
    ax.tick_params(axis="y", colors="#2a7f62"); ax.set_ylim(0, max(gg.smax)*1.7)
    axr = ax.twinx()
    axr.plot(xs, gg.dr.values, "s:", color="#8c6d31", lw=1.4, ms=3.5)
    axr.set_ylabel("$D$ ratio (median)", color="#8c6d31")
    axr.tick_params(axis="y", colors="#8c6d31"); axr.set_ylim(0, max(gg.dr)*1.25)
    ax.set_xticks(xs); ax.set_xticklabels([ST_LAB.get(s, str(s)) for s in arms], fontsize=6)
    ax.set_xlabel("weight-noise SD during training")
    ax.set_title("Not the flat-dispersion cheat", loc="left", fontsize=7.5)
    panel_letter(ax, "c")

    fig.suptitle("The controls that make the claim defensible: stochasticity alone does not "
                 "reproduce the effect", fontsize=8.5, x=0.075, ha="left", y=0.985)
    fig.text(0.075, 0.055, "no noise = deterministic  .  20% noise = weight noise during training "
             "(the claim)  .  post-hoc jitter = deterministic, then one 20% perturbation at the end  "
             ".  wide init = 2x init spread, no training noise",
             fontsize=5.8, color="0.3", ha="left")
    _tag(fig, illustrative)
    p = os.path.join(out, "fig4_controls.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig5_seeds(df, dfine, out, illustrative=True):
    apply_style()
    arms = sorted(df.sigma_train.unique()); xs = np.arange(len(arms))
    tur = df[df.turing]
    fig = plt.figure(figsize=(7.4, 3.2))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.44, left=0.075, right=0.985,
                          top=0.755, bottom=0.185)
    ax = fig.add_subplot(gs[0])
    bins = np.linspace(0.3, 1.0, 26)
    for i, st in enumerate(arms):
        v = tur[tur.sigma_train == st]["rob_0.2"].dropna().values
        h, _ = np.histogram(v, bins=bins, density=True)
        if h.max() > 0:
            ax.fill_between(bins[:-1], i, i + h/h.max()*0.85, color=ST_COL.get(st, "0.5"),
                            alpha=0.75, lw=0.6, edgecolor="w", zorder=10-i)
        ax.text(0.315, i+0.30, ST_LAB.get(st, str(st)), fontsize=6, va="center",
                color=ST_COL.get(st, "0.5") if st != 0.0 else "0.2")
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("fraction still Turing at 20%")
    ax.set_title("Whole seed distribution shifts", loc="left", fontsize=7.2)
    ax.set_xlim(0.30, 1.03); ax.set_ylim(-0.1, len(arms)+0.1)
    panel_letter(ax, "a")

    ax = fig.add_subplot(gs[1])
    sd = tur.groupby(["key", "sigma_train"])["kstar_model"].agg(["median", "std", "count"]).reset_index()
    sd["cv"] = sd["std"]/sd["median"]
    for i, st in enumerate(arms):
        v = sd[(sd.sigma_train == st) & (sd["count"] >= 4)]["cv"].dropna().values*100
        ax.plot(i+np.random.default_rng(i+3).uniform(-.16, .16, len(v)), v, "o",
                ms=2.6, color=ST_COL.get(st, "0.5"), alpha=0.6, mew=0)
        if len(v):
            ax.plot([i-0.28, i+0.28], [np.median(v)]*2, "-", color=ST_COL.get(st, "0.5"), lw=2.4)
    ax.set_xticks(xs); ax.set_xticklabels([ST_LAB.get(s, str(s)) for s in arms], fontsize=6)
    ax.set_xlabel("weight-noise SD during training")
    ax.set_ylabel("seed-to-seed $k^*$ CV (%)")
    ax.set_title("Seed spread widens - the cost", loc="left", fontsize=7.2)
    ax.text(0.03, 0.93, "lower = better", transform=ax.transAxes, fontsize=5.6, color="0.45")
    panel_letter(ax, "b")

    ax = fig.add_subplot(gs[2])
    levels = [s for s in (0.048, 0.10, 0.20) if s in set(dfine.sigma_eval)]
    kcv = dfine[dfine.sigma_eval.isin(levels)].groupby(["sigma_train", "sigma_eval"])["kcv"].median().unstack()
    mk = {0.048: "o", 0.10: "s", 0.20: "^"}
    mf = {0.048: "w", 0.10: "0.6", 0.20: "0.2"}
    for se_ in kcv.columns:
        ax.plot(xs, 100*kcv[se_].values, mk.get(se_, "o")+"-", color="0.3", lw=1.2, ms=3.6,
                mfc=mf.get(se_, "0.5"), mec="0.3")
        ax.text(xs[-1]+0.14, 100*kcv[se_].values[-1], f"{se_*100:.1f}% eval noise",
                fontsize=5.5, color="0.35", va="center")
    ax.set_xticks(xs); ax.set_xticklabels([ST_LAB.get(s, str(s)) for s in arms], fontsize=6)
    ax.set_xlabel("weight-noise SD during training")
    ax.set_ylabel("$k^*$ CV inside surviving cloud (%)")
    ax.set_title("But wavelength stays put", loc="left", fontsize=7.2)
    ax.set_xlim(-0.3, len(arms)+2.2)
    panel_letter(ax, "c")

    fig.suptitle("Reported as a seed distribution, per house convention - never the best seed",
                 fontsize=8.5, x=0.075, ha="left", y=0.985)
    fig.text(0.075, 0.885, f"{df.seed.nunique()} seeds per system per arm; (b) systems with >=4 "
             "Turing seeds; (c) $k^*$ dispersion among perturbed draws that remain Turing",
             fontsize=6, color="0.35", ha="left")
    _tag(fig, illustrative)
    p = os.path.join(out, "fig5_seed_distribution.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig6_per_system(df, baseline, out, illustrative=True):
    apply_style()
    tur = df[df.turing]
    piv = tur.pivot_table(index=["key", "morph"], columns="sigma_train",
                          values="rob_0.2", aggfunc="median")
    pp = piv.dropna(subset=[0.0, 0.20]).copy()
    pp["gain"] = pp[0.20] - pp[0.0]
    pp = pp.reset_index().sort_values(0.0)
    fig = plt.figure(figsize=(7.4, 3.6))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.55, 1, 1], wspace=0.42,
                          left=0.075, right=0.985, top=0.76, bottom=0.19)
    ax = fig.add_subplot(gs[0])
    yy = np.arange(len(pp))
    ax.hlines(yy, pp[0.0], pp[0.20], color="0.80", lw=1.1, zorder=1)
    ax.plot(pp[0.0], yy, "o", ms=3.4, color="0.25", mew=0, zorder=3)
    ax.plot(pp[0.20], yy, "o", ms=3.4, color=FOCAL, mew=0, zorder=4)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=3.4, color="0.25", label="no noise"),
                       plt.Line2D([], [], marker="o", ls="", ms=3.4, color=FOCAL, label="20% train noise")],
              loc="upper left", fontsize=6, handletextpad=0.3, borderaxespad=0.5)
    ax.set_yticks([]); ax.set_ylim(-1, len(pp)); ax.set_xlim(0.32, 1.10)
    ax.set_xlabel("fraction still Turing at 20%")
    ax.set_ylabel(f"{len(pp)} systems, ordered by baseline ->", fontsize=6.2)
    ax.set_title("Fragile systems gain the most", loc="left", fontsize=7.5)
    panel_letter(ax, "a")

    ax = fig.add_subplot(gs[1])
    cx = np.linspace(pp[0.0].min(), 0.99, 50)
    ax.plot(cx, 1.0-cx, "--", color="0.65", lw=1.0, zorder=1)
    ax.scatter(pp[0.0], pp["gain"], c=[MCOL.get(m, "0.5") for m in pp["morph"]], s=16, zorder=3, lw=0)
    ax.axhline(0, color="0.6", lw=0.8, zorder=1)
    rho, pv = stats.spearmanr(pp[0.0], pp["gain"])
    ax.text(0.99, 0.36, f"Spearman $\\rho$ = {rho:.2f}\n(bounded by the ceiling)",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color="0.25")
    ax.text(0.02, 0.96, "ceiling: gain <= 1 - baseline", transform=ax.transAxes,
            fontsize=5.4, color="0.45", va="top")
    for m_ in [m for m in MCOL if m in set(pp["morph"])]:
        ax.plot([], [], "o", ms=4, color=MCOL[m_], label=m_)
    ax.legend(fontsize=5.8, loc="lower left", handletextpad=0.3, borderaxespad=0.4)
    ax.set_xlabel("baseline robustness (no noise)"); ax.set_ylabel("gain at 20% train noise")
    ax.set_title("The weakest benefit most", loc="left", fontsize=7.5)
    ax.margins(0.10)
    panel_letter(ax, "b")

    ax = fig.add_subplot(gs[2])
    sets = []
    if baseline is not None:
        sets.append(("generator\nanswer keys", baseline[baseline.sigma == 0.20]["frac_strict"].values, "0.55"))
    sets += [("recovered,\nno noise", tur[tur.sigma_train == 0.0]["rob_0.2"].dropna().values, "0.25"),
             ("recovered,\n20% noise", tur[tur.sigma_train == 0.20]["rob_0.2"].dropna().values, FOCAL)]
    for i, (lab, v, col) in enumerate(sets):
        ax.plot(i+np.random.default_rng(i+5).uniform(-.14, .14, len(v)), v, "o",
                ms=1.8, color=col, alpha=0.40, mew=0, zorder=2)
        ax.plot([i-0.28, i+0.28], [np.median(v)]*2, "-", color=col, lw=2.6, zorder=4)
        ax.text(i, 1.055, f"{np.median(v):.3f}", ha="center", fontsize=6.2, color=col,
                fontweight="bold" if i == len(sets)-1 else "normal")
    ax.set_xticks(range(len(sets))); ax.set_xticklabels([s[0] for s in sets], fontsize=5.8)
    ax.set_ylabel("fraction still Turing at 20%")
    ax.set_title("Target: beat the training data", loc="left", fontsize=7.5)
    ax.set_ylim(0.15, 1.12); ax.set_xlim(-0.55, len(sets)-0.45)
    panel_letter(ax, "c")

    fig.suptitle("Where the gain comes from - and the reference it has to clear",
                 fontsize=8.5, x=0.075, ha="left", y=0.985)
    fig.text(0.075, 0.885, "(a, b) systems Turing in both arms, median over seeds. (c) generator "
             "answer keys use a different perturbation model ((J,D) direct), so the comparison "
             "bounds rather than matches.", fontsize=6, color="0.35", ha="left")
    _tag(fig, illustrative, "(a)-(b) ILLUSTRATIVE surrogate  .  (c) leftmost group real measured data")
    p = os.path.join(out, "fig6_per_system.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def draw_all(df, dfine, dctrl, systems=None, out_dir="figures/exp13", baseline_csv=None,
             prof=None, matched=None, illustrative=True):
    """Draw the whole deck. `baseline_csv` = exp11_robustness_baseline.csv (optional)."""
    os.makedirs(out_dir, exist_ok=True)
    baseline = pd.read_csv(baseline_csv) if baseline_csv and os.path.exists(baseline_csv) else None
    if matched is None:
        from exp13_noise_robustness_mockup import matched_subset_table
        matched = matched_subset_table(df)
    paths = [fig1_mechanism(df, dfine, baseline, prof or {}, out_dir, illustrative),
             fig2_headline(df, dfine, out_dir, illustrative),
             fig3_tradeoff(df, matched, out_dir, illustrative),
             fig4_controls(df, dctrl, out_dir, illustrative),
             fig5_seeds(df, dfine, out_dir, illustrative),
             fig6_per_system(df, baseline, out_dir, illustrative)]
    for p in paths:
        print(f"[figure] {p}")
    return paths
