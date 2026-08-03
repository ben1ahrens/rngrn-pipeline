#!/usr/bin/env python
"""stage0_figures.py — the FIGURES for the Stage-0 biological-viability check.

The images ARE the deliverable of this unit (the owner asked to "visually verify ... that
our model does in fact form turing patterns under the biological conditions"), so unlike
the per-run sweep figures these are committed, together with the arrays behind every
panel. See the .gitignore negation for experiments/figures_report/stage0/.

Panels that scripts/make_figures.py already draws are REUSED (`fig_pattern` for the part-4
target-vs-recovered comparison). The three genuinely new ones live here: the box-acceptance
summary, the mu (= delta) boundary curve, and the 64x64 pattern gallery.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from rngrn import plotdata as PD                                   # noqa: E402
import make_figures as MF                                          # noqa: E402

OUT = "experiments/figures_report/stage0"
ARR = os.path.join(OUT, "arrays")
FORMS = ("competitive", "nc1")
SCHEMES = ("mobile3", "immobile")
BOX_LO, BOX_HI = 0.4, 5.0


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path} ({os.path.getsize(path)} bytes)")
    return path


# ======================================================================================
# f1 — is the box Turing-EMPTY? the acceptance summary
# ======================================================================================
def fig_box_acceptance(beta_mode):
    z = np.load(os.path.join(ARR, f"part1_box_draws_{beta_mode}.npz"))
    with open(os.path.join(ARR, f"part1_summary_{beta_mode}.json")) as fh:
        summ = json.load(fh)
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.6))

    # (a) acceptance rate per cell, with Wilson 95% intervals and the raw counts on the bars
    ax = axes[0]
    labels, rates, los, his, counts = [], [], [], [], []
    for form in FORMS:
        for scheme in SCHEMES:
            s = summ[f"{form}/{scheme}"]
            labels.append(f"{form}\n{scheme}")
            rates.append(s["rate_strict"])
            los.append(s["wilson95_strict"][0]); his.append(s["wilson95_strict"][1])
            counts.append((s["strict"], s["n"]))
    x = np.arange(len(labels))
    rates = np.array(rates)
    err = np.vstack([rates - np.array(los), np.array(his) - rates])
    ax.bar(x, rates, color=["C0", "C0", "C1", "C1"], width=0.62)
    ax.errorbar(x, rates, yerr=err, fmt="none", ecolor="0.2", capsize=4, lw=1.2)
    for xi, r, (k, n) in zip(x, rates, counts):
        ax.text(xi, max(r, 0) + max(his) * 0.06, f"{k}/{n}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("fraction STRICTLY Turing")
    ax.set_ylim(0, max(his) * 1.35 if max(his) > 0 else 1.0)
    ax.set_title("(a) acceptance INSIDE bio_box.yaml\nstrict test: max Re eig(J) < 0 "
                 "AND max$_{k>0}\\,\\sigma(k)>0$", fontsize=9)

    # (b) strict vs loose — how badly the trace test overcounts here
    ax = axes[1]
    w = 0.36
    strict = [summ[f"{f}/{s}"]["rate_strict"] for f in FORMS for s in SCHEMES]
    loose = [summ[f"{f}/{s}"]["rate_loose"] for f in FORMS for s in SCHEMES]
    ax.bar(x - w / 2, strict, w, label="strict  max Re eig(J) < 0", color="C0")
    ax.bar(x + w / 2, loose, w, label="loose  tr(J) < 0  (eval/analysis.turing_ok)",
           color="C3", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("fraction")
    ax.legend(fontsize=8)
    ax.set_title("(b) the trace test OVERCOUNTS\n(why the strict test is led with)",
                 fontsize=9)

    # (c) where the draws sit: max Re eig(J) against max_{k>0} sigma(k), ZOOMED on the
    # corner that decides the verdict. Turing is the upper-LEFT quadrant. Most draws sit on
    # the diagonal sigma_max = max Re eig(J), i.e. their dispersion peaks at k -> 0 and the
    # instability, when there is one, is a UNIFORM instability and not a Turing one.
    ax = axes[2]
    for form, colour in zip(FORMS, ("C0", "C1")):
        e = np.concatenate([z[f"{form}__{s}__max_re_eig_J"] for s in SCHEMES])
        g = np.concatenate([z[f"{form}__{s}__sig_max_pos"] for s in SCHEMES])
        tt = np.concatenate([z[f"{form}__{s}__turing_strict"] for s in SCHEMES])
        m = np.isfinite(e) & np.isfinite(g)
        ax.scatter(e[m & ~tt], g[m & ~tt], s=2.0, alpha=0.06, color=colour,
                   linewidths=0, label=f"{form} ({int((m & ~tt).sum())})")
        if (m & tt).any():
            ax.scatter(e[m & tt], g[m & tt], s=90, marker="*", color=colour,
                       edgecolor="k", lw=0.6, zorder=5,
                       label=f"{form}: STRICTLY TURING ({int(tt.sum())})")
    ax.axvspan(-1.2, 0.0, ymin=0.5, color="C2", alpha=0.10, zorder=0)
    ax.axvline(0, color="0.4", lw=0.9); ax.axhline(0, color="0.4", lw=0.9)
    ax.set_xlim(-1.2, 0.35); ax.set_ylim(-0.35, 0.35)
    ax.set_xlabel("max Re eig($J$)   (uniform state: stable $<0$)")
    ax.set_ylabel("max$_{k>0}\\,\\sigma(k)$   (patterning: $>0$)")
    ax.set_title("(c) the Turing corner (shaded) — ZOOMED", fontsize=9)
    ax.legend(fontsize=7, loc="lower left", markerscale=1.6)

    # (d) WHERE in the box the survivors live. The box rows are sampled log-uniformly, so
    # the prior is flat on each log axis and any concentration is the Turing region talking.
    # `s` (the KA/KR binding budget) has NO bio_box row at all — it is a FREE direction, and
    # how narrow the survivors are in it is the point of the panel.
    ax = axes[3]
    surv_dr, surv_s, surv_form = [], [], []
    for form in FORMS:
        for sch in SCHEMES:
            tt = z[f"{form}__{sch}__turing_strict"]
            surv_dr += list(z[f"{form}__{sch}__d_ratio"][tt])
            sgm = np.exp(np.log(z[f"{form}__{sch}__s"][tt]).mean(axis=(1, 2)))
            surv_s += list(sgm)
            surv_form += [form] * int(tt.sum())
    surv_dr = np.array(surv_dr); surv_s = np.array(surv_s)
    surv_form = np.array(surv_form)
    for form, colour in zip(FORMS, ("C0", "C1")):
        m = surv_form == form
        if m.any():
            ax.scatter(surv_dr[m], surv_s[m], s=80, marker="*", color=colour,
                       edgecolor="k", lw=0.5, label=f"{form} survivors ({int(m.sum())})")
    ax.axvspan(1.0, 60.0, color="C2", alpha=0.12, zorder=0)
    ax.axhspan(10.0 ** -2, 10.0 ** 2, color="0.85", alpha=0.5, zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.8, 80); ax.set_ylim(10 ** -2.3, 10 ** 2.3)
    ax.set_xlabel("$D$-ratio (bio_box row [1, 60], shaded — sampled log-uniform)")
    ax.set_ylabel("geometric-mean binding budget $s$\n(NO box row: a FREE direction;\n"
                  "grey = the 4-decade sampling prior)", fontsize=8)
    ax.set_title("(d) where the survivors live", fontsize=9)
    ax.legend(fontsize=7, markerscale=1.2)
    n = int(z["n_per_cell"])
    fig.suptitle(f"Stage 0 part 1 — is configs/bio_box.yaml Turing-EMPTY?   "
                 f"{n} draws per (form, D-scheme), {4*n} total, seed {int(z['seed'])}, "
                 f"N=3, n_hill=2, beta_mode={beta_mode}, plausibility_score=1.0 on every "
                 f"draw by construction", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, f"f1_box_acceptance_{beta_mode}.png")


# ======================================================================================
# f2 — the mu (= delta) boundary curve
# ======================================================================================
def fig_mu_boundary(beta_mode):
    z = np.load(os.path.join(ARR, f"part2_mu_sweep_{beta_mode}.npz"))
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), sharex=True)
    for col, form in enumerate(FORMS):
        keys = sorted({k.split("__")[1] for k in z.files
                       if k.startswith(f"{form}__base") and k.endswith("__uniform__d")})
        # --- top: sigma_max(k>0) vs delta, one curve per base system
        ax = axes[0][col]
        ax.axhline(0.0, color="0.5", lw=0.9)
        ax.axvspan(BOX_LO, BOX_HI, color="C2", alpha=0.13, zorder=0)
        for j, b in enumerate(keys):
            p = f"{form}__{b}__uniform__"
            d = z[p + "d"]
            sig = z[p + "sig_max_pos"]
            ok = z[p + "turing_strict"]
            ax.plot(d, sig, lw=1.4, color=f"C{j%10}", label=f"{b}")
            ax.plot(d[ok], sig[ok], "o", ms=3.2, color=f"C{j%10}")
            bp = z[p + "is_base"]
            ax.plot(d[bp], sig[bp], "D", ms=6, mfc="none", mec=f"C{j%10}", mew=1.4)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.axvline(BOX_LO, color="C2", ls="--", lw=1.2)
        ax.axvline(BOX_HI, color="C2", ls="--", lw=1.2)
        ax.set_ylabel("max$_{k>0}\\,\\sigma(k)$")
        ax.set_title(f"{form}", fontsize=10, weight="bold")
        ax.legend(fontsize=7, ncol=2)
        # --- bottom: k* vs delta
        ax = axes[1][col]
        ax.axvspan(BOX_LO, BOX_HI, color="C2", alpha=0.13, zorder=0)
        for j, b in enumerate(keys):
            p = f"{form}__{b}__uniform__"
            d = z[p + "d"]
            ok = z[p + "turing_strict"]
            ks = np.where(ok, z[p + "kstar"], np.nan)
            # marker as well as line: a base system that is Turing at only ONE swept delta
            # draws a zero-length line and would be invisible.
            ax.plot(d, ks, lw=1.4, marker="o", ms=3.5, color=f"C{j%10}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.axvline(BOX_LO, color="C2", ls="--", lw=1.2)
        ax.axvline(BOX_HI, color="C2", ls="--", lw=1.2)
        ax.set_xlabel("$\\delta$ (geometric mean over species) — the generator's $\\mu$")
        ax.set_ylabel("$k^*$  (only where strictly Turing)")
    fig.suptitle(
        "Stage 0 part 2 — does patterning survive at biological degradation rates?\n"
        "$\\delta$ (= the generator's $\\mu$) rescaled so its geometric mean takes each "
        "swept value; 41 log-spaced points on [0.04, 50] plus the base draw's own "
        "$\\delta$.\nShaded band and dashed lines: the cited bio_box `delta` row "
        "[0.4, 5.0].  Dots = STRICTLY Turing.  Open diamond = the base draw itself.  "
        "A break in a curve is a delta at which the steady state did not converge.",
        fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(fig, f"f2_mu_boundary_{beta_mode}.png")


def fig_mu_per_species(beta_mode):
    z = np.load(os.path.join(ARR, f"part2_mu_sweep_{beta_mode}.npz"))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4), sharey=True)
    for col, form in enumerate(FORMS):
        ax = axes[col]
        ax.axhline(0.0, color="0.5", lw=0.9)
        ax.axvspan(BOX_LO, BOX_HI, color="C2", alpha=0.13, zorder=0)
        base = z.get(f"{form}__base_delta")
        for sp in range(3):
            p = f"{form}__base0__species{sp}__"
            if p + "sig_max_pos" not in z.files:
                continue
            d = z[p + "d"]
            sig = z[p + "sig_max_pos"]; ok = z[p + "turing_strict"]
            ax.plot(d, sig, lw=1.5, color=f"C{sp}",
                    label=f"$\\delta_{sp}$ varied (base {base[0][sp]:.2f})"
                          if base is not None else f"$\\delta_{sp}$ varied")
            ax.plot(d[ok], sig[ok], "o", ms=3.5, color=f"C{sp}")
            bp = z[p + "is_base"]
            ax.plot(d[bp], sig[bp], "D", ms=6, mfc="none", mec=f"C{sp}", mew=1.4)
        ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e-3)
        ax.axvline(BOX_LO, color="C2", ls="--", lw=1.2)
        ax.axvline(BOX_HI, color="C2", ls="--", lw=1.2)
        ax.set_xlabel("that species' $\\delta_i$")
        ax.set_title(f"{form} — base0, ONE $\\delta_i$ at a time", fontsize=9)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("max$_{k>0}\\,\\sigma(k)$")
    fig.suptitle("Stage 0 part 2b — PER-SPECIES degradation sweep (the box bounds every "
                 "species, so a uniform rescale is the easy case)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, f"f2b_mu_per_species_{beta_mode}.png")


# ======================================================================================
# f3 — LOOK AT THE PATTERNS
# ======================================================================================
def fig_pattern_gallery(beta_mode):
    z = np.load(os.path.join(ARR, f"part3_rollouts_{beta_mode}.npz"))
    fields = z["fields"]
    forms = z["form"].astype(str)
    paths = []
    for form in FORMS:
        idx = np.where(forms == form)[0]
        if idx.size == 0:
            continue
        ncol = min(4, idx.size)
        nrow = int(np.ceil(idx.size / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 4.9 * nrow),
                                 squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for a, i in enumerate(idx):
            ax = axes[a // ncol][a % ncol]
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
            ch = int(z["channel_shown"][i]) if "channel_shown" in z else 0
            f0 = fields[i][ch]
            im = ax.imshow(f0, cmap="viridis", origin="lower")
            fig.colorbar(im, ax=ax, shrink=0.78)
            patt = bool(z["patterned_any"][i]) if "patterned_any" in z \
                else bool(z["patterned"][i])
            strict = bool(z["turing_strict"][i])
            stop = str(z["stopped_reason"][i])
            # a run that ended on step_budget was TRUNCATED and its field is NOT a
            # statement about the attractor — say so on the panel itself.
            trunc = " TRUNCATED" if stop == "step_budget" else ""
            ax.set_title(
                f"{'PATTERNED' if patt else 'NOT patterned'}{trunc}   "
                f"[{str(z['scheme'][i])}]  channel {ch}\n"
                f"lin. Turing={strict}  $\\sigma_{{max}}$={float(z['sig_max'][i]):+.3g}\n"
                f"$k^*_{{lin}}$={float(z['kstar_lin'][i]):.3f}  "
                f"$k^*_{{field}}$={float(z['kstar_field'][i]):.3f}  "
                f"L={float(z['L'][i]):.1f}\n"
                f"std$_{{max}}$={float(z['std_max'][i]):.3g} "
                f"(ch0 {float(z['std_c0'][i]):.2g}, ch1 {float(z['std_c1'][i]):.2g}, "
                f"ch2 {float(z['std_c2'][i]):.2g})\n"
                # a morphology label on an UNPATTERNED field is a label on float noise
                # (one control below has amplitude 2e-16). The value is still in the
                # arrays; it is simply not asserted on the figure.
                f"morph={str(z['morphology'][i]) if patt else 'n/a (unpatterned)'}  "
                f"stop={stop}\n"
                f"$D$-ratio={float(z['d_ratio'][i]):.2f}  "
                f"$\\delta\\in$[{float(z['delta_min'][i]):.2f}, "
                f"{float(z['delta_max'][i]):.2f}]",
                fontsize=7.2, color=("k" if patt else "C3"))
        fig.suptitle(
            f"Stage 0 part 3 — {form}: {int(z['grid_n'])}x{int(z['grid_n'])} rollouts of "
            f"BOX-CONSTRAINED systems — plausibility_score = 1.0 on EVERY panel.\n"
            f"L = {int(z['periods'])} x 2$\\pi$/$k^*_{{lin}}$, ETDRK4-rfft, "
            f"early_stop=True (checked: 0/12 verdicts change at the full fixed horizon).  "
            f"Red titles are the systems that never pattern.\n"
            f"EACH PANEL SHOWS THE CHANNEL WITH THE LARGEST SPATIAL STD, named in its "
            f"title, and all three channel stds are listed. This is deliberate: the "
            f"sampler shuffles the species order, so\nchannel 0 is arbitrary here, and "
            f"eval/rollout.py's `patterned` — a channel-0-only threshold, correct for a "
            f"real dataset — would call a large pattern one channel over 'unpatterned'.",
            fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.90 if nrow > 1 else 0.82), h_pad=5.0)
        paths.append(_save(fig, f"f3_patterns_{form}_{beta_mode}.png"))
    return paths


# ======================================================================================
# f4 — part 4: the five real targets, prior ON vs OFF
# ======================================================================================
def _recovered_D(runs_root, run_id):
    """The recovered model's own D vector, from the tracked results JSON."""
    with open(os.path.join(runs_root, "runs", run_id, "results",
                           "train_results.json")) as fh:
        return json.load(fh)["recovered"]["D"]


def fig_part4(runs_root="experiments/stage0_bioviab"):
    rows = []
    with open(os.path.join(runs_root, "runs.jsonl")) as fh:
        for line in fh:
            rows.append(json.loads(line))
    rows = [r for r in rows if str(r.get("run_id", "")).startswith("stage0_")]
    if not rows:
        print("  SKIP f4: no stage0_* runs in", runs_root)
        return None
    by = {}
    for r in rows:
        arm = "prior_on" if "prior_on" in r["run_id"] else "prior_off"
        by[(r["sample_key"], arm)] = r
    samples = sorted({k[0] for k in by})
    arms = ("prior_off", "prior_on")

    fig, axes = plt.subplots(len(samples), 3, figsize=(11.4, 3.5 * len(samples)),
                             squeeze=False)
    tbl = []
    for i, sk in enumerate(samples):
        tgt = None
        for j, arm in enumerate(arms):
            r = by.get((sk, arm))
            ax = axes[i][j + 1]
            ax.set_xticks([]); ax.set_yticks([])
            if r is None:
                ax.axis("off"); continue
            zp = PD.load_plot_arrays(
                PD.plot_arrays_path(os.path.join(runs_root, "runs", r["run_id"])))
            if tgt is None:
                tgt = np.asarray(zp["target_frame"], float)[0]
            ps = r.get("plausibility_score")
            patt = r.get("rollout_patterned")
            tbl.append(dict(sample=sk, arm=arm, run_id=r["run_id"],
                            plausibility_score=ps,
                            d_ratio=r.get("plausibility_d_ratio_value"),
                            alpha_in_box=r.get("plausibility_alpha_in_box"),
                            delta_in_box=r.get("plausibility_delta_in_box"),
                            d_ratio_in_box=r.get("plausibility_d_ratio_in_box"),
                            patterned=patt,
                            recovered_D=_recovered_D(runs_root, r["run_id"]),
                            rollout_status=r.get("rollout_status"),
                            stopped_reason=r.get("rollout_stopped_reason"),
                            sig_max=r.get("rollout_sig_max"),
                            kstar_model=r.get("kstar_model"),
                            kstar_fft_rel_err=r.get("kstar_fft_rel_err"),
                            morphology_match=r.get("morphology_match"),
                            loss=r.get("loss")))
            D = np.sort(np.asarray(_recovered_D(runs_root, r["run_id"]), float))
            if "model_field" in zp:
                ax.imshow(np.asarray(zp["model_field"], float)[0], cmap="viridis",
                          origin="lower")
            else:
                # plotdata deliberately writes NO model_field for an unpatterned rollout
                # ("NOTHING HERE IS A FALLBACK") rather than saving decayed noise as if it
                # were a pattern. The empty panel IS the result.
                ax.text(0.5, 0.55, "NO PATTERN FORMED", ha="center", va="center",
                        fontsize=11, color="C3", transform=ax.transAxes)
                ax.text(0.5, 0.42, f"rollout_status={r.get('rollout_status')!r}\n"
                                   f"stopped: {r.get('rollout_stopped_reason')}\n"
                                   f"(no field is saved for an unpatterned rollout —\n"
                                   f"plotdata refuses to store decayed noise)",
                        ha="center", va="center", fontsize=7, color="C3",
                        transform=ax.transAxes)
            ax.set_title(f"{arm}   plausibility_score = {ps:.2f}\n"
                         f"patterned={patt}   "
                         f"$\\sigma_{{max}}$={r.get('rollout_sig_max'):+.4f}\n"
                         f"scored $D$-ratio={r.get('plausibility_d_ratio_value'):.2f}   "
                         f"$D_{{lo}}/D_{{mid}}$={D[0]/D[1]:.3f}",
                         fontsize=8, color=("k" if patt else "C3"))
        ax = axes[i][0]
        ax.set_xticks([]); ax.set_yticks([])
        if tgt is not None:
            ax.imshow(tgt, cmap="viridis", origin="lower")
        ax.set_title(f"TARGET {sk}\n(three_gene_qvar, channel 0)", fontsize=8)
    fig.suptitle("Stage 0 part 4 — five real targets recovered WITH the biological prior "
                 "active (D-ratio log-normal at 7.5 + soft box hinges) against a prior-OFF "
                 "control.\n400 Adam steps, 64 batched restarts, CUDA, cubic dispersion "
                 "backend, one seed each. Indicative, NOT a recovery-quality measurement.",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    os.makedirs(ARR, exist_ok=True)
    with open(os.path.join(ARR, "part4_prior_recovery.json"), "w") as fh:
        json.dump(tbl, fh, indent=2)
    return _save(fig, "f4_part4_prior_recovery.png")


def fig_part4_dispersion(runs_root="experiments/stage0_bioviab"):
    """f5 — THE canonical Turing figure for part 4: sigma(k) recovered against the answer
    key, for every target and both arms. This is where "viable but unpatterned" becomes
    legible: a recovered sigma(k) that is monotone decreasing and everywhere negative cannot
    grow a pattern from noise no matter how long it is integrated, and no amount of
    plausibility score changes that."""
    rows = [json.loads(l) for l in open(os.path.join(runs_root, "runs.jsonl"))]
    rows = [r for r in rows if str(r.get("run_id", "")).startswith("stage0_")]
    if not rows:
        print("  SKIP f5: no stage0_* runs in", runs_root)
        return None
    by = {(r["sample_key"], "prior_on" if "prior_on" in r["run_id"] else "prior_off"): r
          for r in rows}
    samples = sorted({k[0] for k in by})
    fig, axes = plt.subplots(1, len(samples), figsize=(3.6 * len(samples), 3.9),
                             squeeze=False, sharey=False)
    for i, sk in enumerate(samples):
        ax = axes[0][i]
        ax.axhline(0.0, color="0.6", lw=0.8)
        shown_true = False
        for arm, colour in (("prior_off", "C7"), ("prior_on", "C0")):
            r = by.get((sk, arm))
            if r is None:
                continue
            zp = PD.load_plot_arrays(
                PD.plot_arrays_path(os.path.join(runs_root, "runs", r["run_id"])))
            k = np.asarray(zp["dispersion_k"], float)
            ax.plot(k, np.asarray(zp["dispersion_sigma"], float), lw=1.7, color=colour,
                    label=f"{arm} (plaus={r.get('plausibility_score'):.2f})")
            if not shown_true and "dispersion_sigma_true" in zp:
                st = np.asarray(zp["dispersion_sigma_true"], float)
                ax.plot(k, st, lw=1.5, color="C3", label="answer key")
                ax.axvline(zp["meta"]["kstar_dispersion_true"], color="C3", ls="--", lw=1.0)
                shown_true = True
        # ZOOM on the band that decides the verdict, the same trick
        # make_figures.fig_dispersion uses: the answer key's far tail dives to large
        # negative values and would otherwise squash the recovered curves onto sigma = 0,
        # which is exactly the distinction this panel exists to show.
        ys = [np.asarray(l.get_ydata(), float) for l in ax.lines
              if len(l.get_ydata()) > 2]
        xs = [np.asarray(l.get_xdata(), float) for l in ax.lines
              if len(l.get_ydata()) > 2]
        hi = max([float(y.max()) for y in ys] + [1e-3])
        # the lower bound must still contain every RECOVERED curve near k=0, or a model
        # that is stable by a wide margin drops off the axis and reads as absent.
        lo = min([float(y[x <= 0.35].min()) for x, y in zip(xs, ys) if (x <= 0.35).any()]
                 + [-3.0 * hi])
        ax.set_ylim(lo * 1.35, hi * 1.45)
        ax.set_xlim(0, 1.2)
        ax.set_xlabel("$k$ (rad/length)")
        if i == 0:
            ax.set_ylabel("$\\sigma(k)=\\max_i\\mathrm{Re}\\,\\mathrm{eig}(J-k^2D)$")
        ax.set_title(sk, fontsize=9)
        ax.legend(fontsize=6.5, loc="lower left")
    fig.suptitle("Stage 0 part 4 — the recovered dispersion relation against the answer "
                 "key. Both arms sit ENTIRELY BELOW zero and peak at $k\\to0$: the "
                 "recovered models are linearly stable,\nso nothing can grow from noise. "
                 "The prior changes the plausibility score, not the sign of $\\sigma$.",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return _save(fig, "f5_part4_dispersion.png")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta-mode", default="generator")
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["f1", "f2", "f3", "f4", "f5"])
    ap.add_argument("--runs-root", default="experiments/stage0_bioviab")
    a = ap.parse_args(argv)
    want = set(a.only) if a.only else {"f1", "f2", "f3", "f4", "f5"}
    if "f1" in want:
        fig_box_acceptance(a.beta_mode)
    if "f2" in want:
        fig_mu_boundary(a.beta_mode)
        fig_mu_per_species(a.beta_mode)
    if "f3" in want:
        fig_pattern_gallery(a.beta_mode)
    if "f4" in want:
        fig_part4(a.runs_root)
    if "f5" in want:
        fig_part4_dispersion(a.runs_root)


if __name__ == "__main__":
    main()
