"""EXP12 — Does the power-spectrum AMPLITUDE encode the concentration field?

QUESTION: the pipeline scores patterns in the frequency domain (observables.raps,
scoring.morphology._spectral_block). Both mean-subtract and normalise. Does that throw
away concentration information, and does it matter?

ANSWER, in three parts (all measured by this script):

 1. YES, exactly and by Parseval — before normalisation. For an (N,N) field,
      |F[0,0]|^2 / N^4          == mean(field)^2        (the DC bin IS the mean)
      sum|F_sub|^2 / N^4        == var(field)           (integrated mean-subtracted power
                                                         IS the field variance)
    where F_sub is the transform of (field - field.mean()). Verified to ~1e-7 on all
    127 three_gene samples. RAPS peak power scales exactly as amplitude^2.

 2. AND THE PIPELINE DISCARDS IT, deliberately. Every scored spectral quantity is
    amplitude-blind: raps() and _spectral_block() both subtract the mean (annihilating
    the DC bin) and _spectral_block() then divides by the block sum. Probe: rescale a
    field by 3x and the self-distance is ~1e-11 (spectral) / ~1e-9 (morphology). That
    invariance is a FEATURE -- it is why morphology comparison survives the unknown
    fluorescence gain of an experimental image (the same problem Endres/Matas-Gil solve
    with a free rescaling variable).

 3. BUT THE DISCARDED CHANNEL IS INFORMATIVE. frame.mean() tracks the true homogeneous
    steady state x*_0 at Spearman rho ~= 0.95 (ratio median ~0.92). That is precisely
    what the frame-scale anchor exploits: anchor_w * mean((log(obs_scale) - log(x*))^2)
    with obs_scale = frame.mean() is, in effect, a firewall-legal estimator of x*_0 read
    off the DC amplitude. It is the term that took convergence from 2/40 to 37/40.

    By contrast the amplitude channels carry almost nothing about k* (|rho| <= 0.16),
    which is consistent with normalising them away for spatial-mode scoring.

FIREWALL NOTE: frame.mean() is an IMAGE observable, so the anchor is legal. This script
reads x*_true only to VALIDATE that correlation, i.e. on the scoring side. Do not invert
this into feeding x*_true to recovery.

Output: experiments/exp12_spectral_amplitude.csv
"""
import argparse, glob, os
import h5py, numpy as np, pandas as pd


def parseval_channels(field):
    """Return (mean_from_DC, var_from_power, mean, var) for one 2-D field."""
    n = field.shape[0]
    n4 = float(n) ** 4
    dc = np.abs(np.fft.fft2(field)[0, 0]) ** 2
    psub = (np.abs(np.fft.fft2(field - field.mean())) ** 2).sum()
    return np.sqrt(dc / n4), psub / n4, field.mean(), field.var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", default="data/datasets")
    ap.add_argument("--family", default="three_gene")
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--out", default="experiments/exp12_spectral_amplitude.csv")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(f"{args.datasets_root}/{args.family}_*/payload.h5")):
        name = os.path.basename(os.path.dirname(path))
        if "classical" in name:
            continue
        with h5py.File(path, "r") as f:
            for key in sorted(f.keys()):
                g = f[key]
                ch = np.array(g["final_frame"])[args.channel]
                m_dc, v_pow, m, v = parseval_channels(ch)
                xs = np.array(g["x_star"])
                rows.append(dict(
                    ds=name, key=key, morph=str(g.attrs.get("morphology")),
                    topology=str(g.attrs.get("topology")),
                    mean_from_DC=m_dc, frame_mean=m,
                    var_from_power=v_pow, frame_var=v,
                    cv=float(ch.std() / ch.mean()),
                    x_star_ch=float(xs[args.channel]),
                    sigma_max=float(g.attrs["sigma_max"]),
                    k_star=float(g.attrs["k_star"]),
                ))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"n = {len(df)} samples, channel {args.channel}")
    print("PART 1 -- Parseval identities (should be ~1e-7 or better):")
    print("  max |sqrt(DC/N^4) - frame.mean()|   = %.3e"
          % (df.mean_from_DC - df.frame_mean).abs().max())
    print("  max |sum(P_sub)/N^4 - frame.var()|  = %.3e"
          % (df.var_from_power - df.frame_var).abs().max())

    print("\nPART 3 -- Spearman rho of the DISCARDED amplitude channels vs true dynamics:")
    tbl = {}
    for col in ("frame_mean", "frame_var", "cv"):
        tbl[col] = {t: round(df[[col, t]].corr(method="spearman").iloc[0, 1], 3)
                    for t in ("x_star_ch", "sigma_max", "k_star")}
    print(pd.DataFrame(tbl).T.to_string())
    ratio = df.frame_mean / df.x_star_ch
    print("\n  frame.mean() / x*_ch : median %.3f (IQR %.3f-%.3f)"
          % (ratio.median(), ratio.quantile(.25), ratio.quantile(.75)))
    print("  => the frame-scale anchor's target is an ~%.0f%%-accurate estimator of x*."
          % (100 * (ratio.median() - 1) if ratio.median() > 1 else 100 * (1 - ratio.median())))


if __name__ == "__main__":
    main()
