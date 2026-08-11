"""
rngrn_topology.py
================
Flexible "Learned GRN Topology" plot for an RNGRN model. Reference implementation.

Draws the learned network as a gene regulatory diagram:
  * genes as nodes (layout auto-scales with N: a line for N=2, a circle for N>2),
  * activating edges with a pointed arrowhead, repressing edges with a blunt bar end,
  * edge line width encoding the binding-strength magnitude (log-scaled),
  * self-loops for auto-regulation,
  * diffusion drawn as a dashed arrow at each node (width encodes D),
  * a per-node box annotating the learned alpha, beta, delta,
  * edge labels for KA / KR.

N-agnostic: nothing is hard-coded to a particular gene count; the same call works as N grows or
shrinks. Reads ONLY the model's learned parameters (KA, KR, alpha, beta, delta, D) - no analytic
quantity of any ground-truth system.

Packages: networkx (node layout) + matplotlib (drawing). Both are standard; install with
`pip install networkx matplotlib`.

Convention: KA[i, j] / KR[i, j] is the activating / repressing strength of regulator j on target i,
so an edge is drawn from node j to node i.

Usage:
    from rngrn_topology import plot_topology
    fig = plot_topology(model, title="Learned GRN Topology")
    fig.savefig("grn_topology.png", dpi=200, bbox_inches="tight")
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import networkx as nx


def _lighten(color, amount=0.55):
    """Blend a color toward white by `amount` (0=unchanged, 1=white)."""
    import matplotlib.colors as mc
    c = np.array(mc.to_rgb(color))
    return tuple(c + (1.0 - c) * amount)


def _sci(v):
    """Format a value as mathtext 'm x 10^e' (matches the reference image style)."""
    if v is None or not np.isfinite(v) or v == 0:
        return r"$0$"
    e = int(np.floor(np.log10(abs(v))))
    m = v / (10.0 ** e)
    return rf"${m:.1f}\times10^{{{e}}}$"


def _layout(N):
    """Node positions. N=2 -> horizontal line (matches the reference); N>2 -> circle."""
    if N == 2:
        return {0: np.array([-1.0, 0.0]), 1: np.array([1.0, 0.0])}
    G = nx.cycle_graph(N)
    pos = nx.circular_layout(G, scale=1.3)
    return {i: np.asarray(pos[i]) for i in range(N)}


def _width(mag, vmin, vmax, wmin=0.6, wmax=4.0):
    """Log-scaled line width from a magnitude, clamped to [wmin, wmax]."""
    if not np.isfinite(mag) or mag <= 0:
        return wmin
    lo, hi = np.log10(vmin), np.log10(vmax)
    if hi <= lo:
        return 0.5 * (wmin + wmax)
    t = (np.log10(mag) - lo) / (hi - lo)
    return float(np.clip(wmin + t * (wmax - wmin), wmin, wmax))


def _edge(ax, p_src, p_dst, rad, width, kind, color, node_r, sep=0.0):
    """Draw one regulatory edge from p_src to p_dst, trimmed to the node circles.
    kind='act' -> pointed arrow; kind='rep' -> blunt bar end.
    `sep` offsets both endpoints perpendicular to the chord, so two parallel edges (e.g. KA + KR on
    the same ordered pair) attach at DISTINCT points on the node rim and their arrowheads never pile
    up. Returns the label anchor (the arc's mid-peak)."""
    d = p_dst - p_src
    L = np.linalg.norm(d) + 1e-12
    u = d / L
    perp = np.array([-u[1], u[0]])
    a = p_src + u * node_r + perp * sep
    b = p_dst - u * node_r + perp * sep
    style = "-|>" if kind == "act" else "-["
    mut = 6 + 0.5 * width if kind == "act" else 5 + 0.5 * width
    arr = FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}", arrowstyle=style,
                          mutation_scale=mut, lw=width, color=color,
                          shrinkA=0, shrinkB=0, zorder=2, capstyle="butt")
    ax.add_patch(arr)
    mid = 0.5 * (a + b)
    chord = np.linalg.norm(b - a)
    return mid + perp * (rad * chord + np.sign(sep if sep != 0 else rad) * 0.02)


def _label(ax, xy, text, color="#111111"):
    """Edge/loop label with a white background so crossings stay legible."""
    ax.text(xy[0], xy[1], text, fontsize=8, ha="center", va="center", zorder=6,
            color=color, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


def _self_loop(ax, p, width, kind, color, node_r, outdir):
    """Draw an auto-regulation self-loop just outside the node, bulging along `outdir`.
    Returns the label anchor."""
    o = outdir / (np.linalg.norm(outdir) + 1e-9)
    tang = np.array([-o[1], o[0]])
    a = p + node_r * (o * 0.7 - tang * 0.5)
    b = p + node_r * (o * 0.7 + tang * 0.5)
    style = "-|>" if kind == "act" else "-["
    arr = FancyArrowPatch(a, b, connectionstyle="arc3,rad=2.4", arrowstyle=style,
                          mutation_scale=7 + 0.6 * width, lw=width, color=color,
                          shrinkA=0, shrinkB=0, zorder=2)
    ax.add_patch(arr)
    return p + o * (node_r + 0.42)


def plot_topology(model, title="Learned GRN Topology", labels=None, threshold_frac=0.02,
                  show_both=True, figsize=None, node_colors=None):
    """Render the learned RNGRN topology.

    model          : an RNGRN instance (reads .KA .KR .alpha .beta .delta .D .N).
    labels         : optional list of gene labels (default x1..xN).
    threshold_frac : hide edges whose magnitude is below this fraction of the max binding strength.
    show_both      : draw both KA and KR for a pair when both exceed threshold (as in the reference);
                     if False, draw only the dominant regulation per ordered pair.
    node_colors    : optional list of colors, one per node.
    Returns the matplotlib Figure.
    """
    KA = np.asarray(model.KA.detach().cpu().numpy() if hasattr(model.KA, "detach") else model.KA, float)
    KR = np.asarray(model.KR.detach().cpu().numpy() if hasattr(model.KR, "detach") else model.KR, float)
    alpha = np.asarray(model.alpha.detach().cpu().numpy() if hasattr(model.alpha, "detach") else model.alpha, float)
    beta = np.asarray(model.beta.detach().cpu().numpy() if hasattr(model.beta, "detach") else model.beta, float)
    delta = np.asarray(model.delta.detach().cpu().numpy() if hasattr(model.delta, "detach") else model.delta, float)
    D = np.asarray(model.D.detach().cpu().numpy() if hasattr(model.D, "detach") else model.D, float)
    N = int(model.N)
    if labels is None:
        labels = [rf"$x_{{{i+1}}}$" for i in range(N)]

    pos = _layout(N)
    allmag = np.concatenate([KA.ravel(), KR.ravel()])
    allmag = allmag[allmag > 0]
    vmax = allmag.max() if allmag.size else 1.0
    vmin = max(allmag.min(), vmax * 1e-6) if allmag.size else 1e-6
    thr = threshold_frac * vmax

    ACT = "#111111"; REP = "#111111"
    if node_colors is None:
        cmap = plt.get_cmap("coolwarm")
        node_colors = [cmap(0.15 + 0.7 * i / max(1, N - 1)) for i in range(N)]
    node_r = 0.46 if N == 2 else max(0.22, 0.46 * 2 / N)

    if figsize is None:
        figsize = (8.5, 3.6) if N == 2 else (7.5, 7.0)
    # Computer Modern mathtext renders x_i as the classic algebraic (italic serif) x, matching the
    # reference image, rather than the upright DejaVu Sans default.
    prev_fontset = plt.rcParams.get("mathtext.fontset")
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, ax = plt.subplots(figsize=figsize)

    # background rounded panel
    ax.add_patch(FancyBboxPatch((0.012, 0.012), 0.976, 0.976, transform=ax.transAxes,
                                boxstyle="round,pad=0.01,rounding_size=0.03",
                                fc="#e8e8e8", ec="none", zorder=0))

    # ---- edges (off-diagonal) ----
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            ka, kr = KA[i, j], KR[i, j]
            drawn = []
            if show_both:
                if ka > thr: drawn.append(("act", ka))
                if kr > thr: drawn.append(("rep", kr))
            else:
                if max(ka, kr) > thr:
                    drawn.append(("act", ka) if ka >= kr else ("rep", kr))
            for k, (kind, mag) in enumerate(drawn):
                # KA and KR on the SAME ordered pair get equal-and-opposite perpendicular offsets, so
                # their lines run parallel and their arrowheads land at DISTINCT points on the rim.
                # A gentle bow (sign following the offset) keeps a reciprocal j->i / i->j pair apart.
                s = 0.10 if kind == "act" else -0.10
                if len(drawn) == 1:
                    s = 0.0
                rad = 0.12 if kind == "act" else -0.12
                lp = _edge(ax, pos[j], pos[i], rad, _width(mag, vmin, vmax), kind, ACT, node_r, sep=s)
                sym = "K_A" if kind == "act" else "K_R"
                _label(ax, lp, rf"${sym}={_sci(mag)[1:-1]}$")

    # ---- self-loops (diagonal): bulge TANGENTIALLY, clear of the radial diffusion arrow ----
    centroid = np.mean([pos[i] for i in range(N)], axis=0)
    for i in range(N):
        ka, kr = KA[i, i], KR[i, i]
        radial = pos[i] - centroid
        if np.linalg.norm(radial) < 1e-6:
            radial = np.array([0.0, -1.0])
        radial = radial / (np.linalg.norm(radial) + 1e-9)
        if N == 2:
            loopdir = radial + np.array([0.0, -1.2])            # down-outward (matches reference)
            lab_off = node_r + 0.42
        else:
            loopdir = np.array([-radial[1], radial[0]])         # tangential: distinct sector from D
            if loopdir[1] > 0: loopdir = -loopdir               # bias downward, away from title
            lab_off = node_r + 0.55
        for kind, mag in ((("act", ka),) if ka >= kr else (("rep", kr),)):
            if mag > thr:
                _self_loop(ax, pos[i], _width(mag, vmin, vmax), kind, ACT, node_r, loopdir)
                o = loopdir / (np.linalg.norm(loopdir) + 1e-9)
                sym = "K_A" if kind == "act" else "K_R"
                _label(ax, pos[i] + o * lab_off, rf"${sym}={_sci(mag)[1:-1]}$")

    # ---- diffusion arrows (dashed, width ~ D): point UP-outward, clear of the self-loops ----
    Dmin = max(D.min(), D.max() * 1e-6); Dmax = D.max()
    for i in range(N):
        p = pos[i]
        radial = (p - centroid); radial = radial / (np.linalg.norm(radial) + 1e-9) if np.linalg.norm(p - centroid) > 1e-6 else np.array([0.0, 1.0])
        # N=2: up-outward (matches reference). N>2: straight radial-outward, box goes tangential.
        updir = (radial + np.array([0.0, 1.3])); updir = updir / (np.linalg.norm(updir) + 1e-9) if N == 2 else radial
        base = p + node_r * updir
        tip = base + 0.6 * updir
        w = _width(D[i], Dmin, Dmax, 0.6, 2.6)
        ax.add_patch(FancyArrowPatch(base, tip, arrowstyle="-|>", mutation_scale=8,
                                     lw=w, color="#222222", ls=":", shrinkA=0, shrinkB=0, zorder=2))
        lab = tip + updir * 0.22 + np.array([0, 0.12])
        _label(ax, lab, rf"$D_{{{labels[i][1:-1]}}}={D[i]:.3g}$")

    # ---- nodes + per-node parameter boxes ----
    for i in range(N):
        p = pos[i]
        ax.add_patch(Circle(p, node_r, facecolor=node_colors[i], edgecolor="#123",
                            linewidth=2.2, zorder=3))
        ax.text(p[0], p[1], labels[i], color="white", fontsize=18, ha="center", va="center",
                zorder=4)
        # parameter box, placed outward
        outward = p / (np.linalg.norm(p) + 1e-9) if np.linalg.norm(p) > 1e-6 else np.array([0, -1.0])
        if N == 2:
            bx = p + np.array([-1.0 if i == 0 else 1.0, 0]) * (node_r + 0.98)
        else:
            # box goes tangential (perpendicular to the radial diffusion arrow) so they never overlap
            tang = np.array([-outward[1], outward[0]])
            if tang[1] > 0:  # bias downward to avoid the title
                tang = -tang
            bx = p + outward * (node_r + 0.35) + tang * (node_r + 0.85)
        txt = (rf"$\alpha={_sci(alpha[i,i] if alpha.ndim==2 else alpha[i])[1:-1]}$" + "\n"
               rf"$\beta={_sci(beta[i])[1:-1]}$" + "\n"
               rf"$\delta={_sci(delta[i])[1:-1]}$")
        # box tinted to a lighter shade of its node color, for at-a-glance identification
        ax.text(bx[0], bx[1], txt, fontsize=8.5, ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.4", fc=_lighten(node_colors[i], 0.62),
                          ec=node_colors[i], lw=1.4))

    ax.set_title(title, fontsize=14, fontweight="bold")
    xs = [pos[i][0] for i in range(N)]; ys = [pos[i][1] for i in range(N)]
    pad = 2.0 if N == 2 else 1.5
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    plt.rcParams["mathtext.fontset"] = prev_fontset  # restore global state
    return fig
