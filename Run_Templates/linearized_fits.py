"""
linearized_fits.py
==================
Exploratory fit tests requested by Roy, for the scaling panels Fig. 4(b),
4(c) and 4(d).  Panel 4(e) ({8,4}) is Chris's data and is not handled here.

Two tests, both applied to the same three panels:

(A) LINEARIZED POWER LAW
    The scaling ansatz is

        rho(0) = A * delta^beta + B

    where B is the value of rho at delta = 0.  Solving for delta,

        delta = ( (rho(0) - B) / A )^(1/beta)

    so plotting

        Y = ( (rho - B) / A )^(1/beta)     against     X = delta

    must fall on the line Y = X -- slope exactly 1, intercept exactly 0,
    with no free parameters left over.  A and beta are taken from the
    log-log fit of the SAME data, and B from rho at the critical point.

    NOTE ON THE ALGEBRA.  The exponent is 1/beta, not -beta: the email's
    "rho^{-beta} ~ delta" inverts the wrong way.  Dividing by A before
    taking the root matters too -- without it the expected slope is the
    arbitrary number A^(-1/beta) instead of 1, and the "guide line to the
    origin" tests nothing.

(B) ESSENTIAL SINGULARITY (Bethe-lattice form)

        rho(0) = A * exp( -c * delta^(-1/2) )
        =>  ln(rho - B) = ln(A) - c * delta^(-1/2)

    so ln(rho - B) plotted against delta^(-1/2) should be a straight line
    of slope -c.

Both models are two-parameter fits to the same response variable,
ln(rho - B), over the same points, so their residual scatters are
directly comparable -- the one with the smaller sigma_res is the better
description of the data.  That comparison is printed as a summary table.

Usage:
    python linearized_fits.py

Requires KPM_analysis_manuscript.py and plot_utils.py alongside it.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import fixed_box_figure
from KPM_analysis_manuscript import (SYSTEMS, PLOT_BASE_TDOS,
                                     get_ados_data, get_tdos_data)


OUT_DIR = os.path.join(PLOT_BASE_TDOS, "exploratory_fits")
alpha=0.5

# ---------------------------------------------------------------------------
# Panel definitions.  One entry per scaling panel of Fig. 4.
#
#   observable : 'ados' or 'tdos'
#   W_c        : critical disorder strength.  It does NOT have to be one of
#                the simulated weights -- it only enters through delta.
#   side       : 'above' -> delta = (W - W_c)/W_c   (metallic side, DSM-metal)
#                'below' -> delta = (W_c - W)/W_c   (Anderson transition)
#   W_lo, W_hi : inclusive fit window in W
#   B_mode     : 'zero'      -> B = 0
#                'plateau'   -> B = mean of rho over the W in B_window,
#                               i.e. the asymptotic value on the side where
#                               the order parameter vanishes
# ---------------------------------------------------------------------------
PANELS = {
    '4b': dict(label="4(b)  {10,3} DSM-metal",
               sys_name='p10_n7', observable='ados',
               W_c=0.60, side='above', W_lo=0.60, W_hi=1.25,
               B_mode='critical', B_window=(0.00, 0.50)),

    '4c': dict(label="4(c)  {10,3} Anderson",
               sys_name='p10_n7', observable='tdos',
               W_c=10.45, side='below', W_lo=4.00, W_hi=7.00, # power law W_c was 7.00, essential singularity W_c = 10.45
               B_mode='zero'),

    '4d': dict(label="4(d)  {8,3} Anderson",
               sys_name='p8_n9', observable='tdos',
               W_c=8.25, side='below', W_lo=3.00, W_hi=6.00, # power law W_c was 6.00, essential singularity W_c = 8.25
               B_mode='zero'),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_panel(panel):
    """Return (W, rho, delta, B) for one panel, restricted to the fit window."""
    cfg = SYSTEMS[panel['sys_name']]

    if panel['observable'] == 'ados':
        rho_all = np.asarray(get_ados_data(panel['sys_name'], cfg), dtype=float)
        w_all = np.asarray(cfg['wm_list_ados'], dtype=float)
    else:
        rho_all = np.asarray(get_tdos_data(panel['sys_name'], cfg)[0], dtype=float)
        w_all = np.asarray(cfg['wm_list_tdos'], dtype=float)

    if len(rho_all) != len(w_all):
        raise ValueError(f"{panel['label']}: {len(rho_all)} values for "
                         f"{len(w_all)} weights -- cache and weight list "
                         f"are out of sync.")

    # baseline B
    if panel['B_mode'] == 'zero':
        B = 0.0
    elif panel['B_mode'] == 'critical':
        B = float(rho_all[np.isclose(w_all, panel['W_c'])][0])
    else:
        lo, hi = panel['B_window']
        sel = (w_all >= lo) & (w_all <= hi)
        B = float(np.mean(rho_all[sel]))

    m = (w_all >= panel['W_lo']) & (w_all <= panel['W_hi'])
    W, rho = w_all[m], rho_all[m]

    W_c = panel['W_c']
    delta = (W - W_c) / W_c if panel['side'] == 'above' else (W_c - W) / W_c

    keep = (delta > 1e-9) & (rho - B > 1e-12)
    if keep.sum() < 4:
        raise ValueError(f"{panel['label']}: only {keep.sum()} usable points.")

    return W[keep], rho[keep], delta[keep], B


# ---------------------------------------------------------------------------
# Fits.  Both are straight-line least squares on ln(rho - B), so their
# residual scatters can be compared directly.
# ---------------------------------------------------------------------------

def fit_power_law(delta, rho, B):
    """ln(rho - B) = ln A + beta * ln(delta).  Returns dict."""
    x, y = np.log(delta), np.log(rho - B)
    c, V = np.polyfit(x, y, 1, cov=True)
    resid = y - np.polyval(c, x)
    n = len(x)
    return dict(beta=c[0], beta_err=float(np.sqrt(V[0, 0])),
                A=float(np.exp(c[1])),
                sres=float(np.sqrt((resid ** 2).sum() / (n - 2))),
                resid=resid, n=n)


def fit_essential(delta, rho, B):
    """ln(rho - B) = ln A - c * delta^(-alpha).  Returns dict."""
    x, y = delta ** -alpha, np.log(rho - B) #using alpha = 1/2 right now
    c, V = np.polyfit(x, y, 1, cov=True)
    resid = y - np.polyval(c, x)
    n = len(x)
    return dict(c=-c[0], c_err=float(np.sqrt(V[0, 0])),
                A=float(np.exp(c[1])), slope=c[0], intercept=c[1],
                sres=float(np.sqrt((resid ** 2).sum() / (n - 2))),
                resid=resid, n=n)


def slope_through_origin(x, y):
    """Least-squares m for y = m x, plus R^2 about that one-parameter model."""
    m = float((x * y).sum() / (x * x).sum())
    ss_res = float(((y - m * x) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return m, 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Plots.  Same conventions as the manuscript figures: fixed 6x6" box from
# plot_utils, thick spines, inward ticks, SVG output.
# ---------------------------------------------------------------------------

def _style(ax):
    ax.spines[:].set_linewidth(1.5)
    ax.tick_params(axis='both', direction='in', length=4, width=1.5, pad=5)


def plot_linearized(key, panel, delta, rho, B, pl):
    """Y = ((rho - B)/A)^(1/beta) against delta.  Should lie on Y = X."""
    Y = ((rho - B) / pl['A']) ** (1.0 / pl['beta'])
    m, r2 = slope_through_origin(delta, Y)

    fig, ax = fixed_box_figure(right=1.5)
    hi = max(delta.max(), Y.max()) * 1.05
    ax.plot([0, hi], [0, hi], ls='--', color='red', linewidth=1.0,
            label=r"$Y = \delta$ (exact)")
    ax.plot(delta, Y, ls='', marker='o', color='black', markersize=6)

    ax.set_xlabel(r"$\delta$" if panel['side'] == 'above' else r"$\delta_{\rm A}$")
    ax.set_ylabel(r"$\left[(\rho(0)-B)/A\right]^{1/\beta}$", labelpad=12)
    ax.set_title(f"Linearized power law, {panel['label']}\n"
                 rf"$\beta = {pl['beta']:.3f} \pm {pl['beta_err']:.3f}$, "
                 rf"slope $= {m:.3f}$, $R^2 = {r2:.4f}$")
    if key == '4b':
        ax.set_xlim(0.0, 1.1)
        ax.set_xticks([0.0, 0.55, 1.1])
        ax.set_ylim(0.0, 1.1)
        ax.set_yticks([0.0, 0.55, 1.1])
    elif key == '4c':
        ax.set_xlim(0.0, 0.5)
        ax.set_xticks([0.0, 0.25, 0.5])
        ax.set_ylim(0.0, 0.6)
        ax.set_yticks([0.0, 0.3, 0.6])
    elif key == '4d':
        ax.set_xlim(0.0, 0.5)
        ax.set_xticks([0.0, 0.25, 0.5])
        ax.set_ylim(0.0, 0.6)
        ax.set_yticks([0.0, 0.3, 0.6])


    ax.legend(loc='upper left', frameon=False)
    _style(ax)

    path = os.path.join(OUT_DIR, f"Linearized_powerlaw_{key}_{panel['sys_name']}.svg")
    fig.savefig(path)
    plt.close(fig)
    return path, m, r2


def plot_essential(key, panel, delta, rho, B, es):
    """ln(rho - B) against delta^(-1/2).  Should be a straight line."""
    x, y = delta ** -alpha, np.log(rho - B)

    fig, ax = fixed_box_figure(right=1.5)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, es['slope'] * xx + es['intercept'], ls='--', color='red', marker='',
            linewidth=1.0, label="linear fit")
    ax.plot(x, y, ls='', marker='o', color='black', markersize=6)

    ax.set_xlabel(r"$\delta^{-1/2}$" if panel['side'] == 'above'
                  else r"$\delta_{\rm A}^{-1/2}$")
    ax.set_ylabel(r"$\ln\left[\rho(0)-B\right]$", labelpad=12)

   if key == '4b':
        ax.set_xlim(0.0, 1.1)
        ax.set_xticks([0.0, 0.55, 1.1])
        ax.set_ylim(0.0, 1.1)
        ax.set_yticks([0.0, 0.55, 1.1])
    elif key == '4c':
        ax.set_xlim(0.0, 0.5)
        ax.set_xticks([0.0, 0.25, 0.5])
        ax.set_ylim(0.0, 0.6)
        ax.set_yticks([0.0, 0.3, 0.6])
    elif key == '4d':
        ax.set_xlim(0.0, 0.5)
        ax.set_xticks([0.0, 0.25, 0.5])
        ax.set_ylim(0.0, 0.6)
        ax.set_yticks([0.0, 0.3, 0.6])
    ax.set_title(f"Essential singularity, {panel['label']}\n"
                 rf"$c = {es['c']:.3f} \pm {es['c_err']:.3f}$, "
                 rf"$\sigma_{{\rm res}} = {es['sres']:.3f}$")
    ax.legend(loc='upper right', frameon=False)
    _style(ax)

    path = os.path.join(OUT_DIR, f"Essential_singularity_{key}_{panel['sys_name']}.svg")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_residual_comparison(key, panel, delta, pl, es):
    """Both models' log-space residuals on one axes, for the direct comparison."""
    fig, ax = fixed_box_figure(right=1.5)
    ax.axhline(0.0, color='black', linewidth=1.0)
    ax.plot(delta, pl['resid'], ls='', marker='o', color='black',
            markersize=7, label=rf"power law ($\sigma={pl['sres']:.3f}$)")
    ax.plot(delta, es['resid'], ls='', marker='s', color='tab:orange',
            markersize=7, label=rf"essential ($\sigma={es['sres']:.3f}$)")

    ax.set_xlabel(r"$\delta$" if panel['side'] == 'above' else r"$\delta_{\rm A}$")
    ax.set_ylabel(r"$\ln\rho_{\rm data} - \ln\rho_{\rm fit}$", labelpad=12)
    ax.set_title(f"Log-space residuals, {panel['label']}")
    ax.legend(loc='best', frameon=False)
    _style(ax)

    path = os.path.join(OUT_DIR, f"Residual_comparison_{key}_{panel['sys_name']}.svg")
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_panel(key, panel):
    W, rho, delta, B = load_panel(panel)
    pl = fit_power_law(delta, rho, B)
    es = fit_essential(delta, rho, B)

    os.makedirs(OUT_DIR, exist_ok=True)
    p1, m, r2 = plot_linearized(key, panel, delta, rho, B, pl)
    p2 = plot_essential(key, panel, delta, rho, B, es)
    p3 = plot_residual_comparison(key, panel, delta, pl, es)

    print(f"\n=== {panel['label']} ===")
    print(f"  W_c = {panel['W_c']:.2f},  window W in [{panel['W_lo']}, {panel['W_hi']}],"
          f"  B = {B:.3e},  n = {pl['n']}")
    print(f"  power law   : beta = {pl['beta']:.3f} +- {pl['beta_err']:.3f},"
          f"  A = {pl['A']:.4f},  sigma_res = {pl['sres']:.3f}")
    print(f"  linearized  : slope = {m:.3f} (expect 1),  R^2 = {r2:.4f}")
    print(f"  essential   : c    = {es['c']:.3f} +- {es['c_err']:.3f},"
          f"  A = {es['A']:.4f},  sigma_res = {es['sres']:.3f}")
    print(f"  saved: {p1}")
    print(f"         {p2}")
    print(f"         {p3}")

    return dict(key=key, label=panel['label'], n=pl['n'], beta=pl['beta'],
                beta_err=pl['beta_err'], slope=m, r2=r2, c=es['c'],
                sres_pl=pl['sres'], sres_es=es['sres'])


def main():
    rows = []
    for key, panel in PANELS.items():
        try:
            rows.append(run_panel(key, panel))
        except Exception as exc:
            print(f"\n=== {panel['label']} === FAILED: {exc}")

    if not rows:
        return

    print("\n" + "=" * 78)
    print("MODEL COMPARISON  (same data, same response ln(rho-B), 2 parameters each)")
    print("=" * 78)
    print(f"{'panel':<24}{'n':>3}  {'beta':>13}  {'lin.slope':>10}"
          f"  {'s_power':>8}  {'s_essent':>9}  {'preferred':>10}")
    for r in rows:
        pref = "power law" if r['sres_pl'] < r['sres_es'] else "essential"
        print(f"{r['label']:<24}{r['n']:>3}  "
              f"{r['beta']:>6.3f}+-{r['beta_err']:<6.3f}  {r['slope']:>10.3f}"
              f"  {r['sres_pl']:>8.3f}  {r['sres_es']:>9.3f}  {pref:>10}")
    print("=" * 78)


if __name__ == "__main__":
    main()
