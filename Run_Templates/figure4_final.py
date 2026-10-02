"""
figure4_final.py
================
Final, cleaned-up Fig. 4 panels.  One script, twelve figures, the exact
parameters agreed with Bitan and nothing else.

    4(a)   alpha_a  from rho_a(E) ~ |E|^alpha       W = 0.60
    4(b)   beta_a   from rho_a(0) ~ delta^beta      W_c1 = 0.60
    4(b')  beta_t   from rho_t(0) ~ delta^beta      W_c1 = 0.40
    4(c)   beta_A   {10,3} Anderson                 W_c2 = 7.00
    4(d)   beta_A   {8,3}  Anderson                 W_c2 = 6.00

Each of 4(b), 4(b'), 4(c), 4(d) gets three panels: the power law itself,
its linearized form, and the essential-singularity test with the critical
point that form predicts.

Conventions fixed here and not to be changed silently:
  * offsets -- Anderson side B = 0 (rho_t is measured down to 1e-7..1e-8,
    which bounds any constant floor far below anything a fit would want);
    DSM-metal side B is real KPM broadening and must be subtracted.
  * the linearized form is  Y = [(rho - B)/A]^(1/beta)  vs delta, which must
    lie on Y = delta -- exponent 1/beta, and divided by A so that the
    prediction has slope exactly 1 with no free parameters left.
  * the essential-singularity form is rho - B = A exp(-c delta^-alpha) with
    alpha = 1/2 held at the Bethe / mean-field value.
  * every fit line is drawn across the full axis range, never stopping
    short of the frame.

Usage:
    python figure4_final.py

Requires KPM_analysis_manuscript.py and plot_utils.py alongside it.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import fixed_box_figure
from KPM_analysis_manuscript import (SYSTEMS, PLOT_BASE_TDOS,
                                     get_ados_data, get_tdos_data,
                                     get_energy_resolved_data)

OUT_DIR = os.path.join(PLOT_BASE_TDOS, "figure4_final")

# ===========================================================================
# AXIS OVERRIDES -- the one place to adjust framing, one entry per figure.
#
# Each key is exactly the file name the figure is saved under.  Set
#   xlim=(a, b)   -> x-axis runs a..b with ticks at  [a, (a+b)/2, b]
#   ylim=(a, b)   -> same for y
# Leave a value as None to keep the automatic limits for that axis.
# Nothing else about the fit changes: the fit line and its uncertainty
# bands are always redrawn across whatever range you set here, so they
# never stop short of the frame.
#
# Example:
#     'Fig4c_powerlaw': dict(xlim=(0.0, 0.45), ylim=(0.0, 0.032)),
# ===========================================================================
AXES = {
    # --- 4(a) alpha ---------------------------------------------------
    'Fig4a_alpha_ADOS':   dict(xlim=(0.00, 0.02), ylim=(0.000, 0.008)),

    # --- 4(b)  ADOS, W_c1 = 0.55 --------------------------------------
    'Fig4b_powerlaw':     dict(xlim=(0.00, 2.8), ylim=(0.00, 0.04)),
    'Fig4b_linearized':   dict(xlim=(0.00, 3.00), ylim=(0.0, 3.0)),
    'Fig4b_essential':    dict(xlim=(0.68, 1.34), ylim=(-8.4, -4.4)),

    # --- 4(b') TDOS, W_c1 = 0.40 --------------------------------------
    'Fig4bp_powerlaw':    dict(xlim=(0.0, 1.6), ylim=(0.000, 0.036)),
    'Fig4bp_linearized':  dict(xlim=(0.0,1.6), ylim=(0.0, 1.6)),
    'Fig4bp_essential':   dict(xlim=(0.5, 1.6), ylim=(-5.9, -2.9)),

    # --- 4(c)  {10,3} Anderson, W_c2 = 7.00 ---------------------------
    'Fig4c_powerlaw':     dict(xlim=(0.0, 0.46), ylim=(0.0, 0.03)),
    'Fig4c_linearized':   dict(xlim=(0.0, 0.56), ylim=(0.0, 0.56)),
    'Fig4c_essential':    dict(xlim=(1.2, 1.7), ylim=(-8.4, -3.0)),

    # --- 4(d)  {8,3} Anderson, W_c2 = 6.00 ----------------------------
    'Fig4d_powerlaw':     dict(xlim=(0.00, 0.54), ylim=(0.00,0.056)),
    'Fig4d_linearized':   dict(xlim=(0.00, 0.60), ylim=(0.00, 0.62)),
    'Fig4d_essential':    dict(xlim=(1.21, 1.85), ylim=(-7.96, -2.60)),
}

# Number of decimals on the three tick labels; per figure, else the default.
TICK_FMT = {}
TICK_FMT_DEFAULT = None        # None -> let matplotlib choose


MARKER = dict(ls='', marker='o', color='black', markersize=6, zorder=3)
FITLINE = dict(ls='-', marker='', color='red', linewidth=1.2, zorder=2)
BANDLINE = dict(ls='--', marker='', color='red', linewidth=1.0, alpha=0.6, zorder=1)


# ===========================================================================
# Panel definitions -- the green-flagged parameters
# ===========================================================================

# --- 4(a): alpha from the energy-resolved ADOS ----------------------------
# The energy grid must be the pi*a/N_m resolution grid, which
# get_energy_resolved_data builds only when zoom_window is passed -- the dense
# 101-point linspace oversamples below the KPM resolution and shifts alpha.
ALPHA_PANEL = dict(
    sys_name='p10_n7', W=0.55, E_min=0.002, E_max=0.019,
    zoom_window=(-0.04, 0.04),
    title=r"4(a)  $\rho_{\rm a}(E)\sim|E|^{\alpha_{\rm a}}$, $W = 0.60$",
)

# --- 4(b), 4(b'), 4(c), 4(d): power law + linearized + essential ----------
# side   : 'above' -> delta = (W - W_c)/W_c ; 'below' -> (W_c - W)/W_c
# offset : 'critical' | 'plateau' | 'zero'
# ess_grid: (lo, hi, step) of trial W_c for the essential-singularity scan
PANELS = [
    dict(key='4b', label=r"4(b)  $\{10,3\}$ DSM-metal, ADOS",
         sys_name='p10_n7', observable='ados', side='above',
         W_c=0.55, W_lo=0.55, W_hi=2.0, hi_inclusive=True,
         offset='critical', sym=r"\beta_{\rm a}", rho=r"\rho_{\rm a}(0)",
         ess_grid=(0.05, 0.60, 0.002)),

    dict(key='4bp', label=r"4(b$'$)  $\{10,3\}$ DSM-metal, TDOS",
         sys_name='p10_n7', observable='tdos', side='above',
         W_c=0.40, W_lo=0.40, W_hi=1.00, hi_inclusive=True,
         offset='plateau', plateau=(0.00, 0.25),
         sym=r"\beta_{\rm t}", rho=r"\rho_{\rm t}(0)",
         ess_grid=(0.05, 0.40, 0.002), ess_window=(0.41, 1.00)),

    dict(key='4c', label=r"4(c)  $\{10,3\}$ Anderson",
         sys_name='p10_n7', observable='tdos', side='below',
         W_c=7.00, W_lo=4.00, W_hi=6.75, hi_inclusive=True,
         offset='zero', sym=r"\beta_{\rm A}", rho=r"\rho_{\rm t}(0)",
         ess_grid=(6.80, 20.00, 0.05)),

    dict(key='4d', label=r"4(d)  $\{8,3\}$ Anderson",
         sys_name='p8_n9', observable='tdos', side='below',
         W_c=6.00, W_lo=3.00, W_hi=6.00, hi_inclusive=True,
         offset='zero', sym=r"\beta_{\rm A}", rho=r"\rho_{\rm t}(0)",
         # W = 6.00 sits at delta_A = 0 and drops out of the power-law fit;
         # the essential scan is held to the same 11 points so that the two
         # sigma_res are comparable.
         ess_grid=(6.05, 20.00, 0.05), ess_window=(3.00, 5.75)),
]

ESS_ALPHA = 0.5          # Bethe / mean-field value, held fixed
ESS_OFFSET = 'zero'

# ===========================================================================
# Helpers
# ===========================================================================

def _style(ax):
    ax.spines[:].set_linewidth(1.5)
    ax.tick_params(axis='both', direction='in', length=4, width=1.5, pad=5)


def _load(panel):
    """(W, rho) over the full weight list for this panel's observable."""
    cfg = SYSTEMS[panel['sys_name']]
    if panel['observable'] == 'ados':
        rho = np.asarray(get_ados_data(panel['sys_name'], cfg), dtype=float)
        W = np.asarray(cfg['wm_list_ados'], dtype=float)
    else:
        rho = np.asarray(get_tdos_data(panel['sys_name'], cfg)[0], dtype=float)
        W = np.asarray(cfg['wm_list_tdos'], dtype=float)
    if len(rho) != len(W):
        raise ValueError(f"{panel['key']}: {len(rho)} values for {len(W)} "
                         f"weights -- cache and weight list out of sync.")
    return W, rho


def _offset(panel, W, rho):
    if panel['offset'] == 'zero':
        return 0.0
    if panel['offset'] == 'plateau':
        lo, hi = panel['plateau']
        return float(np.mean(rho[(W >= lo) & (W <= hi)]))
    hit = np.isclose(W, panel['W_c'])
    return float(rho[hit][0]) if hit.any() else float(np.interp(panel['W_c'], W, rho))


def _delta(panel, W, W_c=None):
    W_c = panel['W_c'] if W_c is None else W_c
    return (W - W_c) / W_c if panel['side'] == 'above' else (W_c - W) / W_c


def _select(panel, W, rho, B):
    """Points inside the fit window with delta > 0 and rho - B > 0."""
    m = (W >= panel['W_lo'])
    m &= (W <= panel['W_hi']) if panel['hi_inclusive'] else (W < panel['W_hi'])
    Ws, ys = W[m], rho[m] - B
    d = _delta(panel, Ws)
    keep = (d > 1e-9) & (ys > 1e-12)
    return d[keep], ys[keep]


def _powerlaw(d, y):
    """ln y = ln A + beta ln d.  Returns beta, err, A, sigma_res, n, cov."""
    c, cov = np.polyfit(np.log(d), np.log(y), 1, cov=True)
    r = np.log(y) - np.polyval(c, np.log(d))
    n = len(d)
    return (c[0], float(np.sqrt(cov[0, 0])), float(np.exp(c[1])),
            float(np.sqrt((r ** 2).sum() / (n - 2))), n, cov)


def _log_band(x, cov):
    """
    One-sigma uncertainty of ln y for the fitted line ln y = m ln x + b,
    propagating both parameters and their covariance:

        sigma_lny(x)^2 = (ln x)^2 var(m) + var(b) + 2 ln x cov(m, b)

    The band is then y * exp(+/- sigma_lny).  Same expression as the
    error_func inside fit_power_law in the manuscript script.
    """
    lx = np.log(np.clip(x, 1e-12, None))
    var = (lx ** 2) * cov[0, 0] + cov[1, 1] + 2.0 * lx * cov[0, 1]
    return np.sqrt(np.clip(var, 0.0, None))


def _essential(d, y, alpha=ESS_ALPHA):
    """ln y = ln A - c d^-alpha.  Returns c, err, lnA, sigma_res."""
    x = d ** (-alpha)
    coef, cov = np.polyfit(x, np.log(y), 1, cov=True)
    r = np.log(y) - np.polyval(coef, x)
    return (-coef[0], float(np.sqrt(cov[0, 0])), coef[1],
            float(np.sqrt((r ** 2).sum() / (len(d) - 2))))


def _three_ticks(lo, hi, fmt=None):
    """Ticks at the two ends and the midpoint."""
    vals = [lo, 0.5 * (lo + hi), hi]
    if fmt is None:
        return vals, None
    return vals, [format(v, fmt) for v in vals]


def _set_axes(ax, name, xlim=None, ylim=None):
    """
    Apply the limits for figure `name`: whatever AXES specifies, else the
    defaults passed in by the plotting function.  Wherever a range is set,
    exactly three ticks are placed, at both ends and the midpoint.
    Call this BEFORE drawing the fit line so the fit spans the final frame.
    """
    override = AXES.get(name, {})
    xr = override.get('xlim') or xlim
    yr = override.get('ylim') or ylim
    fmt = TICK_FMT.get(name, TICK_FMT_DEFAULT)

    if xr is not None:
        ax.set_xlim(*xr)
        vals, labels = _three_ticks(*xr, fmt=fmt)
        ax.set_xticks(vals)
        if labels:
            ax.set_xticklabels(labels)
    if yr is not None:
        ax.set_ylim(*yr)
        vals, labels = _three_ticks(*yr, fmt=fmt)
        ax.set_yticks(vals)
        if labels:
            ax.set_yticklabels(labels)
    return ax


def _full_span(ax, axis='x', pad=0.0):
    """Sample points spanning the whole current axis, so fits never stop short."""
    lo, hi = ax.get_xlim() if axis == 'x' else ax.get_ylim()
    return np.linspace(lo + pad, hi - pad, 400)


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name + ".svg")
    fig.savefig(path)
    plt.close(fig)
    return path


# ===========================================================================
# 4(a): the DOS exponent
# ===========================================================================

def panel_alpha():
    p = ALPHA_PANEL
    cfg = SYSTEMS[p['sys_name']]
    data = get_energy_resolved_data(p['sys_name'], cfg, mode="ADOS",
                                    zoom_window=p['zoom_window'])
    if p['W'] not in data:
        print(f"[4a] no energy-resolved ADOS at W = {p['W']}; skipping.")
        return None
    e, v = data[p['W']]['energies'], data[p['W']]['vals']

    # Symmetrise the two branches, as in the manuscript script: take the
    # positive energies for the x-axis and average rho_a(+E) with rho_a(-E)
    # to cancel numerical asymmetry.  The pi*a/N_m grid is symmetric about
    # zero, so the flipped negative branch lines up index by index.
    pos = (e >= p['E_min']) & (e <= p['E_max'])
    neg = (e <= -p['E_min']) & (e >= -p['E_max'])
    if pos.sum() != neg.sum():
        raise ValueError(f"[4a] branch mismatch: {pos.sum()} positive vs "
                         f"{neg.sum()} negative energies -- the grid is not "
                         f"symmetric about E = 0.")
    E = e[pos]
    R = (v[pos] + v[neg][::-1]) / 2.0

    coef, cov = np.polyfit(np.log(E), np.log(R), 1, cov=True)
    alpha, err, A = coef[0], float(np.sqrt(cov[0, 0])), float(np.exp(coef[1]))
    r = np.log(R) - np.polyval(coef, np.log(E))
    sres = float(np.sqrt((r ** 2).sum() / (len(E) - 2)))

    name = "Fig4a_alpha_ADOS"
    fig, ax = fixed_box_figure(right=1.2)
    ax.plot(E, R, **MARKER)
    _set_axes(ax, name,
              xlim=(0.0, float(E.max()) * 1.08),
              ylim=(0.0, float(R.max()) * 1.10))
    xx = _full_span(ax)
    yy = A * np.clip(xx, 1e-12, None) ** alpha
    band = _log_band(xx, cov)
    ax.plot(xx, yy * np.exp(band), **BANDLINE)
    ax.plot(xx, yy * np.exp(-band), **BANDLINE)
    ax.plot(xx, yy, **FITLINE,
            label=rf"$\alpha_{{\rm a}} = {alpha:.4f} \pm {err:.4f}$")
    ax.set_xlabel(r"$|E|$")
    ax.set_ylabel(r"$\rho_{\rm a}(E)$", rotation=0, labelpad=30)
    ax.set_title(p['title'])
    ax.legend(loc='upper left', frameon=False)
    _style(ax)
    path = _save(fig, name)

    print(f"4(a)  alpha_a = {alpha:.4f} +- {err:.4f}   "
          f"(n={len(E)}, E in [{p['E_min']}, {p['E_max']}], sres={sres:.4f})")
    return dict(key='4a', name='alpha_a', value=alpha, err=err,
                n=len(E), sres=sres, path=path)


# ===========================================================================
# Power law, linearized, essential singularity
# ===========================================================================

def panel_powerlaw(panel):
    W, rho = _load(panel)
    B = _offset(panel, W, rho)
    d, y = _select(panel, W, rho, B)
    beta, err, A, sres, n, cov = _powerlaw(d, y)

    dlab = r"\delta" if panel['side'] == 'above' else r"\delta_{\rm A}"

    name = f"Fig{panel['key']}_powerlaw"
    fig, ax = fixed_box_figure(right=1.2)
    ax.plot(d, y + B, **MARKER)
    _set_axes(ax, name,
              xlim=(0.0, float(d.max()) * 1.08),
              ylim=(0.0, float((y + B).max()) * 1.10))
    xx = _full_span(ax)
    yy = A * np.clip(xx, 1e-12, None) ** beta
    band = _log_band(xx, cov)
    ax.plot(xx, yy * np.exp(band) + B, **BANDLINE)
    ax.plot(xx, yy * np.exp(-band) + B, **BANDLINE)
    ax.plot(xx, yy + B, **FITLINE,
            label=rf"${panel['sym']} = {beta:.4f} \pm {err:.4f}$")
    ax.set_xlabel(rf"${dlab} = " +
                  (r"(W - W_{\rm c})/W_{\rm c}$" if panel['side'] == 'above'
                   else r"(W_{\rm c} - W)/W_{\rm c}$"))
    ax.set_ylabel(rf"${panel['rho']}$", rotation=0, labelpad=32)
    ax.set_title(panel['label'] + "\n" +
                 rf"$W_{{\rm c}} = {panel['W_c']:.2f}$, $n = {n}$")
    ax.legend(loc='upper left', frameon=False)
    _style(ax)
    path = _save(fig, name)

    print(f"{panel['key']:>5}  beta = {beta:.4f} +- {err:.4f}   "
          f"(W_c={panel['W_c']:.2f}, n={n}, B={B:.3e}, sres={sres:.4f})")
    return dict(key=panel['key'], beta=beta, err=err, A=A, B=B,
                sres=sres, n=n, delta=d, y=y, path=path)


def panel_linearized(panel, fit):
    """Y = [(rho - B)/A]^(1/beta) must lie on Y = delta."""
    d, Y = fit['delta'], (fit['y'] / fit['A']) ** (1.0 / fit['beta'])
    slope = float((d * Y).sum() / (d * d).sum())
    r2 = 1.0 - float(((Y - slope * d) ** 2).sum()) \
             / float(((Y - Y.mean()) ** 2).sum())

    dlab = r"\delta" if panel['side'] == 'above' else r"\delta_{\rm A}"
    hi = max(d.max(), Y.max()) * 1.08

    name = f"Fig{panel['key']}_linearized"
    fig, ax = fixed_box_figure(right=1.2)
    _set_axes(ax, name, xlim=(0.0, hi), ylim=(0.0, hi))
    xx = _full_span(ax)
    ax.plot(xx, xx, **FITLINE, label=rf"$Y = {dlab}$")
    ax.plot(d, Y, **MARKER)
    ax.set_xlabel(rf"${dlab}$")
    ax.set_ylabel(rf"$\left[({panel['rho']}-B)/A\right]^{{1/\beta}}$", labelpad=14)
    ax.set_title(panel['label'] + " linearized" + "\n" +
                 rf"slope $= {slope:.4f}$, $R^2 = {r2:.4f}$")
    ax.legend(loc='upper left', frameon=False)
    _style(ax)
    path = _save(fig, name)

    print(f"{panel['key']:>5}  linearized: slope = {slope:.4f} (expect 1), "
          f"R^2 = {r2:.4f}")
    return dict(slope=slope, r2=r2, path=path)


def panel_essential(panel, fit):
    """
    Scan W_c at fixed alpha = 1/2 and plot the optimum.  The W_c this form
    predicts goes in the title -- that is the number being reported.
    """
    W, rho = _load(panel)
    B =  0.0 if ESS_OFFSET == 'zero' else _offset(panel, W, rho)

    lo, hi, step = panel['ess_grid']
    win = panel.get('ess_window')
    sel = dict(panel)
    if win is not None:
        sel['W_lo'], sel['W_hi'], sel['hi_inclusive'] = win[0], win[1], True

    best = None
    curve = []
    for W_c in np.arange(lo, hi + 1e-12, step):
        probe = dict(sel); probe['W_c'] = W_c
        try:
            d, y = _select(probe, W, rho, B)
        except Exception:
            continue
        if len(d) < 5:
            continue
        c, cerr, lnA, sres = _essential(d, y)
        curve.append((W_c, sres))
        if best is None or sres < best[3]:
            best = (W_c, c, cerr, sres, lnA, d, y)

    if best is None:
        print(f"{panel['key']:>5}  essential: no valid W_c on the grid.")
        return None
    W_c, c, cerr, sres, lnA, d, y = best

    at_edge = (W_c <= lo + step) or (W_c >= hi - step)
    x = d ** (-ESS_ALPHA)
    dlab = r"\delta" if panel['side'] == 'above' else r"\delta_{\rm A}"

    name = f"Fig{panel['key']}_essential"
    fig, ax = fixed_box_figure(right=1.2)
    ax.plot(x, np.log(y), **MARKER)
    xpad = 0.06 * (x.max() - x.min())
    ly = np.log(y)
    ypad = 0.08 * (ly.max() - ly.min())
    _set_axes(ax, name,
              xlim=(float(x.min() - xpad), float(x.max() + xpad)),
              ylim=(float(ly.min() - ypad), float(ly.max() + ypad)))
    xx = _full_span(ax)
    ax.plot(xx, lnA - c * xx, **FITLINE,
            label=rf"$c = {c:.4f} \pm {cerr:.4f}$")
    ax.set_xlabel(rf"${dlab}^{{-1/2}}$")
    ax.set_ylabel(rf"$\ln\left[{panel['rho']}\right] - A$", labelpad=14)
    ax.set_title(panel['label'] + " essential singularity" + "\n" +
                 rf"predicted $W_{{\rm c}} = {W_c:.4f}$, "
                 rf"$\sigma_{{\rm res}} = {sres:.4f}$")
    ax.legend(loc='upper right' if panel['side'] == 'below' else 'lower left',
              frameon=False)
    _style(ax)
    path = _save(fig, name)

    flag = "  [at grid edge -- runaway, not a real optimum]" if at_edge else ""
    print(f"{panel['key']:>5}  essential (alpha=1/2): W_c = {W_c:.4f}, "
          f"c = {c:.4f} +- {cerr:.4f}, sres = {sres:.4f} "
          f"vs power law {fit['sres']:.4f}{flag}")
    return dict(W_c=W_c, c=c, cerr=cerr, sres=sres, at_edge=at_edge,
                curve=curve, path=path)


# ===========================================================================
# Driver
# ===========================================================================

def main():
    print("=" * 74)
    print("Figure 4 -- final panels")
    print("=" * 74)

    rows = []
    a = panel_alpha()
    if a:
        rows.append(("4(a)", r"alpha_a", a['value'], a['err'], a['n'], a['sres']))

    for panel in PANELS:
        try:
            fit = panel_powerlaw(panel)
        except Exception as exc:
            print(f"{panel['key']:>5}  FAILED: {exc}")
            continue
        panel_linearized(panel, fit)
        panel_essential(panel, fit)
        rows.append((panel['key'], panel['sym'], fit['beta'], fit['err'],
                     fit['n'], fit['sres']))
        print()

    print("=" * 74)
    print(f"{'panel':<8}{'exponent':<16}{'value':>10}{'error':>10}"
          f"{'n':>5}{'sigma_res':>12}")
    for key, sym, val, err, n, sres in rows:
        print(f"{key:<8}{sym:<16}{val:>10.4f}{err:>10.4f}{n:>5}{sres:>12.4f}")
    print("=" * 74)
    print(f"figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
