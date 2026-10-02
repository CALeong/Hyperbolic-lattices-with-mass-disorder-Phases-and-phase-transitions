"""
plot_utils.py

Reusable helpers for making matplotlib PLOT BOXES a fixed physical size
(in inches), independent of how much room labels/titles/legends need.

Usage:
    from plot_utils import fixed_box_figure, fixed_box_row

    # single axes, 6x6" box
    fig, ax = fixed_box_figure(box_size=6.0, right=3.0)
    ax.plot(...)
    fig.savefig(...)

    # two axes side by side (e.g. broken-axis plots), each 6x6"
    fig, (ax1, ax2) = fixed_box_row(n=2, box_size=6.0, gap=0.05)
    ax1.plot(...); ax2.plot(...)
    fig.savefig(...)

Import this module *instead of* calling plt.figure()/plt.subplots() in your
plotting functions whenever you want a constant box size. It disables
constrained_layout globally on import so you never need to fight it per-figure.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size

# Kill constrained_layout globally, once, on import. This is what was causing
# the "no gridspecs with layoutgrids" warning — constrained_layout was still
# active at the rcParams level even though individual figures tried to opt out
# after creation. Doing it here means no plotting function needs to touch it.
mpl.rcParams['figure.constrained_layout.use'] = False


BOX_SIZE = 6.0  # the one constant: every plot's main box is 6x6 inches


def fixed_box_figure(left=1.2, right=1.0, bottom=1.0, top=1.0, box_size=BOX_SIZE):
    """
    Create a figure with a single axes whose plotting box is exactly
    box_size x box_size inches, regardless of labels/titles/legends.

    left/right/bottom/top are margins in inches -- this is where legends,
    long titles, wrapped labels etc. get their room. Bump the relevant
    margin if something clips; the box itself will never move or resize.

    Returns: (fig, ax)
    """
    fig_w = left + box_size + right
    fig_h = bottom + box_size + top
    fig = plt.figure(figsize=(fig_w, fig_h), layout=None)

    h = [Size.Fixed(left), Size.Fixed(box_size), Size.Fixed(right)]
    v = [Size.Fixed(bottom), Size.Fixed(box_size), Size.Fixed(top)]
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax = fig.add_axes(divider.get_position(),
                       axes_locator=divider.new_locator(nx=1, ny=1))
    return fig, ax


def fixed_box_row(n=2, width_ratios=None, gap=0.05, left=1.2, right=1.0,
                   bottom=1.0, top=1.0, sharey=False, box_size=BOX_SIZE):
    """
    ONE 6x6" logical plot box (BOX_SIZE), split into n panels side by side --
    for broken-axis plots (the "kink" style), not n separate 6x6 boxes.
    The panels + the gap between them add up to exactly box_size inches wide,
    matching your original `width_ratios=[1,1]` broken-axis setup.

    width_ratios: relative widths of the panels, e.g. [1, 1] for an even
    split (default) or [2, 1] if one side should be wider.
    gap: room between panels, in inches -- just enough for the diagonal
    slash marks, not extra plot area.

    Returns: (fig, [ax1, ax2, ...])
    """
    if width_ratios is None:
        width_ratios = [1] * n
    total_ratio = sum(width_ratios)
    available = box_size - (n - 1) * gap  # panels + gap together = box_size
    panel_widths = [available * (r / total_ratio) for r in width_ratios]

    fig_w = left + box_size + right
    fig_h = bottom + box_size + top
    fig = plt.figure(figsize=(fig_w, fig_h), layout=None)

    h = [Size.Fixed(left)]
    for i, w in enumerate(panel_widths):
        h.append(Size.Fixed(w))
        if i < n - 1:
            h.append(Size.Fixed(gap))
    h.append(Size.Fixed(right))
    v = [Size.Fixed(bottom), Size.Fixed(box_size), Size.Fixed(top)]

    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    axes = []
    for i in range(n):
        nx = 1 + 2 * i  # panel i sits at h-index 1, 3, 5, ... (margins/gaps between)
        ax = fig.add_axes(divider.get_position(),
                           axes_locator=divider.new_locator(nx=nx, ny=1))
        if sharey and axes:
            ax.sharey(axes[0])
        axes.append(ax)
    return fig, axes
