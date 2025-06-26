import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True})

def set_plot_style():
    """Estilo global 'print-ready'."""
    sns.set_theme(context="paper",
                  style="white",
                  font="DejaVu Sans",
                  palette="Set2",
                  rc={
                      "axes.edgecolor": "#444444",
                      "axes.linewidth": 0.7,
                      "axes.grid": True,
                      "grid.alpha": 0.18,
                      "grid.linestyle": "--",
                      "axes.titlesize": 16,
                      "axes.titleweight": "bold",
                      "axes.labelsize": 12,
                      "xtick.labelsize": 11,
                      "ytick.labelsize": 11,
                      "legend.fontsize": 9,
                      "legend.frameon": False,
                      "lines.linewidth": 0.5,
                      "lines.markersize": 3,
                      "figure.dpi": 300,
                  })
    mpl.rcParams["axes.spines.top"]   = False
    mpl.rcParams["axes.spines.right"] = False

set_plot_style()


PALETTE_ARCH  = sns.color_palette("Set2", 8) 
MARKERS_ESC   = ["o", "s", "D", "^", "v", "P", "X"]

INTENSE_ARCH_PALETTE = sns.color_palette("Set1",  9)
INTENSE_MARKERS_ESC  = ["o","s","D","^","v","P","X"]

def arch_color_intense(i):  return INTENSE_ARCH_PALETTE[i % len(INTENSE_ARCH_PALETTE)]
def esc_marker_intense(j):  return INTENSE_MARKERS_ESC[j % len(INTENSE_MARKERS_ESC)]


def arch_color(idx):   return PALETTE_ARCH[idx % len(PALETTE_ARCH)]
def esc_marker(idx):   return MARKERS_ESC[idx % len(MARKERS_ESC)]