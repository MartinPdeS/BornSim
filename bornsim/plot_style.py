"""Shared visual palette for BornSim's Matplotlib and Plotly figures."""

from matplotlib.colors import LinearSegmentedColormap

NAVY = "#102b4e"
BLUE = "#1479bd"
CYAN = "#11bfc1"
GREEN = "#45a98b"
GOLD = "#f6b944"
CORAL = "#dd765c"
TEXT = "#52677d"
GRID = "#dce5ed"

SERIES_COLORS = (NAVY, BLUE, CYAN, GREEN, GOLD, CORAL)

SEQUENTIAL_COLORS = [
    [0.0, NAVY],
    [0.34, BLUE],
    [0.7, CYAN],
    [1.0, GOLD],
]

DIVERGING_COLORS = [
    [0.0, NAVY],
    [0.5, "#f7f9fb"],
    [1.0, GOLD],
]

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("bornsim_sequential", [NAVY, BLUE, CYAN, GOLD])

DIVERGING_CMAP = LinearSegmentedColormap.from_list("bornsim_diverging", [NAVY, "#f7f9fb", GOLD])
