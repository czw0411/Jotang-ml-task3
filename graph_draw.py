# -*- coding: utf-8 -*-
"""Minimal helper that draws a computation graph with matplotlib boxes and arrows.

Two kinds of edges are supported:
    "forward"  -> solid dark arrow (forward pass direction)
    "backward" -> dashed red arrow (gradient flow direction)

Node spec: (node_id, x, y, label, width, height, facecolor)
Edge spec: (src_id, dst_id, label, kind [, (label_dx, label_dy)])

The optional last element moves the edge label away from the midpoint, in data
units. Without it the label is placed just off the line on the side the arc
bends to.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_STYLES = {
    "forward": dict(color="#1f2937", linestyle="-", linewidth=1.6, rad=0.0),
    "backward": dict(color="#b91c1c", linestyle="--", linewidth=1.6, rad=0.18),
}


def _draw_node(ax, node):
    _, x, y, label, width, height, facecolor = node
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.2,
            facecolor=facecolor,
            edgecolor="#334155",
            zorder=3,
        )
    )
    ax.text(x, y, label, ha="center", va="center", fontsize=9, zorder=4, linespacing=1.4)


def _draw_edge(ax, start, end, label, kind, label_offset=None):
    style = _STYLES[kind]
    rad = style["rad"]
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            connectionstyle=f"arc3,rad={rad}",
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            shrinkA=6,
            shrinkB=6,
            zorder=2,
        )
    )
    if not label:
        return
    mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    if label_offset is not None:
        mx, my = mx + label_offset[0], my + label_offset[1]
    else:
        dx, dy = end[0] - start[0], end[1] - start[1]
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        offset = rad * norm * 0.5 + 0.32
        mx, my = mx - dy / norm * offset, my + dx / norm * offset
    ax.text(
        mx,
        my,
        label,
        ha="center",
        va="center",
        fontsize=8,
        color=style["color"],
        zorder=5,
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.9),
    )


def draw_panel(ax, title, nodes, edges, xlim, ylim):
    for node in nodes:
        _draw_node(ax, node)
    positions = {node[0]: (node[1], node[2]) for node in nodes}
    for edge in edges:
        src, dst, label, kind = edge[:4]
        label_offset = edge[4] if len(edge) > 4 else None
        _draw_edge(ax, positions[src], positions[dst], label, kind, label_offset)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def save_figure(path, panels, ncols=1, panel_size=(4.6, 3.0), dpi=170):
    """panels: list of dicts with keys title / nodes / edges / xlim / ylim."""
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows)
    )
    flat = list(getattr(axes, "flat", [axes]))
    for ax, panel in zip(flat, panels):
        draw_panel(ax, **panel)
    for ax in flat[len(panels) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
