# -*- coding: utf-8 -*-
"""Automatic layout checks for matplotlib figures (run before saving).

Checks
  1. text clipped by / outside the figure canvas
  2. text-text overlaps (all visible Text artists incl. tick labels, annotations, legend texts)
  3. text over marker / bar / line-vertex artists (data collisions) unless the text is marked intended
  4. legend box overlapping data markers or text
  5. fonts below the size floor (FS_SMALL)
A text can opt out of (3) with  t.set_gid("allow-overlap")  (e.g. value labels placed on bars).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Text, Annotation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import PathCollection

FLOOR_PT = 8.4


def _hidden_tick_labels(fig):
    """Tick labels whose tick lies outside the visible axis range are never drawn."""
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            try:
                locs = axis.get_majorticklocs(); lo, hi = axis.get_view_interval(); lo, hi = min(lo, hi), max(lo, hi)
                for tick, loc in zip(axis.get_major_ticks(numticks=len(locs)), locs):
                    if not (lo - 1e-9 <= loc <= hi + 1e-9):
                        hidden.add(id(tick.label1)); hidden.add(id(tick.label2))
            except Exception:
                pass
    return hidden


def _texts(fig):
    out = []
    hid = _hidden_tick_labels(fig)
    for t in fig.findobj(Text):
        if id(t) in hid:
            continue
        if not t.get_visible() or not t.get_text().strip():
            continue
        if isinstance(t, Annotation) and t.get_text() == "":
            continue
        out.append(t)
    return out


def _bbox(t, r):
    bb = t.get_window_extent(renderer=r)
    return bb if (bb.width > 0 and bb.height > 0) else None


def _marker_boxes(fig, r, pad_px=1.5):
    boxes = []
    for ax in fig.axes:
        for ln in ax.lines:
            if not ln.get_visible() or ln.get_marker() in (None, "None", "", " ") or ln.get_markersize() <= 0:
                continue
            xy = ax.transData.transform(np.column_stack([ln.get_xdata(orig=False), ln.get_ydata(orig=False)]).astype(float))
            rad = ln.get_markersize() * fig.dpi / 72.0 / 2 + pad_px
            for x, y in xy:
                if np.isfinite(x) and np.isfinite(y):
                    boxes.append((x - rad, y - rad, x + rad, y + rad, "marker"))
        for coll in ax.collections:
            if isinstance(coll, PathCollection) and coll.get_visible():
                offs = coll.get_offsets()
                if len(offs) == 0:
                    continue
                xy = coll.get_offset_transform().transform(np.asarray(offs, float))
                sizes = np.atleast_1d(coll.get_sizes())
                for k, (x, y) in enumerate(xy):
                    s = sizes[k % len(sizes)]
                    rad = (np.sqrt(s) * fig.dpi / 72.0) / 2 + pad_px
                    boxes.append((x - rad, y - rad, x + rad, y + rad, "scatter"))
        for p in ax.patches:
            if isinstance(p, Rectangle) and p.get_visible() and p.get_width() != 0 and p.get_height() != 0 and p.get_alpha() != 0:
                if p.get_facecolor()[3] < 0.5:      # translucent bands are background
                    continue
                bb = p.get_window_extent(renderer=r)
                axb = ax.get_window_extent(renderer=r)          # artists are clipped to their axes
                x0, y0, x1, y1 = max(bb.x0, axb.x0), max(bb.y0, axb.y0), min(bb.x1, axb.x1), min(bb.y1, axb.y1)
                if x1 > x0 and y1 > y0:
                    boxes.append((x0, y0, x1, y1, "bar"))
    return boxes


def check_layout(fig, margin_px=3, text_pad_px=1.0):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    issues = []
    texts = _texts(fig)
    bbs = []
    for t in texts:
        bb = _bbox(t, r)
        if bb is None:
            continue
        bbs.append((t, bb))
        s = t.get_text().replace("\n", " / ")[:38]
        if t.get_fontsize() < FLOOR_PT:
            issues.append(f"font {t.get_fontsize():.1f}pt below floor: '{s}'")
        if bb.x0 < margin_px or bb.y0 < margin_px or bb.x1 > W - margin_px or bb.y1 > H - margin_px:
            if not t.get_clip_on() or True:
                issues.append(f"text outside canvas / clipped: '{s}' bbox=({bb.x0:.0f},{bb.y0:.0f},{bb.x1:.0f},{bb.y1:.0f}) canvas=({W},{H})")
    # text-text overlaps
    for i in range(len(bbs)):
        for j in range(i + 1, len(bbs)):
            a, b = bbs[i][1], bbs[j][1]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0); oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > text_pad_px and oy > text_pad_px:
                issues.append(f"text overlap: '{bbs[i][0].get_text()[:28]}' x '{bbs[j][0].get_text()[:28]}' ({ox:.0f}x{oy:.0f}px)")
    # text vs marker/bar collisions
    boxes = _marker_boxes(fig, r)
    for t, bb in bbs:
        if t.get_gid() == "allow-overlap":
            continue
        for (x0, y0, x1, y1, kind) in boxes:
            ox = min(bb.x1, x1) - max(bb.x0, x0); oy = min(bb.y1, y1) - max(bb.y0, y0)
            if ox > 2 and oy > 2:
                issues.append(f"text over {kind}: '{t.get_text().replace(chr(10), ' / ')[:30]}'")
                break
    # legends
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is None:
            continue
        lbb = lg.get_window_extent(renderer=r)
        for (x0, y0, x1, y1, kind) in boxes:
            ox = min(lbb.x1, x1) - max(lbb.x0, x0); oy = min(lbb.y1, y1) - max(lbb.y0, y0)
            if ox > 1 and oy > 1:
                issues.append(f"legend over {kind}")
                break
        for t, bb in bbs:
            if t in lg.get_texts():
                continue
            ox = min(lbb.x1, bb.x1) - max(lbb.x0, bb.x0); oy = min(lbb.y1, bb.y1) - max(lbb.y0, bb.y0)
            if ox > 1 and oy > 1:
                issues.append(f"legend over text '{t.get_text()[:24]}'")
    return issues
