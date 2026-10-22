# -*- coding: utf-8 -*-
"""Figure 1 (v2.1): framework. SVG scene via paper-figure diagram_lib, rendered by headless Chrome.
Run: python -X utf8 fig1_framework.py   -> out/Figure_1_framework_v6.png
"""
import sys, os
# diagram_lib is a local drawing helper that is not distributed; set DIAGRAM_LIB_DIR to its folder to re-render.
sys.path.insert(0, os.environ.get("DIAGRAM_LIB_DIR", "."))
from diagram_lib import *

SYM = symbols(['file-text', 'brain', 'git-merge', 'shield-check'])
W, H = 1800, 800
OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
b = []
def sub(s, i="i", size=33):          # italic symbol with true subscript (no baseline drift)
    return '<tspan font-style="italic">%s</tspan><tspan font-size="%d" baseline-shift="-10" font-style="italic">%s</tspan>' % (s, size, i)


OKF, BADF = "#DCEFE6", "#F6D9CC"      # light green / light vermilion tints (glyphs carry the meaning in grayscale)
def chip(cx, cy, r, ok):
    o = ['<circle cx="%g" cy="%g" r="%g" fill="%s" stroke="%s" stroke-width="3"/>' % (cx, cy, r, OKF if ok else BADF, INK)]
    d = r * 0.42
    if ok:
        o.append('<path d="M%g %g L%g %g L%g %g" fill="none" stroke="%s" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>' % (cx-d, cy, cx-d*0.25, cy+d*0.75, cx+d, cy-d*0.7, INK))
    else:
        o.append('<path d="M%g %g L%g %g M%g %g L%g %g" stroke="%s" stroke-width="5" stroke-linecap="round"/>' % (cx-d*0.8, cy-d*0.8, cx+d*0.8, cy+d*0.8, cx+d*0.8, cy-d*0.8, cx-d*0.8, cy+d*0.8, INK))
    return ''.join(o)

# ---------------- row 1: pipeline ----------------
Y0, BH = 40, 260
x1, w1 = 30, 330
x2, w2 = 470, 640
x3, w3 = 1220, 250
x4, w4 = 1580, 190
cy = Y0 + BH / 2
# 1 instance
b.append(box(x1, Y0, w1, BH, FILL))
b.append(icon('file-text', x1 + w1/2 - 40, Y0 + 28, 80))
b.append(T(x1 + w1/2, Y0 + 142, 'Evaluation', 36, 'h'))
b.append(T(x1 + w1/2, Y0 + 184, 'instance ' + sub('x'), 36, 'h'))
b.append(T(x1 + w1/2, Y0 + 228, 'prompt, A, B', 33, 's'))
# 2 K judgments
b.append(box(x2, Y0, w2, BH, GRAY))
b.append(T(x2 + w2/2, Y0 + 50, 'K = 5 sampled CoT judgments', 36, 'h'))
votes = ['✓', '✓', '✗', '✓', '✓']
r = 44; gap = (w2 - 5*2*r) / 6
for k, v in enumerate(votes):
    cx = x2 + gap*(k+1) + r*(2*k+1)
    b.append(chip(cx, Y0 + 130, r, v == '✓'))
    b.append(T(cx, Y0 + 214, 'v%d' % (k+1), 33, 's'))
b.append(T(x2 + w2/2, Y0 + 244, '', 33, 's'))
# 3 majority vote
b.append(box(x3, Y0, w3, BH, FILL))
b.append(icon('git-merge', x3 + w3/2 - 40, Y0 + 28, 80))
b.append(T(x3 + w3/2, Y0 + 150, 'Majority', 36, 'h'))
b.append(T(x3 + w3/2, Y0 + 192, 'vote', 36, 'h'))
b.append(T(x3 + w3/2, Y0 + 228, sub('c') + ' > K/2', 33, 's'))
# 4 verdict
b.append(box(x4, Y0, w4, BH, GRAY_D))
b.append(icon('shield-check', x4 + w4/2 - 40, Y0 + 28, 80))
b.append(T(x4 + w4/2, Y0 + 150, 'Verdict', 36, 'h'))
b.append(T(x4 + w4/2, Y0 + 192, sub('ŷ'), 36, 'h'))
for xa, xb in ((x1+w1, x2), (x2+w2, x3), (x3+w3, x4)):
    b.append(arrow(xa + 8, cy, xb - 8, cy))

# ---------------- row 2: three regimes of c_i ----------------
yb = 372
b.append(T(W/2, yb, 'Correct votes ' + sub('c') + ' separate the instances into three regimes', 36, 'h'))
BH2 = 330; y2 = yb + 36
cards = [
    ('0', sub('c') + ' = 0', 'K-persistent error', 'Voting cannot reach gold', GRAY_D),
    ('mid', '0 < ' + sub('c') + ' < K', 'Gold-reachable', 'Voting can help or hurt', FILL),
    ('K', sub('c') + ' = K', 'Stable correct', 'Voting changes nothing', GRAY),
]
cw = 520; cg = (W - 3*cw) / 4
for n, (tag, head, name, desc, fill) in enumerate(cards):
    cx0 = cg + n*(cw + cg)
    b.append(box(cx0, y2, cw, BH2, fill))
    b.append(T(cx0 + cw/2, y2 + 66, head, 40, 'h'))
    b.append(T(cx0 + cw/2, y2 + 130, name, 36, 't'))
    b.append(T(cx0 + cw/2, y2 + 180, desc, 33, 's'))
    nok = [0, 3, 5][n]
    rr_ = 26; gp = 18; tot = 5*2*rr_ + 4*gp; sx = cx0 + cw/2 - tot/2 + rr_
    for k in range(5):
        b.append(chip(sx + k*(2*rr_ + gp), y2 + 262, rr_, k < nok))
# decomposition line
b.append(T(W/2, y2 + BH2 + 72, sub('Δ') + ' = 1[ ' + sub('c') + ' &gt; K/2 ] − ' + sub('c') + ' / K', 40, 't'))
b.append(T(W/2, y2 + BH2 + 124, 'positive and negative voting mass sum to the net effect of voting on accuracy', 33, 's'))

# ---------------- row 3: operational use of label-free vote dispersion ----------------
y3 = y2 + BH2 + 170
b.append(box(30, y3, W - 60, 220, 'none', 14, 3, DASH, dash='12 10'))
b.append(T(W/2, y3 + 52, 'Label-free vote dispersion supports operational decisions', 36, 'h'))
acts = [('Accept', 'unanimous votes', FILL), ('Sample more', 'split votes', GRAY), ('Abstain', 'persistent split', GRAY_D)]
aw = 500; ag = (W - 60 - 3*aw) / 4
for n, (nm, ds, fl) in enumerate(acts):
    ax_ = 30 + ag + n*(aw + ag)
    b.append(box(ax_, y3 + 78, aw, 120, fl))
    b.append(T(ax_ + aw/2, y3 + 128, nm, 36, 'h'))
    b.append(T(ax_ + aw/2, y3 + 176, ds, 33, 's'))
H = int(y3 + 220 + 30)
html = page(chr(10).join(b), SYM, W, H)
render('Figure_1_framework_v6', html, OUTD, column='double', w=W, h=H)
