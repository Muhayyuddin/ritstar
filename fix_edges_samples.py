import re
import numpy as np

with open('generate_abstract_figure.py', 'r') as f:
    text = f.read()

# 1. Update Path E (more graceful, moving from obstacle towards end explicitly):
text = text.replace(
"pth_e_pts = [XS, [0.35, 0.48], [0.55, 0.58], [0.80, 0.65], XG]",
"pth_e_pts = [XS, [0.35, 0.60], [0.45, 0.65], [0.70, 0.70], XG]")

# 2. Add SAMP_E outside the Riemann ellipse to panel B, exactly same color as A 
# AND make bg the same as figure A (euc_fill instead of none)
if "fc=C['euc_fill'], ec=C['euc']" not in text:
    text = text.replace("""# Ghost Euclidean ellipse
ax_rit.add_patch(Ellipse(cen_E, wE, hE, angle=angE,
                           fc='none', ec=C['euc'],
                           lw=1.5, ls='--', alpha=0.3, zorder=0.5))""",
"""# Ghost Euclidean ellipse (matching A's background color)
ax_rit.add_patch(Ellipse(cen_E, wE, hE, angle=angE,
                           fc=C['euc_fill'], ec=C['euc'],
                           lw=1.5, ls='--', alpha=0.3, zorder=0.5))

# Plot 'wasted' Euclidean samples that fall outside the Riemannian ellipse
for p in SAMP_E:
    if not in_ellipse(p, cen_R, wR, hR, angR):
        ax_rit.plot(p[0], p[1], 'o', color=C['wasted'], ms=3.0,
                    alpha=0.55, zorder=0.55, mec='none')""")

# 3. Fix the "eggs" (edges/nodes) for Figure A (Euclidean tree)
# Make nodes visible, increase edge line width
text = text.replace("""# Euclidean Tree Nodes
for p in SAMP_E:
    ax_prob.plot(p[0], p[1], 'o', color=C['tree_euc'], ms=1.5, alpha=0.8, zorder=3, mec='none')

# Euclidean tree edges (real planner data, light grey)
for edge in TREE_EUC_EDGES:
    ax_prob.plot([edge[0, 0], edge[1, 0]], [edge[0, 1], edge[1, 1]],
                 color=C['tree_euc'], lw=0.3, alpha=0.4, zorder=2)""",
"""# Euclidean Tree Nodes
for p in SAMP_E:
    ax_prob.plot(p[0], p[1], 'o', color=C['wasted'], ms=3.0, alpha=0.8, zorder=3, mec='none')

# Euclidean tree edges
for edge in TREE_EUC_EDGES:
    ax_prob.plot([edge[0, 0], edge[1, 0]], [edge[0, 1], edge[1, 1]],
                 color=C['tree_euc'], lw=1.2, alpha=0.6, zorder=2)""")


# For B
text = text.replace("""# CARM Tree Nodes
for p in SAMP_R:
    ax_rit.plot(p[0], p[1], 'o', color='#1976D2', ms=1.5, alpha=0.8, zorder=3, mec='none')

# CARM tree edges
for edge in TREE_CARM_EDGES:
    ax_rit.plot([edge[0, 0], edge[1, 0]], [edge[0, 1], edge[1, 1]],
                 color=C['tree_carm'], lw=0.6, alpha=0.7, zorder=2)""",
"""# CARM Tree Nodes
for p in SAMP_R:
    ax_rit.plot(p[0], p[1], 'o', color='#1976D2', ms=3.0, alpha=0.9, zorder=3, mec='none')

# CARM tree edges
for edge in TREE_CARM_EDGES:
    ax_rit.plot([edge[0, 0], edge[1, 0]], [edge[0, 1], edge[1, 1]],
                 color=C['tree_carm'], lw=1.5, alpha=0.8, zorder=2)""")


with open('generate_abstract_figure.py', 'w') as f:
    f.write(text)

