import re

with open('generate_abstract_figure.py', 'r') as f:
    text = f.read()

# 1. Background color of outer ellipse in Figure B + Wasted Samples
text = re.sub(
    r"# Ghost Euclidean ellipse\nax_rit.add_patch\(Ellipse\(cen_E, wE, hE, angle=angE,\n.*?fc='none', ec=C\['euc'\],\n.*?lw=1\.5, ls='--', alpha=0\.3, zorder=0\.5\)\)",
    """# Ghost Euclidean ellipse (with matching A background color)
ax_rit.add_patch(Ellipse(cen_E, wE, hE, angle=angE,
                           fc=C['euc_fill'], ec=C['euc'],
                           lw=1.5, ls='--', alpha=0.3, zorder=0.5))

# Euclidean 'wasted' samples outside Riemannian ellipse
for p in SAMP_E:
    if not in_ellipse(p, cen_R, wR, hR, angR):
        ax_rit.plot(p[0], p[1], 'o', color=C['wasted'], ms=3.0,
                    alpha=0.55, zorder=0.55, mec='none')""",
    text
)

# 2. Figure A samples and edges (eggs)
text = re.sub(
    r"# Euclidean Tree Nodes\nfor p in SAMP_E:\n\s+ax_prob\.plot.*?# Euclidean tree edges\s*\nfor.*?alpha=0\.6.*?zorder=2\)",
    """# Euclidean Tree Nodes
for p in SAMP_E:
    ax_prob.plot(p[0], p[1], 'o', color=C['wasted'], ms=3.0, alpha=0.8, zorder=3, mec='none')

# Euclidean tree edges
for edge in TREE_EUC_EDGES:
    ax_prob.plot([edge[0, 0], edge[1, 0]], [edge[0, 1], edge[1, 1]],
                 color=C['tree_euc'], lw=1.2, alpha=0.6, zorder=2)""",
    text, flags=re.DOTALL
)

# If it missed the first time because it was still orange dots (like my old test)
text = re.sub(
    r"# Euclidean Tree Nodes.*?ax_prob\.plot.*?zorder=3, mec='none'\)",
    """# Euclidean Tree Nodes
for p in SAMP_E:
    ax_prob.plot(p[0], p[1], 'o', color=C['wasted'], ms=3.0, alpha=0.8, zorder=3, mec='none')""",
    text, flags=re.DOTALL
)

text = re.sub(
    r"# Euclidean tree edges.*?ax_prob\.plot.*?zorder=2\)",
    """# Euclidean tree edges
for edge in TREE_EUC_EDGES:
    ax_prob.plot([edge[0, 0], edge[1, 0]], [edge[0, 1], edge[1, 1]],
                 color=C['tree_euc'], lw=1.2, alpha=0.6, zorder=2)""",
    text, flags=re.DOTALL
)

# 3. Figure B samples and edges (eggs)
text = re.sub(
    r"# CARM Tree Nodes.*?ax_rit\.plot.*?zorder=3, mec='none'\)",
    """# CARM Tree Nodes
for p in SAMP_R:
    ax_rit.plot(p[0], p[1], 'o', color='#1976D2', ms=3.5, alpha=0.9, zorder=3, mec='none')""",
    text, flags=re.DOTALL
)

text = re.sub(
    r"# CARM tree edges\s*\nfor edge in TREE_CARM_EDGES:\n.*?zorder=2\)",
    """# CARM tree edges
for edge in TREE_CARM_EDGES:
    ax_rit.plot([edge[0, 0], edge[1, 0]], [edge[0, 1], edge[1, 1]],
                 color=C['tree_carm'], lw=1.2, alpha=0.8, zorder=2)""",
    text, flags=re.DOTALL
)


with open('generate_abstract_figure.py', 'w') as f:
    f.write(text)

