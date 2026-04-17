import re

with open('generate_abstract_figure.py', 'r') as f:
    text = f.read()

# Fix panel A nodes
text = text.replace("""# Euclidean samples (orange dots like panel b has teal dots)
for p in SAMP_E:
    ax_prob.plot(p[0], p[1], 'o', color=C['wasted'], ms=3.0,
                 alpha=0.55, zorder=3, mec='none')""",
"""# Euclidean Tree Nodes
for p in SAMP_E:
    ax_prob.plot(p[0], p[1], 'o', color=C['tree_euc'], ms=1.5, alpha=0.8, zorder=3, mec='none')""")

# Fix panel B nodes (add them)
text = text.replace("""# CARM tree edges
for edge in TREE_CARM_EDGES:
    ax_rit.plot([edge[0, 0], edge[1, 0]], [edge[0, 1], edge[1, 1]],
                 color=C['tree_carm'], lw=0.4, alpha=0.5, zorder=2)""",
"""# CARM Tree Nodes
for p in SAMP_R:
    ax_rit.plot(p[0], p[1], 'o', color='#1976D2', ms=1.5, alpha=0.8, zorder=3, mec='none')

# CARM tree edges
for edge in TREE_CARM_EDGES:
    ax_rit.plot([edge[0, 0], edge[1, 0]], [edge[0, 1], edge[1, 1]],
                 color=C['tree_carm'], lw=0.6, alpha=0.7, zorder=2)""")

with open('generate_abstract_figure.py', 'w') as f:
    f.write(text)

