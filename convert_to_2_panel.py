import re

with open('generate_abstract_figure.py', 'r') as f:
    code = f.read()

# Instead of the 4 panel figure, we will define a 1x2 panel layout matching the main paper abstract.
new_code = code.replace(
"""fig = plt.figure(figsize=(15, 7.2), facecolor='white')

gs = fig.add_gridspec(
    2, 5,
    width_ratios=[1.15, 0.10, 1.0, 0.10, 1.15],
    height_ratios=[1, 1],
    hspace=0.20, wspace=0.02,
    left=0.02, right=0.98, bottom=0.08, top=0.88,
)""",
"""fig = plt.figure(figsize=(10, 5.5), facecolor='white')

gs = fig.add_gridspec(
    1, 2,
    width_ratios=[1.0, 1.0],
    wspace=0.05,
    left=0.02, right=0.98, bottom=0.12, top=0.85,
)"""
)

# Panel A definition
new_code = new_code.replace(
"""ax_prob = fig.add_subplot(gs[:, 0])""",
"""ax_prob = fig.add_subplot(gs[0, 0])"""
)

# Remove things from gs[0, 2], gs[1, 2] up to gs[:, 4] completely and replace with just one new gs[0, 1] for panel B

to_delete = new_code.split("# ──────────────────────────────────────────────────────────────\n#  (b) Riemannian Informed Set — top-middle\n# ──────────────────────────────────────────────────────────────")[1].split("# ──────────────────────────────────────────────────────────────\n#  Flow arrows between columns\n# ──────────────────────────────────────────────────────────────")[0]

to_replace = """# ──────────────────────────────────────────────────────────────
#  (b) Riemannian Informed Set — top-middle
# ──────────────────────────────────────────────────────────────""" + to_delete

panel_b = """# ──────────────────────────────────────────────────────────────
#  (b) RIT* replaces Euclidean primitives
# ──────────────────────────────────────────────────────────────
ax_rit = fig.add_subplot(gs[0, 1])
setup(ax_rit, margin=0.32)

# Ghost Euclidean ellipse
ax_rit.add_patch(Ellipse(cen_E, wE, hE, angle=angE,
                           fc='none', ec=C['euc'],
                           lw=1.5, ls='--', alpha=0.3, zorder=0.5))

# Riemannian ellipse filled background
ax_rit.add_patch(Ellipse(cen_R, wR, hR, angle=angR,
                           fc=C['rit_fill'], ec='none',
                           lw=0, alpha=0.8, zorder=0.6))

# CARM heatmap clipped to Riemannian ellipse
_ell_clip = Ellipse(cen_R, wR, hR, angle=angR,
                    transform=ax_rit.transData)
_mesh_rit = ax_rit.pcolormesh(Xh, Yh, Fh, cmap='YlOrRd', shading='gouraud',
                              alpha=0.55, zorder=0.7)
_mesh_rit.set_clip_path(_ell_clip)

# Riemannian ellipse border
ax_rit.add_patch(Ellipse(cen_R, wR, hR, angle=angR,
                           fc='none', ec=C['rit'],
                           lw=2.5, alpha=0.7, zorder=0.8))

# Collision points
ax_rit.scatter(COLL_PTS[:, 0], COLL_PTS[:, 1], s=5.0,
                c=C['coll_dot'], alpha=0.8, zorder=4)

# CARM tree edges
for edge in TREE_CARM_EDGES:
    ax_rit.plot([edge[0, 0], edge[1, 0]], [edge[0, 1], edge[1, 1]],
                 color=C['tree_carm'], lw=0.4, alpha=0.5, zorder=2)

# CARM path
ax_rit.plot(PATH_OPT[:, 0], PATH_OPT[:, 1], '-', color=C['path_opt'],
             lw=2.5, alpha=0.9, zorder=8, solid_capstyle='round')

draw_obs(ax_rit, alpha=0.9)
draw_sg(ax_rit)

# Labels
ax_rit.text(0.50, 0.97, r'$\mathcal{I}_R \subset \mathcal{I}_E$',
             fontsize=10, color=C['rit'], ha='center', va='top',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='white',
                       ec=C['rit'], alpha=0.9, lw=0.8))

ax_rit.annotate('path guided\nby learned metric',
                 xy=(0.55, 0.15), xytext=(0.40, 0.04),
                 fontsize=8, color=C['path_opt'], ha='center',
                 fontweight='bold', fontstyle='italic',
                 arrowprops=dict(arrowstyle='->', color=C['path_opt'],
                                 lw=1.3, alpha=0.7))

ax_rit.annotate('tighter area,\ncollision feedback',
                 xy=(0.60, 0.70), xytext=(0.80, 0.85),
                 fontsize=8, color=C['rit'], ha='center',
                 fontweight='bold', fontstyle='italic',
                 arrowprops=dict(arrowstyle='->', color=C['rit'],
                                 lw=1.3, alpha=0.7))

ax_rit.set_title(r'(b)  Riemannian Informed Trees (RIT*)', fontsize=12.5,
                  fontweight='bold', color=C['label'], pad=8)

"""

new_code = new_code.replace(to_replace, panel_b)

# Remove old flow arrows
arrows_delete = new_code.split("# ──────────────────────────────────────────────────────────────\n#  Flow arrows between columns\n# ──────────────────────────────────────────────────────────────")[1].split("# ──────────────────────────────────────────────────────────────\n#  Supertitle\n# ──────────────────────────────────────────────────────────────")[0]
new_code = new_code.replace("# ──────────────────────────────────────────────────────────────\n#  Flow arrows between columns\n# ──────────────────────────────────────────────────────────────" + arrows_delete, "")


with open('generate_abstract_figure.py', 'w') as f:
    f.write(new_code)
