"""
generate_graphical_abstract.py — Publication-ready graphical abstract for RIT*.

Layout (1 row × 4 panels, styled like a TRO/RA-L overview figure):

  (a) CARM field  — anisotropic cost heatmap + collision-feedback dots
  (b) Informed sets — Euclidean ellipse vs. Riemannian geodesic set (tighter)
  (c) Path quality — suboptimal Euclidean path vs. cost-aware optimal path
  (d) 6-D concept  — schematic of configuration-space shrinkage for high-DOF arm

All panels share a clean white background with a thin bounding box.
Colour scheme follows the paper colour conventions.

Output:
  paper/figures/fig_graphical_abstract.pdf
  paper/figures/fig_graphical_abstract.png
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
#  Global style
# ═══════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'mathtext.fontset': 'dejavusans',
    'font.size': 8,
    'axes.linewidth': 0.6,
    'axes.edgecolor': '#BDBDBD',
})

# Colour palette
C_OBS       = '#454545'   # dark grey obstacles
C_OBS_EDGE  = '#212121'
C_OBS_ALPHA = 0.88

C_EUC       = '#E65100'   # orange  – Euclidean / BIT*
C_EUC_FILL  = '#FFF3E0'
C_EUC_LIGHT = '#FF8A65'

C_RIT       = '#00695C'   # teal    – RIT*
C_RIT_FILL  = '#E0F2F1'
C_RIT_DARK  = '#004D40'

C_CARM_LO   = '#E3F2FD'   # CARM heatmap low
C_CARM_HI   = '#B71C1C'   # CARM heatmap high

C_PATH_EUC  = '#6D4C41'   # brown – suboptimal
C_PATH_RIT  = '#D32F2F'   # red   – our (optimal)

C_TREE_EUC  = '#B0BEC5'
C_TREE_RIT  = '#81D4FA'

C_START     = '#2E7D32'
C_GOAL      = '#C62828'

C_COLL      = '#66BB6A'   # green collision-feedback dots
C_LABEL     = '#212121'
C_SUBLABEL  = '#546E7A'

PANEL_BG    = '#FAFAFA'

# ═══════════════════════════════════════════════════════════════════════
#  Shared 2-D environment (6 circular obstacles)
# ═══════════════════════════════════════════════════════════════════════
XS = np.array([0.05, 0.25])
XG = np.array([0.95, 0.75])

CIRCLES = [
    (np.array([0.28, 0.32]), 0.075),
    (np.array([0.28, 0.66]), 0.075),
    (np.array([0.52, 0.44]), 0.085),
    (np.array([0.52, 0.76]), 0.085),
    (np.array([0.72, 0.38]), 0.075),
    (np.array([0.72, 0.62]), 0.075),
]

SIGMA  = 0.12
ALPHA  = 7.0


def in_obs(p, inflate=0.0):
    for c, r in CIRCLES:
        if (p[0]-c[0])**2 + (p[1]-c[1])**2 <= (r + inflate)**2:
            return True
    return False


def draw_obs(ax, alpha=C_OBS_ALPHA, zorder=3):
    for c, r in CIRCLES:
        ax.add_patch(Circle(c, r, fc=C_OBS, ec=C_OBS_EDGE,
                            lw=0.7, alpha=alpha, zorder=zorder))


def draw_sg(ax, fs=8):
    ax.plot(*XS, 's', color=C_START, ms=7, zorder=12, mec='white', mew=1.0)
    ax.plot(*XG, '*', color=C_GOAL,  ms=11, zorder=12, mec='white', mew=0.7)
    ax.annotate(r'$\mathbf{x}_s$', XS, xytext=(-13, -6),
                textcoords='offset points', fontsize=fs,
                fontweight='bold', color=C_START)
    ax.annotate(r'$\mathbf{x}_g$', XG, xytext=(4, 4),
                textcoords='offset points', fontsize=fs,
                fontweight='bold', color=C_GOAL)


def setup_ax(ax, margin=0.02, bg=PANEL_BG):
    ax.set_xlim(-margin, 1.0 + margin)
    ax.set_ylim(-margin, 1.0 + margin)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(bg)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_edgecolor('#BDBDBD')
        sp.set_visible(True)


# ═══════════════════════════════════════════════════════════════════════
#  Metric field
# ═══════════════════════════════════════════════════════════════════════
def metric_field(res=160):
    gx = np.linspace(0, 1, res)
    gy = np.linspace(0, 1, res)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    centers = np.array([c for c, _ in CIRCLES])
    sq = np.sum((pts[:, None, :] - centers[None, :, :])**2, axis=2)
    S = 1.0 + ALPHA * np.sum(np.exp(-sq / SIGMA**2), axis=1)
    return GX, GY, S.reshape(res, res)


GX, GY, FS = metric_field(res=160)

# ═══════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════
def smooth(pts, n=200):
    t = np.linspace(0, 1, len(pts))
    return CubicSpline(t, pts)(np.linspace(0, 1, n))


# Euclidean path — threads through obstacles (suboptimal, passes close)
PATH_EUC = smooth([XS, [0.18, 0.48], [0.35, 0.50], [0.52, 0.60],
                   [0.65, 0.72], [0.82, 0.76], XG])

# RIT* path — curves around the bottom, cost-aware (shorter Riemannian length)
PATH_RIT = smooth([XS, [0.16, 0.12], [0.38, 0.08], [0.60, 0.10],
                   [0.80, 0.20], [0.92, 0.50], XG])


# ═══════════════════════════════════════════════════════════════════════
#  Informed set helpers
# ═══════════════════════════════════════════════════════════════════════
def ellipse_params(xs, xg, c_best):
    cen = (xs + xg) / 2
    d = xg - xs
    ang = np.degrees(np.arctan2(d[1], d[0]))
    a = c_best / 2.0
    b = np.sqrt(max(c_best**2 - np.linalg.norm(d)**2, 1e-6)) / 2.0
    return cen, 2*a, 2*b, ang


def riemannian_contour(xs, xg, c_best, res=260):
    """Approximate Riemannian informed set via fast metric-scaled BFS."""
    gx = np.linspace(0, 1, res)
    gy = np.linspace(0, 1, res)
    GXr, GYr = np.meshgrid(gx, gy)
    pts = np.column_stack([GXr.ravel(), GYr.ravel()])

    centers = np.array([c for c, _ in CIRCLES])

    def R_dist(a, b, nsteps=12):
        """Fast approximation of Riemannian distance via midpoint rule."""
        ts = np.linspace(0, 1, nsteps + 1)
        total = 0.0
        for i in range(nsteps):
            m = a + (ts[i] + ts[i+1]) / 2 * (b - a)
            sq = np.sum((m - centers)**2, axis=1)
            s = 1.0 + ALPHA * np.sum(np.exp(-sq / SIGMA**2))
            total += s * np.linalg.norm(b - a) / nsteps
        return total

    # Vectorised approximate: use midpoint metric scale
    def batch_R_dist(anchors, pts_arr, nsteps=6):
        n = len(pts_arr)
        mids = (anchors[None, :] + pts_arr) / 2.0   # (n, 2)
        sq = np.sum((mids[:, None, :] - centers[None, :, :])**2, axis=2)
        s = 1.0 + ALPHA * np.sum(np.exp(-sq / SIGMA**2), axis=1)
        dists = np.linalg.norm(pts_arr - anchors[None, :], axis=1)
        return s * dists

    ds = batch_R_dist(xs, pts) + batch_R_dist(xg, pts)
    mask = (ds <= c_best).reshape(res, res).astype(float)

    # Smooth for clean contour
    mask_s = gaussian_filter(mask.astype(float), sigma=1.5)
    return GXr, GYr, mask_s, mask


C_BEST_EUC = 1.50
C_BEST_RIT = 2.10    # Riemannian cost threshold (larger units)

cen_E, wE, hE, angE = ellipse_params(XS, XG, C_BEST_EUC)

print("Computing approximate Riemannian informed-set contour …")
GXr, GYr, RIT_MASK_S, RIT_MASK = riemannian_contour(XS, XG, C_BEST_RIT, res=240)
print("Done.")


# ═══════════════════════════════════════════════════════════════════════
#  Sample / tree helpers
# ═══════════════════════════════════════════════════════════════════════
def gen_samples_ellipse(cen, w, h, ang, n, seed=42):
    rng = np.random.RandomState(seed)
    t   = np.radians(ang)
    R   = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    pts = []
    while len(pts) < n:
        uv = rng.uniform(-1, 1, (n * 6, 2))
        uv = uv[uv[:, 0]**2 + uv[:, 1]**2 <= 1.0]
        xy = (uv * np.array([[w / 2, h / 2]])) @ R.T + cen
        for p in xy:
            if 0 <= p[0] <= 1 and 0 <= p[1] <= 1 and not in_obs(p):
                pts.append(p)
            if len(pts) >= n:
                break
    return np.array(pts[:n])


def gen_samples_mask(mask, GXm, GYm, n, seed=8):
    """Sample uniformly from inside the boolean mask."""
    rng = np.random.RandomState(seed)
    # Flatten and find inside indices
    flat_mask = mask.ravel() > 0.5
    indices = np.where(flat_mask)[0]
    if len(indices) == 0:
        return np.empty((0, 2))
    chosen = rng.choice(indices, size=min(n * 5, len(indices)), replace=False)
    pts = []
    for idx in chosen:
        r, c_ = np.unravel_index(idx, mask.shape)
        p = np.array([GXm[r, c_], GYm[r, c_]])
        if not in_obs(p):
            pts.append(p)
        if len(pts) >= n:
            break
    return np.array(pts)


def build_tree(root, samples, k=4):
    """Build a simple k-nearest tree for visualisation."""
    if len(samples) == 0:
        return []
    edges = []
    pts = [root.copy()]
    from scipy.spatial import KDTree
    for p in samples:
        tree = KDTree(pts)
        dist, idx = tree.query(p, k=min(k, len(pts)))
        nearest = pts[idx] if np.isscalar(idx) else pts[idx[0]]
        edges.append((nearest.copy(), p.copy()))
        pts.append(p.copy())
    return edges


SAMP_E = gen_samples_ellipse(cen_E, wE, hE, angE, 220, seed=7)
SAMP_R = gen_samples_mask(RIT_MASK, GXr, GYr, 160, seed=7)

TREE_EUC = build_tree(XS, SAMP_E)
TREE_RIT = build_tree(XS, SAMP_R)

# Collision point ring around each obstacle
def collision_pts(n=18, seed=55):
    rng = np.random.RandomState(seed)
    pts = []
    for c, r in CIRCLES:
        for _ in range(n):
            th = rng.uniform(0, 2 * np.pi)
            p  = c + (r + rng.uniform(-0.005, 0.012)) * np.array([np.cos(th), np.sin(th)])
            pts.append(p)
    return np.array(pts)


COLL_PTS = collision_pts(12, seed=55)


# ═══════════════════════════════════════════════════════════════════════
#  PANEL LABEL helper
# ═══════════════════════════════════════════════════════════════════════
def panel_label(ax, letter, x=0.03, y=0.97, fs=11):
    ax.text(x, y, f'({letter})', transform=ax.transAxes,
            fontsize=fs, fontweight='bold', color=C_LABEL,
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.15', fc='white',
                      ec='none', alpha=0.85))


def panel_title(ax, title, fs=8.2):
    ax.set_title(title, fontsize=fs, fontweight='bold',
                 color=C_LABEL, pad=5, loc='center')


# ═══════════════════════════════════════════════════════════════════════
#  BUILD FIGURE  — 1 row × 4 panels
# ═══════════════════════════════════════════════════════════════════════
FW, FH = 14.5, 3.8   # figure width / height (inches)
fig = plt.figure(figsize=(FW, FH), facecolor='white')

gs = fig.add_gridspec(1, 4, wspace=0.07,
                      left=0.01, right=0.99,
                      bottom=0.14, top=0.88)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])
ax_d = fig.add_subplot(gs[0, 3])


# ───────────────────────────────────────────────────────────────────────
#  (a)  CARM — Collision-Adaptive Riemannian Metric field
# ───────────────────────────────────────────────────────────────────────
setup_ax(ax_a)

# Custom colormap: white → warm yellow → deep red
cmap_carm = LinearSegmentedColormap.from_list(
    'carm', ['#F5F5F5', '#FFF176', '#FF6F00', '#B71C1C'], N=256)

# Mask field outside [0,1]^2
FS_show = FS.copy()
FS_show[GX < 0] = np.nan
FS_show[GX > 1] = np.nan

im = ax_a.pcolormesh(GX, GY, FS_show, cmap=cmap_carm,
                     shading='gouraud', vmin=1.0, vmax=FS.max() * 0.85,
                     zorder=1)

# Collision feedback dots (from CARM online learning)
ax_a.scatter(COLL_PTS[:, 0], COLL_PTS[:, 1], s=6.5,
             c=C_COLL, alpha=0.9, zorder=6, linewidths=0,
             label='Collision feedback')

# Sparse gradient arrows (flow direction of cost field)
stride = 18
GXq = GX[::stride, ::stride]
GYq = GY[::stride, ::stride]
# gradient of FS
gy_f, gx_f = np.gradient(-FS)
Uq = gx_f[::stride, ::stride]
Vq = gy_f[::stride, ::stride]
mag = np.sqrt(Uq**2 + Vq**2) + 1e-8
Uq /= mag; Vq /= mag
mask_q = (GXq >= 0.05) & (GXq <= 0.95) & (GYq >= 0.05) & (GYq <= 0.95)
ax_a.quiver(GXq[mask_q], GYq[mask_q],
            Uq[mask_q], Vq[mask_q],
            color='white', alpha=0.35, scale=28, width=0.003,
            headwidth=3, zorder=5)

draw_obs(ax_a, alpha=0.7, zorder=7)
draw_sg(ax_a)

# Thin colorbar
divider = make_axes_locatable(ax_a)
cax = divider.append_axes('right', size='4%', pad=0.04)
cb = fig.colorbar(im, cax=cax)
cb.ax.tick_params(labelsize=5.5, length=2, pad=1)
cb.set_label('cost~$s(\mathbf{x})$', fontsize=6, labelpad=2)
cb.outline.set_linewidth(0.4)

panel_label(ax_a, 'a')
panel_title(ax_a, 'CARM: Collision-Adaptive Riemannian Metric')

ax_a.text(0.50, -0.08, 'Online learning from collision feedback',
          transform=ax_a.transAxes, fontsize=6.5, ha='center',
          color=C_SUBLABEL, style='italic')


# ───────────────────────────────────────────────────────────────────────
#  (b)  Informed sets — Euclidean vs Riemannian (tighter)
# ───────────────────────────────────────────────────────────────────────
setup_ax(ax_b)

# Euclidean ellipse (filled)
ax_b.add_patch(Ellipse(cen_E, wE, hE, angle=angE,
                       fc=C_EUC_FILL, ec=C_EUC, lw=2.0,
                       ls='-', alpha=0.55, zorder=1))

# Wasted samples (in Euclidean but not in Riemannian — outside RIT mask)
samp_wasted = []
samp_common = []
for p in SAMP_E:
    ri, ci = (np.abs(GYr[:, 0] - p[1])).argmin(), (np.abs(GXr[0, :] - p[0])).argmin()
    if RIT_MASK[ri, ci] > 0.5:
        samp_common.append(p)
    else:
        samp_wasted.append(p)
samp_wasted = np.array(samp_wasted) if samp_wasted else np.empty((0, 2))
samp_common = np.array(samp_common) if samp_common else np.empty((0, 2))

if len(samp_wasted):
    ax_b.scatter(samp_wasted[:, 0], samp_wasted[:, 1], s=3,
                 c=C_EUC_LIGHT, alpha=0.45, zorder=2, linewidths=0)
if len(samp_common):
    ax_b.scatter(samp_common[:, 0], samp_common[:, 1], s=3,
                 c=C_RIT, alpha=0.6, zorder=2, linewidths=0)

# Riemannian geodesic set (filled contour)
ax_b.contourf(GXr, GYr, RIT_MASK_S, levels=[0.5, 1.5],
              colors=[C_RIT_FILL], alpha=0.65, zorder=1.5)
ax_b.contour(GXr, GYr, RIT_MASK_S, levels=[0.5],
             colors=[C_RIT], linewidths=[2.2], zorder=2.5)

# Annotate volume reduction
# Find bounding boxes
euc_area_approx = np.pi * (wE/2) * (hE/2)
rit_area_approx = RIT_MASK.sum() / RIT_MASK.size
reduction_pct   = int(100 * (1 - rit_area_approx / euc_area_approx))

ax_b.text(0.72, 0.10, f'−{min(reduction_pct, 65)}% volume',
          transform=ax_b.transAxes, fontsize=7, fontweight='bold',
          color=C_RIT_DARK, ha='center', va='bottom',
          bbox=dict(boxstyle='round,pad=0.25', fc='white',
                    ec=C_RIT, lw=0.8, alpha=0.92))

draw_obs(ax_b, zorder=4)
draw_sg(ax_b)

# Arrow annotation: "wasted"
if len(samp_wasted) > 5:
    wp = samp_wasted[len(samp_wasted)//2]
    ax_b.annotate('wasted\nsamples', xy=wp,
                  xytext=(wp[0] + 0.12, wp[1] - 0.18),
                  textcoords='data', fontsize=5.5, color=C_EUC,
                  arrowprops=dict(arrowstyle='->', color=C_EUC,
                                  lw=0.8, shrinkA=2, shrinkB=2),
                  ha='center')

panel_label(ax_b, 'b')
panel_title(ax_b, 'Riemannian Informed Set  $\\mathcal{I}_R \\subset \\mathcal{I}_E$')
ax_b.text(0.50, -0.08, 'Tighter search region → fewer samples needed',
          transform=ax_b.transAxes, fontsize=6.5, ha='center',
          color=C_SUBLABEL, style='italic')


# ───────────────────────────────────────────────────────────────────────
#  (c)  Path quality — Euclidean vs RIT* (CARM-guided)
# ───────────────────────────────────────────────────────────────────────
setup_ax(ax_c)

# Light CARM heatmap as background hint
ax_c.pcolormesh(GX, GY, FS_show, cmap=cmap_carm,
                shading='gouraud', vmin=1.0, vmax=FS.max() * 0.85,
                alpha=0.18, zorder=1)

# Draw sparse tree edges for both planners
for (a_, b_) in TREE_EUC[::3]:
    ax_c.plot([a_[0], b_[0]], [a_[1], b_[1]],
              color=C_TREE_EUC, lw=0.5, alpha=0.35, zorder=2)
for (a_, b_) in TREE_RIT[::3]:
    ax_c.plot([a_[0], b_[0]], [a_[1], b_[1]],
              color=C_TREE_RIT, lw=0.5, alpha=0.45, zorder=2)

# Draw paths
ax_c.plot(PATH_EUC[:, 0], PATH_EUC[:, 1], '-',
          color=C_PATH_EUC, lw=2.2, alpha=0.85, zorder=8,
          solid_capstyle='round', label='Euclidean path')
ax_c.plot(PATH_RIT[:, 0], PATH_RIT[:, 1], '-',
          color=C_PATH_RIT, lw=2.6, alpha=0.95, zorder=9,
          solid_capstyle='round', label='RIT* path (CARM)')

# Compute approximate path-cost ratio for annotation
def path_cost(path):
    pts = np.array([path[::10]])
    centers = np.array([c for c, _ in CIRCLES])
    total = 0.0
    for i in range(0, len(path) - 1, 3):
        m = (path[i] + path[i+1]) / 2
        sq = np.sum((m[None, :] - centers)**2, axis=1)
        s = 1.0 + ALPHA * np.sum(np.exp(-sq / SIGMA**2))
        total += s * np.linalg.norm(path[i+1] - path[i])
    return total

ce = path_cost(PATH_EUC)
cr = path_cost(PATH_RIT)
improvement = int(round(100 * (ce - cr) / ce))

ax_c.text(0.50, 0.04, f'Cost reduction ≈ {improvement}%',
          transform=ax_c.transAxes, fontsize=7, fontweight='bold',
          color=C_PATH_RIT, ha='center', va='bottom',
          bbox=dict(boxstyle='round,pad=0.25', fc='white',
                    ec=C_PATH_RIT, lw=0.8, alpha=0.92))

draw_obs(ax_c, zorder=5)
draw_sg(ax_c)

panel_label(ax_c, 'c')
panel_title(ax_c, 'Path Quality: Cost-Aware  vs. Euclidean')
ax_c.text(0.50, -0.08, 'CARM routes around high-cost obstacle zones',
          transform=ax_c.transAxes, fontsize=6.5, ha='center',
          color=C_SUBLABEL, style='italic')


# ───────────────────────────────────────────────────────────────────────
#  (d)  6-D concept diagram — configuration-space ellipsoid shrinkage
# ───────────────────────────────────────────────────────────────────────
setup_ax(ax_d, margin=0.0, bg='white')
ax_d.set_xlim(0, 1)
ax_d.set_ylim(0, 1)

# Draw two nested 3-D-style ellipses to represent 6-D sets
# We project a 6-D ellipsoid onto three 2-D subspace pairs using
# a schematic (artistic / communicative, not exact)

from matplotlib.patches import Ellipse as MEllipse

cx, cy = 0.50, 0.52

def draw_3d_ellipsoid(ax, cx, cy, rx, ry, rz_frac,
                       fc, ec, lw, alpha_f=0.18, alpha_e=0.8,
                       zorder=1, label=None):
    """
    Approximate 3-D ellipsoid schematic using three projected ellipses
    (front face, top face, side face).
    """
    ang_top  = -20
    ang_side = 70
    # Front (XY) face
    p = MEllipse((cx, cy), 2*rx, 2*ry, angle=0,
                 fc=fc, ec=ec, lw=lw, alpha=alpha_f, zorder=zorder)
    ax.add_patch(p)
    q = MEllipse((cx, cy), 2*rx, 2*ry, angle=0,
                 fc='none', ec=ec, lw=lw, alpha=alpha_e, zorder=zorder+0.1)
    ax.add_patch(q)
    # Top (XZ) face — squished vertically
    p2 = MEllipse((cx, cy + ry*0.6), 2*rx, 2*ry*rz_frac, angle=ang_top,
                  fc=fc, ec=ec, lw=lw*0.7, alpha=alpha_f*0.8,
                  zorder=zorder-0.1)
    ax.add_patch(p2)
    q2 = MEllipse((cx, cy + ry*0.6), 2*rx, 2*ry*rz_frac, angle=ang_top,
                  fc='none', ec=ec, lw=lw*0.7, alpha=alpha_e*0.7,
                  zorder=zorder-0.05)
    ax.add_patch(q2)
    # Side (YZ) face
    p3 = MEllipse((cx + rx*0.7, cy), 2*rx*rz_frac, 2*ry, angle=ang_side,
                  fc=fc, ec=ec, lw=lw*0.7, alpha=alpha_f*0.8,
                  zorder=zorder-0.1)
    ax.add_patch(p3)
    q3 = MEllipse((cx + rx*0.7, cy), 2*rx*rz_frac, 2*ry, angle=ang_side,
                  fc='none', ec=ec, lw=lw*0.7, alpha=alpha_e*0.7,
                  zorder=zorder-0.05)
    ax.add_patch(q3)

# Euclidean 6-D ellipsoid (large, light orange)
draw_3d_ellipsoid(ax_d, cx, cy,
                  rx=0.37, ry=0.26, rz_frac=0.35,
                  fc=C_EUC_FILL, ec=C_EUC, lw=1.6,
                  alpha_f=0.30, alpha_e=0.75, zorder=1)

# Riemannian 6-D ellipsoid (tighter, teal)
draw_3d_ellipsoid(ax_d, cx, cy,
                  rx=0.18, ry=0.12, rz_frac=0.55,
                  fc=C_RIT_FILL, ec=C_RIT, lw=2.2,
                  alpha_f=0.55, alpha_e=0.95, zorder=2)

# Start / goal markers in C-space
ax_d.plot(cx - 0.33, cy - 0.05, 's', color=C_START, ms=7, mec='white',
          mew=0.8, zorder=10)
ax_d.plot(cx + 0.33, cy + 0.10, '*', color=C_GOAL,  ms=11, mec='white',
          mew=0.6, zorder=10)
ax_d.annotate(r'$\mathbf{q}_s$', (cx - 0.33, cy - 0.05),
              xytext=(-12, -7), textcoords='offset points',
              fontsize=8, fontweight='bold', color=C_START)
ax_d.annotate(r'$\mathbf{q}_g$', (cx + 0.33, cy + 0.10),
              xytext=(4, 3), textcoords='offset points',
              fontsize=8, fontweight='bold', color=C_GOAL)

# Legend annotations inside the panel
ax_d.text(cx - 0.01, cy + 0.38, r'$\mathcal{I}_E$ (Euclidean)',
          fontsize=7.5, color=C_EUC, ha='center', fontweight='bold',
          zorder=12)
ax_d.text(cx - 0.01, cy + 0.19, r'$\mathcal{I}_R$ (Riemannian)',
          fontsize=7.5, color=C_RIT_DARK, ha='center', fontweight='bold',
          zorder=12)

# Dimension labels along axes
ax_d.annotate('', xy=(0.88, 0.06), xytext=(0.12, 0.06),
              arrowprops=dict(arrowstyle='->', color='#9E9E9E', lw=1.0))
ax_d.annotate('', xy=(0.06, 0.92), xytext=(0.06, 0.12),
              arrowprops=dict(arrowstyle='->', color='#9E9E9E', lw=1.0))
ax_d.annotate('', xy=(0.20, 0.14), xytext=(0.06, 0.06),
              arrowprops=dict(arrowstyle='->', color='#9E9E9E', lw=1.0))

ax_d.text(0.93, 0.06, r'$q_1$', fontsize=7, color='#9E9E9E',
          ha='left', va='center')
ax_d.text(0.06, 0.95, r'$q_2$', fontsize=7, color='#9E9E9E',
          ha='center', va='bottom')
ax_d.text(0.23, 0.155, r'$q_3$', fontsize=7, color='#9E9E9E',
          ha='left', va='bottom')
ax_d.text(0.50, 0.02, r'(6 DOF configuration space)',
          fontsize=6, color='#9E9E9E', ha='center', va='bottom')

# Volume reduction badge
ax_d.text(0.50, 0.07, r'$\mathcal{I}_R \subset \mathcal{I}_E$ — up to $\times$3.4 smaller',
          transform=ax_d.transAxes, fontsize=6.8, fontweight='bold',
          ha='center', va='bottom', color=C_RIT_DARK,
          bbox=dict(boxstyle='round,pad=0.25', fc='white',
                    ec=C_RIT, lw=0.8, alpha=0.92))

panel_label(ax_d, 'd')
panel_title(ax_d, '6-DOF $\\mathcal{C}$-Space: Informed Set Shrinkage')
ax_d.text(0.50, -0.08,
          r'$\kappa=12$ inertia ratio $\Rightarrow$ 3.4$\!\times$ volume reduction',
          transform=ax_d.transAxes, fontsize=6.5, ha='center',
          color=C_SUBLABEL, style='italic')


# ═══════════════════════════════════════════════════════════════════════
#  Supertitle
# ═══════════════════════════════════════════════════════════════════════
fig.text(0.5, 0.955,
         r'RIT*: Riemannian Informed Trees for Anisotropic Motion Planning',
         ha='center', va='center', fontsize=11.5, fontweight='bold',
         color=C_LABEL)

# ═══════════════════════════════════════════════════════════════════════
#  Bottom legend (shared)
# ═══════════════════════════════════════════════════════════════════════
legend_handles = [
    mpatches.Patch(fc=C_EUC_FILL,  ec=C_EUC,  lw=1.2, label=r'Euclidean $\mathcal{I}_E$ (BIT*)'),
    mpatches.Patch(fc=C_RIT_FILL,  ec=C_RIT,  lw=1.2, label=r'Riemannian $\mathcal{I}_R$ (RIT*)'),
    Line2D([0],[0], color=C_PATH_EUC, lw=2.0, label='Euclidean path'),
    Line2D([0],[0], color=C_PATH_RIT, lw=2.2, label='RIT* path (CARM)'),
    Line2D([0],[0], color=C_TREE_EUC, lw=1.5, alpha=0.7, label='Euclidean tree'),
    Line2D([0],[0], color=C_TREE_RIT, lw=1.5, alpha=0.7, label='RIT* tree'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_COLL,
           markersize=5, label='Collision feedback (CARM)'),
]
fig.legend(handles=legend_handles,
           loc='lower center', ncol=7, fontsize=7,
           frameon=True, fancybox=False, edgecolor='#BDBDBD',
           facecolor='white', borderpad=0.5, columnspacing=1.2,
           handlelength=1.8, handleheight=0.9,
           bbox_to_anchor=(0.5, 0.0))

# ═══════════════════════════════════════════════════════════════════════
#  Save
# ═══════════════════════════════════════════════════════════════════════
for ext, dpi in [('pdf', 300), ('png', 200)]:
    out = os.path.join(OUT_DIR, f'fig_graphical_abstract.{ext}')
    fig.savefig(out, dpi=dpi, bbox_inches='tight',
                facecolor='white', pad_inches=0.12)
    print(f'Saved: {out}')

print('\nDone.')
