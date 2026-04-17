"""
generate_abstract_figure.py — Graphical abstract for the RIT* paper.

Uses the 6-circle obstacle environment (same as fig_carm_overview /
fig_carm_informed_set in the paper) for a consistent visual story.

Layout (2-row × 3-column):
  Left   (tall):  Problem — Euclidean tree + path (like overview (c))
  Mid-top:        Solution A — Riemannian informed set shrinks search
  Mid-bottom:     Solution B — CARM field (like overview (a))
  Right  (tall):  Combined — CARM tree + optimal path (like metric_field (c))

Output: paper/figures/fig_abstract.pdf  (+ .png preview)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'paper', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════
#  Color palette
# ════════════════════════════════════════════════════════════════
C = dict(
    obs='#546E7A', obs_e='#37474F',           # dark blue-grey obstacles
    euc='#E65100', euc_fill='#FFF3E0',        # orange = Euclidean
    wasted='#FF8A65',
    rit='#00695C', rit_fill='#E0F2F1',        # teal = RIT*
    rit_samp='#4DB6AC',
    path_euc='#455A64',                       # dark grey Euclidean path
    path_opt='#C62828',                       # red — our approach (CARM)
    tree_euc='#B0BEC5',                       # light grey Euclidean tree
    tree_carm='#90CAF9',                      # light blue CARM tree
    coll_x='#D32F2F',
    coll_dot='#00897B',                       # teal collision dots (like paper)
    start='#2E7D32', goal='#C62828',
    arrow_flow='#455A64',
    label='#37474F', sublabel='#607D8B',
    combined='#1565C0',
)

# ════════════════════════════════════════════════════════════════
#  Environment geometry — 6 circles (matches env_2d_obstacle_inflated)
# ════════════════════════════════════════════════════════════════
XS = np.array([0.05, 0.25])
XG = np.array([0.95, 0.75])

CIRCLES = [
    (np.array([0.30, 0.35]), 0.08),
    (np.array([0.30, 0.65]), 0.08),
    (np.array([0.50, 0.45]), 0.09),
    (np.array([0.50, 0.75]), 0.09),
    (np.array([0.70, 0.40]), 0.08),
    (np.array([0.70, 0.60]), 0.08),
]


def in_obs(p, inflate=0.0):
    for c, r in CIRCLES:
        if (p[0]-c[0])**2 + (p[1]-c[1])**2 <= (r + inflate)**2:
            return True
    return False


def draw_obs(ax, alpha=1.0, zorder=2):
    for cen, r in CIRCLES:
        ax.add_patch(Circle(cen, r, fc=C['obs'], ec=C['obs_e'],
                            lw=0.8, alpha=alpha, zorder=zorder))


def draw_sg(ax, fs=9):
    ax.plot(*XS, 's', color=C['start'], ms=8, zorder=12,
            mec='white', mew=1.2)
    ax.plot(*XG, '*', color=C['goal'], ms=12, zorder=12,
            mec='white', mew=0.8)
    kw = dict(textcoords='offset points', fontsize=fs, fontweight='bold')
    ax.annotate(r'$x_s$', XS, xytext=(-14, -7), color=C['start'], **kw)
    ax.annotate(r'$x_g$', XG, xytext=(5, 5), color=C['goal'], **kw)


def setup(ax, margin=0.02):
    ax.set_xlim(-margin, 1.0 + margin)
    ax.set_ylim(-margin, 1.0 + margin)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(False)


# ════════════════════════════════════════════════════════════════
#  Synthetic beautiful paths and trees 
# ════════════════════════════════════════════════════════════════
def get_smooth_path(pts, samples=100):
    t = np.linspace(0, 1, len(pts))
    cs = CubicSpline(t, pts)
    return cs(np.linspace(0, 1, samples))

# Euclidean path gracefully grazing the obstacle edge directly through the hazardous gap
pth_e_pts = [XS, [0.3, 0.5], [0.5, 0.55], [0.7, 0.7], XG]
PATH_EUC = get_smooth_path(pth_e_pts)

# RIT* / CARM path nicely curving around the wider cost field
pth_c_pts = [XS, [0.3, 0.2], [0.5, 0.2], [0.7, 0.3], [0.9, 0.4], XG]
PATH_OPT = get_smooth_path(pth_c_pts)

def build_radial_tree(root, samples):
    edges = []
    dists = np.linalg.norm(samples - root, axis=1)
    order = np.argsort(dists)
    connected = np.array([root])
    for idx in order:
        pt = samples[idx]
        cdist = np.linalg.norm(connected - pt, axis=1)
        best = np.argmin(cdist)
        edges.append(np.array([connected[best], pt]))
        connected = np.vstack([connected, pt])
    return np.array(edges)

SAMP_E_TREE = None
SAMP_R_TREE = None

# We will build TREE_EUC_EDGES and TREE_CARM_EDGES later after samples are generated!

# ════════════════════════════════════════════════════════════════
#  Informed-set ellipse parameters
# ════════════════════════════════════════════════════════════════
def ell_params(xs, xg, c_best):
    cen = (xs + xg) / 2
    d = xg - xs
    c_min = np.linalg.norm(d)
    ang = np.degrees(np.arctan2(d[1], d[0]))
    a = c_best / 2.0
    b = np.sqrt(max(c_best**2 - c_min**2, 1e-6)) / 2.0
    return cen, 2*a, 2*b, ang


def in_ellipse(p, cen, w, h, ang):
    t = np.radians(ang)
    dx, dy = p[0]-cen[0], p[1]-cen[1]
    u =  dx*np.cos(t) + dy*np.sin(t)
    v = -dx*np.sin(t) + dy*np.cos(t)
    return (u/(w/2))**2 + (v/(h/2))**2 <= 1.0


# Euclidean ellipse (large) — c_best=1.55 so all obstacles fully inside
# Riemannian ellipse (tight) — c_best=1.32 so all obstacles fully inside
# (no ellipse boundary clipping through obstacle circles)
cen_E, wE, hE, angE = ell_params(XS, XG, 1.55)
cen_R, wR, hR, angR = ell_params(XS, XG, 1.32)


def _make_irregular_boundary(cen, w, h, ang, perturb_fn, n=300):
    """Generic helper: build an irregular closed boundary from ellipse params."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    t = np.radians(ang)
    R_mat = np.array([[np.cos(t), -np.sin(t)],
                      [np.sin(t),  np.cos(t)]])
    scale = 1.0 + perturb_fn(theta)
    xy_local = np.column_stack([
        scale * (w / 2) * np.cos(theta),
        scale * (h / 2) * np.sin(theta),
    ])
    return xy_local @ R_mat.T + cen


def _rit_perturb(theta):
    return (0.060 * np.sin(3 * theta + 0.40)
          + 0.040 * np.cos(5 * theta + 1.20)
          + 0.032 * np.sin(2 * theta + 0.60)
          + 0.022 * np.sin(7 * theta + 2.10)
          + 0.012 * np.cos(11 * theta + 0.80))


def _euc_perturb(theta):
    return (0.055 * np.sin(3 * theta + 1.10)
          + 0.038 * np.cos(5 * theta + 2.40)
          + 0.028 * np.sin(2 * theta + 1.80)
          + 0.018 * np.sin(7 * theta + 0.50)
          + 0.010 * np.cos(11 * theta + 1.90))


RIT_BOUNDARY = _make_irregular_boundary(cen_R, wR, hR, angR, _rit_perturb)
EUC_BOUNDARY = _make_irregular_boundary(cen_E, wE, hE, angE, _euc_perturb)


def _make_mpl_path(boundary):
    verts = np.vstack([boundary, boundary[0]])
    codes = np.array(
        [MplPath.MOVETO] + [MplPath.LINETO] * (len(boundary) - 1) + [MplPath.CLOSEPOLY]
    )
    return MplPath(verts, codes), verts, codes


_rit_mpl_path, _rit_verts_global, _rit_codes_global = _make_mpl_path(RIT_BOUNDARY)
_euc_mpl_path, _euc_verts_global, _euc_codes_global = _make_mpl_path(EUC_BOUNDARY)


def in_rit_shape(p):
    return _rit_mpl_path.contains_point(p)


def in_euc_shape(p):
    return _euc_mpl_path.contains_point(p)


# ════════════════════════════════════════════════════════════════
#  Sample generation
# ════════════════════════════════════════════════════════════════
def gen_samples(cen, w, h, ang, n, seed=42):
    rng = np.random.RandomState(seed)
    t = np.radians(ang)
    R = np.array([[np.cos(t), -np.sin(t)],
                  [np.sin(t),  np.cos(t)]])
    pts = []
    while len(pts) < n:
        uv = rng.uniform(-1, 1, (n*5, 2))
        uv = uv[uv[:, 0]**2 + uv[:, 1]**2 <= 1.0]
        sc = uv * np.array([[w/2, h/2]])
        xy = sc @ R.T + cen
        for p in xy:
            if not in_obs(p):
                pts.append(p)
                if len(pts) >= n:
                    break
    return np.array(pts[:n])


def gen_samples_in_shape(boundary, path_fn, n, seed=42):
    """Generate free-space samples inside an arbitrary closed boundary."""
    rng = np.random.RandomState(seed)
    bx_lo = boundary[:, 0].min() - 0.01
    bx_hi = boundary[:, 0].max() + 0.01
    by_lo = boundary[:, 1].min() - 0.01
    by_hi = boundary[:, 1].max() + 0.01
    pts = []
    while len(pts) < n:
        candidates = rng.uniform(
            [bx_lo, by_lo], [bx_hi, by_hi], (n * 10, 2))
        for p in candidates:
            if (path_fn(p) and not in_obs(p)
                    and 0 <= p[0] <= 1 and 0 <= p[1] <= 1):
                pts.append(p)
                if len(pts) >= n:
                    break
    return np.array(pts[:n])


dir_vec = XG - XS
SAMP_E = gen_samples(cen_E, wE, hE, angE, 1800, seed=8)
SAMP_R = gen_samples_in_shape(RIT_BOUNDARY, in_rit_shape, 850, seed=8)

TREE_EUC_EDGES = build_radial_tree(XS, SAMP_E)
TREE_CARM_EDGES = build_radial_tree(XS, SAMP_R)

# Extra samples in Euclidean but outside Riemannian for panel b
SAMP_E_OUT = gen_samples_in_shape(EUC_BOUNDARY, in_euc_shape, 100, seed=10)
SAMP_E_OUT = SAMP_E_OUT[~np.array([in_rit_shape(p) for p in SAMP_E_OUT])]

# ════════════════════════════════════════════════════════════════
#  CARM metric field  (matches ObstacleInflatedMetric formula)
#    s(x) = 1 + α · Σᵢ exp(-‖x − cᵢ‖² / σ²)
#  Rendered with pcolormesh + gouraud shading as in the paper.
# ════════════════════════════════════════════════════════════════
SIGMA_M = 0.12   # metric inflation radius (same as env_2d_obstacle_inflated)
ALPHA_M = 8.0    # inflation strength

def carm_field(res=150):
    """Compute the oracle-style metric scale s(x) on a grid."""
    gx = np.linspace(-0.25, 1.25, res)
    gy = np.linspace(-0.25, 1.25, res)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    centers = np.array([c for c, _ in CIRCLES])
    # Pairwise squared distances  (M, N)
    from scipy.spatial.distance import cdist
    sq = cdist(pts, centers, 'sqeuclidean')
    S = 1.0 + ALPHA_M * np.sum(np.exp(-sq / (SIGMA_M**2)), axis=1)
    return GX, GY, S.reshape(res, res)


def gen_collision_pts(n_per_obs=60, seed=77):
    """Generate collision points scattered around obstacle surfaces."""
    rng = np.random.RandomState(seed)
    pts = []
    for cen, r in CIRCLES:
        for _ in range(n_per_obs):
            th = rng.uniform(0, 2*np.pi)
            p = cen + r * np.array([np.cos(th), np.sin(th)])
            p += rng.uniform(-0.010, 0.010, 2)
            pts.append(p)
    return np.array(pts)


COLL_PTS = gen_collision_pts(15, seed=77)
Xh, Yh, Fh = carm_field()

# ════════════════════════════════════════════════════════════════
#  BUILD FIGURE — 2×3 layout with flow arrows
# ════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'mathtext.fontset': 'dejavusans',
})

fig = plt.figure(figsize=(10, 5.5), facecolor='white')

gs = fig.add_gridspec(
    1, 2,
    width_ratios=[1.0, 1.0],
    wspace=0.05,
    left=0.02, right=0.98, bottom=0.12, top=0.85,
)

# ──────────────────────────────────────────────────────────────
#  (a) Euclidean Informed Set — Euclidean tree + path
# ──────────────────────────────────────────────────────────────
ax_prob = fig.add_subplot(gs[0, 0])
setup(ax_prob, margin=0.32)

# Euclidean set — perfect dashed ellipse (matching panel b outer ellipse style)
ax_prob.add_patch(Ellipse(cen_E, wE, hE, angle=angE,
                          fc=C['euc_fill'], ec=C['euc'],
                          lw=2.5, ls='--', alpha=0.5, zorder=0.6))

# Dense tree — thin green lines, both endpoints inside IE
for edge in TREE_EUC_EDGES:
    if in_ellipse(edge[0], cen_E, wE, hE, angE) and in_ellipse(edge[1], cen_E, wE, hE, angE):
        ax_prob.plot([edge[0, 0], edge[1, 0]], [edge[0, 1], edge[1, 1]],
                     color='#388E3C', lw=0.45, alpha=0.65, zorder=2)

# Euclidean path (dark grey, like overview (c))
ax_prob.plot(PATH_EUC[:, 0], PATH_EUC[:, 1], '-',
             color=C['path_euc'], lw=2.5, alpha=0.9, zorder=8,
             solid_capstyle='round')

draw_obs(ax_prob)
draw_sg(ax_prob)

# Labels
ax_prob.text(0.50, -0.22, r'$\mathcal{I}_E$ — Euclidean informed set',
             fontsize=10, color=C['euc'], ha='center', va='top',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='white',
                       ec=C['euc'], alpha=0.9, lw=0.8))

ax_prob.set_title(r'(a)  Euclidean Informed Set ($\mathcal{I}_E$)', fontsize=12.5,
                  fontweight='bold', color=C['label'], pad=8)

# ──────────────────────────────────────────────────────────────
#  (b) RIT* replaces Euclidean primitives
# ──────────────────────────────────────────────────────────────
ax_rit = fig.add_subplot(gs[0, 1])
setup(ax_rit, margin=0.32)

# Ghost Euclidean ellipse
ax_rit.add_patch(Ellipse(cen_E, wE, hE, angle=angE,
                           fc=C['euc_fill'], ec=C['euc'],
                           lw=1.5, ls='--', alpha=0.3, zorder=0.5))

# Riemannian geodesic set — filled background (arbitrary shape)
ax_rit.fill(RIT_BOUNDARY[:, 0], RIT_BOUNDARY[:, 1],
            fc=C['rit_fill'], ec='none', alpha=0.8, zorder=0.6)

# CARM heatmap clipped to Riemannian shape
_rit_clip_patch = PathPatch(
    MplPath(_rit_verts_global, _rit_codes_global),
    transform=ax_rit.transData, visible=False)
ax_rit.add_patch(_rit_clip_patch)
_mesh_rit = ax_rit.pcolormesh(Xh, Yh, Fh, cmap='YlOrRd', shading='gouraud',
                              alpha=0.55, zorder=0.7)
_mesh_rit.set_clip_path(_rit_clip_patch)

# Riemannian shape border
_bx = np.append(RIT_BOUNDARY[:, 0], RIT_BOUNDARY[0, 0])
_by = np.append(RIT_BOUNDARY[:, 1], RIT_BOUNDARY[0, 1])
ax_rit.plot(_bx, _by, '-', color=C['rit'], lw=2.5, alpha=0.7, zorder=0.8)

# Dense tree edges — thin green lines, both endpoints inside IR
for edge in TREE_CARM_EDGES:
    if in_rit_shape(edge[0]) and in_rit_shape(edge[1]):
        ax_rit.plot([edge[0, 0], edge[1, 0]], [edge[0, 1], edge[1, 1]],
                     color='#388E3C', lw=0.45, alpha=0.65, zorder=2)

# Collision points (boundary)
ax_rit.scatter(COLL_PTS[:, 0], COLL_PTS[:, 1], s=5.0,
                c=C['coll_x'], alpha=0.8, zorder=4)

# A few red samples on top of (inside) the obstacles — represent collision feedback
_rng_obs = np.random.RandomState(42)
_obs_pts = []
for _cen, _r in CIRCLES:
    for _ in range(3):
        _th = _rng_obs.uniform(0, 2 * np.pi)
        _rad = _rng_obs.uniform(0.2, 0.8) * _r
        _obs_pts.append(_cen + _rad * np.array([np.cos(_th), np.sin(_th)]))
_obs_pts = np.array(_obs_pts)
ax_rit.scatter(_obs_pts[:, 0], _obs_pts[:, 1], s=18.0,
               c=C['coll_x'], alpha=0.9, zorder=12, marker='x', linewidths=1.2)

# CARM path
ax_rit.plot(PATH_OPT[:, 0], PATH_OPT[:, 1], '-', color=C['path_opt'],
             lw=2.5, alpha=0.9, zorder=8, solid_capstyle='round')

draw_obs(ax_rit, alpha=0.9)
draw_sg(ax_rit)

# Labels
ax_rit.text(0.50, -0.22, r'$\mathcal{I}_R \subset \mathcal{I}_E$',
             fontsize=10, color=C['rit'], ha='center', va='top',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='white',
                       ec=C['rit'], alpha=0.9, lw=0.8))


ax_rit.set_title(r'(b)  Riemannian Informed Trees (RIT*)', fontsize=12.5,
                  fontweight='bold', color=C['label'], pad=8)

# ──────────────────────────────────────────────────────────────
#  Supertitle
# ──────────────────────────────────────────────────────────────
fig.suptitle('RIT*: Riemannian Informed Trees for Faster Motion Planning',
             fontsize=15, fontweight='bold', color=C['label'], y=0.96)

# ──────────────────────────────────────────────────────────────
#  Legend (bottom)
# ──────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], ls='--', color=C['euc'], lw=1.5,
           label=r'$\mathcal{I}_E$ (Euclidean — too large)'),
    Line2D([0], [0], ls='-', color=C['rit'], lw=2,
           label=r'$\mathcal{I}_R$ (Riemannian — tight)'),
    Line2D([0], [0], color=C['tree_euc'], lw=1.5, alpha=0.6,
           label='Euclidean tree'),
    Line2D([0], [0], color=C['tree_carm'], lw=1.5, alpha=0.6,
           label='CARM tree'),
    Line2D([0], [0], color=C['path_euc'], lw=2.0,
           label='Euclidean path'),
    Line2D([0], [0], color=C['path_opt'], lw=2.0,
           label='Our path (CARM)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=6,
           fontsize=8, frameon=True, fancybox=True,
           edgecolor='#E0E0E0', facecolor='white',
           borderpad=0.5, columnspacing=1.5,
           bbox_to_anchor=(0.5, -0.01))

# ──────────────────────────────────────────────────────────────
#  Save
# ──────────────────────────────────────────────────────────────
fig.savefig(os.path.join(OUT_DIR, 'fig_abstract.pdf'),
            dpi=300, bbox_inches='tight', facecolor='white',
            pad_inches=0.15)
fig.savefig(os.path.join(OUT_DIR, 'fig_abstract.png'),
            dpi=200, bbox_inches='tight', facecolor='white',
            pad_inches=0.15)
print(f'Saved: {OUT_DIR}/fig_abstract.pdf')
print(f'Saved: {OUT_DIR}/fig_abstract.png')
plt.close(fig)
