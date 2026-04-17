"""
generate_abstract_figure_new.py — IMPROVED Graphical abstract for RIT* paper.

Key improvement: Shows the actual curved Riemannian geodesic informed set
boundary instead of a perfect ellipse, highlighting the key difference between
BIT* (perfect ellipse) and RIT* (adaptive geodesic surface).

Output: paper/figures/fig_abstract_new.pdf + .png
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Polygon
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist
import matplotlib.path as mpath

# Add rit_star to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rit_star.metric import ObstacleInflatedMetric
from rit_star.geodesic import GeodesicComputer
from rit_star.informed_set import RiemannianInformedSet, EuclideanInformedSet

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'paper', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════
#  Color palette
# ════════════════════════════════════════════════════════════════
C = dict(
    obs='#546E7A', obs_e='#37474F',
    euc='#E65100', euc_fill='#FFF3E0',
    wasted='#FF8A65',
    rit='#00695C', rit_fill='#E0F2F1',
    rit_samp='#4DB6AC',
    path_euc='#455A64',
    path_opt='#C62828',
    tree_euc='#B0BEC5',
    tree_carm='#90CAF9',
    coll_x='#D32F2F',
    coll_dot='#00897B',
    start='#2E7D32', goal='#C62828',
    arrow_flow='#455A64',
    label='#37474F', sublabel='#607D8B',
    combined='#1565C0',
)

# ════════════════════════════════════════════════════════════════
#  Environment geometry
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

SIGMA_M = 0.12
ALPHA_M = 8.0

# Create the metric and geodesic computer
obstacle_centers = np.array([c for c, _ in CIRCLES])
METRIC = ObstacleInflatedMetric(
    obstacle_centers=obstacle_centers,
    sigma=SIGMA_M,
    alpha=ALPHA_M
)
GEODESIC_COMPUTER = GeodesicComputer(METRIC, tier='diagonal')


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
#  CARM metric field
# ════════════════════════════════════════════════════════════════
def carm_field(res=150):
    gx = np.linspace(-0.25, 1.25, res)
    gy = np.linspace(-0.25, 1.25, res)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    centers = np.array([c for c, _ in CIRCLES])
    sq = cdist(pts, centers, 'sqeuclidean')
    S = 1.0 + ALPHA_M * np.sum(np.exp(-sq / (SIGMA_M**2)), axis=1)
    return GX, GY, S.reshape(res, res)


def get_metric_scale(p):
    """Get the metric scale s(x) at a point."""
    return METRIC.scale(p)


def compute_riemannian_informed_set_contour(xs, xg, c_best, resolution=200):
    """
    Compute the Riemannian geodesic informed set boundary.
    Returns contour points where d_R(xs, x) + d_R(x, xg) ≈ c_best
    """
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    ris = RiemannianInformedSet(xs, xg, c_best, GEODESIC_COMPUTER, bounds=bounds)
    
    # Create grid
    r1 = c_best / 2.0
    center = (xs + xg) / 2.0
    lo = center - r1
    hi = center + r1
    for k in range(2):
        lo[k] = max(lo[k], bounds[k][0])
        hi[k] = min(hi[k], bounds[k][1])
    
    xs_grid = np.linspace(lo[0], hi[0], resolution)
    ys_grid = np.linspace(lo[1], hi[1], resolution)
    XX, YY = np.meshgrid(xs_grid, ys_grid)
    
    # Vectorized membership test
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    member_mask = ris.batch_is_member(pts)
    mask = member_mask.reshape(XX.shape).astype(float)
    
    print(f"  Mask stats: min={mask.min()}, max={mask.max()}, mean={mask.mean():.3f}")
    print(f"  Number of points inside: {member_mask.sum()}/{len(member_mask)}")
    
    # Extract contour at 0.5 level
    fig_temp, ax_temp = plt.subplots()
    # Try different contour levels to see what works
    try:
        cs = ax_temp.contour(XX, YY, mask, levels=[0.5])
    except:
        # If 0.5 doesn't work, try 0.99
        cs = ax_temp.contour(XX, YY, mask, levels=[0.99])
    
    # Extract contour paths
    boundary_points = []
    if hasattr(cs, 'allsegs'):
        # Older matplotlib
        for seg_list in cs.allsegs:
            for seg in seg_list:
                if len(seg) > 0:
                    boundary_points.append(seg)
    else:
        # Newer matplotlib
        for collection in cs.collections:
            for path in collection.get_paths():
                verts = path.vertices
                if len(verts) > 0:
                    boundary_points.append(verts)
    
    plt.close(fig_temp)
    
    if boundary_points:
        # Take the longest contour (main boundary)
        main_boundary = max(boundary_points, key=len)
        return main_boundary, mask, XX, YY
    else:
        return None, mask, XX, YY


# ════════════════════════════════════════════════════════════════
#  Ellipse parameters (for Euclidean case)
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


# Both informed sets use conceptually the same c_best threshold,
# but expressed in their respective metrics
# For visualization: use c_best that shows meaningful comparison
C_BEST_EUCLIDEAN = 1.55  # Euclidean distance threshold
C_BEST_RIEMANNIAN = 2.2  # Riemannian distance threshold (larger due to metric inflation)

# Euclidean ellipse
cen_E, wE, hE, angE = ell_params(XS, XG, C_BEST_EUCLIDEAN)

# ════════════════════════════════════════════════════════════════
#  Paths
# ════════════════════════════════════════════════════════════════
def get_smooth_path(pts, samples=100):
    t = np.linspace(0, 1, len(pts))
    cs = CubicSpline(t, pts)
    return cs(np.linspace(0, 1, samples))


pth_e_pts = [XS, [0.3, 0.5], [0.5, 0.55], [0.7, 0.7], XG]
PATH_EUC = get_smooth_path(pth_e_pts)

pth_c_pts = [XS, [0.3, 0.2], [0.5, 0.2], [0.7, 0.3], [0.9, 0.4], XG]
PATH_OPT = get_smooth_path(pth_c_pts)


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
            if 0 <= p[0] <= 1 and 0 <= p[1] <= 1 and not in_obs(p):
                pts.append(p)
                if len(pts) >= n:
                    break
    return np.array(pts[:n])


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


SAMP_E = gen_samples(cen_E, wE, hE, angE, 250, seed=8)
dir_vec = XG - XS
SAMP_E = np.array([p for p in SAMP_E if np.dot(p - XS, dir_vec) > 0])
TREE_EUC_EDGES = build_radial_tree(XS, SAMP_E)


# ════════════════════════════════════════════════════════════════
#  Compute Riemannian informed set boundary
# ════════════════════════════════════════════════════════════════
print("Computing Riemannian geodesic informed set boundary...")
# Use the SAME c_best for both informed sets (key insight!)
# Both check if cost(xs->x) + cost(x->xg) <= c_best
# The difference is Euclidean vs Riemannian distance measurement
RIT_BOUNDARY, RIT_MASK, RIT_XX, RIT_YY = compute_riemannian_informed_set_contour(XS, XG, c_best=C_BEST_RIEMANNIAN, resolution=200)
print(f"  Found boundary with {len(RIT_BOUNDARY) if RIT_BOUNDARY is not None else 0} points")

# Generate samples inside Riemannian boundary
if RIT_BOUNDARY is not None and len(RIT_BOUNDARY) > 0:
    boundary_path = mpath.Path(RIT_BOUNDARY)
    
    # Generate candidate samples
    rng = np.random.RandomState(8)
    candidates = rng.uniform(0, 1, (5000, 2))
    
    # Filter to those inside boundary and not in obstacles
    inside_mask = boundary_path.contains_points(candidates)
    SAMP_R = []
    for p in candidates[inside_mask]:
        if not in_obs(p) and np.dot(p - XS, dir_vec) > 0:
            SAMP_R.append(p)
            if len(SAMP_R) >= 150:
                break
    SAMP_R = np.array(SAMP_R)
    if len(SAMP_R) > 0:
        TREE_CARM_EDGES = build_radial_tree(XS, SAMP_R)
    else:
        print("  Warning: No samples found insideC_BESTndary, using fallback")
        cen_R, wR, hR, angR = ell_params(XS, XG, C_BEST_RIEMANNIAN)
        SAMP_R = gen_samples(cen_R, wR, hR, angR, 150, seed=8)
        TREE_CARM_EDGES = build_radial_tree(XS, SAMP_R)
else:
    # Fallback to ellipse if boundary computation fails
    print("  Warning: Using fallback ellipse C_BESTRIT* boundary")
    cen_R, wR, hR, angR = ell_params(XS, XG, C_BEST_RIEMANNIAN)
    SAMP_R = gen_samples(cen_R, wR, hR, angR, 150, seed=8)
    TREE_CARM_EDGES = build_radial_tree(XS, SAMP_R)


def gen_collision_pts(n_per_obs=60, seed=77):
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
#  BUILD FIGURE
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
#  (a) BIT* — Euclidean Informed Set (PERFECT ELLIPSE)
# ──────────────────────────────────────────────────────────────
ax_prob = fig.add_subplot(gs[0, 0])
setup(ax_prob, margin=0.32)

# Euclidean ellipse (PERFECT mathematical ellipse, dashed border)
ax_prob.add_patch(Ellipse(cen_E, wE, hE, angle=angE,
                          fc=C['euc_fill'], ec=C['euc'],
                          lw=2.5, ls='--', alpha=0.4, zorder=1))

# Euclidean samples
for p in SAMP_E:
    ax_prob.plot(p[0], p[1], 'o', color=C['wasted'], ms=1.5,
                 alpha=0.55, zorder=3, mec='none')

# Euclidean tree edges
for edge in TREE_EUC_EDGES:
    ax_prob.plot([edge[0, 0], edge[1, 0]], [edge[0, 1], edge[1, 1]],
                 color=C['tree_euc'], lw=0.8, alpha=0.4, zorder=2)

# Euclidean path
ax_prob.plot(PATH_EUC[:, 0], PATH_EUC[:, 1], '-',
             color=C['path_euc'], lw=2.5, alpha=0.9, zorder=8,
             solid_capstyle='round')

draw_obs(ax_prob)
draw_sg(ax_prob)

# Add text to indicate perfect ellipse
ax_prob.text(0.50, -0.22, r'Perfect Ellipse (Euclidean)',
             fontsize=10, color=C['euc'], ha='center', va='top',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='white',
                       ec=C['euc'], alpha=0.9, lw=0.8))

ax_prob.set_title(r'(a)  BIT* — Euclidean Informed Set', fontsize=12.5,
                  fontweight='bold', color=C['label'], pad=8)

# ──────────────────────────────────────────────────────────────
#  (b) RIT* — Riemannian Geodesic Surface (NOT an ellipse!)
# ──────────────────────────────────────────────────────────────
ax_rit = fig.add_subplot(gs[0, 1])
setup(ax_rit, margin=0.32)

# Ghost Euclidean ellipse for comparison
ax_rit.add_patch(Ellipse(cen_E, wE, hE, angle=angE,
                          fc='none', ec=C['euc'],
                          lw=1.5, ls='--', alpha=0.25, zorder=0.5))

# CARM heatmap as background
ax_rit.pcolormesh(Xh, Yh, Fh, cmap='YlOrRd', shading='gouraud',
                   alpha=0.35, zorder=0.6)

# Draw Riemannian geodesic boundary (CURVED SURFACE!)
if RIT_BOUNDARY is not None and len(RIT_BOUNDARY) > 0:
    # Fill the interior using contourf
    ax_rit.contourf(RIT_XX, RIT_YY, RIT_MASK, levels=[0.5, 1.5],
                    colors=[C['rit_fill']], alpha=0.6, zorder=0.7)
    
    # Draw the curved boundary - make it thicker and more prominent
    ax_rit.plot(RIT_BOUNDARY[:, 0], RIT_BOUNDARY[:, 1],
                '-', color=C['rit'], lw=3.0, alpha=0.95, zorder=0.8,
                solid_capstyle='round')
else:
    # Fallback: show that RIT* boundary failed to compute
    ax_rit.text(0.5, 0.5, 'Boundary computation failed', 
                ha='center', va='center', fontsize=10, color='red',
                transform=ax_rit.transAxes)

# Collision points
ax_rit.scatter(COLL_PTS[:, 0], COLL_PTS[:, 1], s=5.0,
               c=C['coll_dot'], alpha=0.8, zorder=4)

# CARM Tree samples
for p in SAMP_R:
    ax_rit.plot(p[0], p[1], 'o', color='#1976D2', ms=1.5, alpha=0.8, zorder=3, mec='none')

# CARM tree edges
for edge in TREE_CARM_EDGES:
    ax_rit.plot([edge[0, 0], edge[1, 0]], [edge[0, 1], edge[1, 1]],
                 color=C['tree_carm'], lw=1.0, alpha=0.7, zorder=2)

# CARM path
ax_rit.plot(PATH_OPT[:, 0], PATH_OPT[:, 1], '-', color=C['path_opt'],
            lw=2.5, alpha=0.9, zorder=8, solid_capstyle='round')

draw_obs(ax_rit, alpha=0.9)
draw_sg(ax_rit)

# Add text to indicate curved geodesic surface
ax_rit.text(0.50, -0.22, r'Curved Geodesic Surface (Riemannian)',
            fontsize=10, color=C['rit'], ha='center', va='top',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec=C['rit'], alpha=0.9, lw=0.8))

ax_rit.set_title(r'(b)  RIT* — Riemannian Geodesic Informed Set', fontsize=12.5,
                 fontweight='bold', color=C['label'], pad=8)

# ──────────────────────────────────────────────────────────────
#  Supertitle
# ──────────────────────────────────────────────────────────────
fig.suptitle('RIT*: Riemannian Informed Trees for Faster Motion Planning',
             fontsize=15, fontweight='bold', color=C['label'], y=0.96)

# ──────────────────────────────────────────────────────────────
#  Legend
# ──────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], ls='--', color=C['euc'], lw=1.5,
           label='BIT* ellipse (ignores obstacles)'),
    Line2D([0], [0], ls='-', color=C['rit'], lw=2.5,
           label='RIT* geodesic (curves around obstacles)'),
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
fig.savefig(os.path.join(OUT_DIR, 'fig_abstract_new.pdf'),
            dpi=300, bbox_inches='tight', facecolor='white',
            pad_inches=0.15)
fig.savefig(os.path.join(OUT_DIR, 'fig_abstract_new.png'),
            dpi=200, bbox_inches='tight', facecolor='white',
            pad_inches=0.15)

print(f"\n✓ Saved improved abstract figure to:")
print(f"  {os.path.join(OUT_DIR, 'fig_abstract_new.pdf')}")
print(f"  {os.path.join(OUT_DIR, 'fig_abstract_new.png')}")
