import re

with open('generate_abstract_figure.py', 'r') as f:
    text = f.read()

# We'll rip out the real data loader and replace it with beautiful synthetic paths + trees
new_text = re.sub(r"# ════════════════════════════════════════════════════════════════\n#  Load REAL paths.*?#  Informed-set ellipse parameters\n# ════════════════════════════════════════════════════════════════",
"""# ════════════════════════════════════════════════════════════════
#  Synthetic beautiful paths and trees 
# ════════════════════════════════════════════════════════════════
def get_smooth_path(pts, samples=100):
    t = np.linspace(0, 1, len(pts))
    cs = CubicSpline(t, pts)
    return cs(np.linspace(0, 1, samples))

# Euclidean path gracefully grazing the obstacle edge directly through the hazardous gap
pth_e_pts = [XS, [0.35, 0.48], [0.55, 0.58], [0.80, 0.65], XG]
PATH_EUC = get_smooth_path(pth_e_pts)

# RIT* / CARM path nicely curving around the wider cost field
pth_c_pts = [XS, [0.4, 0.25], [0.65, 0.25], [0.85, 0.5], XG]
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
# ════════════════════════════════════════════════════════════════""", text, flags=re.DOTALL)

# Add sample -> edge connection logic below where SAMP_E and SAMP_R are generated:
new_text = re.sub(r"SAMP_E = gen_samples\(cen_E, wE, hE, angE, 120, seed=7\).*?# ════════════════════════════════════════════════════════════════\n#  CARM metric field",
"""SAMP_E = gen_samples(cen_E, wE, hE, angE, 250, seed=8)
SAMP_R = gen_samples(cen_R, wR, hR, angR, 150, seed=8)

TREE_EUC_EDGES = build_radial_tree(XS, SAMP_E)
TREE_CARM_EDGES = build_radial_tree(XS, SAMP_R)

# ════════════════════════════════════════════════════════════════
#  CARM metric field""", new_text, flags=re.DOTALL)


with open('generate_abstract_figure.py', 'w') as f:
    f.write(new_text)

