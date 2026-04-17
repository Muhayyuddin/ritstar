with open('generate_abstract_figure.py', 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if skip:
        if "alpha=0.3, zorder=0.5" in line:
            skip = False
        continue
    
    if "# Ghost Euclidean ellipse" in line:
        new_lines.append("# Ghost Euclidean ellipse (matching A's background color)\n")
        new_lines.append("ax_rit.add_patch(Ellipse(cen_E, wE, hE, angle=angE,\n")
        new_lines.append("                           fc=C['euc_fill'], ec=C['euc'],\n")
        new_lines.append("                           lw=1.5, ls='--', alpha=0.5, zorder=0.5))\n\n")
        new_lines.append("# Plot 'wasted' Euclidean samples that fall outside the Riemannian ellipse\n")
        new_lines.append("for p in SAMP_E:\n")
        new_lines.append("    if not in_ellipse(p, cen_R, wR, hR, angR):\n")
        new_lines.append("        ax_rit.plot(p[0], p[1], 'o', color=C['wasted'], ms=3.0, alpha=0.55, zorder=0.55, mec='none')\n")
        skip = True
        continue
        
    new_lines.append(line)

with open('generate_abstract_figure.py', 'w') as f:
    f.writelines(new_lines)
