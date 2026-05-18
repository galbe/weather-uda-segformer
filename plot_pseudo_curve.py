import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

log_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/step1_gta2cs_b2_full/train.log"
out_path = log_path.replace("train.log", "pseudo_loss_curve.png")

seen = set()
iters, pseudos = [], []

with open(log_path) as f:
    for line in f:
        m = re.match(r'\[(\d+)/(\d+)\].*pseudo=([\d.]+)', line)
        if m:
            it, total = int(m.group(1)), int(m.group(2))
            ps = float(m.group(3))
            if it not in seen:
                seen.add(it)
                iters.append(it)
                pseudos.append(ps)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(iters, pseudos, linewidth=1.2, color='steelblue')
ax.set_xlabel('Training iteration', fontsize=12)
ax.set_ylabel('Pseudo-label loss $\\mathcal{L}_{\\mathrm{tgt}}$', fontsize=12)
ax.set_title('Pseudo-label loss vs. iteration (Step 1, B2 full data)', fontsize=13)
ax.set_xlim(0, total)
ax.axhline(0, color='gray', linewidth=0.7, linestyle='--')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}  ({len(iters)} points, up to iter {iters[-1]})")
