import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

# Theo (artifact capture)
try:
    import theo  # type: ignore
    THEO_AVAILABLE = True
except Exception:
    THEO_AVAILABLE = False

# -----------------------
# Basic 2x2 analytic tools
# -----------------------
I2 = np.eye(2, dtype=complex)

sigmax = np.array([[0, 1], [1, 0]], dtype=complex)
sigmay = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigmaz = np.array([[1, 0], [0, -1]], dtype=complex)

psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j])
plus = np.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=complex) / np.sqrt(2.0)


def exp_pauli_combo(a: float, b: float, c: float, t: float) -> np.ndarray:
    """Analytic exp(-i (a*sx + b*sy + c*sz) t) for 2x2 Pauli combinations."""
    a = float(a)
    b = float(b)
    c = float(c)
    t = float(t)
    r = np.sqrt(a * a + b * b + c * c)
    if r < 1e-15 or abs(t) < 1e-15:
        return I2.copy()
    H = a * sigmax + b * sigmay + c * sigmaz
    return np.cos(r * t) * I2 - 1j * np.sin(r * t) * (H / r)


def J_from_U(U: np.ndarray) -> float:
    psi_final = U @ psi0
    inner = np.vdot(plus, psi_final)
    return float(np.abs(inner))


def compute_J_scan(params: np.ndarray, U_builder):
    params = np.asarray(params, dtype=float)
    J = np.empty_like(params, dtype=float)
    for i, x in enumerate(params):
        U = U_builder(float(x))
        J[i] = J_from_U(U)
    return J


def find_local_maxima(x_grid: np.ndarray, y_grid: np.ndarray):
    x = np.asarray(x_grid, dtype=float)
    y = np.asarray(y_grid, dtype=float)
    n = len(y)
    if n < 3:
        return []
    idx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    return [(int(i), float(x[i]), float(y[i])) for i in idx]


def refine_peak_quadratic(x_grid: np.ndarray, y_grid: np.ndarray, idx: int):
    """Quadratic interpolation using points idx-1, idx, idx+1."""
    x = np.asarray(x_grid, dtype=float)
    y = np.asarray(y_grid, dtype=float)
    n = len(x)
    if idx <= 0 or idx >= n - 1:
        return float(x[idx]), float(y[idx])
    xs = np.array([x[idx - 1], x[idx], x[idx + 1]], dtype=float)
    ys = np.array([y[idx - 1], y[idx], y[idx + 1]], dtype=float)
    try:
        a, b, c = np.polyfit(xs, ys, 2)
        if abs(a) < 1e-18:
            return float(x[idx]), float(y[idx])
        x_peak = -b / (2.0 * a)
        y_peak = a * x_peak * x_peak + b * x_peak + c
        return float(x_peak), float(y_peak)
    except Exception:
        return float(x[idx]), float(y[idx])


def top_peaks(refined_peaks, k=3, min_separation=1e-6):
    """Pick top peaks by height with simple dedup by x proximity."""
    candidates = sorted(refined_peaks, key=lambda t: t[1], reverse=True)
    chosen = []
    for xp, yp in candidates:
        if all(abs(xp - xc) > min_separation for xc, _ in chosen):
            chosen.append((xp, yp))
        if len(chosen) >= k:
            break
    return chosen


# -----------------------
# Parameters for the three examples
# -----------------------
d = 1.0

T1 = 2.0  # Example 1 time (fixed)
T2 = np.pi / (2.0 * np.sqrt(2.0) * d)  # Example 2 time (fixed)

TAU1 = 1.0
TAU2 = 2.0

# -----------------------
# Example 1: H = u*sigma_y + w*sigma_z, fixed T = 2
# -----------------------
u_grid = np.linspace(0.0, 10.0, 4000)


def U_example1(u: float, w: float) -> np.ndarray:
    # Hamiltonian coefficients: (0, u, w)
    return exp_pauli_combo(0.0, u, w, T1)


J_u_w1 = compute_J_scan(u_grid, lambda u: U_example1(u, 1.0))
J_u_w0 = compute_J_scan(u_grid, lambda u: U_example1(u, 0.0))

# Leading maxima for w=1 curve
maxima_u_w1 = find_local_maxima(u_grid, J_u_w1)
refined_u_w1 = []
for idx, _, _ in maxima_u_w1:
    xp, yp = refine_peak_quadratic(u_grid, J_u_w1, idx)
    refined_u_w1.append((xp, yp))

near1_threshold_u = 0.995
u_near1 = [(xp, yp) for xp, yp in refined_u_w1 if yp >= near1_threshold_u]
leading_u_w1 = top_peaks(refined_u_w1, k=3, min_separation=0.08)


# -----------------------
# Example 2: H = d*sigma_z + v*sigma_x, fixed T_x
# -----------------------
v_grid = np.linspace(0.0, 10.0, 3000)


def U_example2(v: float) -> np.ndarray:
    # Hamiltonian coefficients: (v, 0, d)
    return exp_pauli_combo(v, 0.0, d, T2)


J_v = compute_J_scan(v_grid, U_example2)

v_hit = d
J_hit = J_from_U(U_example2(v_hit))

# Also find local maxima to show trend
maxima_v = find_local_maxima(v_grid, J_v)
refined_v = []
for idx, _, _ in maxima_v:
    xp, yp = refine_peak_quadratic(v_grid, J_v, idx)
    refined_v.append((xp, yp))
leading_v = top_peaks(refined_v + [(v_hit, J_hit)], k=3, min_separation=0.08)


# -----------------------
# Example 3: two-segment one-parameter model
# U(lambda)=exp[-i(d*sigma_z+lambda*sigma_x)*tau2] exp[-i(d*sigma_z+lambda*sigma_y)*tau1]
# -----------------------
lam_grid = np.linspace(0.0, 15.0, 4500)


def U_example3(lam: float) -> np.ndarray:
    # Segment 1: (0, lam, d) for tau1
    U1 = exp_pauli_combo(0.0, lam, d, TAU1)
    # Segment 2: (lam, 0, d) for tau2
    U2 = exp_pauli_combo(lam, 0.0, d, TAU2)
    return U2 @ U1


J_lam = compute_J_scan(lam_grid, U_example3)

maxima_lam = find_local_maxima(lam_grid, J_lam)
refined_lam = []
for idx, _, _ in maxima_lam:
    xp, yp = refine_peak_quadratic(lam_grid, J_lam, idx)
    refined_lam.append((xp, yp))

near1_threshold = 0.999
near1_lam = [(xp, yp) for xp, yp in refined_lam if yp >= near1_threshold]
leading_lam = top_peaks(refined_lam, k=5, min_separation=0.08)

# For plotting: leading near-1 peaks (prefer the earliest if tied)
near1_lam_sorted = sorted(near1_lam, key=lambda t: t[0])
near1_to_mark = near1_lam_sorted[:4]
if len(near1_to_mark) < 2:
    near1_to_mark = top_peaks(refined_lam, k=3, min_separation=0.08)


# -----------------------
# Plot style (preserve presentation choices)
# -----------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "mathtext.fontset": "dejavusans",
})


def save_figure_svgpng(fig, name_base: str):
    """Save SVG only in Theo; save both SVG+PNG locally (PNG local-only)."""
    svg_buf = io.StringIO()
    fig.savefig(svg_buf, format="svg", bbox_inches="tight")
    svg_text = svg_buf.getvalue()

    if THEO_AVAILABLE:
        theo.save_artifact(name_base + ".svg", svg_text)
    else:
        fig.savefig(name_base + ".svg", format="svg", bbox_inches="tight")
        fig.savefig(name_base + ".png", format="png", bbox_inches="tight", dpi=300)


def _bboxes_intersect(bb1, bb2, margin_px=0):
    return not (
        bb1.x1 + margin_px < bb2.x0
        or bb1.x0 - margin_px > bb2.x1
        or bb1.y1 + margin_px < bb2.y0
        or bb1.y0 - margin_px > bb2.y1
    )


def place_legend_no_overlap(
    ax,
    fig,
    avoid_artists,
    loc_candidates=("lower right", "upper right", "upper left", "lower left"),
    framealpha=0.9,
):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    avoid_bbs = []
    for art in avoid_artists:
        if art is None:
            continue
        try:
            bb = art.get_window_extent(renderer=renderer)
            avoid_bbs.append(bb)
        except Exception:
            pass

    best = None
    for loc in loc_candidates:
        leg = ax.legend(loc=loc, framealpha=framealpha)
        fig.canvas.draw()
        leg_bb = leg.get_window_extent(renderer=renderer)
        overlaps = any(_bboxes_intersect(leg_bb, bb, margin_px=4) for bb in avoid_bbs)
        if not overlaps:
            best = leg
            break
        leg.remove()

    if best is None:
        best = ax.legend(loc=loc_candidates[0], framealpha=framealpha)
    return best


def annotate_peak(
    ax,
    xp,
    yp,
    text,
    i,
    high_peak_threshold=0.97,
    fontsize=8.5,
    bbox_pad=0.22,
):
    # Stagger offsets to avoid annotation stacking and to steer labels away from high peaks.
    # Use different offsets per label index, with sign flipped for near-top peaks.
    offset_sets = [
        (14, 18),  # upper-right
        (14, -28),  # lower-right
        (-90, 18),  # upper-left
        (-90, -28),  # lower-left
        (40, 18),  # far-right
        (40, -28),  # far-right low
    ]

    dx, dy = offset_sets[i % len(offset_sets)]
    if yp >= high_peak_threshold:
        # If peak is near the top, prefer to place text below.
        dy = -abs(dy)

    ann = ax.annotate(
        text,
        xy=(xp, yp),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left",
        va="bottom" if dy >= 0 else "top",
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=%.2f" % bbox_pad, fc="white", ec="0.6", alpha=0.9),
        zorder=6,
    )
    return ann


def style_axes(fig, ax, top_margin=0.80):
    # Presentation-only spacing: avoid constrained_layout; reserve title space.
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=top_margin)
    ax.grid(True)


# -----------------------
# Example 1 figure (separate)
# -----------------------
fig1, ax1 = plt.subplots(1, 1, figsize=(6.2, 4.8), constrained_layout=False)
style_axes(fig1, ax1, top_margin=0.83)

ax1.plot(u_grid, J_u_w1, lw=2.2, color="C0", label=r'$w=1$')
ax1.plot(u_grid, J_u_w0, lw=1.7, color="C0", ls='--', alpha=0.85, label=r'$w=0$')
ax1.set_title("Example 1: constant-H drift", pad=10)
ax1.set_xlabel(r'$u$')
ax1.set_ylabel(r'$J = |\langle +|\psi(T)\rangle|$')
ax1.set_xlim(0.0, 10.0)
ax1.set_ylim(0.0, 1.05)

avoid_for_legend = []
peak_texts = []
for i, (xp, yp) in enumerate(leading_u_w1, start=1):
    ax1.plot([xp], [yp], marker="*", ms=11, color="crimson", zorder=6)
    ann = annotate_peak(
        ax1,
        xp,
        yp,
        f"peak {{{i}}}\nu\u2248{{{xp:.3f}}}\nJ\u2248{{{yp:.3f}}}",
        i=i - 1,
        high_peak_threshold=0.98,
        fontsize=8.2,
        bbox_pad=0.20,
    )
    peak_texts.append(ann)
    avoid_for_legend.append(ann)

near1_text = None
if len(u_near1) > 0:
    ax1.axhline(near1_threshold_u, color="gray", lw=1.0, ls=":", alpha=0.8)
    # Move the label away from the title region.
    near1_text = ax1.text(
        0.02,
        0.14,
        "w=1 near-1",
        transform=ax1.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.6", alpha=0.9),
        zorder=7,
    )
    avoid_for_legend.append(near1_text)

info1 = f"d={d:g}, T={T1:g}"
info_text1 = ax1.text(
    0.02,
    0.02,
    info1,
    transform=ax1.transAxes,
    ha="left",
    va="bottom",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.65", alpha=0.92),
    zorder=7,
)
avoid_for_legend.append(info_text1)

# Legend placement that avoids annotations.
place_legend_no_overlap(
    ax1,
    fig1,
    avoid_artists=avoid_for_legend,
    loc_candidates=("lower right", "upper right", "upper left", "lower left"),
    framealpha=0.9,
)

base1 = "Example1_SigmaY_sigmaZ_fixedT"
save_figure_svgpng(fig1, base1)
plt.close(fig1)


# -----------------------
# Example 2 figure (separate)
# -----------------------
fig2, ax2 = plt.subplots(1, 1, figsize=(6.2, 4.8), constrained_layout=False)
style_axes(fig2, ax2, top_margin=0.83)

ax2.plot(v_grid, J_v, lw=2.0, color="C1", label=r'$J(v)$')
ax2.set_title("Example 2: constant-H (\u03c3x)", pad=10)
ax2.set_xlabel(r'$v$')
ax2.set_ylabel(r'$J = |\langle +|\psi(T_x)\rangle|$')
ax2.set_xlim(0.0, 10.0)
ax2.set_ylim(0.0, 1.05)

avoid_for_legend = []

ax2.plot([v_hit], [J_hit], marker="s", ms=10, color="black", zorder=6)
# Reposition exact-hit annotation to avoid crowding near title/curve peak.
dy = -28 if J_hit >= 0.97 else 16
exact_ann = ax2.annotate(
    f"Exact hit\nv=d={v_hit:g}\nJ\u2248{J_hit:.6f}",
    xy=(v_hit, J_hit),
    xytext=(12, dy),
    textcoords="offset points",
    ha="left",
    va="top" if dy < 0 else "bottom",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.9),
    zorder=7,
)
avoid_for_legend.append(exact_ann)

marked = []
for j, (xp, yp) in enumerate(leading_v):
    if abs(xp - v_hit) < 1e-6:
        continue
    if any(abs(xp - m) < 1e-3 for m in marked):
        continue
    marked.append(xp)
    ax2.plot([xp], [yp], marker="*", ms=9, color="darkorange", zorder=5)
    ann = annotate_peak(
        ax2,
        xp,
        yp,
        f"peak\nv\u2248{{{xp:.3f}}}\nJ\u2248{{{yp:.3f}}}",
        i=(j + 1) % 4,
        high_peak_threshold=0.98,
        fontsize=8.3,
        bbox_pad=0.20,
    )
    avoid_for_legend.append(ann)

info2 = f"d={d:g}, Tx={T2:.6f}"
info_text2 = ax2.text(
    0.02,
    0.02,
    info2,
    transform=ax2.transAxes,
    ha="left",
    va="bottom",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.65", alpha=0.92),
    zorder=7,
)
avoid_for_legend.append(info_text2)

place_legend_no_overlap(
    ax2,
    fig2,
    avoid_artists=avoid_for_legend,
    loc_candidates=("lower right", "upper right", "upper left", "lower left"),
    framealpha=0.9,
)

base2 = "Example2_SigmaX_exact_hit"
save_figure_svgpng(fig2, base2)
plt.close(fig2)


# -----------------------
# Example 3 figure (separate)
# -----------------------
fig3, ax3 = plt.subplots(1, 1, figsize=(6.2, 4.8), constrained_layout=False)
style_axes(fig3, ax3, top_margin=0.83)

ax3.plot(lam_grid, J_lam, lw=2.0, color="C2", label=r'$J(\lambda)$')
ax3.set_title("Example 3: two-segment \u03bb model", pad=10)
ax3.set_xlabel(r'$\lambda$')
ax3.set_ylabel(r'$J = |\langle +|\psi_{{\mathrm{{final}}}}(\lambda)\rangle|$')
ax3.set_xlim(0.0, 15.0)
ax3.set_ylim(0.0, 1.05)
ax3.axhline(near1_threshold, color="gray", lw=1.0, ls=":", alpha=0.8)

avoid_for_legend = []

for i, (xp, yp) in enumerate(near1_to_mark, start=1):
    ax3.plot([xp], [yp], marker="*", ms=12, color="crimson", zorder=6)
    # Use distinct offsets per label so they don't stack.
    ann = annotate_peak(
        ax3,
        xp,
        yp,
        f"near-1 peak {{{i}}}\n\u03bb\u2248{{{xp:.3f}}}\nJ\u2248{{{yp:.6f}}}",
        i=(i - 1),
        high_peak_threshold=0.995,
        fontsize=8.5,
        bbox_pad=0.22,
    )
    avoid_for_legend.append(ann)

info3 = f"d={d:g}, tau1={TAU1:g}, tau2={TAU2:g}"
info_text3 = ax3.text(
    0.02,
    0.02,
    info3,
    transform=ax3.transAxes,
    ha="left",
    va="bottom",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.65", alpha=0.92),
    zorder=7,
)
avoid_for_legend.append(info_text3)

place_legend_no_overlap(
    ax3,
    fig3,
    avoid_artists=avoid_for_legend,
    loc_candidates=("lower right", "upper right", "upper left", "lower left"),
    framealpha=0.9,
)

base3 = "Example3_TwoSegment_Lambda_landscape"
save_figure_svgpng(fig3, base3)
plt.close(fig3)


# -----------------------
# Print compact textual summary
# -----------------------
print("=== Three-Example Summary ===")

if leading_u_w1:
    print(
        "Example 1 (w=1) leading maxima:",
        ", ".join(
            [
                f"u\u2248{x:.3f} (J\u2248{y:.3f})"
                for x, y in leading_u_w1
            ]
        ),
        "; w=0 comparison curve overlaid on same axes.",
    )
else:
    print("Example 1 (w=1): no local maxima found on grid; w=0 curve included for comparison.")

print(f"Example 2 exact hit: v=d={v_hit:g} gives J\u2248{J_hit:.6f}")

if near1_lam_sorted:
    show = near1_lam_sorted[:4]
    print(
        "Example 3 leading near-1 peaks (J>=%.3f):" % near1_threshold,
        ", ".join([f"\u03bb\u2248{x:.3f} (J\u2248{y:.6f})" for x, y in show]),
    )
else:
    show = leading_lam[:3]
    print(
        "Example 3: no peaks above J>=%.3f found on grid; top peaks:" % near1_threshold,
        ", ".join([f"\u03bb\u2248{x:.3f} (J\u2248{y:.6f})" for x, y in show]),
    )