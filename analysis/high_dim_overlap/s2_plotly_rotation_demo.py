#!/usr/bin/env python3
import argparse, math, os
import numpy as np
import pandas as pd

def normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    return X / norms

def sample_cauchy_projected(mu, gamma, n, rng):
    S = rng.standard_cauchy(size=(n,3)) * gamma
    Y = mu.reshape(1,3) + S
    return normalize_rows(Y)

def rotate_z(theta_rad):
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]])

def sample_uniform_s2(n, rng):
    X = rng.normal(size=(n,3))
    return normalize_rows(X)

def Z_beta_s2(beta):
    return 1.0 if beta==0 else math.sinh(beta)/beta

def kde_density_at(X_ref, X_eval, beta, Zb):
    dots = X_eval @ X_ref.T
    return np.mean(np.exp(beta * dots) / Zb, axis=1)

def angular_coverage_overlap(A, B, theta_rad):
    c = math.cos(theta_rad)
    AB = A @ B.T
    cov_A_given_B = np.mean((AB >= c).any(axis=1))
    cov_B_given_A = np.mean((AB.T >= c).any(axis=1))
    return float(0.5*(cov_A_given_B+cov_B_given_A)), float(cov_A_given_B), float(cov_B_given_A)

def spherical_fibonacci_points(N):
    i = np.arange(N)
    phi = (np.pi * (3.0 - np.sqrt(5.0))) * i
    z = 1.0 - (2.0*(i + 0.5) / N)
    r = np.sqrt(np.maximum(0.0, 1.0 - z*z))
    x = r * np.cos(phi); y = r * np.sin(phi)
    return normalize_rows(np.vstack([x,y,z]).T)

def healpix_bin_ids(X, nside=32):
    try:
        import healpy as hp
        theta, phi = hp.vec2ang(X.T)
        pix = hp.ang2pix(nside, theta, phi, nest=False)
        return pix.astype(int), f"HEALPix nside={nside}"
    except Exception:
        Npix = int(12 * nside * nside)
        grid = spherical_fibonacci_points(Npix)
        idx = np.argmax(X @ grid.T, axis=1)
        return idx.astype(int), f"Fibonacci fallback (≈HEALPix {Npix} cells)"

def occupancy_jaccard(idsA, idsB):
    sA = set(idsA.tolist()); sB = set(idsB.tolist())
    inter = len(sA & sB); uni = len(sA | sB)
    return (inter / uni if uni>0 else 0.0), inter, len(sA), len(sB), uni

def hausdorff_geodesic_from_AB(AB):
    max_over_B = AB.max(axis=1)
    max_over_A = AB.max(axis=0)
    d_a = np.arccos(np.clip(max_over_B, -1.0, 1.0))
    d_b = np.arccos(np.clip(max_over_A, -1.0, 1.0))
    return float(max(d_a.max(), d_b.max()))

def wrap_0_2pi(a): return np.mod(a, 2*np.pi)
def min_covering_arc(angles, alpha=0.9):
    n = angles.size; k = max(1, int(math.ceil(alpha*n)))
    arr = np.sort(wrap_0_2pi(angles)); arr2 = np.concatenate([arr, arr+2*np.pi])
    spans = arr2[k-1:n+k-1] - arr
    j = np.argmin(spans)
    return float(arr[j]), float(arr2[j+k-1]), float(spans[j])
def arc_intersection_length(a1,b1,a2,b2):
    def split(a,b):
        if b<=2*np.pi: return [(a,b)]
        else: return [(a,2*np.pi),(0.0,b-2*np.pi)]
    inter=0.0
    for x1,y1 in split(a1,b1):
        for x2,y2 in split(a2,b2):
            lo,hi = max(x1,x2), min(y1,y2)
            if hi>lo: inter += (hi-lo)
    return float(inter)
def chord_overlap_on_common_circle(A, B, alpha=0.9):
    muA = A.mean(axis=0); muB = B.mean(axis=0)
    if np.linalg.norm(muA)<1e-12 or np.linalg.norm(muB)<1e-12:
        return 0.0, 0.0, 0.0
    muA = muA/np.linalg.norm(muA); muB = muB/np.linalg.norm(muB)
    n = np.cross(muA, muB)
    if np.linalg.norm(n)<1e-9:
        tmp = np.array([0.0,0.0,1.0]); 
        if abs(muA@tmp)>0.9: tmp = np.array([1.0,0.0,0.0])
        n = np.cross(muA, tmp)
    n = n/np.linalg.norm(n)
    e1 = muA - (muA@n)*n
    if np.linalg.norm(e1)<1e-12: e1 = muB - (muB@n)*n
    e1 = e1/np.linalg.norm(e1); e2 = np.cross(n, e1)
    angA = np.arctan2(A@e2, A@e1); angB = np.arctan2(B@e2, B@e1)
    a1,b1,w1 = min_covering_arc(angA, alpha); a2,b2,w2 = min_covering_arc(angB, alpha)
    inter = arc_intersection_length(a1,b1,a2,b2); union = w1 + w2 - inter if w1 + w2 - inter > 1e-12 else 1e-12
    return float(inter/union), float(w1), float(w2)

def main():
    ap = argparse.ArgumentParser(description="S^2 Plotly demo with extended metrics")
    ap.add_argument("--mode", choices=["gen","pt"], default="gen")
    ap.add_argument("--pt", type=str, default=None, help="PyTorch .pt/.pth with {'A':tensor,'B':tensor} or tuple/list")
    ap.add_argument("--nA", type=int, default=1000)
    ap.add_argument("--nB", type=int, default=1000)
    ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=12.0)
    ap.add_argument("--theta-deg", type=float, default=60.0)
    ap.add_argument("--fib-N", type=int, default=1200, help="Fibonacci grid size for bin overlap")
    ap.add_argument("--healpix-nside", type=int, default=32)
    ap.add_argument("--alpha-chord", type=float, default=0.9, help="Chord coverage fraction (e.g., 0.9 for 90%)")
    ap.add_argument("--angle-step", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-html", type=str, default="s2_overlap_plotly.html")
    ap.add_argument("--out-csv", type=str, default="s2_rotation_metrics.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.mode=="pt":
        try:
            import torch
        except Exception as e:
            raise RuntimeError("PyTorch required for --mode pt.") from e
        obj = torch.load(args.pt, map_location="cpu")
        if isinstance(obj, dict) and "A" in obj and "B" in obj:
            A = obj["A"].cpu().numpy(); B0 = obj["B"].cpu().numpy()
        elif isinstance(obj, (list,tuple)) and len(obj)==2:
            A = obj[0].cpu().numpy(); B0 = obj[1].cpu().numpy()
        else:
            raise ValueError("Unsupported .pt structure.")
        A = normalize_rows(A); B0 = normalize_rows(B0)
        if A.shape[1]!=3 or B0.shape[1]!=3:
            raise ValueError("Data must have shape (n,3) for S^2 demo.")
    else:
        mu_x = np.array([1.0,0.0,0.0])
        A = sample_cauchy_projected(mu_x, args.gamma, args.nA, rng)
        B0 = sample_cauchy_projected(mu_x, args.gamma, args.nB, rng)

    beta = args.beta; Zb = Z_beta_s2(beta); theta_rad = math.radians(args.theta_deg)
    angles_deg = list(range(0, 181, max(1,int(args.angle_step))))

    X_eval = sample_uniform_s2(2000, rng); pA = kde_density_at(A, X_eval, beta, Zb)
    AA = A @ A.T; np.fill_diagonal(AA, -np.inf)
    kAA = np.sum(np.exp(beta*AA)[AA>-np.inf])/(A.shape[0]*(A.shape[0]-1))
    BB = B0 @ B0.T; np.fill_diagonal(BB, -np.inf)
    kBB = np.sum(np.exp(beta*BB)[BB>-np.inf])/(B0.shape[0]*(B0.shape[0]-1))

    fib_grid = spherical_fibonacci_points(args.fib_N)
    fib_ids_A = np.argmax(A @ fib_grid.T, axis=1)

    healpix_ids_A, hp_label = healpix_bin_ids(A, args.healpix_nside)

    BC_vals=[]; MMD2_vals=[]; OVL_vals=[]; angle_means=[]; cos_means=[]; hausdorff_deg=[]
    fib_jaccard=[]; hp_jaccard=[]; chord_overlap_vals=[]
    for ang in angles_deg:
        R = rotate_z(math.radians(ang)); Bphi = B0 @ R.T
        qB = kde_density_at(Bphi, X_eval, beta, Zb); BC_vals.append(float(np.mean(np.sqrt(pA*qB))))
        AB = A @ Bphi.T; kAB = np.mean(np.exp(beta*AB)); MMD2_vals.append(float((kAA + kBB - 2*kAB)/Zb))
        OVL,_ ,_ = angular_coverage_overlap(A, Bphi, theta_rad); OVL_vals.append(OVL)
        muA = A.mean(axis=0); muB = Bphi.mean(axis=0); muA/=np.linalg.norm(muA); muB/=np.linalg.norm(muB)
        c = float(np.clip(muA@muB, -1.0, 1.0)); cos_means.append(c); angle_means.append(math.degrees(math.acos(c)))
        h = hausdorff_geodesic_from_AB(AB); hausdorff_deg.append(math.degrees(h))
        fib_ids_B = np.argmax(Bphi @ fib_grid.T, axis=1)
        # Jaccard on fib bins
        sA = set(fib_ids_A.tolist()); sB = set(fib_ids_B.tolist())
        inter = len(sA & sB); uni = len(sA | sB); fib_jaccard.append(inter/uni if uni>0 else 0.0)
        # HEALPix or fallback
        hp_ids_B, _ = healpix_bin_ids(Bphi, args.healpix_nside)
        sAh = set(healpix_ids_A.tolist()); sBh = set(hp_ids_B.tolist())
        interh = len(sAh & sBh); unih = len(sAh | sBh); hp_jaccard.append(interh/unih if unih>0 else 0.0)
        ovl_arc, _, _ = chord_overlap_on_common_circle(A, Bphi, args.alpha_chord); chord_overlap_vals.append(ovl_arc)

    df = pd.DataFrame({
        "angle_deg": angles_deg,
        "BC": BC_vals,
        "MMD2_norm": MMD2_vals,
        f"OVL_theta_{args.theta_deg}deg": OVL_vals,
        "mean_angle_deg": angle_means,
        "mean_cosine": cos_means,
        "hausdorff_deg": hausdorff_deg,
        f"fib_Jaccard_N{args.fib_N}": fib_jaccard,
        f"hp_Jaccard_{hp_label}": hp_jaccard,
        f"chord_overlap_alpha{int(100*args.alpha_chord)}pct": chord_overlap_vals,
    })
    df.to_csv(args.out_csv, index=False)

    # Plotly figure (same layout as notebook version)
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from plotly.offline import plot as plotly_save
    except Exception:
        print("Plotly not available; saved CSV only:", args.out_csv)
        return

    # Sphere
    u = np.linspace(0,2*np.pi,60); v = np.linspace(0,np.pi,30); uu,vv = np.meshgrid(u,v)
    xs = np.cos(uu)*np.sin(vv); ys = np.sin(uu)*np.sin(vv); zs = np.cos(vv)

    # Subsample for 3D
    idxA = np.random.choice(A.shape[0], size=min(800,A.shape[0]), replace=False)
    idxB = np.random.choice(B0.shape[0], size=min(800,B0.shape[0]), replace=False)
    A_sub = A[idxA]; B0_sub = B0[idxB]; B_init = (B0_sub @ rotate_z(math.radians(angles_deg[0])).T)

    fig = make_subplots(
        rows=4, cols=3,
        specs=[[{"type":"scene","colspan":3}, None, None],
               [{"type":"xy"}, {"type":"xy"}, {"type":"xy"}],
               [{"type":"xy"}, {"type":"xy"}, {"type":"xy"}],
               [{"type":"xy"}, {"type":"xy"}, {"type":"xy"}]],
        subplot_titles=("S² point clouds (A fixed, B rotated by φ about Z)",
                        "Bhattacharyya coefficient (KDE)",
                        "Kernel MMD² (normalized)",
                        f"Angular coverage overlap θ={args.theta_deg:.0f}°",
                        "Angle between mean directions (deg)",
                        "Cosine similarity of means",
                        "Hausdorff distance (deg, geodesic)",
                        f"Fibonacci bins Jaccard (N={args.fib_N})",
                        f"{hp_label} Jaccard",
                        f"Chord overlap on common great circle (α={int(100*args.alpha_chord)}%)",
                        "", ""))
    fig.update_layout(title={"text":"<b>Overlap on S² — Cauchy-around-μx (projected), B rotated about Z</b>","x":0.5,"xanchor":"center"},
                      title_font_size=20, height=1100, scene=dict(aspectmode="data"),
                      margin=dict(l=10,r=10,t=60,b=10))
    fig.add_trace(go.Surface(x=xs,y=ys,z=zs,opacity=0.2,showscale=False), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=A_sub[:,0], y=A_sub[:,1], z=A_sub[:,2], mode="markers", name="A", marker=dict(size=3)), row=1,col=1)
    fig.add_trace(go.Scatter3d(x=B_init[:,0], y=B_init[:,1], z=B_init[:,2], mode="markers", name="B (rotated)", marker=dict(size=3)), row=1,col=1)

    # Metric lines
    fig.add_trace(go.Scatter(x=angles_deg, y=BC_vals, name="BC (KDE)", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=angles_deg, y=MMD2_vals, name="MMD²", mode="lines"), row=2, col=2)
    fig.add_trace(go.Scatter(x=angles_deg, y=OVL_vals, name="OVL", mode="lines"), row=2, col=3)

    fig.add_trace(go.Scatter(x=angles_deg, y=angle_means, name="Angle(means)", mode="lines"), row=3, col=1)
    fig.add_trace(go.Scatter(x=angles_deg, y=cos_means, name="Cosine(means)", mode="lines"), row=3, col=2)
    fig.add_trace(go.Scatter(x=angles_deg, y=hausdorff_deg, name="Hausdorff(deg)", mode="lines"), row=3, col=3)

    fig.add_trace(go.Scatter(x=angles_deg, y=fib_jaccard, name="Fib Jaccard", mode="lines"), row=4, col=1)
    fig.add_trace(go.Scatter(x=angles_deg, y=hp_jaccard, name="HEALPix Jaccard", mode="lines"), row=4, col=2)
    fig.add_trace(go.Scatter(x=angles_deg, y=chord_overlap_vals, name="Chord overlap", mode="lines"), row=4, col=3)

    # Vertical markers (update via slider frames)
    y1=(min(BC_vals), max(BC_vals)); y2=(min(MMD2_vals), max(MMD2_vals)); y3=(0.0,1.0)
    y4=(min(angle_means), max(angle_means)); y5=(-1.0,1.0); y6=(min(hausdorff_deg), max(hausdorff_deg))
    y7=(0.0,1.0); y8=(0.0,1.0); y9=(0.0,1.0)
    def vline(x0,y): return go.Scatter(x=[x0,x0], y=list(y), mode="lines", showlegend=False, line=dict(dash="dash"))
    fig.add_trace(vline(angles_deg[0],y1), row=2,col=1); fig.add_trace(vline(angles_deg[0],y2), row=2,col=2); fig.add_trace(vline(angles_deg[0],y3), row=2,col=3)
    fig.add_trace(vline(angles_deg[0],y4), row=3,col=1); fig.add_trace(vline(angles_deg[0],y5), row=3,col=2); fig.add_trace(vline(angles_deg[0],y6), row=3,col=3)
    fig.add_trace(vline(angles_deg[0],y7), row=4,col=1); fig.add_trace(vline(angles_deg[0],y8), row=4,col=2); fig.add_trace(vline(angles_deg[0],y9), row=4,col=3)

    # Frames
    frames = []
    for ang in angles_deg:
        Bphi_sub = (B0_sub @ rotate_z(math.radians(ang)).T)
        frames.append(go.Frame(name=str(ang),
            data=[go.Surface(), go.Scatter3d(), go.Scatter3d(x=Bphi_sub[:,0], y=Bphi_sub[:,1], z=Bphi_sub[:,2]),
                  go.Scatter(), go.Scatter(), go.Scatter(),
                  go.Scatter(), go.Scatter(), go.Scatter(),
                  go.Scatter(), go.Scatter(), go.Scatter(),
                  go.Scatter(x=[ang,ang], y=list(y1)), go.Scatter(x=[ang,ang], y=list(y2)), go.Scatter(x=[ang,ang], y=list(y3)),
                  go.Scatter(x=[ang,ang], y=list(y4)), go.Scatter(x=[ang,ang], y=list(y5)), go.Scatter(x=[ang,ang], y=list(y6)),
                  go.Scatter(x=[ang,ang], y=list(y7)), go.Scatter(x=[ang,ang], y=list(y8)), go.Scatter(x=[ang,ang], y=list(y9))]))
    fig.frames = frames

    steps = []
    for ang in angles_deg:
        steps.append(dict(method="animate", args=[[str(ang)], {"mode":"immediate","frame":{"duration":0,"redraw":True},
                                                               "transition":{"duration":0}}], label=f"{ang}°"))
    sliders = [dict(active=0, currentvalue={"prefix":"Rotation φ = "}, pad={"t":30}, steps=steps)]
    fig.update_layout(sliders=sliders, legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5))

    from plotly.offline import plot as plotly_save
    plotly_save(fig, filename=args.out_html, auto_open=False)
    print("Wrote:", args.out_html)
    print("Wrote:", args.out_csv)

if __name__ == "__main__":
    main()
