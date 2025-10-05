#!/usr/bin/env python3
import argparse, math, json, os
import numpy as np

def normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    return X / norms

def sample_cauchy_projected(mu, gamma, n, rng):
    S = rng.standard_cauchy(size=(n, 3)) * gamma
    Y = mu.reshape(1,3) + S
    return normalize_rows(Y)

def sample_uniform_s2(n, rng):
    X = rng.normal(size=(n,3))
    return normalize_rows(X)

def rotate_z(theta_rad):
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[ c, -s, 0.0],[ s,  c, 0.0],[0.0,0.0,1.0]])

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

def main():
    ap = argparse.ArgumentParser(description="S^2 Plotly demo: rotate B around Z; compute overlaps")
    ap.add_argument("--mode", choices=["gen","pt"], default="gen", help="Generate or load base clouds")
    ap.add_argument("--pt", type=str, default=None, help="PyTorch .pt/.pth file with {'A':tensor,'B':tensor} or list/tuple")
    ap.add_argument("--nA", type=int, default=1000)
    ap.add_argument("--nB", type=int, default=1000)
    ap.add_argument("--gamma", type=float, default=0.7, help="Cauchy scale for generator")
    ap.add_argument("--beta", type=float, default=12.0, help="Kernel concentration for KDE/MMD")
    ap.add_argument("--theta-deg", type=float, default=20.0, help="Angular threshold for coverage")
    ap.add_argument("--angle-step", type=int, default=10, help="Degrees between slider steps (0..180)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-html", type=str, default="s2_overlap_plotly.html")
    ap.add_argument("--out-csv", type=str, default="s2_rotation_metrics.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.mode == "pt":
        try:
            import torch
        except Exception as e:
            raise RuntimeError("PyTorch is required for --mode pt.") from e
        obj = torch.load(args.pt, map_location="cpu")
        if isinstance(obj, dict) and "A" in obj and "B" in obj:
            A = obj["A"].cpu().numpy()
            B0 = obj["B"].cpu().numpy()
        elif isinstance(obj, (list,tuple)) and len(obj)==2:
            A = obj[0].cpu().numpy(); B0 = obj[1].cpu().numpy()
        else:
            raise ValueError("Unsupported .pt structure. Use {'A':tensor,'B':tensor} or (A,B).")
        A = normalize_rows(A); B0 = normalize_rows(B0)
        if A.shape[1] != 3 or B0.shape[1] != 3:
            raise ValueError("Loaded data must be on S^2 (shape (n,3)).")
    else:
        mu_x = np.array([1.0, 0.0, 0.0])
        A = sample_cauchy_projected(mu_x, args.gamma, args.nA, rng)
        B0 = sample_cauchy_projected(mu_x, args.gamma, args.nB, rng)

    beta = args.beta
    Zb = 1.0 if beta==0 else math.sinh(beta)/beta
    theta_rad = math.radians(args.theta_deg)
    angles_deg = list(range(0, 181, max(1, int(args.angle_step))))

    # Precompute BC pA on shared eval grid
    X_eval = sample_uniform_s2(2000, rng)
    pA = kde_density_at(A, X_eval, beta, Zb)

    # Precompute kAA and kBB for MMD^2
    AA = A @ A.T; np.fill_diagonal(AA, -np.inf)
    kAA = np.sum(np.exp(beta*AA)[AA>-np.inf])/(A.shape[0]*(A.shape[0]-1))
    BB = B0 @ B0.T; np.fill_diagonal(BB, -np.inf)
    kBB = np.sum(np.exp(beta*BB)[BB>-np.inf])/(B0.shape[0]*(B0.shape[0]-1))

    BC_vals=[]; MMD2_vals=[]; OVL_vals=[]; covAB_vals=[]; covBA_vals=[]
    for ang in angles_deg:
        R = rotate_z(math.radians(ang))
        Bphi = B0 @ R.T
        qB = kde_density_at(Bphi, X_eval, beta, Zb)
        BC_vals.append(float(np.mean(np.sqrt(pA*qB))))
        AB = A @ Bphi.T
        kAB = np.mean(np.exp(beta*AB))
        MMD2_vals.append(float((kAA + kBB - 2*kAB)/Zb))
        OVL, covAB, covBA = angular_coverage_overlap(A, Bphi, theta_rad)
        OVL_vals.append(OVL); covAB_vals.append(covAB); covBA_vals.append(covBA)

    import pandas as pd
    df = pd.DataFrame({
        "angle_deg": angles_deg,
        "BC": BC_vals,
        "MMD2_norm": MMD2_vals,
        f"OVL_theta_{args.theta_deg}deg": OVL_vals,
        f"Cov(A|B)_theta_{args.theta_deg}deg": covAB_vals,
        f"Cov(B|A)_theta_{args.theta_deg}deg": covBA_vals,
    })
    df.to_csv(args.out_csv, index=False)

    # Build Plotly fig
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from plotly.offline import plot as plotly_save
    except Exception as e:
        print("Plotly not available; wrote CSV:", args.out_csv)
        return

    # Sphere surface
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    uu, vv = np.meshgrid(u, v)
    xs = np.cos(uu)*np.sin(vv); ys = np.sin(uu)*np.sin(vv); zs = np.cos(vv)

    # Subsample for 3D scatter
    nA = A.shape[0]; nB = B0.shape[0]
    subA = min(1200, nA); subB = min(1200, nB)
    idxA = np.random.choice(nA, size=subA, replace=False)
    idxB = np.random.choice(nB, size=subB, replace=False)
    A_sub = A[idxA]; B0_sub = B0[idxB]

    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{"type":"scene","colspan":3}, None, None],
               [{"type":"xy"}, {"type":"xy"}, {"type":"xy"}]],
        subplot_titles=("S² point clouds (A fixed, B rotated by φ about Z)",
                        "Bhattacharyya coefficient (KDE)",
                        "Kernel MMD² (normalized)",
                        f"Angular coverage overlap θ={args.theta_deg:.0f}°")
    )

    sphere = go.Surface(x=xs, y=ys, z=zs, opacity=0.2, showscale=False)
    A_tr = go.Scatter3d(x=A_sub[:,0], y=A_sub[:,1], z=A_sub[:,2], mode='markers', name='A', marker=dict(size=3))
    B_init = (B0_sub @ rotate_z(math.radians(angles_deg[0])).T)
    B_tr = go.Scatter3d(x=B_init[:,0], y=B_init[:,1], z=B_init[:,2], mode='markers', name='B (rotated)', marker=dict(size=3))

    fig.add_trace(sphere, row=1, col=1)
    fig.add_trace(A_tr, row=1, col=1)
    fig.add_trace(B_tr, row=1, col=1)

    fig.add_trace(go.Scatter(x=angles_deg, y=BC_vals, name="BC (KDE)", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=angles_deg, y=MMD2_vals, name="MMD²", mode="lines"), row=2, col=2)
    fig.add_trace(go.Scatter(x=angles_deg, y=OVL_vals, name="OVL", mode="lines"), row=2, col=3)

    # Vertical lines
    y1min,y1max = float(min(BC_vals)), float(max(BC_vals))
    y2min,y2max = float(min(MMD2_vals)), float(max(MMD2_vals))
    y3min,y3max = float(min(OVL_vals)), float(max(OVL_vals))

    def vline(x0, ymin, ymax):
        return go.Scatter(x=[x0,x0], y=[ymin,ymax], mode="lines", showlegend=False, line=dict(dash="dash"))

    v1 = vline(angles_deg[0], y1min, y1max); fig.add_trace(v1, row=2, col=1)
    v2 = vline(angles_deg[0], y2min, y2max); fig.add_trace(v2, row=2, col=2)
    v3 = vline(angles_deg[0], y3min, y3max); fig.add_trace(v3, row=2, col=3)

    frames = []
    for ang in angles_deg:
        Bphi_sub = (B0_sub @ rotate_z(math.radians(ang)).T)
        frames.append(go.Frame(
            name=str(ang),
            data=[go.Surface(), go.Scatter3d(), go.Scatter3d(x=Bphi_sub[:,0], y=Bphi_sub[:,1], z=Bphi_sub[:,2]),
                  go.Scatter(), go.Scatter(), go.Scatter(),
                  go.Scatter(x=[ang,ang], y=[y1min,y1max]),
                  go.Scatter(x=[ang,ang], y=[y2min,y2max]),
                  go.Scatter(x=[ang,ang], y=[y3min,y3max])]
        ))
    fig.frames = frames

    steps = []
    for ang in angles_deg:
        steps.append(dict(method="animate",
                          args=[[str(ang)], {"mode":"immediate", "frame":{"duration":0,"redraw":True},
                                             "transition":{"duration":0}}],
                          label=f"{ang}°"))
    sliders = [dict(active=0, currentvalue={"prefix":"Rotation φ = "}, pad={"t":30}, steps=steps)]

    fig.update_layout(title="Overlap on S²: Cauchy-around-μx projected; B rotated about Z",
                      height=800, sliders=sliders, scene=dict(aspectmode="data"),
                      margin=dict(l=10,r=10,t=50,b=10))

    fig.update_xaxes(title_text="φ (deg)", row=2, col=1)
    fig.update_yaxes(title_text="BC (0–1)", row=2, col=1)
    fig.update_xaxes(title_text="φ (deg)", row=2, col=2)
    fig.update_yaxes(title_text="MMD²", row=2, col=2)
    fig.update_xaxes(title_text="φ (deg)", row=2, col=3)
    fig.update_yaxes(title_text="OVL", row=2, col=3)

    from plotly.offline import plot as plotly_save
    plotly_save(fig, filename=args.out_html, auto_open=False)
    print("Wrote:", args.out_html)
    print("Wrote:", args.out_csv)

if __name__ == "__main__":
    main()
