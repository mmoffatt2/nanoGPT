#!/usr/bin/env python3
import argparse, math, numpy as np, pandas as pd

# ---------- Sampling utilities ----------
def normalize_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n==0] = 1.0; return X / n

def sample_uniform_s2(n, rng):
    return normalize_rows(rng.normal(size=(n,3)))

def sample_cauchy_projected(mu, gamma, n, rng):
    Y = mu.reshape(1,3) + rng.standard_cauchy(size=(n,3)) * gamma
    return normalize_rows(Y)

def sample_gaussian_projected(mu, sigma, n, rng):
    Y = mu.reshape(1,3) + rng.normal(size=(n,3)) * sigma
    return normalize_rows(Y)

def _orthonormal_basis_from_mu(mu):
    mu = mu / np.linalg.norm(mu)
    # pick a helper not colinear
    if abs(mu[2]) < 0.9:
        h = np.array([0.0, 0.0, 1.0])
    else:
        h = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(mu, h); e1 /= np.linalg.norm(e1)
    e2 = np.cross(mu, e1)
    return mu, e1, e2

def sample_vmf(mu, kappa, n, rng):
    """von Mises–Fisher on S^2 with mean direction mu (|mu|=1) and concentration kappa.
    Uses inverse CDF for cos(theta): t ~ proportional to exp(kappa t) on [-1,1]."""
    mu = mu / np.linalg.norm(mu)
    if kappa <= 1e-8:
        # uniform on sphere
        return sample_uniform_s2(n, rng)
    u = rng.random(size=n)
    # Stable inverse CDF: t = -1 + (1/kappa) * log(1 + u * (exp(2kappa)-1))
    e2k = np.exp(2.0 * kappa)
    t = -1.0 + (np.log1p(u * (e2k - 1.0)) / kappa)  # cos(theta)
    t = np.clip(t, -1.0, 1.0)
    phi = rng.random(size=n) * (2.0*np.pi)
    sint = np.sqrt(np.clip(1.0 - t*t, 0.0, 1.0))
    mu_hat, e1, e2 = _orthonormal_basis_from_mu(mu)
    X = (t[:,None] * mu_hat[None,:] +
         (sint*np.cos(phi))[:,None] * e1[None,:] +
         (sint*np.sin(phi))[:,None] * e2[None,:])
    return normalize_rows(X)

# ---------- Kernels / metrics (same as before) ----------
def Z_beta_s2(beta): return 1.0 if beta==0 else math.sinh(beta)/beta
def kde_density_at(X_ref, X_eval, beta, Zb): return np.mean(np.exp(beta*(X_eval@X_ref.T))/Zb, axis=1)
def angular_coverage_overlap(A,B,theta_rad):
    c=math.cos(theta_rad); AB=A@B.T; return float(0.5*(np.mean((AB>=c).any(1))+np.mean((AB.T>=c).any(1)))), float(np.mean((AB>=c).any(1))), float(np.mean((AB.T>=c).any(1)))
def spherical_fibonacci_points(N):
    i=np.arange(N); phi=(np.pi*(3.0-np.sqrt(5.0)))*i; z=1.0-(2.0*(i+0.5)/N); r=np.sqrt(np.maximum(0.0,1.0-z*z)); x=r*np.cos(phi); y=r*np.sin(phi); return normalize_rows(np.vstack([x,y,z]).T)
def healpix_bin_ids(X, nside=32):
    try:
        import healpy as hp
        theta,phi=hp.vec2ang(X.T); pix=hp.ang2pix(nside, theta, phi, nest=False); return pix.astype(int), f"HEALPix nside={nside}"
    except Exception:
        Npix=int(12*nside*nside); grid=spherical_fibonacci_points(Npix); idx=np.argmax(X@grid.T,axis=1); return idx.astype(int), f"Fibonacci fallback (≈HEALPix {Npix} cells)"
def occupancy_jaccard(idsA, idsB):
    sA=set(idsA.tolist()); sB=set(idsB.tolist()); inter=len(sA&sB); uni=len(sA|sB); return (inter/uni if uni>0 else 0.0), inter, len(sA), len(sB), uni
def hausdorff_directed_from_AB(AB):
    dA=np.arccos(np.clip(AB.max(1),-1,1)).max(); dB=np.arccos(np.clip(AB.max(0),-1,1)).max(); return float(dA), float(dB)
def kde_kl_directional(A,B,beta,Zb,eps=1e-12):
    nA=A.shape[0]; AA=A@A.T; KAA=np.exp(beta*AA)/Zb; self_k=math.exp(beta)/Zb; pA=(KAA.sum(1)-self_k)/max(nA-1,1)
    qB=np.mean(np.exp(beta*(A@B.T))/Zb, axis=1)
    return float(np.mean(np.log(np.maximum(pA,eps))-np.log(np.maximum(qB,eps))))

def chord_overlap_details(A, B, alpha=0.9):
    def wrap(a): return np.mod(a,2*np.pi)
    def min_arc(angles, alpha=0.9):
        n=angles.size; k=max(1,int(math.ceil(alpha*n))); arr=np.sort(wrap(angles)); arr2=np.concatenate([arr,arr+2*np.pi]); spans=arr2[k-1:n+k-1]-arr; j=int(np.argmin(spans)); return float(arr[j]), float(arr2[j+k-1]), float(spans[j])
    def inter_len(a1,b1,a2,b2):
        def split(a,b): return [(a,b)] if b<=2*np.pi else [(a,2*np.pi),(0.0,b-2*np.pi)]
        inter=0.0
        for x1,y1 in split(a1,b1):
            for x2,y2 in split(a2,b2):
                lo,hi=max(x1,x2),min(y1,y2)
                if hi>lo: inter += (hi-lo)
        return float(inter)
    muA=A.mean(0); muB=B.mean(0)
    if np.linalg.norm(muA)<1e-12 or np.linalg.norm(muB)<1e-12: return (0.0,0.0,0.0,0.0,0.0)
    muA/=np.linalg.norm(muA); muB/=np.linalg.norm(muB)
    n=np.cross(muA,muB)
    if np.linalg.norm(n)<1e-9:
        tmp=np.array([0,0,1.0]); 
        if abs(muA@tmp)>0.9: tmp=np.array([1.0,0,0])
        n=np.cross(muA,tmp)
    n/=np.linalg.norm(n)
    e1=muA-(muA@n)*n
    if np.linalg.norm(e1)<1e-12: e1=muB-(muB@n)*n
    e1/=np.linalg.norm(e1); e2=np.cross(n,e1)
    angA=np.arctan2(A@e2,A@e1); angB=np.arctan2(B@e2,B@e1)
    a1,b1,w1=min_arc(angA,0.9); a2,b2,w2=min_arc(angB,0.9)
    inter=inter_len(a1,b1,a2,b2)
    union=w1+w2-inter if w1+w2-inter>1e-12 else 1e-12
    ovl=inter/union; mA=inter/max(w1,1e-12); mB=inter/max(w2,1e-12)
    return ovl,mA,mB,w1,w2

# ---------- Main ----------
def main():
    ap=argparse.ArgumentParser(description="S^2 Plotly demo with overlap/merge modes + isotropic generators")
    ap.add_argument("--mode", choices=["gen","pt"], default="gen")
    ap.add_argument("--pt", type=str, default=None)
    ap.add_argument("--nA", type=int, default=1000); ap.add_argument("--nB", type=int, default=1000)
    # generator model
    ap.add_argument("--gen-model", choices=["vmf","gaussian","cauchy"], default="vmf", help="Generator used in --mode gen")
    # vMF parameters
    ap.add_argument("--kappa1", type=float, default=20.0, help="vMF concentration for A (larger = tighter)")
    ap.add_argument("--kappa2", type=float, default=20.0, help="vMF concentration for B")
    # Gaussian parameters
    ap.add_argument("--sigma1", type=float, default=0.25, help="Gaussian sigma for A")
    ap.add_argument("--sigma2", type=float, default=0.25, help="Gaussian sigma for B")
    # Cauchy parameters (kept for back-compat)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--gamma1", type=float, default=0.7); ap.add_argument("--gamma2", type=float, default=0.7)
    # colors / point size / draw subset
    ap.add_argument("--color-A", type=str, default="#1f77b4"); ap.add_argument("--color-B", type=str, default="#ff7f0e")
    ap.add_argument("--marker-size", type=int, default=4); ap.add_argument("--plot-subset", type=int, default=900)
    # layout sizing
    ap.add_argument("--fig-height", type=int, default=1100); ap.add_argument("--scene-frac", type=float, default=0.45)
    # metrics params
    ap.add_argument("--beta", type=float, default=12.0); ap.add_argument("--theta-deg", type=float, default=60.0)
    ap.add_argument("--prox-deg", type=float, default=15.0)
    ap.add_argument("--fib-N", type=int, default=1200); ap.add_argument("--healpix-nside", type=int, default=32)
    ap.add_argument("--alpha-chord", type=float, default=0.9)
    ap.add_argument("--angle-step", type=int, default=10); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--metric-mode", choices=["overlap","merge"], default="merge")
    ap.add_argument("--out-html", type=str, default="s2_merge_plotly.html"); ap.add_argument("--out-csv", type=str, default="s2_rotation_metrics.csv")
    args=ap.parse_args()

    rng=np.random.default_rng(args.seed)
    mu_x=np.array([1.0,0,0])

    # --- Load or generate ---
    if args.mode=="pt":
        import torch
        obj=torch.load(args.pt, map_location="cpu")
        if isinstance(obj,dict) and "A" in obj and "B" in obj: A=obj["A"].cpu().numpy(); B0=obj["B"].cpu().numpy()
        elif isinstance(obj,(list,tuple)) and len(obj)==2: A=obj[0].cpu().numpy(); B0=obj[1].cpu().numpy()
        else: raise ValueError("Unsupported .pt structure.")
        A=normalize_rows(A); B0=normalize_rows(B0)
        if A.shape[1]!=3 or B0.shape[1]!=3: raise ValueError("Data must be (n,3) for S^2 demo.")
    else:
        if args.gen_model == "vmf":
            A = sample_vmf(mu_x, args.kappa1, args.nA, rng)
            B0 = sample_vmf(mu_x, args.kappa2, args.nB, rng)
        elif args.gen_model == "gaussian":
            A = sample_gaussian_projected(mu_x, args.sigma1, args.nA, rng)
            B0 = sample_gaussian_projected(mu_x, args.sigma2, args.nB, rng)
        else: # cauchy
            g1 = args.gamma1 if args.gamma is None else (args.gamma1 if args.gamma1!=0.7 else args.gamma)
            g2 = args.gamma2 if args.gamma is None else (args.gamma2 if args.gamma2!=0.7 else args.gamma)
            if g1 is None: g1 = args.gamma1
            if g2 is None: g2 = args.gamma2
            A = sample_cauchy_projected(mu_x, g1, args.nA, rng)
            B0 = sample_cauchy_projected(mu_x, g2, args.nB, rng)

    # --- Metrics ---
    beta=args.beta; Zb=Z_beta_s2(beta); theta_rad=math.radians(args.theta_deg); prox_rad=math.radians(args.prox_deg)
    angles_deg=list(range(0,181,max(1,int(args.angle_step))))

    X_eval=sample_uniform_s2(2000,rng); pA_eval=kde_density_at(A,X_eval,beta,Zb)
    AA=A@A.T; np.fill_diagonal(AA,-np.inf); kAA=np.sum(np.exp(beta*AA)[AA>-np.inf])/(A.shape[0]*(A.shape[0]-1))
    BB=B0@B0.T; np.fill_diagonal(BB,-np.inf); kBB=np.sum(np.exp(beta*BB)[BB>-np.inf])/(B0.shape[0]*(B0.shape[0]-1))

    fib_grid=spherical_fibonacci_points(args.fib_N); fib_ids_A=np.argmax(A@fib_grid.T,axis=1)
    hp_ids_A, hp_label=healpix_bin_ids(A, args.healpix_nside)

    BC_vals=[]; MMD2_vals=[]; OVL_vals=[]; KL_AtoB_vals=[]; KL_BtoA_vals=[]; Cov_A_given_B=[]; Cov_B_given_A=[]
    Haus_AtoB=[]; Haus_BtoA=[]; Fib_Jaccard=[]; Fib_rec_AtoB=[]; Fib_rec_BtoA=[]; HP_Jaccard=[]; HP_rec_AtoB=[]; HP_rec_BtoA=[]
    Chord_sym=[]; Chord_AtoB=[]; Chord_BtoA=[]; ang_means=[]; cos_means=[]
    ProxCov_AinB=[]; ProxCov_BinA=[]; MeanNN_AtoB_deg=[]; MeanNN_BtoA_deg=[]

    for ang in angles_deg:
        R=np.array([[math.cos(math.radians(ang)),-math.sin(math.radians(ang)),0.0],
                    [math.sin(math.radians(ang)), math.cos(math.radians(ang)),0.0],
                    [0.0,0.0,1.0]])
        B=B0@R.T
        qB_eval=kde_density_at(B,X_eval,beta,Zb); BC_vals.append(float(np.mean(np.sqrt(pA_eval*qB_eval))))
        AB=A@B.T; kAB=np.mean(np.exp(beta*AB)); MMD2_vals.append(float((kAA+kBB-2*kAB)/Zb))
        cth=math.cos(theta_rad); covA=float(np.mean((AB>=cth).any(1))); covB=float(np.mean((AB.T>=cth).any(1)))
        Cov_A_given_B.append(covA); Cov_B_given_A.append(covB); OVL_vals.append(0.5*(covA+covB))
        muA=A.mean(0); muB=B.mean(0); muA/=np.linalg.norm(muA); muB/=np.linalg.norm(muB)
        c=float(np.clip(muA@muB,-1,1)); cos_means.append(c); ang_means.append(math.degrees(math.acos(c)))
        hAB,hBA=hausdorff_directed_from_AB(AB); Haus_AtoB.append(math.degrees(hAB)); Haus_BtoA.append(math.degrees(hBA))
        KL_AtoB_vals.append(kde_kl_directional(A,B,beta,Zb)); KL_BtoA_vals.append(kde_kl_directional(B,A,beta,Zb))
        fib_ids_B=np.argmax(B@fib_grid.T,axis=1); jacc, inter, nAocc, nBocc, uni=occupancy_jaccard(fib_ids_A,fib_ids_B)
        Fib_Jaccard.append(jacc); Fib_rec_AtoB.append(inter/max(nAocc,1)); Fib_rec_BtoA.append(inter/max(nBocc,1))
        hp_ids_B,_=healpix_bin_ids(B, args.healpix_nside); jh, ih, nAocc_h, nBocc_h, uh=occupancy_jaccard(hp_ids_A,hp_ids_B)
        HP_Jaccard.append(jh); HP_rec_AtoB.append(ih/max(nAocc_h,1)); HP_rec_BtoA.append(ih/max(nBocc_h,1))
        ovl_sym,mA,mB,_,_=chord_overlap_details(A,B,args.alpha_chord); Chord_sym.append(ovl_sym); Chord_AtoB.append(mA); Chord_BtoA.append(mB)
        nearest_A=np.arccos(np.clip(AB.max(1),-1,1)); nearest_B=np.arccos(np.clip(AB.max(0),-1,1))
        ProxCov_AinB.append(float(np.mean(nearest_A <= prox_rad))); ProxCov_BinA.append(float(np.mean(nearest_B <= prox_rad)))
        MeanNN_AtoB_deg.append(float(np.degrees(nearest_A.mean()))); MeanNN_BtoA_deg.append(float(np.degrees(nearest_B.mean())))

    # Save CSV
    df=pd.DataFrame({
        "angle_deg": angles_deg,
        "BC": BC_vals, "MMD2_norm": MMD2_vals, f"OVL_theta_{args.theta_deg}deg": OVL_vals,
        "mean_angle_deg": ang_means, "mean_cosine": cos_means,
        "Hausdorff_sym_deg": np.maximum(Haus_BtoA, Haus_AtoB),
        f"Fib_Jaccard_N{args.fib_N}": Fib_Jaccard, f"HP_Jaccard_{hp_label}": HP_Jaccard,
        f"Chord_overlap_alpha{int(100*args.alpha_chord)}pct": Chord_sym,
        "KL_AtoB": KL_AtoB_vals, "KL_BtoA": KL_BtoA_vals,
        f"Cov_A|B_theta_{args.theta_deg}deg": Cov_A_given_B, f"Cov_B|A_theta_{args.theta_deg}deg": Cov_B_given_A,
        "Haus_AtoB_deg": Haus_AtoB, "Haus_BtoA_deg": Haus_BtoA,
        f"Fib_recall_AinB_N{args.fib_N}": Fib_rec_AtoB, f"Fib_recall_BinA_N{args.fib_N}": Fib_rec_BtoA,
        f"HP_recall_AinB_{hp_label}": HP_rec_AtoB, f"HP_recall_BinA_{hp_label}": HP_rec_BtoA,
        f"Chord_merge_AtoB_alpha{int(100*args.alpha_chord)}pct": Chord_AtoB, f"Chord_merge_BtoA_alpha{int(100*args.alpha_chord)}pct": Chord_BtoA,
        f"Prox{int(args.prox_deg)}_AinB": ProxCov_AinB, f"Prox{int(args.prox_deg)}_BinA": ProxCov_BinA,
        "MeanNN_AtoB_deg": MeanNN_AtoB_deg, "MeanNN_BtoA_deg": MeanNN_BtoA_deg
    })
    df.to_csv(args.out_csv, index=False)

    # --- Plotly ---
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from plotly.offline import plot as plotly_save
    except Exception:
        print("Plotly not available; saved CSV only:", args.out_csv); return

    sf=max(0.05, min(0.9, float(args.scene_frac)))
    rest=(1.0-sf)/4.0
    row_heights=[sf, rest, rest, rest, rest]

    rows, cols = 5, 3
    if args.metric_mode=="overlap":
        titles=(
            "S² point clouds (A fixed, B rotated by φ about Z)",
            # row 2
            "Bhattacharyya coefficient (KDE)",
            "Kernel MMD² (normalized)",
            f"Angular coverage overlap θ={args.theta_deg:.0f}°",
            # row 3
            "Angle between mean directions (deg)",
            "Cosine similarity of means",
            "Hausdorff distance (deg, geodesic)",
            # row 4
            f"Fibonacci bins Jaccard (N={args.fib_N})",
            f"{hp_label} Jaccard",
            f"Chord overlap on common great circle (α={int(100*args.alpha_chord)}%)",
            # row 5 (new panels)
            f"Proximity coverage @ δ={args.prox_deg:.0f}° (A|B & B|A)",
            "Mean nearest angle (deg)",
            ""  # keep grid (empty slot at row5,col3)
        )
    else:
        titles=(
            "S² point clouds (A fixed, B rotated by φ about Z)",
            # row 2
            "KDE-KL merge: KL(A||B) & KL(B||A)",
            f"Angular coverage merges θ={args.theta_deg:.0f}°: Cov(A|B), Cov(B|A)",
            "Directed Hausdorff distances (deg)",
            # row 3
            "Fibonacci bin recall: |A∩B|/|A| and /|B|",
            "HEALPix bin recall: |A∩B|/|A| and /|B|",
            f"Chord merges (α={int(100*args.alpha_chord)}%): inter/width_A and inter/width_B",
            # row 4
            "Angle between mean directions (deg)",
            "Cosine similarity of means",
            "",
            # row 5 (new panels)
            f"Proximity coverage @ δ={args.prox_deg:.0f}° (A|B & B|A)",
            "Mean nearest angle (deg)",
            ""
        )

    fig=make_subplots(rows=rows, cols=cols, row_heights=row_heights,
                      specs=[[{"type":"scene","colspan":3}, None, None],
                             [{"type":"xy"},{"type":"xy"},{"type":"xy"}],
                             [{"type":"xy"},{"type":"xy"},{"type":"xy"}],
                             [{"type":"xy"},{"type":"xy"},{"type":"xy"}],
                             [{"type":"xy"},{"type":"xy"},{"type":"xy"}]],
                      subplot_titles=titles)

    fig.update_layout(title={"text":"<b>Overlap on S² — Cauchy-around-μx (projected), B rotated about Z</b>","x":0.5,"xanchor":"center"},
                      title_font_size=20, height=args.fig_height, scene=dict(aspectmode="data"),
                      margin=dict(l=10,r=10,t=60,b=80), showlegend=False)

    # 3D sphere and points
    u=np.linspace(0,2*np.pi,60); v=np.linspace(0,np.pi,30); uu,vv=np.meshgrid(u,v); xs=np.cos(uu)*np.sin(vv); ys=np.sin(uu)*np.sin(vv); zs=np.cos(vv)
    idxA=np.random.choice(A.shape[0], size=min(args.plot_subset, A.shape[0]), replace=False)
    idxB=np.random.choice(B0.shape[0], size=min(args.plot_subset, B0.shape[0]), replace=False)
    A_sub=A[idxA]; B0_sub=B0[idxB]; B_init=(B0_sub@np.eye(3).T)
    fig.add_trace(go.Surface(x=xs,y=ys,z=zs,opacity=0.2,showscale=False), row=1,col=1)
    fig.add_trace(go.Scatter3d(x=A_sub[:,0],y=A_sub[:,1],z=A_sub[:,2],mode="markers",
                               marker=dict(size=args.marker_size, color=args.color_A)), row=1,col=1)
    fig.add_trace(go.Scatter3d(x=B_init[:,0],y=B_init[:,1],z=B_init[:,2],mode="markers",
                               marker=dict(size=args.marker_size, color=args.color_B)), row=1,col=1)

    # Curves
    def add_lines_overlap():
        c1, c2, c3 = "#6C7A89", "#34495E", "#95A5A6"
        fig.add_trace(go.Scatter(x=angles_deg,y=BC_vals,mode="lines",line=dict(color=c1)), row=2,col=1)
        fig.add_trace(go.Scatter(x=angles_deg,y=MMD2_vals,mode="lines",line=dict(color=c2)), row=2,col=2)
        fig.add_trace(go.Scatter(x=angles_deg,y=OVL_vals,mode="lines",line=dict(color=c3)), row=2,col=3)
        fig.add_trace(go.Scatter(x=angles_deg,y=ang_means,mode="lines",line=dict(color=c1)), row=3,col=1)
        fig.add_trace(go.Scatter(x=angles_deg,y=cos_means,mode="lines",line=dict(color=c2)), row=3,col=2)
        fig.add_trace(go.Scatter(x=angles_deg,y=np.maximum(Haus_AtoB,Haus_BtoA),mode="lines",line=dict(color=c3)), row=3,col=3)
        fig.add_trace(go.Scatter(x=angles_deg,y=Fib_Jaccard,mode="lines",line=dict(color=c1)), row=4,col=1)
        fig.add_trace(go.Scatter(x=angles_deg,y=HP_Jaccard,mode="lines",line=dict(color=c2)), row=4,col=2)
        fig.add_trace(go.Scatter(x=angles_deg,y=Chord_sym,mode="lines",line=dict(color=c3)), row=4,col=3)
        # Proximity + mean NN
        fig.add_trace(go.Scatter(x=angles_deg,y=ProxCov_AinB,mode="lines",line=dict(color=c1)), row=5,col=1)
        fig.add_trace(go.Scatter(x=angles_deg,y=ProxCov_BinA,mode="lines",line=dict(color=c2)), row=5,col=1)
        fig.add_trace(go.Scatter(x=angles_deg,y=MeanNN_AtoB_deg,mode="lines",line=dict(color=c1)), row=5,col=2)
        fig.add_trace(go.Scatter(x=angles_deg,y=MeanNN_BtoA_deg,mode="lines",line=dict(color=c2)), row=5,col=2)
    def add_lines_merge():
        fig.add_trace(go.Scatter(x=angles_deg,y=KL_AtoB_vals,mode="lines",line=dict(color=args.color_A)), row=2,col=1)
        fig.add_trace(go.Scatter(x=angles_deg,y=KL_BtoA_vals,mode="lines",line=dict(color=args.color_B)), row=2,col=1)
        fig.add_trace(go.Scatter(x=angles_deg,y=Cov_A_given_B,mode="lines",line=dict(color=args.color_A)), row=2,col=2)
        fig.add_trace(go.Scatter(x=angles_deg,y=Cov_B_given_A,mode="lines",line=dict(color=args.color_B)), row=2,col=2)
        fig.add_trace(go.Scatter(x=angles_deg,y=Haus_AtoB,mode="lines",line=dict(color=args.color_A)), row=2,col=3)
        fig.add_trace(go.Scatter(x=angles_deg,y=Haus_BtoA,mode="lines",line=dict(color=args.color_B)), row=2,col=3)
        fig.add_trace(go.Scatter(x=angles_deg,y=Fib_rec_AtoB,mode="lines",line=dict(color=args.color_A)), row=3,col=1)
        fig.add_trace(go.Scatter(x=angles_deg,y=Fib_rec_BtoA,mode="lines",line=dict(color=args.color_B)), row=3,col=1)
        fig.add_trace(go.Scatter(x=angles_deg,y=HP_rec_AtoB,mode="lines",line=dict(color=args.color_A)), row=3,col=2)
        fig.add_trace(go.Scatter(x=angles_deg,y=HP_rec_BtoA,mode="lines",line=dict(color=args.color_B)), row=3,col=2)
        fig.add_trace(go.Scatter(x=angles_deg,y=Chord_AtoB,mode="lines",line=dict(color=args.color_A)), row=3,col=3)
        fig.add_trace(go.Scatter(x=angles_deg,y=Chord_BtoA,mode="lines",line=dict(color=args.color_B)), row=3,col=3)
        fig.add_trace(go.Scatter(x=angles_deg,y=ang_means,mode="lines",line=dict(color="#7F8C8D")), row=4,col=1)
        fig.add_trace(go.Scatter(x=angles_deg,y=cos_means,mode="lines",line=dict(color="#7F8C8D")), row=4,col=2)
        # Proximity + mean NN
        fig.add_trace(go.Scatter(x=angles_deg,y=ProxCov_AinB,mode="lines",line=dict(color=args.color_A)), row=5,col=1)
        fig.add_trace(go.Scatter(x=angles_deg,y=ProxCov_BinA,mode="lines",line=dict(color=args.color_B)), row=5,col=1)
        fig.add_trace(go.Scatter(x=angles_deg,y=MeanNN_AtoB_deg,mode="lines",line=dict(color=args.color_A)), row=5,col=2)
        fig.add_trace(go.Scatter(x=angles_deg,y=MeanNN_BtoA_deg,mode="lines",line=dict(color=args.color_B)), row=5,col=2)

    if args.metric_mode=="overlap": add_lines_overlap()
    else: add_lines_merge()

    # Axes labels for last row panels
    fig.update_xaxes(title_text="φ (deg)", row=5, col=1); fig.update_yaxes(title_text="Prox. cov", row=5, col=1)
    fig.update_xaxes(title_text="φ (deg)", row=5, col=2); fig.update_yaxes(title_text="Mean NN angle (deg)", row=5, col=2)

    # Vertical lines and frames
    if args.metric_mode=="overlap":
        y_ranges=[(min(BC_vals),max(BC_vals)),(min(MMD2_vals),max(MMD2_vals)),(0,1),
                  (min(ang_means),max(ang_means)),(-1,1),(min(np.maximum(Haus_AtoB,Haus_BtoA)),max(np.maximum(Haus_AtoB,Haus_BtoA))),
                  (0,1),(0,1),(0,1),(0,1),(min(MeanNN_AtoB_deg+MeanNN_BtoA_deg),max(MeanNN_AtoB_deg+MeanNN_BtoA_deg))]
        v_coords=[(2,1),(2,2),(2,3),(3,1),(3,2),(3,3),(4,1),(4,2),(4,3),(5,1),(5,2)]
    else:
        y_ranges=[(min(KL_AtoB_vals+KL_BtoA_vals),max(KL_AtoB_vals+KL_BtoA_vals)),(0,1),
                  (min(Haus_AtoB+Haus_BtoA),max(Haus_AtoB+Haus_BtoA)),
                  (0,1),(0,1),(0,1),(min(ang_means),max(ang_means)),(-1,1),(0,1),
                  (min(MeanNN_AtoB_deg+MeanNN_BtoA_deg),max(MeanNN_AtoB_deg+MeanNN_BtoA_deg))]
        v_coords=[(2,1),(2,2),(2,3),(3,1),(3,2),(3,3),(4,1),(4,2),(5,1),(5,2)]

    vline_indices=[]
    for (row,col),yr in zip(v_coords,y_ranges):
        fig.add_trace(go.Scatter(x=[angles_deg[0], angles_deg[0]], y=list(yr), mode="lines",
                                 line=dict(dash="dash", width=1), showlegend=False), row=row, col=col)
        vline_indices.append(len(fig.data)-1)

    frames=[]
    for ang in angles_deg:
        R=np.array([[math.cos(math.radians(ang)),-math.sin(math.radians(ang)),0.0],
                    [math.sin(math.radians(ang)), math.cos(math.radians(ang)),0.0],
                    [0.0,0.0,1.0]])
        B_sub=(B0_sub@R.T)
        updates=[go.Scatter3d(x=B_sub[:,0], y=B_sub[:,1], z=B_sub[:,2])]
        updates += [go.Scatter(x=[ang,ang], y=list(yr)) for yr in y_ranges]
        frames.append(go.Frame(name=str(ang), data=updates, traces=[2]+vline_indices))
    fig.frames=frames

    # Slider (progress bar) with extra bottom padding; add autoscale/reset buttons
    steps=[dict(method="animate", args=[[str(ang)],{"mode":"immediate","frame":{"duration":0,"redraw":True},"transition":{"duration":0}}], label=f"{ang}°") for ang in angles_deg]
    sliders=[dict(active=0, currentvalue={"prefix":"Rotation φ = "}, pad={"t":30,"b":15}, steps=steps)]
    fig.update_layout(sliders=sliders)
    config=dict(displaylogo=False, modeBarButtonsToAdd=["autoScale2d","resetScale2d"], scrollZoom=True)

    from plotly.offline import plot as plotly_save
    plotly_save(fig, filename=args.out_html, auto_open=False, config=config)
    print("Wrote:", args.out_html); print("Wrote:", args.out_csv)

if __name__=="__main__":
    main()

