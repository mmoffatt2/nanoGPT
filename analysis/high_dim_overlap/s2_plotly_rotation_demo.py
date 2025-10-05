#!/usr/bin/env python3
import argparse, math, numpy as np, pandas as pd

def normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms==0]=1.0; return X/norms
def sample_cauchy_projected(mu, gamma, n, rng):
    S = rng.standard_cauchy(size=(n,3)) * gamma; Y = mu.reshape(1,3) + S; return normalize_rows(Y)
def rotate_z(theta_rad):
    c,s = math.cos(theta_rad), math.sin(theta_rad); return np.array([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]])
def sample_uniform_s2(n, rng): return normalize_rows(rng.normal(size=(n,3)))
def Z_beta_s2(beta): return 1.0 if beta==0 else math.sinh(beta)/beta
def kde_density_at(X_ref, X_eval, beta, Zb): return np.mean(np.exp(beta*(X_eval@X_ref.T))/Zb, axis=1)
def angular_coverage_overlap(A,B,theta_rad):
    c=math.cos(theta_rad); AB=A@B.T; return float(0.5*(np.mean((AB>=c).any(1))+np.mean((AB.T>=c).any(1)))), float(np.mean((AB>=c).any(1))), float(np.mean((AB.T>=c).any(1)))
def spherical_fibonacci_points(N):
    i=np.arange(N); phi=(np.pi*(3.0-np.sqrt(5.0)))*i; z=1.0-(2.0*(i+0.5)/N); r=np.sqrt(np.maximum(0.0,1.0-z*z)); x=r*np.cos(phi); y=r*np.sin(phi); return normalize_rows(np.vstack([x,y,z]).T)
def healpix_bin_ids(X, nside=32):
    try:
        import healpy as hp
        theta,phi=hp.vec2ang(X.T); pix=hp.ang2pix(nside,theta,phi,nest=False); return pix.astype(int), f"HEALPix nside={nside}"
    except Exception:
        Npix=int(12*nside*nside); grid=spherical_fibonacci_points(Npix); idx=np.argmax(X@grid.T,axis=1); return idx.astype(int), f"Fibonacci fallback (≈HEALPix {Npix} cells)"
def occupancy_jaccard(idsA, idsB):
    sA=set(idsA.tolist()); sB=set(idsB.tolist()); inter=len(sA&sB); uni=len(sA|sB); return (inter/uni if uni>0 else 0.0), inter, len(sA), len(sB), uni
def hausdorff_directed_from_AB(AB):
    dA=np.arccos(np.clip(AB.max(1),-1,1)).max(); dB=np.arccos(np.clip(AB.max(0),-1,1)).max(); return float(dA), float(dB)
def wrap_0_2pi(a): return np.mod(a,2*np.pi)
def min_covering_arc(angles, alpha=0.9):
    n=angles.size; k=max(1,int(math.ceil(alpha*n))); arr=np.sort(wrap_0_2pi(angles)); arr2=np.concatenate([arr,arr+2*np.pi]); spans=arr2[k-1:n+k-1]-arr; j=int(np.argmin(spans)); return float(arr[j]), float(arr2[j+k-1]), float(spans[j])
def arc_intersection_length(a1,b1,a2,b2):
    def split(a,b): return [(a,b)] if b<=2*np.pi else [(a,2*np.pi),(0.0,b-2*np.pi)]
    inter=0.0
    for x1,y1 in split(a1,b1):
        for x2,y2 in split(a2,b2):
            lo,hi=max(x1,x2),min(y1,y2)
            if hi>lo: inter += (hi-lo)
    return float(inter)
def chord_overlap_details(A,B,alpha=0.9):
    muA=A.mean(0); muB=B.mean(0)
    if np.linalg.norm(muA)<1e-12 or np.linalg.norm(muB)<1e-12: return (0.0,0.0,0.0,0.0,0.0)
    muA/=np.linalg.norm(muA); muB/=np.linalg.norm(muB)
    n=np.cross(muA,muB)
    if np.linalg.norm(n)<1e-9:
        tmp=np.array([0.0,0.0,1.0]); 
        if abs(muA@tmp)>0.9: tmp=np.array([1.0,0.0,0.0])
        n=np.cross(muA,tmp)
    n/=np.linalg.norm(n)
    e1=muA-(muA@n)*n
    if np.linalg.norm(e1)<1e-12: e1=muB-(muB@n)*n
    e1/=np.linalg.norm(e1); e2=np.cross(n,e1)
    angA=np.arctan2(A@e2,A@e1); angB=np.arctan2(B@e2,B@e1)
    a1,b1,w1=min_covering_arc(angA,alpha); a2,b2,w2=min_covering_arc(angB,alpha)
    inter=arc_intersection_length(a1,b1,a2,b2); union=w1+w2-inter if w1+w2-inter>1e-12 else 1e-12; ovl=inter/union; mA=inter/max(w1,1e-12); mB=inter/max(w2,1e-12)
    return ovl,mA,mB,w1,w2
def kde_kl_directional(A,B,beta,Zb,eps=1e-12):
    nA=A.shape[0]; AA=A@A.T; KAA=np.exp(beta*AA)/Zb; self_k=math.exp(beta)/Zb; pA=(KAA.sum(1)-self_k)/max(nA-1,1)
    qB=np.mean(np.exp(beta*(A@B.T))/Zb, axis=1)
    return float(np.mean(np.log(np.maximum(pA,eps))-np.log(np.maximum(qB,eps))))

def main():
    ap=argparse.ArgumentParser(description="S^2 Plotly demo with overlap/merge modes (polished)")
    ap.add_argument("--mode", choices=["gen","pt"], default="gen")
    ap.add_argument("--pt", type=str, default=None)
    # point counts
    ap.add_argument("--nA", type=int, default=1000)
    ap.add_argument("--nB", type=int, default=1000)
    # Cauchy scales (distinct); --gamma kept for back-compat if you want to set both at once
    ap.add_argument("--gamma", type=float, default=None, help="If set, applies to both distributions unless gamma1/gamma2 are given")
    ap.add_argument("--gamma1", type=float, default=0.7, help="Cauchy scale for A")
    ap.add_argument("--gamma2", type=float, default=0.7, help="Cauchy scale for B")
    # colors and plotting
    ap.add_argument("--color-A", type=str, default="#E74C3C", help="Color for A (CSS/hex)")
    ap.add_argument("--color-B", type=str, default="#2ECC71", help="Color for B (CSS/hex)")
    ap.add_argument("--marker-size", type=int, default=4, help="3D marker size")
    ap.add_argument("--plot-subset", type=int, default=900, help="Max #points per cloud to display")
    # metrics params
    ap.add_argument("--beta", type=float, default=12.0)
    ap.add_argument("--theta-deg", type=float, default=60.0)
    ap.add_argument("--fib-N", type=int, default=1200)
    ap.add_argument("--healpix-nside", type=int, default=32)
    ap.add_argument("--alpha-chord", type=float, default=0.9)
    ap.add_argument("--angle-step", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--metric-mode", choices=["overlap","merge"], default="merge")
    ap.add_argument("--out-html", type=str, default="s2_merge_plotly.html")
    ap.add_argument("--out-csv", type=str, default="s2_rotation_metrics.csv")
    args=ap.parse_args()

    rng=np.random.default_rng(args.seed)
    # choose gammas
    g1 = args.gamma1 if args.gamma is None else args.gamma1 if args.gamma1!=0.7 else args.gamma
    g2 = args.gamma2 if args.gamma is None else args.gamma2 if args.gamma2!=0.7 else args.gamma
    if g1 is None: g1 = args.gamma1
    if g2 is None: g2 = args.gamma2

    if args.mode=="pt":
        import torch
        obj=torch.load(args.pt, map_location="cpu")
        if isinstance(obj,dict) and "A" in obj and "B" in obj: A=obj["A"].cpu().numpy(); B0=obj["B"].cpu().numpy()
        elif isinstance(obj,(list,tuple)) and len(obj)==2: A=obj[0].cpu().numpy(); B0=obj[1].cpu().numpy()
        else: raise ValueError("Unsupported .pt structure.")
        A=normalize_rows(A); B0=normalize_rows(B0)
        if A.shape[1]!=3 or B0.shape[1]!=3: raise ValueError("Data must be (n,3) for S^2 demo.")
    else:
        mu_x=np.array([1.0,0.0,0.0])
        A=sample_cauchy_projected(mu_x, g1, args.nA, rng)
        B0=sample_cauchy_projected(mu_x, g2, args.nB, rng)

    beta=args.beta; Zb=Z_beta_s2(beta); theta_rad=math.radians(args.theta_deg)
    angles_deg=list(range(0,181,max(1,int(args.angle_step))))

    X_eval=sample_uniform_s2(2000,rng); pA_eval=kde_density_at(A,X_eval,beta,Zb)
    AA=A@A.T; np.fill_diagonal(AA,-np.inf); kAA=np.sum(np.exp(beta*AA)[AA>-np.inf])/(A.shape[0]*(A.shape[0]-1))
    BB=B0@B0.T; np.fill_diagonal(BB,-np.inf); kBB=np.sum(np.exp(beta*BB)[BB>-np.inf])/(B0.shape[0]*(B0.shape[0]-1))

    fib_grid=spherical_fibonacci_points(args.fib_N); fib_ids_A=np.argmax(A@fib_grid.T,axis=1)
    hp_ids_A, hp_label=healpix_bin_ids(A, args.healpix_nside)

    # collect metrics across angles
    BC_vals=[]; MMD2_vals=[]; OVL_vals=[]; KL_AtoB_vals=[]; KL_BtoA_vals=[]; Cov_A_given_B=[]; Cov_B_given_A=[]
    Haus_AtoB=[]; Haus_BtoA=[]; Fib_Jaccard=[]; Fib_rec_AtoB=[]; Fib_rec_BtoA=[]; HP_Jaccard=[]; HP_rec_AtoB=[]; HP_rec_BtoA=[]
    Chord_sym=[]; Chord_AtoB=[]; Chord_BtoA=[]; ang_means=[]; cos_means=[]
    for ang in angles_deg:
        R=rotate_z(math.radians(ang)); B=B0@R.T
        qB_eval=kde_density_at(B,X_eval,beta,Zb); BC_vals.append(float(np.mean(np.sqrt(pA_eval*qB_eval))))
        AB=A@B.T; kAB=np.mean(np.exp(beta*AB)); MMD2_vals.append(float((kAA+kBB-2*kAB)/Zb))
        ovl,cAB,cBA=angular_coverage_overlap(A,B,theta_rad); OVL_vals.append(ovl); Cov_A_given_B.append(cAB); Cov_B_given_A.append(cBA)
        muA=A.mean(0); muB=B.mean(0); muA/=np.linalg.norm(muA); muB/=np.linalg.norm(muB); c=float(np.clip(muA@muB,-1,1))
        cos_means.append(c); ang_means.append(math.degrees(math.acos(c)))
        hAB,hBA=hausdorff_directed_from_AB(AB); Haus_AtoB.append(math.degrees(hAB)); Haus_BtoA.append(math.degrees(hBA))
        KL_AtoB_vals.append(kde_kl_directional(A,B,beta,Zb)); KL_BtoA_vals.append(kde_kl_directional(B,A,beta,Zb))
        fib_ids_B=np.argmax(B@fib_grid.T,axis=1); jacc, inter, nAocc, nBocc, uni=occupancy_jaccard(fib_ids_A,fib_ids_B)
        Fib_Jaccard.append(jacc); Fib_rec_AtoB.append(inter/max(nAocc,1)); Fib_rec_BtoA.append(inter/max(nBocc,1))
        hp_ids_B,_=healpix_bin_ids(B,args.healpix_nside); jh, ih, nAocc_h, nBocc_h, uh = occupancy_jaccard(hp_ids_A,hp_ids_B)
        HP_Jaccard.append(jh); HP_rec_AtoB.append(ih/max(nAocc_h,1)); HP_rec_BtoA.append(ih/max(nBocc_h,1))
        ovl_sym,mA,mB,wA,wB=chord_overlap_details(A,B,args.alpha_chord); Chord_sym.append(ovl_sym); Chord_AtoB.append(mA); Chord_BtoA.append(mB)

    df=pd.DataFrame({
        "angle_deg": angles_deg,
        "BC": BC_vals, "MMD2_norm": MMD2_vals, f"OVL_theta_{args.theta_deg}deg": OVL_vals,
        "mean_angle_deg": ang_means, "mean_cosine": cos_means,
        "Hausdorff_sym_deg": np.maximum(Haus_AtoB, Haus_BtoA),
        f"Fib_Jaccard_N{args.fib_N}": Fib_Jaccard, f"HP_Jaccard_{hp_label}": HP_Jaccard,
        f"Chord_overlap_alpha{int(100*args.alpha_chord)}pct": Chord_sym,
        "KL_AtoB": KL_AtoB_vals, "KL_BtoA": KL_BtoA_vals,
        f"Cov_A|B_theta_{args.theta_deg}deg": Cov_A_given_B, f"Cov_B|A_theta_{args.theta_deg}deg": Cov_B_given_A,
        "Haus_AtoB_deg": Haus_AtoB, "Haus_BtoA_deg": Haus_BtoA,
        f"Fib_recall_AinB_N{args.fib_N}": Fib_rec_AtoB, f"Fib_recall_BinA_N{args.fib_N}": Fib_rec_BtoA,
        f"HP_recall_AinB_{hp_label}": HP_rec_AtoB, f"HP_recall_BinA_{hp_label}": HP_rec_BtoA,
        f"Chord_merge_AtoB_alpha{int(100*args.alpha_chord)}pct": Chord_AtoB, f"Chord_merge_BtoA_alpha{int(100*args.alpha_chord)}pct": Chord_BtoA,
    })
    df.to_csv(args.out_csv, index=False)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from plotly.offline import plot as plotly_save
    except Exception:
        print("Plotly not available; saved CSV only:", args.out_csv); return

    # 3D sphere and subsample
    u=np.linspace(0,2*np.pi,60); v=np.linspace(0,np.pi,30); uu,vv=np.meshgrid(u,v); xs=np.cos(uu)*np.sin(vv); ys=np.sin(uu)*np.cos(vv); zs=np.cos(vv)
    idxA=np.random.choice(A.shape[0], size=min(args.plot_subset, A.shape[0]), replace=False)
    idxB=np.random.choice(B0.shape[0], size=min(args.plot_subset, B0.shape[0]), replace=False)
    A_sub=A[idxA]; B0_sub=B0[idxB]; B_init=(B0_sub@rotate_z(math.radians(angles_deg[0])).T)

    rows, cols = 4, 3
    if args.metric_mode=="overlap":
        titles=("S² point clouds (A fixed, B rotated by φ about Z)","Bhattacharyya coefficient (KDE)","Kernel MMD² (normalized)",f"Angular coverage overlap θ={args.theta_deg:.0f}°",
                "Angle between mean directions (deg)","Cosine similarity of means","Hausdorff distance (deg, geodesic)",
                f"Fibonacci bins Jaccard (N={args.fib_N})", f"{hp_label} Jaccard", f"Chord overlap on common great circle (α={int(100*args.alpha_chord)}%)","", "")
    else:
        titles=("S² point clouds (A fixed, B rotated by φ about Z)","KDE-KL merge: KL(A||B) & KL(B||A)",
                f"Angular coverage merges θ={args.theta_deg:.0f}°: Cov(A|B), Cov(B|A)","Directed Hausdorff distances (deg)",
                "Fibonacci bin recall: |A∩B|/|A| and /|B|","HEALPix bin recall: |A∩B|/|A| and /|B|",
                f"Chord merges (α={int(100*args.alpha_chord)}%): inter/width_A and inter/width_B","Angle between mean directions (deg)","Cosine similarity of means","")
    fig=make_subplots(rows=rows, cols=cols,
                      specs=[[{"type":"scene","colspan":3}, None, None],
                             [{"type":"xy"},{"type":"xy"},{"type":"xy"}],
                             [{"type":"xy"},{"type":"xy"},{"type":"xy"}],
                             [{"type":"xy"},{"type":"xy"},{"type":"xy"}]],
                      subplot_titles=titles)
    fig.update_layout(title={"text":"<b>Overlap on S² — Cauchy-around-μx (projected), B rotated about Z</b>","x":0.5,"xanchor":"center"},
                      title_font_size=20, height=1100, scene=dict(aspectmode="data"),
                      margin=dict(l=10,r=10,t=60,b=10), showlegend=False)

    fig.add_trace(go.Surface(x=xs,y=ys,z=zs,opacity=0.2,showscale=False), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=A_sub[:,0], y=A_sub[:,1], z=A_sub[:,2], mode="markers", name="A",
                               marker=dict(size=args.marker_size, color=args.color_A), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=B_init[:,0], y=B_init[:,1], z=B_init[:,2], mode="markers", name="B",
                               marker=dict(size=args.marker_size, color=args.color_B), showlegend=False), row=1, col=1)

    if args.metric_mode=="overlap":
        c1, c2, c3 = "#6C7A89", "#34495E", "#95A5A6"
        fig.add_trace(go.Scatter(x=angles_deg, y=BC_vals, mode="lines", line=dict(color=c1), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=angles_deg, y=MMD2_vals, mode="lines", line=dict(color=c2), showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=angles_deg, y=OVL_vals, mode="lines", line=dict(color=c3), showlegend=False), row=2, col=3)
        fig.add_trace(go.Scatter(x=angles_deg, y=ang_means, mode="lines", line=dict(color=c1), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=angles_deg, y=cos_means, mode="lines", line=dict(color=c2), showlegend=False), row=3, col=2)
        fig.add_trace(go.Scatter(x=angles_deg, y=np.maximum(Haus_AtoB, Haus_BtoA), mode="lines", line=dict(color=c3), showlegend=False), row=3, col=3)
        fig.add_trace(go.Scatter(x=angles_deg, y=Fib_Jaccard, mode="lines", line=dict(color=c1), showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=angles_deg, y=HP_Jaccard, mode="lines", line=dict(color=c2), showlegend=False), row=4, col=2)
        fig.add_trace(go.Scatter(x=angles_deg, y=Chord_sym, mode="lines", line=dict(color=c3), showlegend=False), row=4, col=3)
        y_ranges=[(min(BC_vals),max(BC_vals)),(min(MMD2_vals),max(MMD2_vals)),(0,1),
                  (min(ang_means),max(ang_means)),(-1,1),(min(np.maximum(Haus_AtoB,Haus_BtoA)),max(np.maximum(Haus_AtoB,Haus_BtoA))),
                  (0,1),(0,1),(0,1)]
        v_coords=[(2,1),(2,2),(2,3),(3,1),(3,2),(3,3),(4,1),(4,2),(4,3)]
    else:
        fig.add_trace(go.Scatter(x=angles_deg, y=KL_AtoB_vals, mode="lines", line=dict(color=args.color_A), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=angles_deg, y=KL_BtoA_vals, mode="lines", line=dict(color=args.color_B), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=angles_deg, y=Cov_A_given_B, mode="lines", line=dict(color=args.color_A), showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=angles_deg, y=Cov_B_given_A, mode="lines", line=dict(color=args.color_B), showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=angles_deg, y=Haus_AtoB, mode="lines", line=dict(color=args.color_A), showlegend=False), row=2, col=3)
        fig.add_trace(go.Scatter(x=angles_deg, y=Haus_BtoA, mode="lines", line=dict(color=args.color_B), showlegend=False), row=2, col=3)
        fig.add_trace(go.Scatter(x=angles_deg, y=Fib_rec_AtoB, mode="lines", line=dict(color=args.color_A), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=angles_deg, y=Fib_rec_BtoA, mode="lines", line=dict(color=args.color_B), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=angles_deg, y=HP_rec_AtoB, mode="lines", line=dict(color=args.color_A), showlegend=False), row=3, col=2)
        fig.add_trace(go.Scatter(x=angles_deg, y=HP_rec_BtoA, mode="lines", line=dict(color=args.color_B), showlegend=False), row=3, col=2)
        fig.add_trace(go.Scatter(x=angles_deg, y=Chord_AtoB, mode="lines", line=dict(color=args.color_A), showlegend=False), row=3, col=3)
        fig.add_trace(go.Scatter(x=angles_deg, y=Chord_BtoA, mode="lines", line=dict(color=args.color_B), showlegend=False), row=3, col=3)
        fig.add_trace(go.Scatter(x=angles_deg, y=ang_means, mode="lines", line=dict(color="#7F8C8D"), showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=angles_deg, y=cos_means, mode="lines", line=dict(color="#7F8C8D"), showlegend=False), row=4, col=2)
        y_ranges=[(min(KL_AtoB_vals+KL_BtoA_vals),max(KL_AtoB_vals+KL_BtoA_vals)),(0,1),(min(Haus_AtoB+Haus_BtoA),max(Haus_AtoB+Haus_BtoA)),
                  (0,1),(0,1),(0,1),(min(ang_means),max(ang_means)),(-1,1)]
        v_coords=[(2,1),(2,2),(2,3),(3,1),(3,2),(3,3),(4,1),(4,2)]

    vline_indices=[]
    for (row,col),yr in zip(v_coords,y_ranges):
        tr = go.Scatter(x=[angles_deg[0], angles_deg[0]], y=list(yr), mode="lines",
                        line=dict(dash="dash", width=1), showlegend=False)
        fig.add_trace(tr, row=row, col=col); vline_indices.append(len(fig.data)-1)

    # Frames updating B scatter + vlines only
    frames=[]
    for ang in angles_deg:
        B_sub=(B0_sub@rotate_z(math.radians(ang)).T)
        updates=[go.Scatter3d(x=B_sub[:,0], y=B_sub[:,1], z=B_sub[:,2])]
        updates += [go.Scatter(x=[ang,ang], y=list(yr)) for yr in y_ranges]
        frames.append(go.Frame(name=str(ang), data=updates, traces=[2]+vline_indices))
    fig.frames=frames

    steps=[dict(method="animate", args=[[str(ang)],{"mode":"immediate","frame":{"duration":0,"redraw":True},"transition":{"duration":0}}], label=f"{ang}°") for ang in angles_deg]
    sliders=[dict(active=0, currentvalue={"prefix":"Rotation φ = "}, pad={"t":30}, steps=steps)]
    fig.update_layout(sliders=sliders)

    out_html=args.out_html if args.out_html else ("s2_merge_plotly.html" if args.metric_mode=="merge" else "s2_overlap_plotly.html")
    from plotly.offline import plot as plotly_save
    plotly_save(fig, filename=out_html, auto_open=False)
    print("Wrote:", out_html); print("Wrote:", args.out_csv)

if __name__=="__main__":
    main()
