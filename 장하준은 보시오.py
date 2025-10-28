"""
Unified 1D (y-axis) Silicon Photonic Modulator — Single File
C-Option (No GAN Teacher):
HF(1D numeric) → OperatorNet1D → PINN Correction (Poisson + Drift–Diffusion + Helmholtz)
Soft/Adaptive BC + RAR, Best-only checkpoints, Simple visualizations

Why 1D (y-axis)?
- Collapse the 3D problem to a vertical cross-section only. Geometry is layered along y.
- Treat the silicon core between 0..wg_height (μm); oxide above/below.
- Bias is applied between bottom/top (Dirichlet ±V/2). Doping is uniform in the silicon core.

Notes:
- Units: geometry in μm (converted to cm for physics), doping in cm^-3.
- Helmholtz is solved as a 1D inhomogeneous problem with a Gaussian source and Dirichlet boundaries.
- OperatorNet pre-trains on HF(1D) labels; PINN correction then enforces PDE/BC consistency.

Author: 1D-yaxis-patched
Date: 2025-10-28
"""

# ============== Imports & Setup ==============
import os, math, time, warnings, json
warnings.filterwarnings("ignore")
import numpy as np
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

# SciPy for sparse solves (HF stage)
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============== Config ==============
@dataclass
class Cfg:
    # physics (SI / CGS mix)
    q: float = 1.602e-19
    kb: float = 1.380649e-23
    T: float = 300.0
    eps0: float = 8.854e-12           # F/m
    Vt: float = (1.380649e-23*300.0)/1.602e-19

    # materials
    eps_si: float = 11.7*8.854e-12
    eps_sio2: float = 3.9*8.854e-12
    n_i: float = 1.45e10              # cm^-3
    mu_n: float = 0.135               # m^2/Vs (≈1350 cm^2/Vs)
    mu_p: float = 0.048               # m^2/Vs (≈480 cm^2/Vs)

    # Soref–Bennett @1550nm
    an: float = 8.8e-22               # cm^3
    ap: float = 8.5e-18               # cm^3
    gp: float = 0.8

    # optics
    wavelength_um: float = 1.55
    n_si_opt: float = 3.476
    n_sio2_opt: float = 1.444

    # geometry (μm)
    box_th: float = 2.0
    wg_height: float = 0.22
    clad_th: float = 2.0

    # doping (cm^-3) — uniform inside silicon
    ND: float = 5e18
    NA: float = 5e18

    # grid (1D)
    NY: int = 256

    # HF solver
    max_iter: int = 50
    relax: float = 0.6
    tol_phi: float = 5e-5

    # Helmholtz
    pml_cells: int = 6      # implemented as imag shift band
    pml_strength: float = 3.
    src_amp: float = 1.0

    # dataset
    out_dir: str = "hf1d_out"
    n_samples: int = 32
    V_range: tuple = (0.0, 5.0)
    seed: int = 42

    # training
    BATCH: int = 4
    EPOCH_OP: int = 8
    EPOCH_PINN: int = 12
    LR_OP: float = 2e-3
    LR_PINN: float = 1e-3

    # loss weights (PINN)
    W_H: float = 0.3
    W_DATA: float = 0.05
    W_DD: float = 1.0
    # BC
    W_BC_PHI_DIR: float = 1.0
    W_BC_DD_FLUX: float = 0.5
    W_BC_PSI: float = 0.2

CFG = Cfg()
np.random.seed(CFG.seed)

def y_grid(cfg: Cfg):
    # y in μm; total = box + core + clad
    Ly = cfg.box_th + cfg.wg_height + cfg.clad_th
    y = np.linspace(0.0, Ly, cfg.NY)
    dy = y[1]-y[0]
    return y, dy, Ly

# ============== HF 1D Solver (Poisson + Gummel + Helmholtz) ==============

def make_masks_1d(cfg: Cfg):
    y, dy, Ly = y_grid(cfg)
    core = ((y >= cfg.box_th) & (y <= cfg.box_th + cfg.wg_height)).astype(np.float64)
    eps = cfg.eps_sio2 + (cfg.eps_si - cfg.eps_sio2)*core
    n_base = cfg.n_sio2_opt + (cfg.n_si_opt - cfg.n_sio2_opt)*core
    return y, dy, Ly, core, eps, n_base


def assemble_poisson_1d(cfg: Cfg, eps):
    NY = cfg.NY; y, dy, Ly = y_grid(cfg)
    rows=[]; cols=[]; data=[]; b=np.zeros(NY, dtype=np.float64)
    def add(i,j,v): rows.append(i); cols.append(j); data.append(v)
    for j in range(NY):
        if j==0 or j==NY-1:  # Dirichlet φ(0)=-V/2, φ(L)=+V/2 -> set via b later
            add(j,j,1.0)
            continue
        eps_m = 0.5*(eps[j]+eps[j-1])
        eps_p = 0.5*(eps[j]+eps[j+1])
        cm = eps_m/(dy*dy); cp = eps_p/(dy*dy)
        add(j,j, cm+cp)
        add(j,j-1,-cm)
        add(j,j+1,-cp)
    A = coo_matrix((data,(rows,cols)), shape=(NY,NY)).tocsr()
    return A, b


def init_quasi_fermi(cfg: Cfg, V):
    # top (y=Ly) at +V/2 (n-contact), bottom (y=0) at -V/2 (p-contact)
    Fn = +V/2 - cfg.Vt*np.log(max(cfg.ND,1.0)/cfg.n_i)
    Fp = -V/2 + cfg.Vt*np.log(max(cfg.NA,1.0)/cfg.n_i)
    return Fn, Fp


def carriers_from_phi(cfg: Cfg, phi, Fn, Fp):
    n = cfg.n_i * np.exp((phi - Fn)/cfg.Vt)
    p = cfg.n_i * np.exp((Fp - phi)/cfg.Vt)
    return n, p


def poiss_rhs(cfg: Cfg, n, p, ND, NA):
    rho = cfg.q*(p - n + ND - NA)   # C/m^3 if densities in m^-3
    return -rho*1e6  # convert cm^-3 → m^-3 (×1e6)


def solve_poisson_1d(cfg: Cfg, eps, ND, NA, V):
    NY = cfg.NY; y, dy, Ly = y_grid(cfg)
    A, b = assemble_poisson_1d(cfg, eps)
    # Boundary φ
    b[:] = 0.0
    b[0] = -V/2; b[-1] = +V/2
    # init linear
    phi = np.linspace(-V/2, +V/2, NY).astype(np.float64)
    Fn,Fp = init_quasi_fermi(cfg, V)
    iters=0
    for it in range(cfg.max_iter):
        n,p = carriers_from_phi(cfg, phi, Fn, Fp)
        rhs = poiss_rhs(cfg, n, p, ND, NA)
        rhs[0] = -V/2; rhs[-1] = +V/2
        phi_new = spsolve(A, b - rhs)
        dphi = np.max(np.abs(phi_new-phi))
        phi = (1-cfg.relax)*phi + cfg.relax*phi_new
        iters=it+1
        if dphi < cfg.tol_phi: break
    return phi, n, p, iters


def soref_delta_n(cfg: Cfg, n_cm3, p_cm3):
    dN = np.maximum(n_cm3 - cfg.n_i, 0.0)
    dP = np.maximum(p_cm3 - cfg.n_i, 0.0)
    dn = -(cfg.an*dN + cfg.ap*(dP**cfg.gp))
    return dn


def assemble_helmholtz_1d(cfg: Cfg, n_tot):
    NY = cfg.NY; y, dy, Ly = y_grid(cfg)
    k0 = 2*math.pi/cfg.wavelength_um
    rows=[]; cols=[]; data=[]; rhs=np.zeros(NY, dtype=np.complex128)
    def add(i,j,v): rows.append(i); cols.append(j); data.append(v)
    for j in range(NY):
        if j==0 or j==NY-1:
            add(j,j,1.0+0j); rhs[j]=0.0
            continue
        diag = -2.0/(dy*dy) + (k0**2)*(n_tot[j]**2)
        add(j,j,diag)
        add(j,j-1,1.0/(dy*dy))
        add(j,j+1,1.0/(dy*dy))
    # Simple absorbing: imag shift in boundary bands
    for j in range(CFG.pml_cells):
        s = CFG.pml_strength*(1 - j/max(1,CFG.pml_cells))
        add(j,j, -1j*s)
        add(NY-1-j,NY-1-j, -1j*s)
    A = coo_matrix((data,(rows,cols)), shape=(NY,NY), dtype=np.complex128).tocsr()
    return A, rhs


def add_gaussian_source_1d(cfg: Cfg, rhs, center_idx, sigma_cells=2.0, amp=1.0):
    y = np.arange(cfg.NY)
    G = np.exp(-((y-center_idx)**2)/(2*sigma_cells**2))
    rhs[:] += amp*G


def make_sample_1d(cfg: Cfg, V):
    y, dy, Ly, core, eps, n_base = make_masks_1d(cfg)
    # doping (cm^-3): only inside silicon
    ND = cfg.ND*core
    NA = cfg.NA*core
    phi, n_cm3, p_cm3, iters = solve_poisson_1d(cfg, eps, ND, NA, V)
    dn = soref_delta_n(cfg, n_cm3, p_cm3)
    n_tot = np.clip(n_base + dn, 1.0, None)
    A, rhs = assemble_helmholtz_1d(cfg, n_tot)
    add_gaussian_source_1d(cfg, rhs, center_idx=max(2, CFG.pml_cells+1), sigma_cells=3.0, amp=CFG.src_amp)
    psi = spsolve(A, rhs)
    # Poisson residual check: d/dy(ε dφ/dy)+ρ
    dphi_dy = np.gradient(phi, dy)
    J = eps*dphi_dy
    divJ = np.gradient(J, dy)
    rho = CFG.q*(p_cm3 - n_cm3 + ND - NA)*1e6
    poisson_res = divJ + rho
    meta = dict(V=float(V), iters=int(iters), dy=float(dy), Ny=int(CFG.NY))
    sample = dict(
        inputs=dict(
            y_um=y.astype(np.float32),
            core=core.astype(np.float32),
            ND=ND.astype(np.float32),
            NA=NA.astype(np.float32),
            V=np.full_like(core, V, dtype=np.float32)
        ),
        labels=dict(
            phi=phi.astype(np.float32),
            n=n_cm3.astype(np.float32),
            p=p_cm3.astype(np.float32),
            dn=dn.astype(np.float32),
            n_tot=n_tot.astype(np.float32),
            psi=np.asarray(psi, dtype=np.complex64)
        ),
        checks=dict(poisson_res=poisson_res.astype(np.float64)),
        meta=meta
    )
    return sample


def save_npz_sample(path, sample):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out={}
    for k,v in sample['inputs'].items(): out[f"inputs/{k}"]=v
    for k,v in sample['labels'].items(): out[f"labels/{k}"]=v
    for k,v in sample['checks'].items(): out[f"checks/{k}"]=v
    out['meta/json']=np.string_(json.dumps(sample['meta']))
    np.savez_compressed(path, **out)

# ============== PyTorch Dataset (from HF 1D) ==============
class HF1DDataset(Dataset):
    def __init__(self, n_samples=CFG.n_samples, V_range=CFG.V_range):
        self.samples=[]
        for s in range(n_samples):
            V = float(np.random.uniform(*V_range))
            self.samples.append(make_sample_1d(CFG, V))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        S = self.samples[i]
        core = S['inputs']['core']
        V = S['inputs']['V']
        ND = S['inputs']['ND']
        NA = S['inputs']['NA']
        # x: [C, NY]
        x = np.stack([
            V/5.0,
            core,
            ND/1e19,
            NA/1e19
        ], axis=0).astype(np.float32)
        # y: [C, NY] — phi, logn, logp, |psi|
        phi = S['labels']['phi'].astype(np.float32)
        n   = np.clip(S['labels']['n'].astype(np.float32), 1e10, None)
        p   = np.clip(S['labels']['p'].astype(np.float32), 1e10, None)
        logn = np.log10(n); logp=np.log10(p)
        psi_mag = np.abs(S['labels']['psi']).astype(np.float32)
        y = np.stack([phi, logn, logp, psi_mag], axis=0)
        return torch.from_numpy(x), torch.from_numpy(y)

# ============== OperatorNet1D & CorrNet1D ==============
class OperatorNet1D(nn.Module):
    def __init__(self, in_c=4, out_c=4, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_c, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, out_c, 1)
        )
    def forward(self, x): return self.net(x)

class CorrNet1D(nn.Module):
    def __init__(self, in_c=8, out_c=4, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_c, width, 3, padding=1), nn.GELU(),
            nn.Conv1d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv1d(width, out_c, 1)
        )
    def forward(self, x): return self.net(x)

# ============== Finite-diff helpers (1D) ==============

def diff1(u, dy):
    # central differences, same length, Neumann at ends via replicate
    u = u.unsqueeze(0).unsqueeze(0) if u.dim()==1 else u
    pad = F.pad(u, (1,1), mode='replicate')
    du = (pad[:,:,2:] - pad[:,:,:-2])/(2*dy)
    return du.squeeze(0)

def lap1(u, dy):
    u = u.unsqueeze(0).unsqueeze(0) if u.dim()==1 else u
    pad = F.pad(u, (1,1), mode='replicate')
    d2 = (pad[:,:,2:] - 2*pad[:,:,1:-1] + pad[:,:,:-2])/(dy*dy)
    return d2.squeeze(0)

# ============== Physics Residuals (1D) ==============

def physics_residuals_1d(pred, x_in):
    """pred: [B,4,NY] = [phi, logn, logp, psi]; x_in: [B,4,NY] = [Vn, core, NDn, NAn]"""
    B,_,NY = pred.shape
    # recover variables
    phi = pred[:,0]
    n   = (10**pred[:,1]).clamp(min=1e10)
    p   = (10**pred[:,2]).clamp(min=1e10)
    psi = pred[:,3]
    core = x_in[:,1]
    # material
    eps = torch.from_numpy(CFG.eps_sio2*np.ones(NY)).to(pred.device).float()[None,:]
    eps = eps + (CFG.eps_si - CFG.eps_sio2)*core
    dy = (CFG.box_th + CFG.wg_height + CFG.clad_th)/ (CFG.NY-1)

    # Poisson: d/dy(ε dφ/dy) + ρ = 0,  ρ = q(p - n + ND - NA)
    dphidy = diff1(phi, dy)
    J = eps*dphidy
    divJ = diff1(J, dy)
    ND = x_in[:,2]*1e19*core
    NA = x_in[:,3]*1e19*core
    rho = CFG.q*(p - n + ND - NA)*1e6
    rP = divJ + rho

    # Drift–Diffusion: ∂y Jn = 0, ∂y Jp = 0
    dn_dy = diff1(n, dy); dp_dy = diff1(p, dy)
    Ex = -dphidy
    Jn = CFG.q*(CFG.mu_n*n*Ex + (CFG.mu_n*CFG.Vt)*dn_dy)  # Einstein D=μVt
    Jp = CFG.q*(CFG.mu_p*p*Ex - (CFG.mu_p*CFG.Vt)*dp_dy)
    rN = diff1(Jn, dy)
    rPp= diff1(Jp, dy)

    # Helmholtz: ψ'' + k0^2 n(y)^2 ψ = 0, with n(y)=n_base+Δn (Δn from (n,p) via Soref-Bennett)
    n_base = torch.from_numpy((CFG.n_sio2_opt + (CFG.n_si_opt - CFG.n_sio2_opt)*core.cpu().numpy())).to(pred.device).float()
    dN = (n - CFG.n_i).clamp(min=0)
    dP = (p - CFG.n_i).clamp(min=0)
    dn = -(CFG.an*dN + CFG.ap*(dP**CFG.gp))
    n_tot = (n_base + dn).clamp(min=1.0)
    k0 = 2*math.pi/CFG.wavelength_um
    lap_psi = lap1(psi, dy)
    rH = lap_psi + (k0**2)*(n_tot**2)*psi

    return rP, rN, rPp, rH

# ============== Training Loops ==============

def save_best(path, state, score, best):
    if score < best[0]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)
        best[0] = score


def train_operator():
    ds = HF1DDataset(CFG.n_samples, CFG.V_range)
    dl = DataLoader(ds, batch_size=CFG.BATCH, shuffle=True)
    net = OperatorNet1D().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG.LR_OP)
    best=[1e30]
    for ep in range(1, CFG.EPOCH_OP+1):
        net.train(); tl=0
        for x,y in dl:
            x=x.to(DEVICE); y=y.to(DEVICE)
            pred = net(x)
            loss = (pred - y).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()
        tl/=len(dl)
        print(f"[OP] ep{ep}: L1={tl:.4f}")
        save_best('checkpoints/op1d_best.pt', {'model':net.state_dict()}, tl, best)
    return net


def train_pinn(op_net):
    for p in op_net.parameters(): p.requires_grad=False
    corr = CorrNet1D().to(DEVICE)
    opt = torch.optim.AdamW(corr.parameters(), lr=CFG.LR_PINN)
    ds = HF1DDataset(CFG.n_samples, CFG.V_range)
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    best=[1e30]
    for ep in range(1, CFG.EPOCH_PINN+1):
        corr.train(); tphys=0; ttot=0
        for x,y in dl:
            x=x.to(DEVICE); y=y.to(DEVICE)
            with torch.no_grad():
                op = op_net(x)
            out = op + corr(torch.cat([x,op], dim=1))
            # physics residuals
            rP, rN, rPp, rH = physics_residuals_1d(out, x)
            # RAR weighting by mixed residual
            with torch.no_grad():
                r = 0.5*rP.abs() + 0.5*rH.abs()
                w = 1.0 + 9.0*(r >= torch.quantile(r, 0.98)).float()  # top 2%
                w = w * (w.numel()/w.sum().clamp(min=1e-6))
            LP = (w*(rP**2)).mean()
            LDD = (w*(rN**2 + rPp**2)).mean()
            LH = (w*(rH**2)).mean()
            data_l1 = (out - y).abs().mean()
            # BCs: Dirichlet for φ, zero-flux for Jn/Jp, absorbing band for ψ (Dirichlet enforced weakly)
            Vabs = x[:,0]*5.0
            phi = out[:,0]
            psi = out[:,3]
            dy = (CFG.box_th + CFG.wg_height + CFG.clad_th)/ (CFG.NY-1)
            # Dirichlet φ(0)=-V/2, φ(L)=+V/2
            bc_phi = ((phi[:,0] + 0.5*Vabs[:,0])**2 + (phi[:,-1] - 0.5*Vabs[:,-1])**2).mean()
            # No-flux Jn/Jp at ends
            n = (10**out[:,1]).clamp(min=1e10); p = (10**out[:,2]).clamp(min=1e10)
            dphidy = diff1(phi, dy)
            Ex = -dphidy
            dn_dy = diff1(n, dy); dp_dy = diff1(p, dy)
            Jn = CFG.q*(CFG.mu_n*n*Ex + (CFG.mu_n*CFG.Vt)*dn_dy)
            Jp = CFG.q*(CFG.mu_p*p*Ex - (CFG.mu_p*CFG.Vt)*dp_dy)
            bc_flux = (Jn[:,0]**2 + Jn[:,-1]**2 + Jp[:,0]**2 + Jp[:,-1]**2).mean()
            # ψ boundary suppression (Dirichlet-like)
            band = 4
            wL = torch.arange(CFG.NY, device=DEVICE).float()
            mask = ((wL<band) | (wL>CFG.NY-1-band)).float()[None,:]
            bc_psi = (mask*psi.abs()).pow(2).mean()

            loss = LP + CFG.W_DD*LDD + CFG.W_H*LH + CFG.W_DATA*data_l1 \
                 + CFG.W_BC_PHI_DIR*bc_phi + CFG.W_BC_DD_FLUX*bc_flux + CFG.W_BC_PSI*bc_psi

            opt.zero_grad(); loss.backward(); clip_grad_norm_(corr.parameters(), 1.0); opt.step()
            tphys += (LP + CFG.W_DD*LDD + CFG.W_H*LH).item(); ttot += loss.item()
        print(f"[PINN] ep{ep}: phys={tphys/len(dl):.4e} total={ttot/len(dl):.4e}")
        save_best('checkpoints/pinn1d_best.pt', {'corr':corr.state_dict()}, tphys/len(dl), best)
    return corr

# ============== Visualization ==============

def visualize_1d(op, corr):
    op.eval(); corr.eval()
    with torch.no_grad():
        # single evaluation at V=5
        core = make_sample_1d(CFG, V=5.0)['inputs']['core'].astype(np.float32)
        ND = (CFG.ND*core).astype(np.float32)
        NA = (CFG.NA*core).astype(np.float32)
        V = np.full_like(core, 5.0, dtype=np.float32)
        x = np.stack([V/5.0, core, ND/1e19, NA/1e19], axis=0)
        x_t = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
        out = op(x_t) + corr(torch.cat([x_t, op(x_t)], dim=1))
        phi, logn, logp, psi = [t.squeeze(0).cpu().numpy() for t in out]
    y, dy, Ly = y_grid(CFG)
    fig,axs=plt.subplots(2,2,figsize=(10,6))
    axs=axs.ravel()
    axs[0].plot(y, phi); axs[0].set_title('φ(y) [V]'); axs[0].set_xlabel('y [μm]')
    axs[1].plot(y, logn, label='log10 n'); axs[1].plot(y, logp, label='log10 p'); axs[1].legend(); axs[1].set_title('Carriers')
    axs[2].plot(y, psi); axs[2].set_title('ψ(y) (arb)')
    axs[3].plot(y, core, label='core'); axs[3].set_ylim(-0.1,1.1); axs[3].legend(); axs[3].set_title('Core mask')
    plt.tight_layout(); os.makedirs('figs', exist_ok=True); plt.savefig('figs/unified1d_viz.png', dpi=160)
    print('Saved figs/unified1d_viz.png')

# ============== Main ==============
if __name__=='__main__':
    os.makedirs('checkpoints', exist_ok=True)
    print('== HF(1D) pre-generation (in-memory) ==')
    # (Dataset internally runs HF solver per-sample)
    print(asdict(CFG))

    print('== Train OperatorNet1D ==')
    op = train_operator()
    # load best
    if os.path.exists('checkpoints/op1d_best.pt'):
        op.load_state_dict(torch.load('checkpoints/op1d_best.pt')['model'])

    print('== Train PINN Correction (1D Poisson+DD+Helmholtz + BC + RAR) ==')
    corr = train_pinn(op)
    if os.path.exists('checkpoints/pinn1d_best.pt'):
        corr.load_state_dict(torch.load('checkpoints/pinn1d_best.pt')['corr'])

    print('== Visualize ==')
    visualize_1d(op, corr)
