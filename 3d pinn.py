"""
Unified 3D Photonic Modulator — Single File (Patched + Res-SIREN PINN)
HF(analytic) → cGAN Teacher (UNet3D + SE + FiLM, WGAN-GP) → SA-PINN Student (Res-SIREN)
Full 3D physics (Poisson + Drift–Diffusion + Helmholtz) + Soft/Adaptive BC + RAR

Key patches (from feedback):
1) Teacher grad loss re-scaled by physical grid spacing (dx, dy, dz)
2) Analytic HF labels smoothed (tanh) at depletion & junction to avoid gradient blow-ups
3) Optional Teacher boundary Neumann penalty (∂φ/∂n=0) for better consistency with PINN
4) Training voltage range extended to [0, 5] V (avoid extrapolation for V=5 visualizations)
5) PINN boundary handling upgraded: boundary sampling schedule, boundary RAR, data masking near boundary
6) DD residual re-normalized by characteristic length; Helmholtz warm-up scheduling
7) SA-PINN lambda init tuned (stronger BC at start) + faster lambda optimizer
8) "Best-only" checkpoint saving for Teacher (G) and PINN

Additional architecture upgrades (applied here):
9) PINN backbone → Res-SIREN (shallow, wider, residual skips), w0_first=30, w0_inner=15
10) PINN heads split: electrical head [φ, log n, log p, Δn, α] and optical head [ψ]
11) Optional AMP + gradient checkpointing (safe-by-default off)

Author: unified-patched
Date: 2025-10-27
"""

# ============== Safety / Determinism ==============
import warnings, math, random, os
warnings.filterwarnings("ignore")
import numpy as np
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import spectral_norm as SN
from torch.nn.utils import clip_grad_norm_
from torch.utils.checkpoint import checkpoint_sequential
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import Normalize
from matplotlib.cm import RdBu_r

# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = False

SEED = 1234
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✓ Using device: {device}")

# ==============================
# 0) Simple runtime config
# ==============================
class Cfg:
    use_amp: bool = False            # AMP for forward/loss (second-order grads can be finicky; off by default)
    ckpt_segments: int = 0           # gradient checkpointing over Res-SIREN trunk (0 = off)
    add_fourier_feats: bool = False  # add low-order Fourier features to coords (safe default: off)
    fourier_bands: Tuple[int,...] = (1, 2, 4)  # used when add_fourier_feats=True

cfg = Cfg()

# ==============================
# 1) Physical parameters
# ==============================
class Params:
    # fundamentals
    q = 1.602e-19
    kb = 1.380649e-23
    T = 300.0
    Vt = kb*T/q
    eps0 = 8.854e-14  # F/cm

    # material
    eps_si = 11.7
    eps_sio2 = 3.9
    n_si = 3.476
    n_sio2 = 1.444
    n_air = 1.0

    # semiconductor (units: cm^-3)
    ni = 1.45e10
    NA = 5e18
    ND = 5e18

    # geometry (units: μm)
    wg_width  = 0.5
    wg_height = 0.22
    wg_length = 10.0

    # optics
    wavelength = 1.55  # μm
    k0_um = 2*math.pi/wavelength  # 1/μm
    k0 = k0_um * 1e4  # 1/cm

    # Soref-Bennett @1550nm (units: cm^3)
    an = 8.8e-22
    ap = 8.5e-18
    bn = 8.5e-18
    bp = 6.0e-18

    # mobilities (cm^2/Vs) and Einstein (cm^2/s)
    mu_n = 800.0
    mu_p = 250.0
    Dn = mu_n*Vt
    Dp = mu_p*Vt

params = Params()

# Convert geometry to cm for internal calculations
params.wg_width_cm = params.wg_width * 1e-4
params.wg_height_cm = params.wg_height * 1e-4
params.wg_length_cm = params.wg_length * 1e-4

# ==============================
# 2) HF analytic 3D generator (Using cm units internally) + smoothing
# ==============================

def smooth_heaviside(x: np.ndarray, w: float):
    w = max(float(w), 1e-12)
    return 0.5*(1.0 + np.tanh(x/(w)))

def smooth_tanh(x: np.ndarray, w: float):
    w = max(float(w), 1e-12)
    return np.tanh(x/w)

class HFDataGenerator:
    def __init__(self, nx=32, ny=16, nz=32):
        self.nx, self.ny, self.nz = nx, ny, nz
        # --- cm grid (internal calc) ---
        self.x_cm = np.linspace(-params.wg_width_cm/2, params.wg_width_cm/2, nx)
        self.y_cm = np.linspace(0, params.wg_height_cm, ny)
        self.z_cm = np.linspace(0, params.wg_length_cm, nz)
        self.X_cm, self.Y_cm, self.Z_cm = np.meshgrid(self.x_cm, self.y_cm, self.z_cm, indexing='ij')
        # --- μm grid (for reference/plots) ---
        self.x_um = np.linspace(-params.wg_width/2, params.wg_width/2, nx)
        self.y_um = np.linspace(0, params.wg_height, ny)
        self.z_um = np.linspace(0, params.wg_length, nz)
        self.X_um, self.Y_um, self.Z_um = np.meshgrid(self.x_um, self.y_um, self.z_um, indexing='ij')
        print(f"✓ HF grid (cm): {nx}×{ny}×{nz}")

    def label(self, V: float, NA=None, ND=None):
        NA = NA or params.NA  # cm^-3
        ND = ND or params.ND  # cm^-3
        X = self.X_cm; Y = self.Y_cm; Z = self.Z_cm  # cm grid

        # depletion widths (simple, cm units)
        Vbi = params.Vt*np.log(NA*ND/(params.ni**2))
        Vt_eff = max(Vbi - V, 1e-6)
        W = np.sqrt(2*params.eps_si*params.eps0*Vt_eff/params.q*(1/NA+1/ND))
        W = float(np.clip(W, 1e-11, 0.3*params.wg_width_cm))  # cm

        # Smooth region indicators
        Hn = smooth_heaviside(X - W/2, W*0.5)
        Hp = smooth_heaviside(-X - W/2, W*0.5)
        Hj = smooth_heaviside(-np.abs(X) + W/2, W*0.5)  # junction vicinity
        mid = np.clip(1.0 - (Hn + Hp), 0.0, 1.0)

        # Carrier densities in cm^-3 (smoothed)
        n = Hn*ND + Hp*(params.ni**2/NA) + mid*(0.05*ND)
        p = Hp*NA + Hn*(params.ni**2/ND) + mid*(0.05*NA)

        # forward injection (smoothed via Hj to avoid abrupt global scaling)
        if V > 0:
            inj = np.exp(np.clip(V/params.Vt, 0, 40))
            n = n*(1.0 + Hj*(inj-1.0)*(X < -W/2))
            p = p*(1.0 + Hj*(inj-1.0)*(X >  W/2))

        # core mask (cm)
        core = ((np.abs(X) <= params.wg_width_cm/2) & (Y >= 0) & (Y <= params.wg_height_cm)).astype(np.float32)
        n = n*core + params.ni*(1-core)
        p = p*core + params.ni*(1-core)

        # Soref-Bennett
        dN = np.maximum(n-params.ni, 0)
        dP = np.maximum(p-params.ni, 0)
        dn = -(params.an*dN + params.ap*(dP**0.8))          # Δn (dimensionless)
        da = (params.bn*dN + params.bp*dP)                  # α [cm^-1]

        # Potential (smoothed tanh instead of sign)
        phi = 0.5*V*smooth_tanh(X, W)*core  # V

        mask = core > 0.5
        dn_avg = dn[mask].mean() if mask.any() else dn.mean()
        delta_n_eff = 0.7*dn_avg
        phase = params.k0*delta_n_eff*params.wg_length_cm
        phase_deg = np.degrees(phase)
        loss_dB = (da[mask].mean() if mask.any() else da.mean())*params.wg_length_cm * 4.343
        Vpi = (params.wavelength*V)/(2*abs(delta_n_eff)) if abs(delta_n_eff) > 1e-10 else np.inf
        VpiL = Vpi*params.wg_length_cm if np.isfinite(Vpi) else np.inf

        return {
            'phi_vol': phi.astype(np.float32),
            'n_vol':   n.astype(np.float32),
            'p_vol':   p.astype(np.float32),
            'dn_vol':  dn.astype(np.float32),
            'da_vol':  da.astype(np.float32),
            'scalars': {'delta_n_eff': delta_n_eff, 'phase_deg': phase_deg,
                        'loss_dB': loss_dB, 'V_pi': Vpi, 'V_pi_L': VpiL},
            # μm reference grids
            'X_um': self.X_um.astype(np.float32),
            'Y_um': self.Y_um.astype(np.float32),
            'Z_um': self.Z_um.astype(np.float32),
            'voltage': float(V), 'NA': float(NA), 'ND': float(ND)
        }

class HFDataset(Dataset):
    def __init__(self, n_samples=100, grid=(32,16,32), v_max=5.0):
        nx, ny, nz = grid
        self.gen = HFDataGenerator(nx, ny, nz)
        self.samples = []
        self.v_max = float(v_max)
        print(f"Generating {n_samples} HF samples...")
        for _ in tqdm(range(n_samples)):
            V = float(np.random.uniform(0.0, self.v_max))
            NA = 10**np.random.uniform(17.5, 19.0)  # cm^-3
            ND = 10**np.random.uniform(17.5, 19.0)  # cm^-3
            self.samples.append(self.gen.label(V, NA, ND))
        print("✓ HF dataset ready")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        L = self.samples[i]
        phi = torch.tensor(L['phi_vol'])
        logn = torch.log10(torch.clamp(torch.tensor(L['n_vol']), 1e1, 1e25))
        logp = torch.log10(torch.clamp(torch.tensor(L['p_vol']), 1e1, 1e25))
        dn  = torch.tensor(L['dn_vol'])
        da  = torch.tensor(L['da_vol'])
        y = torch.stack([phi, logn, logp, dn, da], dim=0)  # [5,D,H,W]
        cond = torch.tensor([L['voltage'], L['NA'], L['ND']], dtype=torch.float32)
        return {'y': y.float(), 'cond': cond.float()}

# ==============================
# 3) Condition builder 3D (Using cm units internally)
# ==============================
class ConditionBuilder3D:
    def __init__(self, nx, ny, nz, device):
        self.nx, self.ny, self.nz = nx, ny, nz
        x_cm = np.linspace(-params.wg_width_cm/2, params.wg_width_cm/2, nx)
        y_cm = np.linspace(0, params.wg_height_cm, ny)
        z_cm = np.linspace(0, params.wg_length_cm, nz)
        X_cm, Y_cm, Z_cm = np.meshgrid(x_cm, y_cm, z_cm, indexing='ij')
        self.X_cm = torch.tensor(X_cm, dtype=torch.float32, device=device)
        self.Y_cm = torch.tensor(Y_cm, dtype=torch.float32, device=device)
        self.Z_cm = torch.tensor(Z_cm, dtype=torch.float32, device=device)
        core = ((torch.abs(self.X_cm) <= params.wg_width_cm/2) &
                (self.Y_cm >= 0) & (self.Y_cm <= params.wg_height_cm)).float()
        self.core = core

    def grid_cond(self, V, NA, ND, v_max=5.0):
        Vn  = V/float(v_max)
        NAn = (math.log10(NA)-18.0)/1.5
        NDn = (math.log10(ND)-18.0)/1.5
        v  = torch.full_like(self.X_cm, Vn)
        na = torch.full_like(self.X_cm, NAn)
        nd = torch.full_like(self.X_cm, NDn)
        xnorm = self.X_cm/(params.wg_width_cm/2)
        ynorm = self.Y_cm/(params.wg_height_cm/2) - 1.0
        znorm = self.Z_cm/(params.wg_length_cm/2) - 1.0
        return torch.stack([v, na, nd, self.core, xnorm, ynorm, znorm], dim=0)  # [7,D,H,W]

    def style_vec(self, V, NA, ND, v_max=5.0):
        return torch.tensor(
            [V/float(v_max), (math.log10(NA)-18.0)/1.5, (math.log10(ND)-18.0)/1.5],
            dtype=torch.float32, device=self.X_cm.device
        ).unsqueeze(0)  # [1,3]

# ==============================
# 4) UNet3D with SE + FiLM (Teacher G)
# ==============================
class SE3D(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, max(4, c//r)), nn.ReLU(False),
            nn.Linear(max(4, c//r), c), nn.Sigmoid()
        )
    def forward(self, x):
        b,c,_,_,_ = x.shape
        s = self.avg(x).view(b,c)
        w = self.fc(s).view(b,c,1,1,1)
        return x*w

class FiLMLayer(nn.Module):
    def __init__(self, c, style_dim=64):
        super().__init__()
        self.to_gamma = nn.Linear(style_dim, c)
        self.to_beta  = nn.Linear(style_dim, c)
    def forward(self, x, s):
        gamma = self.to_gamma(s).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta  = self.to_beta(s).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x*(1+gamma) + beta

def conv_block(in_c, out_c, style_dim=None):
    layers = [
        nn.Conv3d(in_c, out_c, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, out_c), out_c),
        nn.LeakyReLU(0.2, inplace=False),
        nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, out_c), out_c),
        nn.LeakyReLU(0.2, inplace=False),
        SE3D(out_c),
    ]
    blk = nn.Sequential(*layers)
    film = FiLMLayer(out_c, style_dim) if style_dim is not None else None
    return blk, film

class UNet3D_G(nn.Module):
    def __init__(self, zc=8, cond_c=7, out_c=5, style_dim=64, base_c=32):
        super().__init__()
        self.style_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(False),
            nn.Linear(64, style_dim), nn.ReLU(False)
        )
        c1, c2, c3, c4 = base_c, base_c*2, base_c*4, base_c*8
        # enc
        self.e1, self.f1 = conv_block(zc+cond_c, c1, style_dim)
        self.e2, self.f2 = conv_block(c1, c2, style_dim)
        self.e3, self.f3 = conv_block(c2, c3, style_dim)
        self.pool = nn.MaxPool3d(2)
        # bottleneck
        self.b , self.fb = conv_block(c3, c4, style_dim)
        # dec
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3, self.fd3 = conv_block(c4+c3, c3, style_dim)
        self.d2, self.fd2 = conv_block(c3+c2, c2, style_dim)
        self.d1, self.fd1 = conv_block(c2+c1, c1, style_dim)
        self.out = nn.Conv3d(c1, out_c, 1)

    def apply_film(self, x, film, s):
        return film(x, s) if film is not None else x

    def forward(self, z, cond_grid, style_vec):
        s = self.style_mlp(style_vec)  # [B,style_dim]
        x = torch.cat([z, cond_grid], dim=1)
        e1 = self.apply_film(self.e1(x), self.f1, s)
        p1 = self.pool(e1)
        e2 = self.apply_film(self.e2(p1), self.f2, s)
        p2 = self.pool(e2)
        e3 = self.apply_film(self.e3(p2), self.f3, s)
        p3 = self.pool(e3)
        b  = self.apply_film(self.b(p3),  self.fb, s)
        u3 = self.up(b)
        if u3.shape[2:]!= e3.shape[2:]:
            u3 = F.interpolate(u3, size=e3.shape[2:], mode='trilinear', align_corners=True)
        d3 = self.apply_film(self.d3(torch.cat([u3,e3],dim=1)), self.fd3, s)
        u2 = self.up(d3)
        if u2.shape[2:]!= e2.shape[2:]:
            u2 = F.interpolate(u2, size=e2.shape[2:], mode='trilinear', align_corners=True)
        d2 = self.apply_film(self.d2(torch.cat([u2,e2],dim=1)), self.fd2, s)
        u1 = self.up(d2)
        if u1.shape[2:]!= e1.shape[2:]:
            u1 = F.interpolate(u1, size=e1.shape[2:], mode='trilinear', align_corners=True)
        d1 = self.apply_film(self.d1(torch.cat([u1,e1],dim=1)), self.fd1, s)
        return self.out(d1)  # [B,5,D,H,W]

class PatchD3D(nn.Module):
    """SpectralNorm Discriminator"""
    def __init__(self, in_c=5, cond_c=7, base=32):
        super().__init__()
        gc = lambda c: min(8, c)
        self.c1 = nn.Sequential(SN(nn.Conv3d(in_c+cond_c, base, 4, 2, 1)), nn.LeakyReLU(0.2, False))
        self.c2 = nn.Sequential(SN(nn.Conv3d(base, base*2, 4, 2, 1)), nn.GroupNorm(gc(base*2), base*2), nn.LeakyReLU(0.2, False))
        self.c3 = nn.Sequential(SN(nn.Conv3d(base*2, base*4, 4, 2, 1)), nn.GroupNorm(gc(base*4), base*4), nn.LeakyReLU(0.2, False))
        self.c4 = nn.Sequential(SN(nn.Conv3d(base*4, base*8, 3, 1, 1)), nn.GroupNorm(gc(base*8), base*8), nn.LeakyReLU(0.2, False))
        self.out = SN(nn.Conv3d(base*8, 1, 3, 1, 1))

    def forward(self, y, cond):
        x = torch.cat([y, cond], dim=1)
        x = self.c1(x); x = self.c2(x); x = self.c3(x); x = self.c4(x)
        return self.out(x)

# ==============================
# 5) 3D finite differences helpers for volumes
# ==============================

def grad_central_3d(t, dx, dy, dz):
    """Central-diff gradients on common interior grid."""
    B, C, D, H, W = t.shape
    inner = t[:, :, 1:-1, 1:-1, 1:-1]
    def zeros_like_inner(): return torch.zeros_like(inner)
    if D <= 2 or H <= 2 or W <= 2:
        return zeros_like_inner(), zeros_like_inner(), zeros_like_inner()
    gx = (t[:, :, 2:, 1:-1, 1:-1] - t[:, :, :-2, 1:-1, 1:-1]) / (2*dx + 1e-12)
    gy = (t[:, :, 1:-1, 2:, 1:-1] - t[:, :, 1:-1, :-2, 1:-1]) / (2*dy + 1e-12)
    gz = (t[:, :, 1:-1, 1:-1, 2:] - t[:, :, 1:-1, 1:-1, :-2]) / (2*dz + 1e-12)
    return gx, gy, gz


def laplacian_central_3d(t, dx, dy, dz):
    """Central-diff 3D Laplacian on interior grid (D,H,W >= 3)."""
    B, C, D, H, W = t.shape
    assert D >= 3 and H >= 3 and W >= 3, "Need D,H,W >= 3 for Laplacian."
    d2x = (t[:, :, 2:, 1:-1, 1:-1] - 2*t[:, :, 1:-1, 1:-1, 1:-1] + t[:, :, :-2, 1:-1, 1:-1]) / (dx*dx + 1e-12)
    d2y = (t[:, :, 1:-1, 2:, 1:-1] - 2*t[:, :, 1:-1, 1:-1, 1:-1] + t[:, :, 1:-1, :-2, 1:-1]) / (dy*dy + 1e-12)
    d2z = (t[:, :, 1:-1, 1:-1, 2:] - 2*t[:, :, 1:-1, 1:-1, 1:-1] + t[:, :, 1:-1, 1:-1, :-2]) / (dz*dz + 1e-12)
    return d2x + d2y + d2z

# ==============================
# 6) Teacher (WGAN-GP) trainer — STABILIZED + boundary penalty + best-save
# ==============================
class CGANTeacher:
    def __init__(self, grid=(32,16,32), device='cpu', v_max=5.0, save_best_path: Optional[str] = 'teacher_best.pt'):
        nx, ny, nz = grid
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dev = device
        self.v_max = float(v_max)
        self.G = UNet3D_G(cond_c=7).to(device)
        self.D = PatchD3D(in_c=5, cond_c=7).to(device)

        # TTUR
        self.optG = torch.optim.Adam(self.G.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=4e-4, betas=(0.5, 0.9))

        self.cb = ConditionBuilder3D(nx, ny, nz, device=device)

        # WGAN-GP parameters
        self.lambda_gp = 10.0
        self.n_critic = 5

        self.lambda_L1 = 20.0
        self.lambda_grad = 5.0
        self.lambda_phys = 0.5
        self.lambda_bc   = 0.5  # optional boundary Neumann penalty weight

        self.best_score = float('inf')
        self.save_best_path = save_best_path
        print(f"✓ Teacher init (WGAN-GP Full 3D): {nx}×{ny}×{nz}")

    def _channelwise_l1(self, y_fake, y_real, eps=1e-6):
        diff  = y_fake - y_real
        std   = y_real.flatten(2).std(dim=2, unbiased=False).clamp_min(eps)
        diff  = diff / std.view(std.size(0), std.size(1), 1, 1, 1)
        return diff.abs().mean()

    def _physics_penalty_3d(self, y, cond):
        B,_,D,H,W = y.shape
        if D < 3 or H < 3 or W < 3:
            return torch.tensor(0.0, device=y.device)

        phi = y[:,0:1]
        n   = (10.0**y[:,1:2]).clamp(params.ni, 1e25)
        p   = (10.0**y[:,2:3]).clamp(params.ni, 1e25)
        core = cond[:,3:4]

        dx = params.wg_width_cm / max(D-1, 1)
        dy = params.wg_height_cm / max(H-1, 1)
        dz = params.wg_length_cm / max(W-1, 1)

        lap = laplacian_central_3d(phi, dx, dy, dz)

        def crop_inner(t): return t[:, :, 1:-1, 1:-1, 1:-1]
        core_i = crop_inner(core)
        n_i = crop_inner(n)
        p_i = crop_inner(p)

        X_cm = self.cb.X_cm.view(1,1,D,H,W)
        is_n = (X_cm >= 0).float()
        is_p = 1.0 - is_n
        doping = core * (is_n * params.ND - is_p * params.NA)
        doping_i = crop_inner(doping)

        eps_eff = params.eps_si * params.eps0
        rho_i = params.q * (p_i - n_i + doping_i)
        res = eps_eff * lap + rho_i

        L_poisson = (res / (params.q * params.NA + 1e-12))**2
        L_poisson = L_poisson.mean()

        is_n_i = crop_inner(is_n)
        is_p_i = 1.0 - is_n_i
        L_bulk_p = ((is_p_i * core_i * (p_i - params.NA)) / (params.NA + 1e-12))**2
        L_bulk_n = ((is_n_i * core_i * (n_i - params.ND)) / (params.ND + 1e-12))**2
        L_bulk = (L_bulk_p + L_bulk_n).mean()

        return L_poisson + L_bulk

    def _boundary_neumann_phi(self, y, cond):
        """Compute simple ∂φ/∂n=0 penalty on 6 faces using 1st-order differences."""
        phi = y[:,0:1]
        _,_,D,H,W = phi.shape
        if D<3 or H<3 or W<3:
            return torch.tensor(0.0, device=phi.device)
        dx = params.wg_width_cm / max(D-1, 1)
        dy = params.wg_height_cm / max(H-1, 1)
        dz = params.wg_length_cm / max(W-1, 1)
        # x-faces
        dphi_dx_left  = (phi[:,:,1:2,1:-1,1:-1] - phi[:,:,0:1,1:-1,1:-1])/(dx)
        dphi_dx_right = (phi[:,:, -1:,1:-1,1:-1] - phi[:,:, -2:-1,1:-1,1:-1])/(dx)
        # y-faces
        dphi_dy_bot = (phi[:,:,1:-1,1:2,1:-1] - phi[:,:,1:-1,0:1,1:-1])/(dy)
        dphi_dy_top = (phi[:,:,1:-1, -1:,1:-1] - phi[:,:,1:-1, -2:-1,1:-1])/(dy)
        # z-faces
        dphi_dz_front = (phi[:,:,1:-1,1:-1,1:2] - phi[:,:,1:-1,1:-1,0:1])/(dz)
        dphi_dz_back  = (phi[:,:,1:-1,1:-1, -1:] - phi[:,:,1:-1,1:-1, -2:-1])/(dz)
        L = (dphi_dx_left.pow(2).mean() + dphi_dx_right.pow(2).mean() +
             dphi_dy_bot.pow(2).mean()  + dphi_dy_top.pow(2).mean()  +
             dphi_dz_front.pow(2).mean()+ dphi_dz_back.pow(2).mean())/6.0
        return L

    def _gradient_penalty(self, y_real, y_fake, cond):
        B = y_real.shape[0]
        alpha = torch.rand(B, 1, 1, 1, 1, device=self.dev)
        inter = (alpha * y_real + (1 - alpha) * y_fake).requires_grad_(True)
        d_inter = self.D(inter, cond)
        grad_outputs = torch.ones_like(d_inter, requires_grad=False)
        gradients = torch.autograd.grad(
            outputs=d_inter, inputs=inter, grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.reshape(B, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self, ds:Dataset, epochs=3, batch=2, grad_clip=1.0):
        loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
        step = 0
        for ep in range(1, epochs+1):
            pbar = tqdm(loader, desc=f"Teacher Epoch {ep}/{epochs}")
            running_score = 0.0; n_steps = 0
            for bt in pbar:
                step += 1
                warm = min(1.0, step/500.0)

                y_real = bt['y'].to(self.dev)   # [B,5,D,H,W]
                cond_pack = bt['cond'].to(self.dev)  # [B,3]
                B = y_real.size(0)

                conds, styles = [], []
                for i in range(B):
                    V = float(cond_pack[i,0].item())
                    NA = float(cond_pack[i,1].item())
                    ND = float(cond_pack[i,2].item())
                    conds.append(self.cb.grid_cond(V, NA, ND, v_max=5.0))
                    styles.append(self.cb.style_vec(V, NA, ND, v_max=5.0))
                cond = torch.stack(conds, dim=0)           # [B,7,D,H,W]
                style = torch.cat(styles, dim=0)           # [B,3]

                _, _, D, H, W = y_real.shape
                dx = params.wg_width_cm / max(D-1, 1)
                dy = params.wg_height_cm / max(H-1, 1)
                dz = params.wg_length_cm / max(W-1, 1)

                # --- D step (Critic) ---
                for _ in range(self.n_critic):
                    self.optD.zero_grad()
                    z = torch.randn(B, 8, self.nx, self.ny, self.nz, device=self.dev)
                    with torch.no_grad():
                        y_fake = self.G(z, cond, style)

                    real_logit = self.D(y_real, cond)
                    fake_logit = self.D(y_fake, cond)
                    gp = self._gradient_penalty(y_real, y_fake, cond)
                    lossD = -(torch.mean(real_logit) - torch.mean(fake_logit)) + self.lambda_gp * gp
                    lossD.backward()
                    if grad_clip > 0: clip_grad_norm_(self.D.parameters(), grad_clip)
                    self.optD.step()

                # --- G step ---
                self.optG.zero_grad()
                z = torch.randn(B, 8, self.nx, self.ny, self.nz, device=self.dev)
                y_fake = self.G(z, cond, style)
                fake_logit = self.D(y_fake, cond)
                adv_loss = -torch.mean(fake_logit)

                l1_loss = self._channelwise_l1(y_fake, y_real)

                # PATCH (A): gradient loss with physical scaling
                gxf, gyf, gzf = grad_central_3d(y_fake, dx, dy, dz)
                gxr, gyr, gzr = grad_central_3d(y_real, dx, dy, dz)
                grad_loss = (
                    F.l1_loss(gxf*dx, gxr*dx) +
                    F.l1_loss(gyf*dy, gyr*dy) +
                    F.l1_loss(gzf*dz, gzr*dz)
                ) / 3.0

                phys_loss = self._physics_penalty_3d(y_fake, cond)
                bc_loss   = self._boundary_neumann_phi(y_fake, cond)

                lossG = (adv_loss + self.lambda_L1*l1_loss + self.lambda_grad*grad_loss +
                         (self.lambda_phys*warm)*phys_loss + (self.lambda_bc*warm)*bc_loss)
                lossG.backward()
                if grad_clip > 0: clip_grad_norm_(self.G.parameters(), grad_clip)
                self.optG.step()

                # A simple score for best checkpoint (physics + L1 + grad)
                score = (self.lambda_L1*l1_loss + self.lambda_grad*grad_loss + self.lambda_phys*phys_loss).item()
                running_score += score; n_steps += 1
                avg_score = running_score / n_steps
                if avg_score < self.best_score and self.save_best_path:
                    self.best_score = avg_score
                    torch.save({'G': self.G.state_dict(), 'meta': {'avg_score': avg_score, 'step': step}}, self.save_best_path)

                pbar.set_postfix(D=f"{lossD.item():.3f}", G_adv=f"{adv_loss.item():.3f}",
                                 L1=f"{l1_loss.item():.3f}", Grad=f"{grad_loss.item():.3f}",
                                 Phys=f"{phys_loss.item():.3f}", BC=f"{bc_loss.item():.3f}", Warm=f"{warm:.2f}")

    @torch.no_grad()
    def generate(self, V, NA=None, ND=None):
        NA = NA or params.NA; ND = ND or params.ND
        self.G.eval()
        cond = self.cb.grid_cond(V, NA, ND, v_max=5.0).unsqueeze(0)
        style = self.cb.style_vec(V, NA, ND, v_max=5.0)
        z = torch.randn(1, 8, self.nx, self.ny, self.nz, device=self.dev)
        y = self.G(z, cond, style)  # [1,5,D,H,W]
        phi = y[0,0].cpu().numpy()
        n   = torch.pow(10.0, y[0,1]).clamp_min(params.ni).cpu().numpy()
        p   = torch.pow(10.0, y[0,2]).clamp_min(params.ni).cpu().numpy()
        dn  = y[0,3].cpu().numpy()
        da  = y[0,4].cpu().numpy()
        self.G.train()
        return {'phi_vol':phi, 'n_vol':n, 'p_vol':p, 'dn_vol':dn, 'da_vol':da}

# ==============================
# 7) Teacher-supervised sampler (cm coords)
# ==============================
class TeacherSampler:
    """Sample continuous labels from teacher volume via grid_sample (cm units)."""
    def __init__(self, teacher: CGANTeacher):
        self.t = teacher
        self.dev = teacher.dev
        self.nx, self.ny, self.nz = teacher.nx, teacher.ny, teacher.nz

    @torch.no_grad()
    def sample(self, coords_cm):  # [N,6] = [x,y,z,V,NA,ND]
        V = float(coords_cm[0,3].item())
        NA = float(coords_cm[0,4].item())
        ND = float(coords_cm[0,5].item())

        out = self.t.generate(V, NA, ND)

        def vol(np_array): return torch.tensor(np_array, device=self.dev).view(1, 1, self.nx, self.ny, self.nz)
        phi_vol = vol(out['phi_vol'])
        logn_vol = torch.log10(torch.clamp(vol(out['n_vol']), 1e1, 1e25))
        logp_vol = torch.log10(torch.clamp(vol(out['p_vol']), 1e1, 1e25))
        dn_vol  = vol(out['dn_vol'])
        da_vol  = vol(out['da_vol'])

        x_norm = (coords_cm[:,0:1] / (params.wg_width_cm / 2)).clamp(-1, 1)
        y_norm = (coords_cm[:,1:2] / (params.wg_height_cm / 2) - 1).clamp(-1, 1)
        z_norm = (coords_cm[:,2:3] / (params.wg_length_cm / 2) - 1).clamp(-1, 1)

        # grid: [N, out_d, out_h, out_w, 3] = [1,1,1,M,3]
        grid = torch.stack([z_norm, y_norm, x_norm], dim=-1).view(1, 1, 1, -1, 3)

        def gs(T_vol):
            out = F.grid_sample(T_vol, grid, mode='bilinear', padding_mode='border', align_corners=True)
            return out.view(-1, 1)  # [M,1]

        phi_s  = gs(phi_vol)
        logn_s = gs(logn_vol)
        logp_s = gs(logp_vol)
        dn_s   = gs(dn_vol)
        da_s   = gs(da_vol)

        return torch.cat([phi_s, logn_s, logp_s, dn_s, da_s], dim=1)  # [N,5]

# ==============================
# 8) PINN (Res-SIREN) with 3D physics + Soft BC + boundary RAR + best-save
# ==============================
class Sine(nn.Module):
    def __init__(self, w0=30.0): super().__init__(); self.w0=w0
    def forward(self, x): return torch.sin(self.w0*x)

def siren_linear(in_dim, out_dim, w0=30.0, is_first=False):
    lin = nn.Linear(in_dim, out_dim)
    with torch.no_grad():
        if is_first: lin.weight.uniform_(-1/in_dim, 1/in_dim)
        else:
            bound = math.sqrt(6/in_dim)/w0
            lin.weight.uniform_(-bound, bound)
    return lin

class ResSirenBlock(nn.Module):
    def __init__(self, H: int, w0: float = 15.0):
        super().__init__()
        self.lin1 = siren_linear(H, H, w0)
        self.act1 = Sine(w0)
        self.lin2 = siren_linear(H, H, w0)
        self.act2 = Sine(w0)
    def forward(self, x):
        h = self.act1(self.lin1(x))
        h = self.act2(self.lin2(h))
        return x + h

class PINN3D_ResSIREN(nn.Module):
    """Res-SIREN trunk + split heads. Optional low-order Fourier features for coords."""
    def __init__(self, hidden=256, blocks=3, w0_first=30.0, w0_inner=15.0, add_fourier=False, fourier_bands=(1,2,4)):
        super().__init__()
        self.add_fourier = add_fourier
        self.fourier_bands = tuple(int(b) for b in fourier_bands)
        in_dim_base = 6  # [x,y,z,V,NA,ND]
        self.feat_dim = in_dim_base
        if self.add_fourier:
            # for x,y,z only; sin & cos per band
            self.feat_dim = in_dim_base + 3*2*len(self.fourier_bands)
        self.first = nn.Sequential(
            siren_linear(self.feat_dim, hidden, w0_first, is_first=True), Sine(w0_first)
        )
        self.blocks = nn.ModuleList([ResSirenBlock(hidden, w0_inner) for _ in range(blocks)])
        # heads
        self.head_elec = nn.Linear(hidden, 5)  # [phi, logn, logp, dn, da]
        self.head_opt  = nn.Linear(hidden, 1)  # [psi]

    def _norm_inputs(self, coords_cm):
        x_norm = coords_cm[:,0:1] / (params.wg_width_cm / 2)
        y_norm = coords_cm[:,1:2] / (params.wg_height_cm / 2) - 1
        z_norm = coords_cm[:,2:3] / (params.wg_length_cm / 2) - 1
        v_norm = coords_cm[:,3:4] / 5.0
        na_norm = (torch.log10(coords_cm[:,4:5]) - 18.0) / 1.5
        nd_norm = (torch.log10(coords_cm[:,5:6]) - 18.0) / 1.5
        base = [x_norm, y_norm, z_norm, v_norm, na_norm, nd_norm]
        if not self.add_fourier:
            return torch.cat(base, dim=1)
        # low-order Fourier features on spatial coords (μm/cm normalized already)
        feats = [x_norm, y_norm, z_norm]
        four = []
        for b in self.fourier_bands:
            for t in feats:
                four.append(torch.sin(2*math.pi*b*t))
                four.append(torch.cos(2*math.pi*b*t))
        return torch.cat(base + four, dim=1)

    def forward(self, coords_cm):
        x = self._norm_inputs(coords_cm)
        h = self.first(x)
        if cfg.ckpt_segments and self.training:
            # checkpoint across residual stack to save memory
            h = checkpoint_sequential(self.blocks, cfg.ckpt_segments, h)
        else:
            for blk in self.blocks:
                h = blk(h)
        elec = self.head_elec(h)
        opt  = self.head_opt(h)
        out = torch.cat([elec, opt], dim=1)
        return out  # [N, 6]

# --------- Soft BC helpers ----------

def sample_boundary_points(N, V, NA, ND, dev):
    """Axis-aligned box faces; returns coords(cm) with requires_grad and outward normals n_hat."""
    xs = torch.rand(N,1, device=dev)*params.wg_width_cm - params.wg_width_cm/2
    ys = torch.rand(N,1, device=dev)*params.wg_height_cm
    zs = torch.rand(N,1, device=dev)*params.wg_length_cm
    face = torch.randint(0,6,(N,1), device=dev)
    xs[face.eq(0)] = -params.wg_width_cm/2
    xs[face.eq(1)] =  params.wg_width_cm/2
    ys[face.eq(2)] = 0.0
    ys[face.eq(3)] = params.wg_height_cm
    zs[face.eq(4)] = 0.0
    zs[face.eq(5)] = params.wg_length_cm
    Vc = torch.full((N,1), float(V), device=dev)
    NAc= torch.full((N,1), float(NA), device=dev)
    NDc= torch.full((N,1), float(ND), device=dev)
    coords = torch.cat([xs, ys, zs, Vc, NAc, NDc], dim=1).requires_grad_(True)

    n_hat = torch.zeros(N,3, device=dev)
    idx = face.squeeze()
    n_hat[idx==0,0] = -1
    n_hat[idx==1,0] =  1
    n_hat[idx==2,1] = -1
    n_hat[idx==3,1] =  1
    n_hat[idx==4,2] = -1
    n_hat[idx==5,2] =  1
    return coords, n_hat

class PINNTrainer:
    def __init__(self, pinn: nn.Module, teacher: CGANTeacher, device='cpu', save_best_path: Optional[str] = 'pinn_best.pt'):
        self.pinn = pinn.to(device)
        self.teacher = teacher
        self.dev = device
        self.ts  = TeacherSampler(teacher)

        # Self-adaptive weights for each loss component
        self.log_lambda_data = nn.Parameter(torch.tensor(0.0, device=device))
        self.log_lambda_p = nn.Parameter(torch.tensor(0.0, device=device))
        self.log_lambda_dd = nn.Parameter(torch.tensor(0.0, device=device))
        self.log_lambda_h = nn.Parameter(torch.tensor(0.0, device=device))
        self.log_lambda_phiN = nn.Parameter(torch.tensor(0.0, device=device))
        self.log_lambda_flux = nn.Parameter(torch.tensor(0.0, device=device))
        self.log_lambda_hrob = nn.Parameter(torch.tensor(0.0, device=device))
        self.log_lambda_gauge = nn.Parameter(torch.tensor(0.0, device=device))

        self.adaptive_weights = [
            self.log_lambda_data, self.log_lambda_p, self.log_lambda_dd, self.log_lambda_h,
            self.log_lambda_phiN, self.log_lambda_flux, self.log_lambda_hrob, self.log_lambda_gauge
        ]

        # Dual optimizers for SA-PINN (faster lambda)
        self.optimizer_net = torch.optim.Adam(self.pinn.parameters(), lr=1e-3)
        self.optimizer_lambda = torch.optim.Adam(self.adaptive_weights, lr=1e-3)

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and (device=='cuda')))

        # constants
        self.eps_si_const = params.eps_si * params.eps0
        self.q_const = params.q
        self.k0_sq_const = params.k0**2
        self.k0_const = params.k0
        self.n_si_const = params.n_si
        self.mu_n_const = params.mu_n
        self.mu_p_const = params.mu_p
        self.Dn_const = params.Dn
        self.Dp_const = params.Dp

        # init lambdas (stronger BC start)
        self.log_lambda_phiN.data.fill_(math.log(5.0))
        self.log_lambda_flux.data.fill_(math.log(5.0))
        self.log_lambda_hrob.data.fill_(math.log(10.0))
        self.log_lambda_gauge.data.fill_(math.log(1.0))

        self.save_best_path = save_best_path
        self.best_epoch_loss = float('inf')

        print("✓ PINN Trainer ready (SA-PINN Full 3D Physics + Soft BC + RAR + Res-SIREN)")

    def sample_points(self, N, V, NA, ND):
        x_cm = (torch.rand(N,1, device=self.dev) * params.wg_width_cm) - (params.wg_width_cm / 2)
        y_cm = torch.rand(N,1, device=self.dev) * params.wg_height_cm
        z_cm = torch.rand(N,1, device=self.dev) * params.wg_length_cm
        v = torch.full((N,1), float(V), device=self.dev)
        na = torch.full((N,1), float(NA), device=self.dev)
        nd = torch.full((N,1), float(ND), device=self.dev)
        coords_cm = torch.cat([x_cm, y_cm, z_cm, v, na, nd], dim=1)
        coords_cm.requires_grad_(True)
        return coords_cm

    def _grad_wrt(self, y, coords_cm, idx):
        g = torch.autograd.grad(y.sum(), coords_cm, create_graph=True, retain_graph=True)[0]
        return g[:, idx:idx+1]

    def physics_loss(self, coords_cm, out):
        """Interior PDE residuals: Poisson + Drift–Diffusion + Helmholtz."""
        # Avoid autocast here to keep second-order grads stable
        phi = out[:,0:1]
        n   = torch.pow(10.0, out[:,1:2]).clamp(params.ni, 1e25)
        p   = torch.pow(10.0, out[:,2:3]).clamp(params.ni, 1e25)
        dn  = out[:,3:4]
        psi = out[:,5:6]

        # Fields, doping split
        dphidx = self._grad_wrt(phi, coords_cm, 0)
        dphidy = self._grad_wrt(phi, coords_cm, 1)
        dphidz = self._grad_wrt(phi, coords_cm, 2)
        Ex, Ey, Ez = -dphidx, -dphidy, -dphidz

        x_cm = coords_cm[:,0:1]
        is_n = (x_cm >= 0).float()
        is_p = 1.0 - is_n
        NA = coords_cm[:,4:5]
        ND = coords_cm[:,5:6]

        # Poisson
        d2phi_dx2 = self._grad_wrt(dphidx, coords_cm, 0)
        d2phi_dy2 = self._grad_wrt(dphidy, coords_cm, 1)
        d2phi_dz2 = self._grad_wrt(dphidz, coords_cm, 2)
        lap_phi = d2phi_dx2 + d2phi_dy2 + d2phi_dz2
        rho = self.q_const * (p - n + (is_n * ND - is_p * NA))
        poisson_res = (self.eps_si_const * lap_phi + rho) / (self.q_const * (NA + 1e-12))
        L_poisson = (poisson_res**2).mean()

        # Drift–Diffusion (continuity: ∇·Jn=0, ∇·Jp=0) — rescaled by characteristic length
        dn_dx = self._grad_wrt(n, coords_cm, 0); dn_dy = self._grad_wrt(n, coords_cm, 1); dn_dz = self._grad_wrt(n, coords_cm, 2)
        dp_dx = self._grad_wrt(p, coords_cm, 0); dp_dy = self._grad_wrt(p, coords_cm, 1); dp_dz = self._grad_wrt(p, coords_cm, 2)
        Jn_x = self.q_const * (self.mu_n_const * n * Ex + self.Dn_const * dn_dx)
        Jn_y = self.q_const * (self.mu_n_const * n * Ey + self.Dn_const * dn_dy)
        Jn_z = self.q_const * (self.mu_n_const * n * Ez + self.Dn_const * dn_dz)
        Jp_x = self.q_const * (self.mu_p_const * p * Ex - self.Dp_const * dp_dx)
        Jp_y = self.q_const * (self.mu_p_const * p * Ey - self.Dp_const * dp_dy)
        Jp_z = self.q_const * (self.mu_p_const * p * Ez - self.Dp_const * dp_dz)
        divJn = self._grad_wrt(Jn_x, coords_cm, 0) + self._grad_wrt(Jn_y, coords_cm, 1) + self._grad_wrt(Jn_z, coords_cm, 2)
        divJp = self._grad_wrt(Jp_x, coords_cm, 0) + self._grad_wrt(Jp_y, coords_cm, 1) + self._grad_wrt(Jp_z, coords_cm, 2)
        Lc = params.wg_width_cm
        scale_dd = self.q_const * self.mu_n_const * params.ni / (Lc + 1e-12)
        L_dd = ((divJn/ (scale_dd+1e-12))**2 + (divJp/(scale_dd+1e-12))**2).mean()

        # Helmholtz
        dpsi_dx = self._grad_wrt(psi, coords_cm, 0)
        dpsi_dy = self._grad_wrt(psi, coords_cm, 1)
        dpsi_dz = self._grad_wrt(psi, coords_cm, 2)
        d2psi_dx2 = self._grad_wrt(dpsi_dx, coords_cm, 0)
        d2psi_dy2 = self._grad_wrt(dpsi_dy, coords_cm, 1)
        d2psi_dz2 = self._grad_wrt(dpsi_dz, coords_cm, 2)
        lap_psi = d2psi_dx2 + d2psi_dy2 + d2psi_dz2
        n_eff = self.n_si_const + dn
        helm_res = lap_psi + self.k0_sq_const * (n_eff**2) * psi
        L_helm = (helm_res / (self.k0_sq_const + 1e-12))**2
        L_helm = L_helm.mean()

        return L_poisson, L_dd, L_helm

    def bc_losses(self, coords_b, n_hat, out_b):
        """Soft BCs: Poisson Neumann (∂φ/∂n=0), no-flux, Helmholtz Robin, gauge fix."""
        phi = out_b[:,0:1]
        n   = torch.pow(10.0, out_b[:,1:2]).clamp(params.ni, 1e25)
        p   = torch.pow(10.0, out_b[:,2:3]).clamp(params.ni, 1e25)
        dn  = out_b[:,3:4]
        psi = out_b[:,5:6]

        # normal derivative for φ
        dphidx = self._grad_wrt(phi, coords_b, 0); dphidy = self._grad_wrt(phi, coords_b, 1); dphidz = self._grad_wrt(phi, coords_b, 2)
        grad_phi_n = dphidx*n_hat[:,0:1] + dphidy*n_hat[:,1:2] + dphidz*n_hat[:,2:3]
        L_phi_neu = (grad_phi_n**2).mean()

        # no-flux: (Jn·n̂)=0, (Jp·n̂)=0
        Ex, Ey, Ez = -dphidx, -dphidy, -dphidz
        dn_dx = self._grad_wrt(n, coords_b, 0); dn_dy = self._grad_wrt(n, coords_b, 1); dn_dz = self._grad_wrt(n, coords_b, 2)
        dp_dx = self._grad_wrt(p, coords_b, 0); dp_dy = self._grad_wrt(p, coords_b, 1); dp_dz = self._grad_wrt(p, coords_b, 2)
        Jn_x = self.q_const * (self.mu_n_const * n * Ex + self.Dn_const * dn_dx)
        Jn_y = self.q_const * (self.mu_n_const * n * Ey + self.Dn_const * dn_dy)
        Jn_z = self.q_const * (self.mu_n_const * n * Ez + self.Dn_const * dn_dz)
        Jp_x = self.q_const * (self.mu_p_const * p * Ex - self.Dp_const * dp_dx)
        Jp_y = self.q_const * (self.mu_p_const * p * Ey - self.Dp_const * dp_dy)
        Jp_z = self.q_const * (self.mu_p_const * p * Ez - self.Dp_const * dp_dz)
        Jn_dot_n = Jn_x*n_hat[:,0:1] + Jn_y*n_hat[:,1:2] + Jn_z*n_hat[:,2:3]
        Jp_dot_n = Jp_x*n_hat[:,0:1] + Jp_y*n_hat[:,1:2] + Jp_z*n_hat[:,2:3]
        L_no_flux = (Jn_dot_n**2 + Jp_dot_n**2).mean()

        # Helmholtz Robin: ∂ψ/∂n = k0 n_eff ψ (real-ψ approx)
        dpsi_dx = self._grad_wrt(psi, coords_b, 0); dpsi_dy = self._grad_wrt(psi, coords_b, 1); dpsi_dz = self._grad_wrt(psi, coords_b, 2)
        dpsi_dn = dpsi_dx*n_hat[:,0:1] + dpsi_dy*n_hat[:,1:2] + dpsi_dz*n_hat[:,2:3]
        n_eff = self.n_si_const + dn
        L_helm_robin = ((dpsi_dn - self.k0_const*n_eff*psi)**2).mean()

        # gauge fix: mean φ = 0
        L_gauge = (phi.mean()**2)

        return L_phi_neu, L_no_flux, L_helm_robin, L_gauge

    def _poisson_res_pointwise(self, coords_cm, _unused_out=None):
        # recompute with grads on (RAR sampling)
        with torch.enable_grad():
            coords_cm_req = coords_cm.detach().clone().requires_grad_(True)
            out_req = self.pinn(coords_cm_req)
            phi = out_req[:, 0:1]
            n   = torch.pow(10.0, out_req[:,1:2]).clamp_min(params.ni)
            p   = torch.pow(10.0, out_req[:,2:3]).clamp_min(params.ni)

            x   = coords_cm_req[:,0:1]
            is_n = (x>=0).float(); is_p = 1.0 - is_n
            NA = coords_cm_req[:,4:5]; ND = coords_cm_req[:,5:6]

            dphidx = self._grad_wrt(phi, coords_cm_req, 0)
            dphidy = self._grad_wrt(phi, coords_cm_req, 1)
            dphidz = self._grad_wrt(phi, coords_cm_req, 2)
            d2phi_dx2 = self._grad_wrt(dphidx, coords_cm_req, 0)
            d2phi_dy2 = self._grad_wrt(dphidy, coords_cm_req, 1)
            d2phi_dz2 = self._grad_wrt(dphidz, coords_cm_req, 2)
            lap = d2phi_dx2 + d2phi_dy2 + d2phi_dz2

            rho = self.q_const*(p - n + (is_n*ND - is_p*NA))
            res = (self.eps_si_const * lap + rho) / (self.q_const * (NA + 1e-12))
            return res.abs().detach().view(-1)

    def train(self,
              epochs=20, points=800, configs=3,
              rar_frac=0.2, helm_warmup_steps=2000.0, grad_clip=1.0):
        step = 0

        for ep in range(1,epochs+1):
            ep_losses = {'total': 0.0, 'data': 0.0, 'P': 0.0, 'DD': 0.0, 'H': 0.0,
                         'phiN':0.0, 'flux':0.0, 'hrob':0.0, 'gauge':0.0}
            pbar = tqdm(range(configs), desc=f"PINN Epoch {ep}/{epochs}")
            epoch_total = 0.0

            for ci in pbar:
                step += 1
                V = float(np.random.uniform(0.0, 5.0))
                NA= 10**np.random.uniform(17.5, 19.0)
                ND= 10**np.random.uniform(17.5, 19.0)

                # ---- RAR selection on interior points ----
                pool_size = int(points * (1.0 + rar_frac))
                coords_pool = self.sample_points(pool_size, V, NA, ND)
                res = self._poisson_res_pointwise(coords_pool, self.pinn(coords_pool))
                k = max(1, int(points * rar_frac))
                topk_indices = torch.topk(res, k=k).indices
                mask = torch.ones(pool_size, dtype=torch.bool, device=self.dev)
                mask[topk_indices] = False
                rest_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
                perm = rest_indices[torch.randperm(rest_indices.numel(), device=self.dev)[:points-k]]
                chosen_indices = torch.cat([topk_indices, perm], dim=0)
                coords = coords_pool[chosen_indices].detach().clone().requires_grad_(True)

                # ---- Teacher data loss with boundary masking ----
                with torch.cuda.amp.autocast(enabled=(cfg.use_amp and (self.dev=='cuda'))):
                    pred = self.pinn(coords)
                    target = self.ts.sample(coords.detach())

                # boundary-aware weight (downweight points too close to faces)
                x,y,z = coords[:,0], coords[:,1], coords[:,2]
                m = 0.05
                wx = m*params.wg_width_cm/2; wy = m*params.wg_height_cm; wz = m*params.wg_length_cm
                dists = torch.stack([
                    (params.wg_width_cm/2 - x.abs()),
                    (params.wg_height_cm - y).abs().clamp_min(0), y,
                    (params.wg_length_cm - z).abs().clamp_min(0), z
                ], dim=1)
                dmin = torch.min(dists, dim=1).values
                w_data = (dmin > torch.min(torch.tensor([wx,wy,wz], device=self.dev))).float()
                loss_data = ((pred[:,:5] - target)**2 * w_data.view(-1,1)).mean()

                # ---- Interior physics ---- (keep in fp32)
                Lp, Ld, Lh = self.physics_loss(coords, pred)

                # ---- Boundary conditions with schedule + boundary RAR ----
                bc_mult = min(3.0, 1.0 + step/1000.0)
                N_b = int(max(16, points//4) * bc_mult)

                # boundary RAR: sample 2× and pick hardest by Neumann violation
                pool_b, n_hat_pool = sample_boundary_points(N=int(N_b*2), V=V, NA=NA, ND=ND, dev=self.dev)
                out_pool = self.pinn(pool_b)
                dphidx = self._grad_wrt(out_pool[:,0:1], pool_b, 0)
                dphidy = self._grad_wrt(out_pool[:,0:1], pool_b, 1)
                dphidz = self._grad_wrt(out_pool[:,0:1], pool_b, 2)
                viol = (dphidx*n_hat_pool[:,0:1] + dphidy*n_hat_pool[:,1:2] + dphidz*n_hat_pool[:,2:3]).abs().view(-1)
                topk = torch.topk(viol, k=N_b).indices
                coords_b = pool_b[topk].detach().clone().requires_grad_(True)
                n_hat = n_hat_pool[topk]

                out_b = self.pinn(coords_b)
                L_phi_neu, L_no_flux, L_helm_robin, L_gauge = self.bc_losses(coords_b, n_hat, out_b)

                # ---- SA-PINN: scheduling for Helmholtz terms ----
                alpha = min(1.0, step / helm_warmup_steps)

                # ---- SA-PINN: Network Update (Minimization) ----
                self.optimizer_net.zero_grad()
                loss_net = (torch.exp(-self.log_lambda_data) * loss_data + self.log_lambda_data +
                            torch.exp(-self.log_lambda_p) * Lp + self.log_lambda_p +
                            torch.exp(-self.log_lambda_dd) * Ld + self.log_lambda_dd +
                            torch.exp(-self.log_lambda_h) * (alpha*Lh) + self.log_lambda_h +
                            torch.exp(-self.log_lambda_phiN) * L_phi_neu + self.log_lambda_phiN +
                            torch.exp(-self.log_lambda_flux) * L_no_flux + self.log_lambda_flux +
                            torch.exp(-self.log_lambda_hrob) * (alpha*L_helm_robin) + self.log_lambda_hrob +
                            torch.exp(-self.log_lambda_gauge) * L_gauge + self.log_lambda_gauge)
                loss_net.backward(retain_graph=True)
                if grad_clip > 0: clip_grad_norm_(self.pinn.parameters(), grad_clip)
                self.optimizer_net.step()

                # ---- SA-PINN: Lambda Update (Maximization via minimizing negative loss) ----
                self.optimizer_lambda.zero_grad()
                loss_lambda = - (
                    torch.exp(-self.log_lambda_data) * loss_data.detach() + self.log_lambda_data +
                    torch.exp(-self.log_lambda_p)    * Lp.detach()        + self.log_lambda_p +
                    torch.exp(-self.log_lambda_dd)   * Ld.detach()        + self.log_lambda_dd +
                    torch.exp(-self.log_lambda_h)    * (alpha*Lh).detach()+ self.log_lambda_h +
                    torch.exp(-self.log_lambda_phiN) * L_phi_neu.detach() + self.log_lambda_phiN +
                    torch.exp(-self.log_lambda_flux) * L_no_flux.detach() + self.log_lambda_flux +
                    torch.exp(-self.log_lambda_hrob) * (alpha*L_helm_robin).detach() + self.log_lambda_hrob +
                    torch.exp(-self.log_lambda_gauge)* L_gauge.detach()   + self.log_lambda_gauge
                )
                loss_lambda.backward()
                self.optimizer_lambda.step()

                # logging
                ep_losses['total'] += loss_net.item()
                ep_losses['data']  += loss_data.item()
                ep_losses['P']     += Lp.item()
                ep_losses['DD']    += Ld.item()
                ep_losses['H']     += Lh.item()
                ep_losses['phiN']  += L_phi_neu.item()
                ep_losses['flux']  += L_no_flux.item()
                ep_losses['hrob']  += L_helm_robin.item()
                ep_losses['gauge'] += L_gauge.item()
                epoch_total += loss_net.item()

                pbar.set_postfix({k: f'{v/(ci+1):.3e}' for k,v in ep_losses.items()})

            avg_losses = {k: v / configs for k, v in ep_losses.items()}
            print(f"  → Avg: L={avg_losses['total']:.4e} | Data={avg_losses['data']:.4e} | "
                  f"P={avg_losses['P']:.4e} | DD={avg_losses['DD']:.4e} | H={avg_losses['H']:.4e} | "
                  f"BCs(N,F,R,G)={avg_losses['phiN']:.2e},{avg_losses['flux']:.2e},{avg_losses['hrob']:.2e},{avg_losses['gauge']:.2e}")
            print(f"  → Lambdas(data,P,DD,H) = "
                  f"{torch.exp(self.log_lambda_data).item():.2e}, {torch.exp(self.log_lambda_p).item():.2e}, "
                  f"{torch.exp(self.log_lambda_dd).item():.2e}, {torch.exp(self.log_lambda_h).item():.2e}")

            # Save best epoch
            if avg_losses['total'] < self.best_epoch_loss and self.save_best_path:
                self.best_epoch_loss = avg_losses['total']
                torch.save({'pinn': self.pinn.state_dict(), 'meta': {'loss': self.best_epoch_loss, 'epoch': ep}}, self.save_best_path)

# ==============================
# 9) Orchestration
# ==============================

def run_unified_full3d(
    n_samples=200, grid=(32,16,32), v_max=5.0,
    teacher_epochs=5, teacher_bs=2,
    pinn_epochs=10, pinn_points=2000, pinn_configs=5,
    pinn_hidden=256, pinn_blocks=3, w0_first=30.0, w0_inner=15.0,
    add_fourier=False, fourier_bands=(1,2,4)
):
    ds = HFDataset(n_samples=n_samples, grid=grid, v_max=v_max)
    teacher = CGANTeacher(grid=grid, device=device, v_max=v_max, save_best_path='teacher_best.pt')
    teacher.train(ds, epochs=teacher_epochs, batch=teacher_bs)

    tch_out = teacher.generate(1.2)
    print("Teacher gen shapes:", {k:np.array(v).shape for k,v in tch_out.items()})

    pinn = PINN3D_ResSIREN(hidden=pinn_hidden, blocks=pinn_blocks, w0_first=w0_first, w0_inner=w0_inner,
                           add_fourier=add_fourier, fourier_bands=fourier_bands)
    trainer = PINNTrainer(pinn, teacher, device=device, save_best_path='pinn_best.pt')
    trainer.train(epochs=pinn_epochs, points=pinn_points, configs=pinn_configs)
    return {'teacher':teacher, 'pinn':pinn, 'dataset':ds, 'trainer':trainer}

# ==============================
# 10) Visualization Helpers (μm)
# ==============================

def visualize_mid_slices(teacher: CGANTeacher, V: float, NA=None, ND=None, save_path: Optional[str]=None):
    print("🔎 Visualizing Teacher mid-slices...")
    NA = NA or params.NA; ND = ND or params.ND
    out = teacher.generate(V, NA, ND)
    phi_vol, n_vol, p_vol, dn_vol = out['phi_vol'], out['n_vol'], out['p_vol'], out['dn_vol']
    nx, ny, nz = teacher.nx, teacher.ny, teacher.nz

    x_um = np.linspace(-params.wg_width/2, params.wg_width/2, nx)
    y_um = np.linspace(0, params.wg_height, ny)
    z_um = np.linspace(0, params.wg_length, nz)

    mid_z_idx = nz // 2
    phi_slice = phi_vol[:, :, mid_z_idx]
    logn_slice = np.log10(np.clip(n_vol[:, :, mid_z_idx], 1e1, 1e25))
    logp_slice = np.log10(np.clip(p_vol[:, :, mid_z_idx], 1e1, 1e25))
    dn_slice = dn_vol[:, :, mid_z_idx]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['φ (V)', 'log10 n (cm⁻³)', 'log10 p (cm⁻³)', 'Δn']
    data_slices = [phi_slice, logn_slice, logp_slice, dn_slice]
    cmaps = ['coolwarm', 'viridis', 'viridis', RdBu_r]
    extent_um = [x_um.min(), x_um.max(), y_um.min(), y_um.max()]

    for i, ax in enumerate(axes):
        im = ax.imshow(data_slices[i].T, origin='lower', aspect='auto', extent=extent_um, cmap=cmaps[i])
        ax.set_title(titles[i], fontsize=10)
        ax.set_xlabel('X [μm]'); ax.set_ylabel('Y [μm]')
        fig.colorbar(im, ax=ax)

    fig.suptitle(f'Teacher Predicted Mid-Plane (z={z_um[mid_z_idx]:.1f} μm) — V={V:.1f} V', fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight'); print(f"✓ Saved Teacher mid-slice to: {save_path}")
    plt.show()


def plot_3d_waveguide_outline(ax, x_min, x_max, y_min, y_max, z_min, z_max):
    verts = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]
    ])
    edges = [
        [0,1],[1,2],[2,3],[3,0],   # bottom
        [4,5],[5,6],[6,7],[7,4],   # top
        [0,4],[1,5],[2,6],[3,7]    # pillars
    ]
    for e in edges:
        ax.plot(verts[e,0], verts[e,1], verts[e,2], 'k--', lw=0.7, alpha=0.6)

@torch.no_grad()
def visualize_pinn_3d_slices(
    pinn_trainer: 'PINNTrainer',
    voltages_to_plot: List[float] = [0.0, 1.0, 3.0, 5.0],
    grid_res: Tuple[int, int, int] = (48, 24, 48),
    save_path: str = 'pinn_3d_slices.png'
):
    print(f"🔬 Visualizing PINN 3D Δn (μm) for V in {voltages_to_plot} ...")
    pinn = pinn_trainer.pinn; pinn.eval(); dev = pinn_trainer.dev
    nx, ny, nz = grid_res

    x_um = np.linspace(-params.wg_width/2, params.wg_width/2, nx)
    y_um = np.linspace(0, params.wg_height, ny)
    z_um = np.linspace(0, params.wg_length, nz)
    X_um, Y_um, Z_um = np.meshgrid(x_um, y_um, z_um, indexing='ij')

    X_cm = X_um * 1e-4; Y_cm = Y_um * 1e-4; Z_cm = Z_um * 1e-4

    core_mask_np = ((np.abs(X_um) <= params.wg_width / 2) & (Y_um >= 0) & (Y_um <= params.wg_height))
    n_voltages = len(voltages_to_plot)
    fig = plt.figure(figsize=(20, 5 * n_voltages))
    fig.suptitle('PINN Predicted Δn in 3D Waveguide Context (μm)', fontsize=16, y=1.00)

    plot_vmin = -2e-3; plot_vmax = 1e-4
    norm = Normalize(vmin=plot_vmin, vmax=plot_vmax); cmap = RdBu_r

    for i, V in enumerate(voltages_to_plot):
        N = nx*ny*nz
        coords_flat_cm = np.stack([
            X_cm.ravel(),
            Y_cm.ravel(),
            Z_cm.ravel(),
            np.full(N, V, dtype=np.float32),
            np.full(N, params.NA, dtype=np.float32),
            np.full(N, params.ND, dtype=np.float32)
        ], axis=1)
        coords_t = torch.tensor(coords_flat_cm, dtype=torch.float32, device=dev)

        pinn_out = pinn(coords_t)
        dn_flat = pinn_out[:, 3].detach().cpu().numpy()
        dn_3d = dn_flat.reshape(nx, ny, nz)
        dn_3d_masked = np.where(core_mask_np, dn_3d, np.nan)

        mid_x_idx, mid_y_idx, mid_z_idx = nx // 2, ny // 2, nz // 2

        ax_3d = fig.add_subplot(n_voltages, 4, i*4 + 1, projection='3d')
        ax_3d.set_title(f'V = {V:.1f} V (3D View)', fontsize=10)
        plot_3d_waveguide_outline(ax_3d, x_um.min(), x_um.max(), y_um.min(), y_um.max(), z_um.min(), z_um.max())

        Y_slice_0_um, Z_slice_0_um = np.meshgrid(y_um, z_um)
        ax_3d.contourf(Y_slice_0_um, Z_slice_0_um, dn_3d_masked[mid_x_idx, :, :].T,
                       zdir='x', offset=x_um[mid_x_idx], cmap=cmap, norm=norm, alpha=0.7, levels=15)
        X_slice_1_um, Z_slice_1_um = np.meshgrid(x_um, z_um)
        ax_3d.contourf(X_slice_1_um, Z_slice_1_um, dn_3d_masked[:, mid_y_idx, :].T,
                       zdir='y', offset=y_um[mid_y_idx], cmap=cmap, norm=norm, alpha=0.7, levels=15)
        X_slice_2_um, Y_slice_2_um = np.meshgrid(x_um, y_um)
        ax_3d.contourf(X_slice_2_um, Y_slice_2_um, dn_3d_masked[:, :, mid_z_idx].T,
                       zdir='z', offset=z_um[mid_z_idx], cmap=cmap, norm=norm, alpha=0.7, levels=15)

        ax_3d.set_xlim(x_um.min(), x_um.max()); ax_3d.set_ylim(y_um.min(), y_um.max()); ax_3d.set_zlim(z_um.min(), z_um.max())
        ax_3d.set_xlabel('X [μm]'); ax_3d.set_ylabel('Y [μm]'); ax_3d.set_zlabel('Z [μm]')
        ax_3d.grid(False); ax_3d.xaxis.pane.fill = False; ax_3d.yaxis.pane.fill = False; ax_3d.zaxis.pane.fill = False

        extent_yz = [y_um.min(), y_um.max(), z_um.min(), z_um.max()]
        extent_xz = [x_um.min(), x_um.max(), z_um.min(), z_um.max()]
        extent_xy = [x_um.min(), x_um.max(), y_um.min(), y_um.max()]

        ax_yz = fig.add_subplot(n_voltages, 4, i*4 + 2)
        slice_yz = dn_3d_masked[mid_x_idx, :, :]
        im = ax_yz.imshow(slice_yz.T, origin='lower', aspect='auto', cmap=cmap, norm=norm, extent=extent_yz)
        ax_yz.set_title(f'V={V:.1f} V (x={x_um[mid_x_idx]:.2f} μm)', fontsize=10)
        ax_yz.set_xlabel('Y [μm]'); ax_yz.set_ylabel('Z [μm]'); plt.colorbar(im, ax=ax_yz, label='Δn')

        ax_xz = fig.add_subplot(n_voltages, 4, i*4 + 3)
        slice_xz = dn_3d_masked[:, mid_y_idx, :]
        im = ax_xz.imshow(slice_xz.T, origin='lower', aspect='auto', cmap=cmap, norm=norm, extent=extent_xz)
        ax_xz.set_title(f'V={V:.1f} V (y={y_um[mid_y_idx]:.2f} μm)', fontsize=10)
        ax_xz.set_xlabel('X [μm]'); ax_xz.set_ylabel('Z [μm]'); plt.colorbar(im, ax=ax_xz, label='Δn')

        ax_xy = fig.add_subplot(n_voltages, 4, i*4 + 4)
        slice_xy = dn_3d_masked[:, :, mid_z_idx]
        im = ax_xy.imshow(slice_xy.T, origin='lower', aspect='auto', cmap=cmap, norm=norm, extent=extent_xy)
        ax_xy.set_title(f'V={V:.1f} V (z={z_um[mid_z_idx]:.1f} μm)', fontsize=10)
        ax_xy.set_xlabel('X [μm]'); ax_xy.set_ylabel('Y [μm]'); plt.colorbar(im, ax=ax_xy, label='Δn')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight'); print(f"✓ Saved PINN 3D slices to: {save_path}")
    plt.show()

@torch.no_grad()
def visualize_pinn_mid_slices(pinn_trainer: PINNTrainer, V: float, NA=None, ND=None, grid_res=(32, 16, 32), save_path: Optional[str]=None):
    print("🔎 Visualizing PINN mid-slices...")
    NA = NA or params.NA; ND = ND or params.ND
    pinn = pinn_trainer.pinn; pinn.eval(); dev = pinn_trainer.dev
    nx, ny, nz = grid_res

    x_um = np.linspace(-params.wg_width/2, params.wg_width/2, nx)
    y_um = np.linspace(0, params.wg_height, ny)
    z_um = np.linspace(0, params.wg_length, nz)
    X_um, Y_um, Z_um = np.meshgrid(x_um, y_um, z_um, indexing='ij')
    X_cm = X_um * 1e-4; Y_cm = Y_um * 1e-4; Z_cm = Z_um * 1e-4

    N = nx*ny*nz
    coords_flat_cm = np.stack([
        X_cm.ravel(),
        Y_cm.ravel(),
        Z_cm.ravel(),
        np.full(N, V, dtype=np.float32),
        np.full(N, NA, dtype=np.float32),
        np.full(N, ND, dtype=np.float32)
    ], axis=1)
    coords_t = torch.tensor(coords_flat_cm, dtype=torch.float32, device=dev)

    pinn_out = pinn(coords_t)
    phi_vol  = pinn_out[:, 0].detach().cpu().numpy().reshape(nx, ny, nz)
    logn_vol = pinn_out[:, 1].detach().cpu().numpy().reshape(nx, ny, nz)
    logp_vol = pinn_out[:, 2].detach().cpu().numpy().reshape(nx, ny, nz)
    dn_vol   = pinn_out[:, 3].detach().cpu().numpy().reshape(nx, ny, nz)

    mid_z_idx = nz // 2
    phi_slice, logn_slice, logp_slice, dn_slice = \
        phi_vol[:,:,mid_z_idx], logn_vol[:,:,mid_z_idx], logp_vol[:,:,mid_z_idx], dn_vol[:,:,mid_z_idx]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['φ (V)', 'log10 n (cm⁻³)', 'log10 p (cm⁻³)', 'Δn']
    data_slices = [phi_slice, logn_slice, logp_slice, dn_slice]
    cmaps = ['coolwarm', 'viridis', 'viridis', RdBu_r]
    extent_um = [x_um.min(), x_um.max(), y_um.min(), y_um.max()]

    for i, ax in enumerate(axes):
        plot_data = np.clip(data_slices[i], 10, 21) if titles[i].startswith('log') else data_slices[i]
        im = ax.imshow(plot_data.T, origin='lower', aspect='auto', extent=extent_um, cmap=cmaps[i])
        ax.set_title(titles[i], fontsize=10)
        ax.set_xlabel('X [μm]'); ax.set_ylabel('Y [μm]')
        fig.colorbar(im, ax=ax)

    fig.suptitle(f'PINN Predicted Mid-Plane (z={z_um[mid_z_idx]:.1f} μm) — V={V:.1f} V', fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight'); print(f"✓ Saved PINN mid-slice to: {save_path}")
    plt.show()

# ==============================
# 11) Execution
# ==============================
if __name__ == "__main__":
    print("--- Starting Unified 3D Modulator Training (Patched + Res-SIREN) ---")

    # QUICK DEMO (adjust as needed)
    DEMO_GRID = (72, 48, 72)  # smaller grid for quick checks
    models = run_unified_full3d(
        n_samples=2000,
        grid=DEMO_GRID,
        teacher_epochs=50,
        teacher_bs=4,
        pinn_epochs=1000,
        pinn_points=12000,
        pinn_configs=15,
        pinn_hidden=256,
        pinn_blocks=3,
        w0_first=30.0,
        w0_inner=15.0,
        add_fourier=True,
        fourier_bands=(1,2,4)
    )

    print("--- Training Complete. Generating Visualizations ---")

    # 1) Visualize Teacher output (within trained range 0..5 V)
    try:
        visualize_mid_slices(
            models['teacher'],
            V=5.0,
            save_path='teacher_midslice_V5.0.png'
        )
    except Exception as e:
        print(f"Error visualizing teacher: {e}")

    # 2) Visualize PINN 2D mid-slices
    try:
        visualize_pinn_mid_slices(
            models['trainer'],
            V=5.0,
            grid_res=DEMO_GRID,
            save_path='pinn_midslice_V5.0.png'
        )
    except Exception as e:
        print(f"Error visualizing PINN 2D slices: {e}")

    # 3) Visualize PINN 3D slices
    try:
        visualize_pinn_3d_slices(
            models['trainer'],
            voltages_to_plot=[0.0, 1.0, 3.0, 5.0],
            grid_res=(72, 48, 72),
            save_path='pinn_3d_slices_all_V.png'
        )
    except Exception as e:
        print(f"Error visualizing PINN 3D slices: {e}")

    print("--- Script Finished ---")
