# hf3d_solver_dataset.py
# ------------------------------------------------------------
# 3D HF Dataset Solver: Poisson + Gummel-DD(near-zero-current) + Helmholtz
# - Rib/Slab silicon geometry on rectilinear grid
# - Dirichlet contacts at x-min/x-max (±V/2), natural Neumann elsewhere
# - Gummel-like update using constant quasi-Fermi levels Fn/Fp from contacts
# - Soref–Bennett Δn, scalar Helmholtz with simple absorbing layer
# - Saves NPZ dataset samples (inputs + labels + checks)
# ------------------------------------------------------------
import os, math, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from dataclasses import dataclass, asdict

# SciPy sparse solvers
from scipy.sparse import coo_matrix, csc_matrix, eye
from scipy.sparse.linalg import spsolve, splu

# ------------------------- Config -------------------------
@dataclass
class Cfg:
    # physical constants (SI unless noted)
    q: float = 1.602e-19
    kb: float = 1.380649e-23
    T: float = 300.0
    eps0: float = 8.854e-12
    Vt: float = (1.380649e-23*300.0)/1.602e-19

    # materials
    eps_si: float = 11.7*8.854e-12
    eps_sio2: float = 3.9*8.854e-12
    n_i: float = 1.45e10   # cm^-3
    mu_n: float = 0.135    # m^2/Vs (silicon @300K order-of-mag)
    mu_p: float = 0.048    # m^2/Vs

    # Soref–Bennett @1550nm
    an: float = 8.8e-22    # cm^3
    ap: float = 8.5e-18    # cm^3
    gp: float = 0.8

    # optics
    wavelength_um: float = 1.55
    n_si_opt: float = 3.476
    n_sio2_opt: float = 1.444

    # doping (cm^-3)
    ND: float = 5e18
    NA: float = 5e18

    # geometry (um)
    rib_w: float = 0.4
    rib_h: float = 0.22
    slab_h: float = 0.14
    Lx: float = 0.50
    Ly: float = 0.24
    Lz: float = 0.50

    # grid
    NX: int = 72
    NY: int = 48
    NZ: int = 72

    # gummel
    max_iter: int = 30
    relax: float = 0.5
    tol_phi: float = 5e-5  # V

    # helmholtz
    pml_cells: int = 4       # absorbing thickness in cells
    pml_strength: float = 3  # imag shift coefficient
    src_amp: float = 1.0

    # dataset
    out_dir: str = "hf3d_out"
    n_samples: int = 4
    V_range: tuple = (0.0, 5.0)
    seed: int = 42

CFG = Cfg()

# ------------------------- Grid & Indexing -------------------------
def linspace3(Lx, Ly, Lz, NX, NY, NZ):
    x = np.linspace(-Lx/2, Lx/2, NX)
    y = np.linspace(0, Ly, NY)
    z = np.linspace(-Lz/2, Lz/2, NZ)
    return np.meshgrid(x, y, z, indexing='ij')

def idx(i,j,k, NX,NY,NZ):
    return i + NX*(j + NY*k)

def neighbors(i,j,k, NX,NY,NZ):
    xm = (max(i-1,0), j, k); xp = (min(i+1,NX-1), j, k)
    ym = (i, max(j-1,0), k); yp = (i, min(j+1,NY-1), k)
    zm = (i, j, max(k-1,0)); zp = (i, j, min(k+1,NZ-1))
    return xm,xp,ym,yp,zm,zp

# ------------------------- Geometry & Masks -------------------------
def make_masks(cfg: Cfg):
    X,Y,Z = linspace3(cfg.Lx, cfg.Ly, cfg.Lz, cfg.NX, cfg.NY, cfg.NZ)
    rib  = (np.abs(X) <= cfg.rib_w/2) & (Y <= cfg.rib_h)
    slab = (np.abs(X) <= cfg.rib_w/2 + 0.2) & (Y <= cfg.slab_h)
    core = (rib | slab).astype(np.float32)
    left = (X < 0).astype(np.float32)
    right= 1.0 - left
    # optical base index
    n_base = cfg.n_sio2_opt + (cfg.n_si_opt - cfg.n_sio2_opt)*core
    # permittivity for Poisson
    eps = cfg.eps_sio2 + (cfg.eps_si - cfg.eps_sio2)*core
    return X,Y,Z, core.astype(np.float32), rib.astype(np.float32), slab.astype(np.float32), left, right, n_base.astype(np.float32), eps.astype(np.float64)

# ------------------------- Poisson Assembly -------------------------
def assemble_poisson(cfg: Cfg, eps, bc_left, bc_right):
    NX,NY,NZ = cfg.NX, cfg.NY, cfg.NZ
    dx = cfg.Lx/(NX-1); dy = cfg.Ly/(NY-1); dz = cfg.Lz/(NZ-1)

    rows=[]; cols=[]; data=[]; b = np.zeros(NX*NY*NZ, dtype=np.float64)

    def add(ii,jj,v): rows.append(ii); cols.append(jj); data.append(v)

    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                p = idx(i,j,k,NX,NY,NZ)

                # Dirichlet contacts on x-faces
                if i==0:
                    add(p,p,1.0); b[p]=bc_left; continue
                if i==NX-1:
                    add(p,p,1.0); b[p]=bc_right; continue

                # interior / Neumann on y,z via natural divergence form
                # face-centered eps
                eps_xm = 0.5*(eps[i,j,k] + eps[i-1,j,k])
                eps_xp = 0.5*(eps[i,j,k] + eps[i+1,j,k]) if i+1<NX else eps[i,j,k]
                eps_ym = 0.5*(eps[i,j,k] + eps[i,j-1,k]) if j-1>=0 else eps[i,j,k]
                eps_yp = 0.5*(eps[i,j,k] + eps[i,j+1,k]) if j+1<NY else eps[i,j,k]
                eps_zm = 0.5*(eps[i,j,k] + eps[i,j,k-1]) if k-1>=0 else eps[i,j,k]
                eps_zp = 0.5*(eps[i,j,k] + eps[i,j,k+1]) if k+1<NZ else eps[i,j,k]

                cxm = eps_xm/(dx*dx); cxp = eps_xp/(dx*dx)
                cym = eps_ym/(dy*dy); cyp = eps_yp/(dy*dy)
                czm = eps_zm/(dz*dz); czp = eps_zp/(dz*dz)

                diag = cxm+cxp+cym+cyp+czm+czp
                add(p,p,diag)

                # x-
                q = idx(i-1,j,k,NX,NY,NZ); add(p,q,-cxm)
                # x+
                q = idx(i+1,j,k,NX,NY,NZ); add(p,q,-cxp)
                # y-
                if j-1>=0:
                    q = idx(i,j-1,k,NX,NY,NZ); add(p,q,-cym)
                # y+
                if j+1<NY:
                    q = idx(i,j+1,k,NX,NY,NZ); add(p,q,-cyp)
                # z-
                if k-1>=0:
                    q = idx(i,j,k-1,NX,NY,NZ); add(p,q,-czm)
                # z+
                if k+1<NZ:
                    q = idx(i,j,k+1,NX,NY,NZ); add(p,q,-czp)

    A = coo_matrix((data,(rows,cols)), shape=(NX*NY*NZ, NX*NY*NZ)).tocsr()
    return A, b

# ------------------------- Gummel-like Update -------------------------
def init_quasi_fermi(cfg: Cfg, phi_left, phi_right):
    # contacts: choose Fn from n-contact (right) and Fp from p-contact (left)
    Fn = phi_right - cfg.Vt*np.log(max(cfg.ND,1.0)/cfg.n_i)  # n = ND at right
    Fp = phi_left  + cfg.Vt*np.log(max(cfg.NA,1.0)/cfg.n_i)  # p = NA at left
    return Fn, Fp

def carriers_from_phi(cfg: Cfg, phi, Fn, Fp):
    # n = ni * exp((phi - Fn)/Vt), p = ni * exp((Fp - phi)/Vt)
    n = cfg.n_i * np.exp((phi - Fn)/cfg.Vt)
    p = cfg.n_i * np.exp((Fp - phi)/cfg.Vt)
    return n, p

def poiss_rhs(cfg: Cfg, n, p, ND, NA):
    # ∇·(ε∇φ) = -ρ,  ρ = q(p - n + ND - NA)
    rho = cfg.q*(p - n + ND - NA)  # C/m^3 (ND/NA in cm^-3, scaled below)
    # convert ND, NA, n, p from cm^-3 to m^-3 if provided in cm^-3:
    # Here we assume ND/NA/n/p are in cm^-3 -> multiply by 1e6 for m^-3
    return -rho*1e6

def solve_poisson(cfg: Cfg, eps, ND, NA, Fn, Fp, Vbias):
    NX,NY,NZ = cfg.NX, cfg.NY, cfg.NZ
    A, b = assemble_poisson(cfg, eps, bc_left=-Vbias/2, bc_right=+Vbias/2)
    # initial phi (linear between contacts)
    phi = np.linspace(-Vbias/2, +Vbias/2, NX).reshape(NX,1,1)
    phi = np.broadcast_to(phi, (NX,NY,NZ)).copy()

    for it in range(cfg.max_iter):
        # update carriers from current phi
        n, p = carriers_from_phi(cfg, phi, Fn, Fp)
        rhs = poiss_rhs(cfg, n, p, ND, NA).reshape(-1)

        phi_new = spsolve(A, b - rhs)
        phi_new = phi_new.reshape(NX,NY,NZ)

        dphi = np.max(np.abs(phi_new - phi))
        phi = (1-cfg.relax)*phi + cfg.relax*phi_new
        if dphi < cfg.tol_phi:
            # converged
            break
    return phi, n, p, it+1

# ------------------------- Soref–Bennett Δn -------------------------
def soref_delta_n(cfg: Cfg, n, p):
    dN = np.maximum(n - cfg.n_i, 0.0)
    dP = np.maximum(p - cfg.n_i, 0.0)
    # n,p in cm^-3 for coefficients; ensure they are
    # If you want strict units, pass cm^-3 arrays. Here n,p are in cm^-3 already.
    dn = -(cfg.an*dN + cfg.ap*(dP**cfg.gp))
    return dn

# ------------------------- Helmholtz Solve -------------------------
def assemble_helmholtz(cfg: Cfg, n_tot, pml_cells, pml_strength):
    NX,NY,NZ = cfg.NX, cfg.NY, cfg.NZ
    dx = cfg.Lx/(NX-1); dy = cfg.Ly/(NY-1); dz = cfg.Lz/(NZ-1)
    k0 = 2*math.pi/(cfg.wavelength_um)  # um^-1
    # convert grid spacing to um
    dx_um, dy_um, dz_um = dx, dy, dz

    # simple complex shift as absorbing layer near boundaries
    sigma = np.zeros((NX,NY,NZ), dtype=np.float64)
    for i in range(NX):
        di = min(i, NX-1-i)
        if di < pml_cells: sigma[i,:,:] += pml_strength*(1 - di/pml_cells)
    for j in range(NY):
        dj = min(j, NY-1-j)
        if dj < pml_cells: sigma[:,j,:] += pml_strength*(1 - dj/pml_cells)
    for k in range(NZ):
        dk = min(k, NZ-1-k)
        if dk < pml_cells: sigma[:,:,k] += pml_strength*(1 - dk/pml_cells)

    N = NX*NY*NZ
    rows=[]; cols=[]; data=[]; rhs = np.zeros(N, dtype=np.complex128)

    def add(ii,jj,v): rows.append(ii); cols.append(jj); data.append(v)

    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                p = idx(i,j,k,NX,NY,NZ)

                # Dirichlet at boundary (psi=0) (PML still helps suppress reflection)
                if i in (0,NX-1) or j in (0,NY-1) or k in (0,NZ-1):
                    add(p,p,1.0+0j); rhs[p]=0.0
                    continue

                cx = 1.0/(dx_um*dx_um)
                cy = 1.0/(dy_um*dy_um)
                cz = 1.0/(dz_um*dz_um)
                diag = -(2*cx + 2*cy + 2*cz)

                # Laplacian stencil
                add(p,p,diag + (k0**2)*(n_tot[i,j,k]**2) - 1j*sigma[i,j,k])

                q = idx(i-1,j,k,NX,NY,NZ); add(p,q,cx)
                q = idx(i+1,j,k,NX,NY,NZ); add(p,q,cx)
                q = idx(i,j-1,k,NX,NY,NZ); add(p,q,cy)
                q = idx(i,j+1,k,NX,NY,NZ); add(p,q,cy)
                q = idx(i,j,k-1,NX,NY,NZ); add(p,q,cz)
                q = idx(i,j,k+1,NX,NY,NZ); add(p,q,cz)

    A = coo_matrix((data,(rows,cols)), shape=(N,N), dtype=np.complex128).tocsr()
    return A, rhs

def add_gaussian_source(cfg: Cfg, rhs, where, amp=1.0, sigma_cells=2.0):
    NX,NY,NZ = cfg.NX, cfg.NY, cfg.NZ
    cx,cy,cz = where
    X,Y,Z = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing='ij')
    G = np.exp(-((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2)/(2*sigma_cells**2))
    rhs[:] += (amp*G).reshape(-1)

# ------------------------- Dataset Generation -------------------------
def make_sample(cfg: Cfg, Vbias):
    X,Y,Z, core, rib, slab, left, right, n_base_opt, eps = make_masks(cfg)

    # doping (cm^-3)
    ND = np.zeros_like(core) + cfg.ND*right
    NA = np.zeros_like(core) + cfg.NA*left

    # quasi-Fermi levels from contacts
    Fn, Fp = init_quasi_fermi(cfg, phi_left=-Vbias/2, phi_right=+Vbias/2)

    # solve Poisson self-consistently
    phi, n_cm3, p_cm3, iters = solve_poisson(cfg, eps, ND, NA, Fn, Fp, Vbias)

    # Soref-Bennett dn (Δn)
    dn = soref_delta_n(cfg, n_cm3, p_cm3)

    # optical total index
    n_tot = n_base_opt + dn
    n_tot = np.clip(n_tot, 1.0, None)

    # Helmholtz
    A, rhs = assemble_helmholtz(cfg, n_tot, cfg.pml_cells, cfg.pml_strength)
    # source near x-min core region
    cx = 2; cy = max(int(cfg.NY*0.5),1); cz = int(cfg.NZ/2)
    add_gaussian_source(cfg, rhs, (cx,cy,cz), amp=cfg.src_amp, sigma_cells=2.0)
    psi = spsolve(A, rhs).reshape(cfg.NX, cfg.NY, cfg.NZ)

    # checks: Poisson residual
    # ∇·(ε∇φ)+ρ = 0
    dx = cfg.Lx/(cfg.NX-1); dy = cfg.Ly/(cfg.NY-1); dz = cfg.Lz/(cfg.NZ-1)
    def ddx(u): return (np.roll(u,-1,0)-np.roll(u,1,0))/(2*dx)
    def ddy(u): return (np.roll(u,-1,1)-np.roll(u,1,1))/(2*dy)
    def ddz(u): return (np.roll(u,-1,2)-np.roll(u,1,2))/(2*dz)
    Jx = eps*ddx(phi); Jy = eps*ddy(phi); Jz = eps*ddz(phi)
    divJ = (np.roll(Jx,-1,0)-np.roll(Jx,1,0))/(2*dx) \
         + (np.roll(Jy,-1,1)-np.roll(Jy,1,1))/(2*dy) \
         + (np.roll(Jz,-1,2)-np.roll(Jz,1,2))/(2*dz)
    rho = cfg.q*(p_cm3 - n_cm3 + ND - NA)*1e6
    poisson_res = divJ + rho

    meta = dict(
        Vbias=float(Vbias),
        gummel_iters=int(iters),
        Fn=float(Fn), Fp=float(Fp),
        shape=[int(cfg.NX),int(cfg.NY),int(cfg.NZ)],
        dx=float(dx), dy=float(dy), dz=float(dz),
        wavelength_um=float(cfg.wavelength_um)
    )

    sample = dict(
        inputs=dict(
            V=np.full_like(core, Vbias, dtype=np.float32),
            mask_core=core.astype(np.float32),
            mask_rib=rib.astype(np.float32),
            mask_slab=slab.astype(np.float32),
            mask_left=left.astype(np.float32),
            mask_right=right.astype(np.float32),
            ND=ND.astype(np.float32),
            NA=NA.astype(np.float32)
        ),
        labels=dict(
            phi=phi.astype(np.float32),
            n=n_cm3.astype(np.float32),   # cm^-3
            p=p_cm3.astype(np.float32),   # cm^-3
            dn=dn.astype(np.float32),     # Δn
            n_tot=n_tot.astype(np.float32),
            psi=psi.astype(np.complex64)
        ),
        checks=dict(
            poisson_res=poisson_res.astype(np.float64)
        ),
        meta=meta
    )
    return sample

def save_npz_sample(path, sample):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # flatten nested dict keys
    out = {}
    for k,v in sample["inputs"].items():
        out[f"inputs/{k}"] = v
    for k,v in sample["labels"].items():
        out[f"labels/{k}"] = v
    for k,v in sample["checks"].items():
        out[f"checks/{k}"] = v
    out["meta/json"] = np.string_(json.dumps(sample["meta"]))
    np.savez_compressed(path, **out)

def main():
    cfg = CFG
    np.random.seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    print("== 3D HF dataset generation ==")
    print(asdict(cfg))

    for s in range(cfg.n_samples):
        V = np.random.uniform(*cfg.V_range)
        t0=time.time()
        sample = make_sample(cfg, Vbias=V)
        path = os.path.join(cfg.out_dir, f"sample_V{V:.2f}.npz")
        save_npz_sample(path, sample)
        dt=time.time()-t0
        print(f"  saved {path}  ({dt:.1f}s)  gummel_iters={sample['meta']['gummel_iters']}")
    print("Done.")

if __name__ == "__main__":
    main()
