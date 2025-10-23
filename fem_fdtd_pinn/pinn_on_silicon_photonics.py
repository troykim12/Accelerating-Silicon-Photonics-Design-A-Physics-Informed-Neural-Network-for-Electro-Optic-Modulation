"""
Physics-Informed Neural Network (PINN) Framework for Silicon Photonics
완전한 멀티피직스 시뮬레이션 + PINN 학습 + 추론 파이프라인

✅ FINAL PATCHED VERSION:
- [FIXED] Critical derivative bug: All losses now computed in PHYSICAL space.
- [FIXED] Removed artificial output scaling from model.forward.
- [MERGED] Added Hard Physics Constraints (Constraint Loss).
- [MERGED] Added Residual Adaptive Refinement (RAR) sampling.
- [MERGED] Added physical-scale normalization to Loss functions.
- [MERGED] Implemented 3-stage pre-training curriculum.
- [MERGED] Using stable MANUAL loss weights.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import pandas as pd
from tqdm import tqdm

# ==================== Physical Constants (CGS units) ====================
q = 1.602e-19
eps0 = 8.854e-14
eps_si = 11.7 * eps0
eps_sio2 = 3.9 * eps0
n_si = np.sqrt(11.7)
n_sio2 = np.sqrt(3.9)
kB, T = 1.380649e-23, 300.0
VT = kB * T / q  # Thermal voltage ~0.026 V

# Geometry (cm)
um = 1e-4
W_clad = 3.0 * um
H_clad = 1.8 * um
L_clad = 6.0 * um
w_core = 0.5 * um
h_core = 0.22 * um

cx0 = (W_clad - w_core) / 2
cx1 = (W_clad + w_core) / 2
cy0 = (H_clad - h_core) / 2
cy1 = (H_clad + h_core) / 2
cxm = (cx0 + cx1) / 2

# Optical
lambda_cm = 1.55e-4
k0 = 2 * np.pi / lambda_cm

# Doping
NA = 1e17  # cm^-3
ND = 1e17
ni = 1e10

# ==================== Soref-Bennett Model ====================

def soref_bennett_dn_torch(dN, dP):
    """Δn from carrier changes (PyTorch version for autograd)"""
    return -(8.8e-22 * dN + 8.5e-18 * torch.pow(torch.clamp(dP, min=0.0), 0.8))


# ==================== PINN Network Architecture ====================

class FourierFeatureEmbedding(nn.Module):
    def __init__(self, input_dim: int, embedding_size: int, scale: float = 1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(input_dim, embedding_size) * scale,
                              requires_grad=False)

    def forward(self, x_normalized):
        # x_normalized: (batch, input_dim) - already normalized to [0,1]
        x_proj = 2 * np.pi * x_normalized @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MultiPhysicsPINN(nn.Module):
    """
    Inputs: (x, y, z, V_bias) - NORMALIZED to [0,1]
    Outputs: (φ, n, p, ...) - In PHYSICAL units
    """
    def __init__(self,
                 hidden_dims: List[int] = [256, 256, 256, 256],
                 fourier_features: int = 256,
                 fourier_scale: float = 10.0):
        super().__init__()

        self.fourier = FourierFeatureEmbedding(4, fourier_features, fourier_scale)
        input_dim = 2 * fourier_features

        # Electrical sub-network
        elec_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            elec_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.Tanh(),
            ])
            in_dim = h_dim
        self.elec_net = nn.Sequential(*elec_layers)
        self.elec_out = nn.Linear(in_dim, 3) # φ, log(n), log(p)

        # Optical sub-network
        opt_layers = []
        in_dim = input_dim + 3
        for h_dim in hidden_dims:
            opt_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.Tanh(),
            ])
            in_dim = h_dim
        self.opt_net = nn.Sequential(*opt_layers)
        self.opt_out = nn.Linear(in_dim, 4) # E_x, E_y, E_z_real, E_z_imag

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, coords_normalized):
        embedded = self.fourier(coords_normalized)

        # Electrical physics
        elec_features = self.elec_net(embedded)
        elec_raw = self.elec_out(elec_features)

        # [FIXED] Removed artificial output scaling.
        # Let the network learn the physical scales.
        phi = elec_raw[:, 0:1]
        log_n = elec_raw[:, 1:2]
        log_p = elec_raw[:, 2:3]

        n = torch.exp(torch.clamp(log_n, -50, 50)) # Clamp for stability
        p = torch.exp(torch.clamp(log_p, -50, 50))

        # Optical physics
        opt_input = torch.cat([embedded, phi, log_n, log_p], dim=-1)
        opt_features = self.opt_net(opt_input)
        opt_raw = self.opt_out(opt_features)

        Ex = opt_raw[:, 0:1]
        Ey = opt_raw[:, 1:2]
        Ez_real = opt_raw[:, 2:3]
        Ez_imag = opt_raw[:, 3:4]

        return {
            'phi': phi,
            'n': n,
            'p': p,
            'Ex': Ex,
            'Ey': Ey,
            'Ez_real': Ez_real,
            'Ez_imag': Ez_imag
        }

# ==================== Physics Loss with Correct Derivatives ====================

class PhysicsLoss:
    """
    [FIXED] All methods take PHYSICAL coordinates for derivatives.
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.NA = NA
        self.ND = ND
        self.ni = ni

        # [MERGED] Physical scaling factors for losses
        self.poisson_scale = q * self.NA # Typical charge density
        self.continuity_scale = (q * 1400.0 * self.NA * 0.01) / w_core # Typical current density
        self.helmholtz_scale = k0**2
        self.bc_scale = 0.01 # Typical voltage

    def is_in_core(self, coords_phys):
        x = coords_phys[:, 0:1]
        y = coords_phys[:, 1:2]
        mask = ((x >= cx0) & (x <= cx1) & (y >= cy0) & (y <= cy1))
        return mask.float()

    def doping_profile(self, coords_phys):
        x = coords_phys[:, 0:1]
        in_core = self.is_in_core(coords_phys)
        is_p = (x < cxm).float()
        is_n = (x >= cxm).float()
        doping = in_core * (is_n * self.ND - is_p * self.NA)
        return doping

    def poisson_loss(self, coords_phys, fields):
        phi = fields['phi']
        n = fields['n']
        p = fields['p']

        # [FIXED] Derivatives w.r.t. PHYSICAL coordinates
        grad_phi = self.gradient(phi, coords_phys)

        in_core = self.is_in_core(coords_phys)
        eps = in_core * eps_si + (1 - in_core) * eps_sio2

        flux = eps.unsqueeze(-1) * grad_phi
        div_flux = self.divergence(flux, coords_phys)

        doping = self.doping_profile(coords_phys)
        rho = q * (p - n + doping)

        residual = div_flux + rho

        # [MERGED] Apply physical loss normalization
        return torch.mean((residual / self.poisson_scale) ** 2)

    def continuity_loss(self, coords_phys, fields):
        n = fields['n']
        p = fields['p']
        phi = fields['phi']

        mu_n, D_n = 1400.0, 36.0
        mu_p, D_p = 450.0, 12.0

        # [FIXED] Derivatives w.r.t. PHYSICAL coordinates
        grad_phi = self.gradient(phi, coords_phys)
        grad_n = self.gradient(n, coords_phys)
        grad_p = self.gradient(p, coords_phys)

        J_n = q * mu_n * n.unsqueeze(-1) * grad_phi + q * D_n * grad_n
        div_Jn = self.divergence(J_n, coords_phys)

        J_p = -q * mu_p * p.unsqueeze(-1) * grad_phi - q * D_p * grad_p
        div_Jp = self.divergence(J_p, coords_phys)

        # [MERGED] Apply physical loss normalization
        loss_n = torch.mean((div_Jn / self.continuity_scale) ** 2)
        loss_p = torch.mean((div_Jp / self.continuity_scale) ** 2)

        return loss_n + loss_p

    def helmholtz_loss(self, coords_phys, fields):
        Ez_real = fields['Ez_real']
        Ez_imag = fields['Ez_imag']
        Ez = torch.complex(Ez_real, Ez_imag)

        # [FIXED] Derivatives w.r.t. PHYSICAL coordinates
        laplacian_Ez = self.laplacian(Ez, coords_phys)

        n_field = fields['n']
        p_field = fields['p']

        n_eq = torch.full_like(n_field, self.ni)
        p_eq = torch.full_like(p_field, self.ni)

        dN = n_field - n_eq
        dP = p_field - p_eq
        delta_n_tensor = soref_bennett_dn_torch(dN, dP)

        in_core = self.is_in_core(coords_phys)
        n_total = in_core * (n_si + delta_n_tensor) + (1 - in_core) * n_sio2
        n_total = torch.clamp(n_total, 1.0, 5.0)

        eps_r = n_total ** 2
        k_squared = k0 ** 2 * eps_r

        residual = laplacian_Ez + k_squared.unsqueeze(-1) * Ez

        # [MERGED] Apply physical loss normalization
        return torch.mean(torch.abs(residual / self.helmholtz_scale) ** 2)

    def boundary_loss(self, coords_phys_bc, fields_bc):
        phi_bc = fields_bc['phi']
        V_bias = coords_phys_bc[:, 3:4]
        z_phys = coords_phys_bc[:, 2:3]

        # Simple BC: 0V at z=0, V_bias at z=L
        target_phi = torch.where(z_phys < L_clad / 2, 0.0, V_bias)
        loss = torch.mean(((phi_bc - target_phi) / self.bc_scale) ** 2)

        return loss

    # [MERGED] Hard physics constraints
    def physics_constraints(self, coords_phys, fields):
        """HARD CONSTRAINTS: Force carriers to match doping in bulk regions"""
        x = coords_phys[:, 0:1]
        in_core = self.is_in_core(coords_phys)

        is_p = (x < (cxm - 0.1*um)).float() * in_core # Far from junction
        is_n = (x > (cxm + 0.1*um)).float() * in_core # Far from junction

        p_bulk = torch.full_like(fields['p'], self.NA)
        n_bulk = torch.full_like(fields['n'], self.ND)

        # Constraint: p = NA in p-region, n = ND in n-region
        constraint_p = torch.mean((is_p * (fields['p'] - p_bulk) / self.NA)**2)
        constraint_n = torch.mean((is_n * (fields['n'] - n_bulk) / self.ND)**2)

        # Constraint: n = ni^2/NA in p-region, p = ni^2/ND in n-region
        constraint_n_min = torch.mean((is_p * (fields['n'] - (self.ni**2 / self.NA)) / self.ni)**2)
        constraint_p_min = torch.mean((is_n * (fields['p'] - (self.ni**2 / self.ND)) / self.ni)**2)

        return constraint_p + constraint_n + constraint_n_min + constraint_p_min

    # Utility functions for derivatives
    def gradient(self, y, x_phys):
        grad_outputs = torch.ones_like(y)
        grad = torch.autograd.grad(outputs=y, inputs=x_phys, grad_outputs=grad_outputs,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        return grad[:, :3]

    def divergence(self, vector_field, x_phys):
        div = torch.zeros(vector_field.shape[0], 1, device=self.device)
        for i in range(3):
            grad_i = self.gradient(vector_field[:, i:i+1], x_phys)
            div += grad_i[:, i:i+1]
        return div

    def laplacian(self, scalar_field, x_phys):
        grad = self.gradient(scalar_field, x_phys)
        lap = torch.zeros_like(scalar_field)
        for i in range(3):
            grad_i = self.gradient(grad[:, i:i+1], x_phys)
            lap += grad_i[:, i:i+1]
        return lap

# ==================== Trainer with All Fixes ====================

class PINNTrainer:
    def __init__(self, model: MultiPhysicsPINN,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.model = model.to(device)
        self.device = device
        self.physics_loss = PhysicsLoss(device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Stable LR
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=100
        )

        # [FIXED] Stable MANUAL weights
        self.loss_weights = {
            'poisson': 1.0,
            'continuity': 0.0, # Start at 0
            'helmholtz': 0.0, # Start at 0
            'bc': 10.0,
            'constraint': 0.0 # Start at 0
        }

        self.loss_history = []

        # [MERGED] 3-stage pre-training schedule
        self.schedule = {
            'stage1_epochs': 500,  # Poisson + BC
            'stage2_epochs': 1000, # + Continuity + Constraint
            'stage3_epochs': 1500  # + Helmholtz
        }

    def normalize_coords(self, coords_phys):
        coords_norm = coords_phys.clone()
        coords_norm[:, 0] /= W_clad
        coords_norm[:, 1] /= H_clad
        coords_norm[:, 2] /= L_clad
        coords_norm[:, 3] /= 0.05
        return coords_norm

    # [MERGED] RAR logic included
    def generate_training_points(self, n_points: int = 10000, adaptive: bool = False):
        """Generate collocation points with optional adaptive refinement"""
        x = torch.rand(n_points, 1) * W_clad
        y = torch.rand(n_points, 1) * H_clad
        z = torch.rand(n_points, 1) * L_clad
        V_bias = torch.rand(n_points, 1) * 0.05

        interior_phys = torch.cat([x, y, z, V_bias], dim=1).to(self.device)

        # Boundary points
        n_bc = n_points // 10
        x_bc = torch.rand(n_bc, 1) * W_clad
        y_bc = torch.rand(n_bc, 1) * H_clad
        z_bc = torch.cat([torch.zeros(n_bc//2, 1),
                          torch.full((n_bc - n_bc//2, 1), L_clad)], dim=0)
        V_bc = torch.rand(n_bc, 1) * 0.05

        boundary_phys = torch.cat([x_bc, y_bc, z_bc, V_bc], dim=1).to(self.device)

        # ADAPTIVE REFINEMENT: More points near junction
        if adaptive:
            n_junc = n_points // 5
            x_junc = torch.randn(n_junc, 1) * 0.2 * um + cxm # Near junction
            x_junc = torch.clamp(x_junc, cx0, cx1)
            y_junc = torch.rand(n_junc, 1) * h_core + cy0
            z_junc = torch.rand(n_junc, 1) * L_clad
            V_junc = torch.rand(n_junc, 1) * 0.05

            junction_points = torch.cat([x_junc, y_junc, z_junc, V_junc], dim=1).to(self.device)
            interior_phys = torch.cat([interior_phys, junction_points], dim=0)

        return interior_phys, boundary_phys

    # [MERGED] 3-stage curriculum
    def update_schedule(self, epoch):
        if epoch < self.schedule['stage1_epochs']:
            # Stage 1: Learn potential
            self.loss_weights['poisson'] = 1.0
            self.loss_weights['continuity'] = 0.0
            self.loss_weights['helmholtz'] = 0.0
            self.loss_weights['constraint'] = 0.0
            self.loss_weights['bc'] = 10.0
        elif epoch < self.schedule['stage2_epochs']:
            # Stage 2: Add carrier transport
            self.loss_weights['poisson'] = 1.0
            self.loss_weights['continuity'] = 1.0
            self.loss_weights['helmholtz'] = 0.0
            self.loss_weights['constraint'] = 10.0 # Turn on constraints
            self.loss_weights['bc'] = 10.0
        elif epoch < self.schedule['stage3_epochs']:
            # Stage 3: Gradually add optical
            self.loss_weights['poisson'] = 1.0
            self.loss_weights['continuity'] = 1.0
            progress = (epoch - self.schedule['stage2_epochs']) / (self.schedule['stage3_epochs'] - self.schedule['stage2_epochs'])
            self.loss_weights['helmholtz'] = 1e-5 * progress # Start very small
            self.loss_weights['constraint'] = 5.0
            self.loss_weights['bc'] = 5.0
        else:
            # Full multi-physics
            self.loss_weights['helmholtz'] = 1e-5 # Final weight

    def compute_loss(self, coords_int_phys, coords_bc_phys):
        """
        [FIXED] Correct derivative workflow
        """
        coords_int_phys.requires_grad = True
        coords_bc_phys.requires_grad = True

        coords_int_norm = self.normalize_coords(coords_int_phys)
        coords_bc_norm = self.normalize_coords(coords_bc_phys)

        fields_int = self.model(coords_int_norm)
        fields_bc = self.model(coords_bc_norm)

        # [FIXED] All losses computed w.r.t. PHYSICAL coordinates
        L_poisson = self.physics_loss.poisson_loss(coords_int_phys, fields_int)
        L_continuity = self.physics_loss.continuity_loss(coords_int_phys, fields_int)
        L_helmholtz = self.physics_loss.helmholtz_loss(coords_int_phys, fields_int)
        L_bc = self.physics_loss.boundary_loss(coords_bc_phys, fields_bc)
        L_constraint = self.physics_loss.physics_constraints(coords_int_phys, fields_int) # [MERGED]

        total_loss = (
            self.loss_weights['poisson'] * L_poisson +
            self.loss_weights['continuity'] * L_continuity +
            self.loss_weights['helmholtz'] * L_helmholtz +
            self.loss_weights['bc'] * L_bc +
            self.loss_weights['constraint'] * L_constraint
        )

        return total_loss, {
            'poisson': L_poisson.item(),
            'continuity': L_continuity.item(),
            'helmholtz': L_helmholtz.item(),
            'bc': L_bc.item(),
            'constraint': L_constraint.item(),
            'total': total_loss.item()
        }

    def train(self, epochs: int = 3000, n_points: int = 20000):
        print(f"Training PINN on {self.device} with FINAL PATCH")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("\nTraining Schedule:")
        print(f"  Stage 1 (0-{self.schedule['stage1_epochs']}): Poisson + BC")
        print(f"  Stage 2 ({self.schedule['stage1_epochs']}-{self.schedule['stage2_epochs']}): + Continuity + Constraints")
        print(f"  Stage 3 ({self.schedule['stage2_epochs']}+): + Helmholtz")

        pbar = tqdm(range(epochs), desc="Training")

        for epoch in pbar:
            self.update_schedule(epoch)

            # [MERGED] Use RAR after stage 1
            adaptive = epoch > self.schedule['stage1_epochs']
            coords_int_phys, coords_bc_phys = self.generate_training_points(n_points, adaptive=adaptive)

            self.optimizer.zero_grad()
            loss, loss_dict = self.compute_loss(coords_int_phys, coords_bc_phys)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.scheduler.step(loss)
            self.loss_history.append(loss_dict)

            if epoch % 20 == 0:
                pbar.set_postfix({
                    'Loss': f"{loss_dict['total']:.2e}",
                    'Pois': f"{loss_dict['poisson']:.2e}",
                    'Cont': f"{loss_dict['continuity']:.2e}",
                    'Cons': f"{loss_dict['constraint']:.2e}",
                    'LR': f"{self.optimizer.param_groups[0]['lr']:.1e}"
                })

    def plot_loss_history(self):
        df = pd.DataFrame(self.loss_history)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Individual losses
        for ax, col in zip(axes.flat, ['poisson', 'continuity', 'helmholtz', 'bc', 'constraint']):
            if col in df.columns and df[col].max() > 0:
                ax.semilogy(df[col], linewidth=0.5, alpha=0.7)
                ax.set_title(f'{col.capitalize()} Loss', fontsize=14)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.grid(True, alpha=0.3)

        # Plot total loss
        axes[1, 2].semilogy(df['total'], linewidth=0.5, alpha=0.7, color='r')
        axes[1, 2].set_title('Total Loss', fontsize=14)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].grid(True, alpha=0.3)


        plt.tight_layout()
        plt.savefig('pinn_training_FINAL_PATCHED.png', dpi=200)
        plt.show()

# ==================== Inference and Visualization ====================

def inference_slice(model, V_bias=0.02, z_slice=None, resolution=150):
    if z_slice is None:
        z_slice = L_clad / 2

    # Create grid in PHYSICAL units
    x = torch.linspace(0, W_clad, resolution)
    y = torch.linspace(0, H_clad, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    Z = torch.full_like(X, z_slice)
    V = torch.full_like(X, V_bias)

    coords_phys = torch.stack([X.flatten(), Y.flatten(), Z.flatten(), V.flatten()], dim=1)

    # Normalize coords for model input
    coords_norm = coords_phys.clone()
    coords_norm[:, 0] /= W_clad
    coords_norm[:, 1] /= H_clad
    coords_norm[:, 2] /= L_clad
    coords_norm[:, 3] /= 0.05

    model.eval()
    with torch.no_grad():
        coords_norm = coords_norm.to(next(model.parameters()).device)
        # Model gets normalized coords, outputs physical values
        fields = model(coords_norm)

    results = {}
    for key, val in fields.items():
        results[key] = val.cpu().numpy().reshape(resolution, resolution)

    results['X'] = X.numpy()
    results['Y'] = Y.numpy()

    return results

def visualize_results(results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    extent = [0, W_clad/um, 0, H_clad/um]

    # Potential
    im0 = axes[0,0].imshow(results['phi'].T * 1000, origin='lower', extent=extent, cmap='RdBu')
    axes[0,0].set_title('φ [mV]', fontsize=14)
    plt.colorbar(im0, ax=axes[0,0], label='mV')

    # Carriers
    n_plot = np.clip(results['n'], 1e6, 1e20)
    im1 = axes[0,1].imshow(np.log10(n_plot).T, origin='lower', extent=extent, cmap='viridis', vmin=8, vmax=18)
    axes[0,1].set_title('log₁₀(n) [cm⁻³]', fontsize=14)
    plt.colorbar(im1, ax=axes[0,1])

    p_plot = np.clip(results['p'], 1e6, 1e20)
    im2 = axes[0,2].imshow(np.log10(p_plot).T, origin='lower', extent=extent, cmap='plasma', vmin=8, vmax=18)
    axes[0,2].set_title('log₁₀(p) [cm⁻³]', fontsize=14)
    plt.colorbar(im2, ax=axes[0,2])

    # Optical fields
    im3 = axes[1,0].imshow(results['Ez_real'].T, origin='lower', extent=extent, cmap='RdBu')
    axes[1,0].set_title('Re(E_z)', fontsize=14)
    plt.colorbar(im3, ax=axes[1,0])

    im4 = axes[1,1].imshow(results['Ez_imag'].T, origin='lower', extent=extent, cmap='RdBu')
    axes[1,1].set_title('Im(E_z)', fontsize=14)
    plt.colorbar(im4, ax=axes[1,1])

    intensity = results['Ez_real']**2 + results['Ez_imag']**2
    im5 = axes[1,2].imshow(intensity.T, origin='lower', extent=extent, cmap='hot')
    axes[1,2].set_title('|E_z|² (Intensity)', fontsize=14)
    plt.colorbar(im5, ax=axes[1,2])

    # Mark structures
    for ax in axes.flat:
        ax.axvline(cx0/um, color='cyan', linestyle='--', linewidth=0.8, alpha=0.6, label='Core')
        ax.axvline(cx1/um, color='cyan', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axhline(cy0/um, color='cyan', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axhline(cy1/um, color='cyan', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axvline(cxm/um, color='yellow', linestyle=':', linewidth=1.2, alpha=0.8, label='p-n Junction')
        ax.set_xlabel('x [μm]', fontsize=12)
        ax.set_ylabel('y [μm]', fontsize=12)

    axes[0,0].legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig('pinn_results_FINAL_PATCHED.png', dpi=200)
    plt.show()

# ==================== Main Execution ====================

if __name__ == "__main__":
    print("=" * 70)
    print("PINN Framework - FINAL PATCHED VERSION")
    print("This version corrects the critical derivative bug.")
    print("=" * 70)

    # Initialize model
    model = MultiPhysicsPINN(
        hidden_dims=[256, 256, 256, 256],
        fourier_features=256,
        fourier_scale=10.0
    )

    # Initialize trainer
    trainer = PINNTrainer(model)

    # Train
    print("\n[1] Training PINN...")
    trainer.train(epochs=3000, n_points=20000)

    # Plot training history
    print("\n[2] Plotting training history...")
    trainer.plot_loss_history()

    # Inference
    print("\n[3] Running inference...")
    results = inference_slice(model, V_bias=0.02, resolution=150)

    # Visualize
    print("\n[4] Visualizing results...")
    visualize_results(results)

    # Save model
    torch.save(model.state_dict(), 'pinn_silicon_FINAL_PATCHED.pth')
    print("\n✓ Model saved: pinn_silicon_FINAL_PATCHED.pth")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - FINAL REPORT")
    print("=" * 70)
    if trainer.loss_history:
        final = trainer.loss_history[-1]
        print(f"Final Losses:")
        print(f"  Poisson:    {final['poisson']:.3e}")
        print(f"  Continuity: {final['continuity']:.3e}")
        print(f"  Helmholtz:  {final['helmholtz']:.3e}")
        print(f"  BC:         {final['bc']:.3e}")
        print(f"  Constraint: {final['constraint']:.3e}")
        print(f"  Total:      {final['total']:.3e}")
    print("=" * 70)
    print("Please check 'pinn_results_FINAL_PATCHED.png' for results.")
    print("If results are still not good, consider:")
    print("  1. Increasing epochs (5000+)")
    print("  2. Tuning curriculum schedule (e.g., more epochs per stage)")
    print("  3. Tuning constraint/helmholtz loss weights")
