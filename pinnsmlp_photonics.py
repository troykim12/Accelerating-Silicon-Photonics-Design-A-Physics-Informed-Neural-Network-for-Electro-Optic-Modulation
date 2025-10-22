"""
Physics-Informed Neural Network (PINN) Framework for Silicon Photonics
- Feedback from pinn_advanced_feedback_ko.md applied
- Patched Version:
  ✓ Step 1: Reverted to simpler baseline (pinn_modified.py logic)
  ✓ Step 2: Correct Normalization (Loss calculated in PHYSICAL space)
  ✓ Step 3: Removed Output Scaling (from model.forward)
  ✓ Step 4: Stable Manual Weights + Curriculum Learning
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
n_si = np.sqrt(11.7)      # ≈ 3.42
n_sio2 = np.sqrt(3.9)     # ≈ 1.975

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
cxm = (cx0 + cx1) / 2  # p-n junction

# Optical
lambda_cm = 1.55e-4  # 1.55 μm
k0 = 2 * np.pi / lambda_cm

# ==================== Soref-Bennett Model ====================

# [MODIFIED] Added a PyTorch-native version of the Soref-Bennett model
def soref_bennett_dn_torch(dN, dP):
    """Δn from carrier changes (PyTorch version)"""
    # Use torch.clamp to ensure dP is non-negative before applying power
    return -(8.8e-22 * dN + 8.5e-18 * torch.pow(torch.clamp(dP, min=0.0), 0.8))

# ==================== PINN Network Architecture ====================

class FourierFeatureEmbedding(nn.Module):
    """
    Fourier Feature Networks to overcome spectral bias
    Tancik et al., NeurIPS 2020
    """
    def __init__(self, input_dim: int, embedding_size: int, scale: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_size = embedding_size
        # Random Fourier features
        self.B = nn.Parameter(torch.randn(input_dim, embedding_size) * scale,
                              requires_grad=False)

    def forward(self, x_normalized):
        # x_normalized: (batch, input_dim) - already normalized to [0,1]
        x_proj = 2 * np.pi * x_normalized @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MultiPhysicsPINN(nn.Module):
    """
    Physics-Informed Neural Network for coupled electro-optics

    Inputs: (x, y, z, V_bias) - NORMALIZED to [0,1]
    Outputs: (φ, n, p, E_x, E_y, E_z_real, E_z_imag) - In PHYSICAL units
    """

    def __init__(self,
                 hidden_dims: List[int] = [256, 256, 256, 256],
                 fourier_features: int = 256,
                 fourier_scale: float = 10.0): # F-feature scale
        super().__init__()

        # Fourier feature embedding
        self.fourier = FourierFeatureEmbedding(4, fourier_features, fourier_scale)
        input_dim = 2 * fourier_features

        # Electrical sub-network (φ, n, p)
        elec_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            elec_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.Tanh(),
            ])
            in_dim = h_dim
        self.elec_net = nn.Sequential(*elec_layers)
        self.elec_out = nn.Linear(in_dim, 3)  # φ, log(n), log(p)

        # Optical sub-network (E fields)
        opt_layers = []
        in_dim = input_dim + 3  # + (φ, log_n, log_p) from electrical
        for h_dim in hidden_dims:
            opt_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.Tanh(),
            ])
            in_dim = h_dim
        self.opt_net = nn.Sequential(*opt_layers)
        self.opt_out = nn.Linear(in_dim, 4)  # E_x, E_y, E_z_real, E_z_imag

        # Initialize weights (Xavier for stability)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, coords_normalized):
        """
        coords_normalized: (batch, 4) - [x, y, z, V_bias] in [0,1]
        returns: dict with all physical fields in physical units
        """
        # Fourier embedding
        embedded = self.fourier(coords_normalized)

        # Electrical physics
        elec_features = self.elec_net(embedded)
        elec_raw = self.elec_out(elec_features)

        # [PATCHED] Step 3: Removed all artificial output scaling.
        # Network now learns the physical scales directly.
        phi = elec_raw[:, 0:1]      # Potential
        log_n = elec_raw[:, 1:2]    # log(electron density)
        log_p = elec_raw[:, 2:3]    # log(hole density)

        n = torch.exp(log_n)  # Enforce positivity
        p = torch.exp(log_p)

        # Optical physics (conditioned on electrical)
        # Pass the raw log_n, log_p as they are more stable features
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

# ==================== Physics-Based Loss Functions ====================

class PhysicsLoss:
    """
    Implements all physics-informed loss terms.
    [PATCHED] All methods now take PHYSICAL coordinates for derivatives.
    """

    def __init__(self, device='cpu'):
        self.device = device

        # Doping profile
        self.NA = 1e17  # cm^-3
        self.ND = 1e17
        self.ni = 1e10

    def is_in_core(self, coords_phys):
        """Check if points are in Si core (uses physical coords)"""
        x, y = coords_phys[:, 0:1], coords_phys[:, 1:2]
        mask = ((x >= cx0) & (x <= cx1) & (y >= cy0) & (y <= cy1))
        return mask.float()

    def doping_profile(self, coords_phys):
        """N_D+ - N_A- (uses physical coords)"""
        x = coords_phys[:, 0:1]
        in_core = self.is_in_core(coords_phys)

        # Left side: p-type (NA), Right side: n-type (ND)
        is_p = (x < cxm).float()
        is_n = (x >= cxm).float()

        doping = in_core * (is_n * self.ND - is_p * self.NA)
        return doping

    def poisson_loss(self, coords_phys, fields):
        """
        Poisson equation: ∇·(ε∇φ) = -q(p - n + N_D - N_A)
        """
        phi = fields['phi']
        n = fields['n']
        p = fields['p']

        # [PATCHED] Compute gradients w.r.t. PHYSICAL coordinates
        grad_phi = self.gradient(phi, coords_phys)

        # Material-dependent permittivity
        in_core = self.is_in_core(coords_phys)
        eps = in_core * eps_si + (1 - in_core) * eps_sio2

        # Divergence of (ε∇φ)
        flux = eps.unsqueeze(-1) * grad_phi
        div_flux = self.divergence(flux, coords_phys)

        # Charge density
        doping = self.doping_profile(coords_phys)
        rho = q * (p - n + doping)

        # Residual
        residual = div_flux + rho
        return torch.mean(residual ** 2)

    def continuity_loss(self, coords_phys, fields):
        """
        Steady-state continuity: ∇·J_n = 0, ∇·J_p = 0
        """
        n = fields['n']
        p = fields['p']
        phi = fields['phi']

        # Electron current: J_n = q μ_n n ∇φ + q D_n ∇n
        mu_n, D_n = 1400.0, 36.0  # cm^2/V·s, cm^2/s

        # [PATCHED] Compute gradients w.r.t. PHYSICAL coordinates
        grad_phi = self.gradient(phi, coords_phys)
        grad_n = self.gradient(n, coords_phys)

        J_n = q * mu_n * n.unsqueeze(-1) * grad_phi + q * D_n * grad_n
        div_Jn = self.divergence(J_n, coords_phys)

        # Hole current: J_p = -q μ_p p ∇φ - q D_p ∇p
        mu_p, D_p = 450.0, 12.0
        grad_p = self.gradient(p, coords_phys)

        J_p = -q * mu_p * p.unsqueeze(-1) * grad_phi - q * D_p * grad_p
        div_Jp = self.divergence(J_p, coords_phys)

        loss_n = torch.mean(div_Jn ** 2)
        loss_p = torch.mean(div_Jp ** 2)

        return loss_n + loss_p

    def helmholtz_loss(self, coords_phys, fields):
        """
        Helmholtz equation: ∇²E + k₀²ε_r E = 0
        """
        Ez_real = fields['Ez_real']
        Ez_imag = fields['Ez_imag']
        Ez = torch.complex(Ez_real, Ez_imag)

        # [PATCHED] Compute gradients w.r.t. PHYSICAL coordinates
        laplacian_Ez = self.laplacian(Ez, coords_phys)

        # Modified refractive index from Soref-Bennett
        n_field = fields['n']
        p_field = fields['p']

        # Equilibrium (approximate)
        n_eq = torch.full_like(n_field, self.ni)
        p_eq = torch.full_like(p_field, self.ni)

        # Autograd-safe torch-native Soref-Bennett calculation
        dN = n_field - n_eq
        dP = p_field - p_eq
        delta_n_tensor = soref_bennett_dn_torch(dN, dP)

        in_core = self.is_in_core(coords_phys)
        n_total = in_core * (n_si + delta_n_tensor) + (1 - in_core) * n_sio2

        eps_r = n_total ** 2
        k_squared = k0 ** 2 * eps_r

        # Residual
        residual = laplacian_Ez + k_squared.unsqueeze(-1) * Ez
        return torch.mean(torch.abs(residual) ** 2)

    def boundary_loss(self, coords_phys_bc, fields_bc):
        """
        Boundary conditions at z=0 and z=L contacts (uses physical coords)
        """
        phi_bc = fields_bc['phi']
        V_bias = coords_phys_bc[:, 3:4]  # V_bias is already physical
        z_phys = coords_phys_bc[:, 2:3]  # z is already physical

        # At z=0: φ = 0, at z=L: φ = V_bias
        # [PATCHED] Simpler BC: 0V at z=0, V_bias at z=L
        target_phi = torch.where(z_phys < L_clad / 2, 0.0, V_bias)
        loss = torch.mean((phi_bc - target_phi) ** 2)

        return loss

    # ========== Utility functions for derivatives ==========

    def gradient(self, y, x_phys):
        """Compute ∇y with respect to PHYSICAL x_phys"""
        grad_outputs = torch.ones_like(y)
        grad = torch.autograd.grad(
            outputs=y,
            inputs=x_phys, # [PATCHED] Derivative w.r.t. physical coordinates
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        return grad[:, :3]  # (x, y, z) components only

    def divergence(self, vector_field, x_phys):
        """Compute ∇·F w.r.t. PHYSICAL x_phys"""
        div = torch.zeros(vector_field.shape[0], 1, device=self.device)
        for i in range(3):  # x, y, z
            grad_i = self.gradient(vector_field[:, i:i+1], x_phys)
            div += grad_i[:, i:i+1]
        return div

    def laplacian(self, scalar_field, x_phys):
        """Compute ∇²f w.r.t. PHYSICAL x_phys"""
        grad = self.gradient(scalar_field, x_phys)
        lap = torch.zeros_like(scalar_field)
        for i in range(3):
            grad_i = self.gradient(grad[:, i:i+1], x_phys)
            lap += grad_i[:, i:i+1]
        return lap

# ==================== Training Pipeline ====================

class PINNTrainer:
    """
    Complete training pipeline with stable settings
    """

    def __init__(self,
                 model: MultiPhysicsPINN,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.model = model.to(device)
        self.device = device
        self.physics_loss = PhysicsLoss(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Lower LR

        # [PATCHED] Step 4: Stable Manual Weights + Curriculum Learning
        self.loss_weights = {
            'poisson': 1.0,
            'continuity': 1.0,
            'helmholtz': 0.0,  # Start with 0 (curriculum learning)
            'bc': 10.0
        }

        # History
        self.loss_history = []

        # Curriculum learning schedule
        self.curriculum_schedule = {
            'electrical_only_epochs': 500,  # Train electrical part first
            'helmholtz_final_weight': 1e-20   # Final weight after curriculum
        }

    def normalize_coords(self, coords_phys):
        """Normalize physical coordinates to [0, 1]"""
        coords_norm = coords_phys.clone()
        coords_norm[:, 0] /= W_clad
        coords_norm[:, 1] /= H_clad
        coords_norm[:, 2] /= L_clad
        coords_norm[:, 3] /= 0.05  # V_bias: 0-50mV -> [0,1]
        return coords_norm

    def generate_training_points(self, n_points: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate collocation points in PHYSICAL units
        Returns: (interior_points_phys, boundary_points_phys)
        """
        # Interior points
        x = torch.rand(n_points, 1) * W_clad
        y = torch.rand(n_points, 1) * H_clad
        z = torch.rand(n_points, 1) * L_clad
        V_bias = torch.rand(n_points, 1) * 0.05  # 0-50mV range

        interior_phys = torch.cat([x, y, z, V_bias], dim=1).to(self.device)

        # Boundary points (z=0 and z=L contacts)
        n_bc = n_points // 10
        x_bc = torch.rand(n_bc, 1) * W_clad
        y_bc = torch.rand(n_bc, 1) * H_clad
        z_bc = torch.cat([torch.zeros(n_bc//2, 1),
                          torch.full((n_bc - n_bc//2, 1), L_clad)], dim=0)
        V_bc = torch.rand(n_bc, 1) * 0.05

        boundary_phys = torch.cat([x_bc, y_bc, z_bc, V_bc], dim=1).to(self.device)

        return interior_phys, boundary_phys

    def update_curriculum(self, epoch):
        """Update loss weights based on curriculum schedule"""
        if epoch < self.curriculum_schedule['electrical_only_epochs']:
            self.loss_weights['helmholtz'] = 0.0
        else:
            # Gradually ramp up or set directly
            self.loss_weights['helmholtz'] = self.curriculum_schedule['helmholtz_final_weight']

    def compute_loss(self, coords_int_phys, coords_bc_phys):
        """
        [PATCHED] Step 2: Correct Normalization/Derivative Workflow
        1. Normalize physical coords -> model
        2. Model -> physical outputs
        3. Loss(physical outputs, physical coords)
        """

        # Enable gradients on PHYSICAL coordinates for derivative calculation
        coords_int_phys.requires_grad = True
        coords_bc_phys.requires_grad = True

        # 1. Normalize coords to feed into the model
        coords_int_norm = self.normalize_coords(coords_int_phys)
        coords_bc_norm = self.normalize_coords(coords_bc_phys)

        # 2. Forward pass
        fields_int = self.model(coords_int_norm)
        fields_bc = self.model(coords_bc_norm)

        # 3. Compute loss using PHYSICAL coordinates and outputs
        L_poisson = self.physics_loss.poisson_loss(coords_int_phys, fields_int)
        L_continuity = self.physics_loss.continuity_loss(coords_int_phys, fields_int)
        L_helmholtz = self.physics_loss.helmholtz_loss(coords_int_phys, fields_int)
        L_bc = self.physics_loss.boundary_loss(coords_bc_phys, fields_bc)

        # Weighted sum
        total_loss = (
            self.loss_weights['poisson'] * L_poisson +
            self.loss_weights['continuity'] * L_continuity +
            self.loss_weights['helmholtz'] * L_helmholtz +
            self.loss_weights['bc'] * L_bc
        )

        return total_loss, {
            'poisson': L_poisson.item(),
            'continuity': L_continuity.item(),
            'helmholtz': L_helmholtz.item(),
            'bc': L_bc.item(),
            'total': total_loss.item()
        }

    def train(self, epochs: int = 1000, n_points: int = 5000):
        """Main training loop"""
        print(f"Training PINN on {self.device} with STABILIZED logic")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Curriculum: Electrical (0-{self.curriculum_schedule['electrical_only_epochs']}) -> Full Multi-physics")

        pbar = tqdm(range(epochs), desc="Training")

        for epoch in pbar:
            # Update curriculum weights
            self.update_curriculum(epoch)

            # Generate new collocation points (physical units)
            coords_int_phys, coords_bc_phys = self.generate_training_points(n_points)

            # Zero gradients
            self.optimizer.zero_grad()

            # Compute loss
            loss, loss_dict = self.compute_loss(coords_int_phys, coords_bc_phys)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Log
            self.loss_history.append(loss_dict)

            if epoch % 10 == 0:
                pbar.set_postfix({
                    'Loss': f"{loss_dict['total']:.3e}",
                    'Pois': f"{loss_dict['poisson']:.3e}",
                    'Cont': f"{loss_dict['continuity']:.3e}",
                    'Helm': f"{loss_dict['helmholtz']:.3e}",
                    'BC': f"{loss_dict['bc']:.3e}",
                })

    def plot_loss_history(self):
        """Visualize training progress"""
        df = pd.DataFrame(self.loss_history)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        for ax, col in zip(axes.flat, ['poisson', 'continuity', 'helmholtz', 'bc']):
            if df[col].max() > 0: # Only plot if loss was active
                ax.semilogy(df[col])
                ax.set_title(f'{col.capitalize()} Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('pinn_training_loss_stabilized.png', dpi=200)
        plt.show()

# ==================== Inference and Visualization ====================

def inference_slice(model: MultiPhysicsPINN,
                    V_bias: float = 0.02,
                    z_slice: float = None,
                    resolution: int = 100):
    """
    Run inference on 2D slice for visualization
    """
    if z_slice is None:
        z_slice = L_clad / 2

    # Create grid (physical units)
    x = torch.linspace(0, W_clad, resolution)
    y = torch.linspace(0, H_clad, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    Z = torch.full_like(X, z_slice)
    V = torch.full_like(X, V_bias)

    coords_phys = torch.stack([X.flatten(), Y.flatten(),
                               Z.flatten(), V.flatten()], dim=1)

    # Normalize physical coords for model input
    coords_norm = coords_phys.clone()
    coords_norm[:, 0] /= W_clad
    coords_norm[:, 1] /= H_clad
    coords_norm[:, 2] /= L_clad
    coords_norm[:, 3] /= 0.05

    # Inference
    model.eval()
    with torch.no_grad():
        coords_norm = coords_norm.to(next(model.parameters()).device)
        fields = model(coords_norm) # Model gets normalized coords

    # Reshape outputs (which are in physical units)
    results = {}
    for key, val in fields.items():
        results[key] = val.cpu().numpy().reshape(resolution, resolution)

    results['X'] = X.numpy()
    results['Y'] = Y.numpy()

    return results

def visualize_results(results):
    """Comprehensive visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    X, Y = results['X'], results['Y']
    extent = [0, W_clad/um, 0, H_clad/um]

    # Potential
    im0 = axes[0,0].imshow(results['phi'].T, origin='lower', extent=extent, cmap='RdBu')
    axes[0,0].set_title('φ [V]')
    plt.colorbar(im0, ax=axes[0,0])

    # Electron density
    # Add clip for stability if n is 0
    n_plot = np.clip(results['n'], 1e-10, None)
    im1 = axes[0,1].imshow(np.log10(n_plot).T, origin='lower', extent=extent, cmap='viridis')
    axes[0,1].set_title('log₁₀(n) [cm⁻³]')
    plt.colorbar(im1, ax=axes[0,1])

    # Hole density
    p_plot = np.clip(results['p'], 1e-10, None)
    im2 = axes[0,2].imshow(np.log10(p_plot).T, origin='lower', extent=extent, cmap='plasma')
    axes[0,2].set_title('log₁₀(p) [cm⁻³]')
    plt.colorbar(im2, ax=axes[0,2])

    # E-field components
    im3 = axes[1,0].imshow(results['Ez_real'].T, origin='lower', extent=extent, cmap='RdBu')
    axes[1,0].set_title('Re(E_z)')
    plt.colorbar(im3, ax=axes[1,0])

    im4 = axes[1,1].imshow(results['Ez_imag'].T, origin='lower', extent=extent, cmap='RdBu')
    axes[1,1].set_title('Im(E_z)')
    plt.colorbar(im4, ax=axes[1,1])

    # Intensity
    intensity = results['Ez_real']**2 + results['Ez_imag']**2
    im5 = axes[1,2].imshow(intensity.T, origin='lower', extent=extent, cmap='hot')
    axes[1,2].set_title('|E_z|²')
    plt.colorbar(im5, ax=axes[1,2])

    for ax in axes.flat:
        ax.set_xlabel('x [μm]')
        ax.set_ylabel('y [μm]')
        # Mark core region
        ax.axvline(cx0/um, color='cyan', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(cx1/um, color='cyan', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axhline(cy0/um, color='cyan', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axhline(cy1/um, color='cyan', linestyle='--', linewidth=0.5, alpha=0.5)
        # Mark p-n junction
        ax.axvline(cxm/um, color='yellow', linestyle=':', linewidth=1.0, alpha=0.7)

    plt.tight_layout()
    plt.savefig('pinn_inference_results_stabilized.png', dpi=200)
    plt.show()

# ==================== Main Execution ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Stabilized PINN Framework for Silicon Photonic Modulator")
    print("Applying feedback from pinn_advanced_feedback_ko.md")
    print("=" * 70)

    # Initialize model
    model = MultiPhysicsPINN(
        hidden_dims=[256, 256, 256, 256],
        fourier_features=256,
        fourier_scale=10.0 # Scale for normalized inputs
    )

    # Initialize trainer
    trainer = PINNTrainer(model)

    # Train
    print("\n[1] Training PINN...")
    trainer.train(epochs=2000, n_points=10000) # Increased epochs/points

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
    torch.save(model.state_dict(), 'pinn_silicon_modulator_stabilized.pth')
    print("\n✓ Model saved: pinn_silicon_modulator_stabilized.pth")

    print("\n" + "=" * 70)
    print("PINN Training Complete!")
    print("=" * 70)
    print(f"Device: {trainer.device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if trainer.loss_history:
        print(f"Final total loss: {trainer.loss_history[-1]['total']:.3e}")
