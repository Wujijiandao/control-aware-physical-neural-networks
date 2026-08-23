"""Physics-constrained substrate models. Project-original implementation."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InterferometricOracle(nn.Module):
    """Trainable coherent multi-path interference model with intensity readout.

    The model is a project-original generic physical function family and does not copy third-party research source code. It is
    not copied from any external research codebase.
    """

    def __init__(self, d: int = 3, paths: int = 64, detectors: int = 16):
        super().__init__()
        self.K = nn.Parameter(torch.randn(paths, d) * 4.0)
        self.b = nn.Parameter(torch.rand(paths) * 2.0 * math.pi)
        self.cre = nn.Parameter(torch.randn(detectors, paths) / math.sqrt(paths))
        self.cim = nn.Parameter(torch.randn(detectors, paths) / math.sqrt(paths))
        self.w = nn.Parameter(torch.randn(detectors) * 0.1)
        self.bias = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * math.pi * (x @ self.K.T) + self.b
        c, s = torch.cos(phase), torch.sin(phase)
        norm = math.sqrt(self.K.shape[0])
        re = (c @ self.cre.T - s @ self.cim.T) / norm
        im = (s @ self.cre.T + c @ self.cim.T) / norm
        intensity = re.square() + im.square()
        return F.softplus(intensity @ self.w + self.bias)


class NonlinearOscillatorOracle(nn.Module):
    """Differentiable damped coupled-oscillator physical oracle.

    Each input vector parametrizes constant generalized forces applied to a chain
    of Duffing-type oscillators.  The network is integrated for a fixed physical
    horizon using a semi-implicit Euler scheme.  The non-negative readout is a
    trainable linear combination of final displacement, velocity and local
    oscillator energy features.

    This implementation is project-original and intentionally does not import
    third-party research implementations.
    """

    def __init__(self, d: int = 3, oscillators: int = 20, integration_steps: int = 14, dt: float = 0.055):
        super().__init__()
        self.d = int(d)
        self.oscillators = int(oscillators)
        self.integration_steps = int(integration_steps)
        self.dt = float(dt)

        # State-to-force transduction and static force offsets.
        self.force = nn.Parameter(torch.randn(oscillators, d) * 0.55)
        self.force_bias = nn.Parameter(torch.zeros(oscillators))

        # Positive physical parameters are represented through softplus.
        self.raw_omega = nn.Parameter(torch.randn(oscillators) * 0.10)
        self.raw_gamma = nn.Parameter(torch.randn(oscillators) * 0.10 - 1.0)
        self.raw_alpha = nn.Parameter(torch.randn(oscillators) * 0.10 - 0.8)
        self.raw_coupling = nn.Parameter(torch.tensor(-0.4))

        # Readout over displacement, velocity and positive energy-like features.
        self.readout = nn.Parameter(torch.randn(oscillators * 4) * 0.05)
        self.bias = nn.Parameter(torch.tensor(0.05))

    def physical_parameters(self):
        omega = 0.65 + 0.55 * F.softplus(self.raw_omega)
        gamma = 0.08 + 0.20 * F.softplus(self.raw_gamma)
        alpha = 0.04 + 0.16 * F.softplus(self.raw_alpha)
        coupling = 0.03 + 0.12 * F.softplus(self.raw_coupling)
        return omega, gamma, alpha, coupling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        omega, gamma, alpha, coupling = self.physical_parameters()
        # Centered input prevents a large DC bias while retaining a physical
        # affine force map; tanh represents actuator saturation.
        drive = 1.25 * torch.tanh((x - 0.5) @ self.force.T + self.force_bias)
        q = torch.zeros_like(drive)
        v = torch.zeros_like(drive)
        dt = self.dt
        for _ in range(self.integration_steps):
            # Periodic nearest-neighbour elastic coupling.
            lap = torch.roll(q, 1, dims=-1) + torch.roll(q, -1, dims=-1) - 2.0 * q
            acc = drive - gamma * v - omega.square() * q - alpha * q.pow(3) + coupling * lap
            v = v + dt * acc
            q = q + dt * v
        energy = 0.5 * (v.square() + omega.square() * q.square()) + 0.25 * alpha * q.pow(4)
        features = torch.cat([q, v, q.square(), energy], dim=-1)
        return F.softplus(features @ self.readout + self.bias)
