//! Burgers' equation: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
//!
//! Nonlinear agent dynamics — beliefs steepen and form shocks,
//! moderated by viscosity ν.

use crate::{Grid1D, DVector};
use serde::{Serialize, Deserialize};

/// Solver for Burgers' equation in 1D.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurgersSolver {
    pub grid: Grid1D,
    /// Viscosity ν ≥ 0.
    pub nu: f64,
}

impl BurgersSolver {
    pub fn new(grid: Grid1D, nu: f64) -> Self {
        assert!(nu >= 0.0, "viscosity must be non-negative");
        Self { grid, nu }
    }

    /// One step using explicit Euler with upwind advection + central diffusion.
    pub fn step(&self, u: &DVector<f64>, dt: f64) -> DVector<f64> {
        let n = self.grid.n;
        let dx = self.grid.dx;
        let dx2 = dx * dx;

        let mut u_new = u.clone();
        for i in 0..n {
            let u_left = if i == 0 { u[0] } else { u[i - 1] };
            let u_right = if i == n - 1 { u[n - 1] } else { u[i + 1] };

            // Upwind advection based on sign of u[i]
            let advection = if u[i] >= 0.0 {
                u[i] * (u[i] - u_left) / dx
            } else {
                u[i] * (u_right - u[i]) / dx
            };

            // Central diffusion
            let diffusion = self.nu * (u_right - 2.0 * u[i] + u_left) / dx2;

            u_new[i] = u[i] + dt * (-advection + diffusion);
        }
        u_new
    }

    /// One step using Lax-Friedrichs for the advective part + central diffusion.
    pub fn step_lax_friedrichs(&self, u: &DVector<f64>, dt: f64) -> DVector<f64> {
        let n = self.grid.n;
        let dx = self.grid.dx;
        let dx2 = dx * dx;

        let mut u_new = u.clone();
        for i in 0..n {
            let u_left = if i == 0 { u[0] } else { u[i - 1] };
            let u_right = if i == n - 1 { u[n - 1] } else { u[i + 1] };

            // Lax-Friedrichs diffusion + flux
            let flux_right = 0.5 * u[i] * u[i]; // F(u) = u²/2
            let flux_left = if i == 0 { flux_right } else { 0.5 * u_left * u_left };
            let fr = if i == n - 1 { 0.5 * u[i] * u[i] } else { 0.5 * u_right * u_right };
            let fl = flux_right;

            let avg = 0.5 * (u_left + u_right);
            let dflux = (fr - fl) / (2.0 * dx);

            let visc = self.nu * (u_right - 2.0 * u[i] + u_left) / dx2;

            u_new[i] = avg - dt * dflux + dt * visc;
        }
        u_new
    }

    /// Solve to steady state.
    pub fn solve(&self, u0: &DVector<f64>, dt: f64, steps: usize) -> Vec<DVector<f64>> {
        let mut traj = Vec::with_capacity(steps + 1);
        let mut u = u0.clone();
        traj.push(u.clone());
        for _ in 0..steps {
            u = self.step(&u, dt);
            traj.push(u.clone());
        }
        traj
    }

    /// Compute the total "energy" ∫u² dx (L² norm squared).
    pub fn energy(&self, u: &DVector<f64>) -> f64 {
        u.iter().map(|x| x * x).sum::<f64>() * self.grid.dx
    }

    /// Compute the total variation TV(u) = Σ|u_{i+1} - u_i|.
    pub fn total_variation(&self, u: &DVector<f64>) -> f64 {
        (0..u.len() - 1).map(|i| (u[i + 1] - u[i]).abs()).sum()
    }

    /// Reynolds number: Re = U·L/ν where U = max|u|, L = domain length.
    pub fn reynolds(&self, u: &DVector<f64>) -> f64 {
        if self.nu < 1e-15 { return f64::INFINITY; }
        let u_max = u.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        let length = self.grid.b - self.grid.a;
        u_max * length / self.nu
    }
}
