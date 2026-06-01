//! Transport equation: ∂u/∂t + v·∂u/∂x = 0
//!
//! Agent belief propagation — beliefs advect at velocity v without changing shape.
//! Numerical methods: upwind, Lax-Friedrichs, Lax-Wendroff.

use crate::{Grid1D, DVector};
use serde::{Serialize, Deserialize};

/// Solver for the 1D transport equation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportSolver {
    pub grid: Grid1D,
    /// Advection velocity.
    pub velocity: f64,
}

impl TransportSolver {
    pub fn new(grid: Grid1D, velocity: f64) -> Self {
        Self { grid, velocity }
    }

    /// One step using first-order upwind scheme.
    pub fn step_upwind(&self, u: &DVector<f64>, dt: f64) -> DVector<f64> {
        let n = self.grid.n;
        let dx = self.grid.dx;
        let c = self.velocity * dt / dx; // Courant number

        let mut u_new = u.clone();
        for i in 0..n {
            let (u_left, u_right) = if self.velocity >= 0.0 {
                let ul = if i == 0 { u[0] } else { u[i - 1] };
                (ul, u[i])
            } else {
                let ur = if i == n - 1 { u[n - 1] } else { u[i + 1] };
                (u[i], ur)
            };
            u_new[i] = u[i] - c * (u_right - u_left);
        }
        u_new
    }

    /// One step using Lax-Friedrichs scheme.
    pub fn step_lax_friedrichs(&self, u: &DVector<f64>, dt: f64) -> DVector<f64> {
        let n = self.grid.n;
        let dx = self.grid.dx;
        let c = self.velocity * dt / dx;

        let mut u_new = u.clone();
        for i in 0..n {
            let u_left = if i == 0 { u[0] } else { u[i - 1] };
            let u_right = if i == n - 1 { u[n - 1] } else { u[i + 1] };
            u_new[i] = 0.5 * (u_left + u_right) - 0.5 * c * (u_right - u_left);
        }
        u_new
    }

    /// One step using Lax-Wendroff scheme (second-order).
    pub fn step_lax_wendroff(&self, u: &DVector<f64>, dt: f64) -> DVector<f64> {
        let n = self.grid.n;
        let dx = self.grid.dx;
        let c = self.velocity * dt / dx;
        let c2 = c * c;

        let mut u_new = u.clone();
        for i in 0..n {
            let u_left = if i == 0 { u[0] } else { u[i - 1] };
            let u_right = if i == n - 1 { u[n - 1] } else { u[i + 1] };
            u_new[i] = u[i] - 0.5 * c * (u_right - u_left)
                + 0.5 * c2 * (u_right - 2.0 * u[i] + u_left);
        }
        u_new
    }

    /// CFL number.
    pub fn courant(&self, dt: f64) -> f64 {
        self.velocity.abs() * dt / self.grid.dx
    }

    /// Solve with the upwind scheme.
    pub fn solve_upwind(&self, u0: &DVector<f64>, dt: f64, steps: usize) -> Vec<DVector<f64>> {
        let mut traj = Vec::with_capacity(steps + 1);
        let mut u = u0.clone();
        traj.push(u.clone());
        for _ in 0..steps {
            u = self.step_upwind(&u, dt);
            traj.push(u.clone());
        }
        traj
    }

    /// Solve with Lax-Friedrichs.
    pub fn solve_lax_friedrichs(&self, u0: &DVector<f64>, dt: f64, steps: usize) -> Vec<DVector<f64>> {
        let mut traj = Vec::with_capacity(steps + 1);
        let mut u = u0.clone();
        traj.push(u.clone());
        for _ in 0..steps {
            u = self.step_lax_friedrichs(&u, dt);
            traj.push(u.clone());
        }
        traj
    }
}
