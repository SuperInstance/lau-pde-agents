//! Poisson equation: Δu = f
//!
//! Agent equilibrium with external forcing — beliefs reach steady state
//! under continuous influence from source terms.

use crate::{Grid1D, DVector};
use serde::{Serialize, Deserialize};

/// Solver for the Poisson equation in 1D.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoissonSolver {
    pub grid: Grid1D,
}

impl PoissonSolver {
    pub fn new(grid: Grid1D) -> Self {
        Self { grid }
    }

    /// Solve Δu = f with Dirichlet BCs using Jacobi iteration.
    pub fn solve(&self, f: &DVector<f64>, bc_left: f64, bc_right: f64, tol: f64, max_iters: usize) -> (DVector<f64>, usize) {
        let n = self.grid.n;
        let dx2 = self.grid.dx * self.grid.dx;
        assert_eq!(f.len(), n, "forcing term must match grid size");
        let mut u = DVector::zeros(n);

        for k in 0..max_iters {
            let mut max_change: f64 = 0.0;
            for i in 0..n {
                let left = if i == 0 { bc_left } else { u[i - 1] };
                let right = if i == n - 1 { bc_right } else { u[i + 1] };
                let new_val = 0.5 * (left + right - dx2 * f[i]);
                max_change = max_change.max((new_val - u[i]).abs());
                u[i] = new_val;
            }
            if max_change < tol {
                return (u, k + 1);
            }
        }
        (u, max_iters)
    }

    /// Solve using Gauss-Seidel.
    pub fn solve_gs(&self, f: &DVector<f64>, bc_left: f64, bc_right: f64, tol: f64, max_iters: usize) -> (DVector<f64>, usize) {
        let n = self.grid.n;
        let dx2 = self.grid.dx * self.grid.dx;
        assert_eq!(f.len(), n);
        let mut u = DVector::zeros(n);

        for k in 0..max_iters {
            let mut max_change: f64 = 0.0;
            for i in 0..n {
                let left = if i == 0 { bc_left } else { u[i - 1] };
                let right = if i == n - 1 { bc_right } else { u[i + 1] };
                let new_val = 0.5 * (left + right - dx2 * f[i]);
                max_change = max_change.max((new_val - u[i]).abs());
                u[i] = new_val;
            }
            if max_change < tol {
                return (u, k + 1);
            }
        }
        (u, max_iters)
    }

    /// Solve with mixed (Robin) boundary conditions: α·u + β·u' = γ at each boundary.
    pub fn solve_robin(&self, f: &DVector<f64>, alpha_l: f64, beta_l: f64, gamma_l: f64,
                       alpha_r: f64, beta_r: f64, gamma_r: f64, tol: f64, max_iters: usize) -> (DVector<f64>, usize) {
        let n = self.grid.n;
        let dx = self.grid.dx;
        let dx2 = dx * dx;
        let mut u = DVector::zeros(n);

        for k in 0..max_iters {
            let mut max_change: f64 = 0.0;
            for i in 0..n {
                let (left, right) = if i == 0 {
                    // Ghost point approach: u_ghost = (γ_l - α_l * u[0]) * dx / β_l + u[0]
                    let ghost_left = if beta_l.abs() > 1e-12 {
                        (gamma_l - alpha_l * u[0]) * dx / beta_l + u[0]
                    } else { u[0] };
                    (ghost_left, if i < n - 1 { u[i + 1] } else { 0.0 })
                } else if i == n - 1 {
                    let ghost_right = if beta_r.abs() > 1e-12 {
                        (gamma_r - alpha_r * u[i]) * dx / beta_r + u[i]
                    } else { u[i] };
                    (u[i - 1], ghost_right)
                } else {
                    (u[i - 1], u[i + 1])
                };
                let new_val = 0.5 * (left + right - dx2 * f[i]);
                max_change = max_change.max((new_val - u[i]).abs());
                u[i] = new_val;
            }
            if max_change < tol {
                return (u, k + 1);
            }
        }
        (u, max_iters)
    }
}
