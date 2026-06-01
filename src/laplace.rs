//! Laplace equation: Δu = 0
//!
//! Models agent equilibrium — steady-state belief distribution with no forcing.
//! Solutions are harmonic functions satisfying the maximum principle.

use crate::{Grid1D, Grid2D, DVector};
use serde::{Serialize, Deserialize};

/// Solver for the Laplace equation in 1D.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaplaceSolver {
    pub grid: Grid1D,
}

impl LaplaceSolver {
    pub fn new(grid: Grid1D) -> Self {
        Self { grid }
    }

    /// Solve analytically: u is linear between boundary values.
    pub fn solve(&self, bc_left: f64, bc_right: f64) -> DVector<f64> {
        let n = self.grid.n;
        DVector::from_fn(n, |i, _| {
            let x = self.grid.x(i);
            let t = (x - self.grid.a) / (self.grid.b - self.grid.a);
            bc_left * (1.0 - t) + bc_right * t
        })
    }

    /// Solve using Jacobi iteration.
    pub fn solve_jacobi(&self, bc_left: f64, bc_right: f64, tol: f64, max_iters: usize) -> (DVector<f64>, usize) {
        let n = self.grid.n;
        let mut u = DVector::zeros(n);
        let mut iters = 0;

        for k in 0..max_iters {
            let mut u_new = DVector::zeros(n);
            for i in 0..n {
                let left = if i == 0 { bc_left } else { u[i - 1] };
                let right = if i == n - 1 { bc_right } else { u[i + 1] };
                u_new[i] = 0.5 * (left + right);
            }
            let change = (&u_new - &u).abs().max();
            u = u_new;
            iters = k + 1;
            if change < tol {
                break;
            }
        }
        (u, iters)
    }

    /// Solve using Gauss-Seidel iteration.
    pub fn solve_gauss_seidel(&self, bc_left: f64, bc_right: f64, tol: f64, max_iters: usize) -> (DVector<f64>, usize) {
        let n = self.grid.n;
        let mut u = DVector::zeros(n);

        for k in 0..max_iters {
            let mut max_change: f64 = 0.0;
            for i in 0..n {
                let left = if i == 0 { bc_left } else { u[i - 1] };
                let right = if i == n - 1 { bc_right } else { u[i + 1] };
                let new_val = 0.5 * (left + right);
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

/// Solver for the Laplace equation in 2D using Jacobi iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaplaceSolver2D {
    pub grid: Grid2D,
}

impl LaplaceSolver2D {
    pub fn new(grid: Grid2D) -> Self {
        Self { grid }
    }

    /// Solve with Dirichlet boundary condition `bc_val` on all edges.
    pub fn solve(&self, bc_val: f64, tol: f64, max_iters: usize) -> (DVector<f64>, usize) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let mut u = DVector::zeros(nx * ny);

        for k in 0..max_iters {
            let mut max_change = 0.0f64;
            let mut u_new = u.clone();
            for j in 0..ny {
                for i in 0..nx {
                    let idx = j * nx + i;
                    let left = if i > 0 { u[idx - 1] } else { bc_val };
                    let right = if i < nx - 1 { u[idx + 1] } else { bc_val };
                    let below = if j > 0 { u[idx - nx] } else { bc_val };
                    let above = if j < ny - 1 { u[idx + nx] } else { bc_val };
                    let new_val = 0.25 * (left + right + below + above);
                    max_change = max_change.max((new_val - u[idx]).abs());
                    u_new[idx] = new_val;
                }
            }
            u = u_new;
            if max_change < tol {
                return (u, k + 1);
            }
        }
        (u, max_iters)
    }
}
