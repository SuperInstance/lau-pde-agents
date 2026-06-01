//! Heat equation: ∂u/∂t = α·Δu
//!
//! Models diffusion of agent beliefs — beliefs smooth out over time and space,
//! analogous to heat spreading through a medium.
//!
//! Numerical method: Forward Euler (explicit) or Backward Euler (implicit).

use crate::{Grid1D, Grid2D, laplacian_1d, DMatrix, DVector};
use serde::{Serialize, Deserialize};

/// Solver for the heat/diffusion equation in 1D.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatSolver {
    /// Grid.
    pub grid: Grid1D,
    /// Diffusion coefficient α > 0.
    pub alpha: f64,
    /// Laplacian matrix.
    #[serde(skip)]
    pub laplacian: Option<DMatrix<f64>>,
}

impl HeatSolver {
    /// Create a new heat solver.
    pub fn new(grid: Grid1D, alpha: f64) -> Self {
        assert!(alpha > 0.0, "diffusion coefficient must be positive");
        Self { grid, alpha, laplacian: None }
    }

    /// Initialize the Laplacian matrix.
    pub fn init(&mut self) {
        self.laplacian = Some(laplacian_1d(&self.grid));
    }

    /// Maximum stable time step for explicit Euler: dt < dx²/(2α).
    pub fn max_stable_dt(&self) -> f64 {
        self.grid.dx * self.grid.dx / (2.0 * self.alpha)
    }

    /// Advance one step using forward (explicit) Euler.
    /// `u` has length n (interior points), `bc` has length 2 (left, right boundary).
    pub fn step_explicit(&self, u: &DVector<f64>, dt: f64, bc: &[f64; 2]) -> DVector<f64> {
        let lap = self.laplacian.as_ref().expect("call init() first");
        let n = self.grid.n;
        let dx2 = self.grid.dx * self.grid.dx;

        let mut rhs = lap * u;
        // Boundary contributions
        rhs[0] += bc[0] / dx2;
        rhs[n - 1] += bc[1] / dx2;

        u + self.alpha * dt * rhs
    }

    /// Solve for T total time with given initial condition, returning trajectory.
    pub fn solve_explicit(&mut self, u0: &DVector<f64>, dt: f64, steps: usize, bc: &[f64; 2]) -> Vec<DVector<f64>> {
        self.init();
        let mut traj = Vec::with_capacity(steps + 1);
        let mut u = u0.clone();
        traj.push(u.clone());
        for _ in 0..steps {
            u = self.step_explicit(&u, dt, bc);
            traj.push(u.clone());
        }
        traj
    }

    /// Solve to steady state (when max change < tolerance).
    pub fn solve_to_steady(&mut self, u0: &DVector<f64>, dt: f64, bc: &[f64; 2], tol: f64, max_steps: usize) -> (DVector<f64>, usize) {
        self.init();
        let mut u = u0.clone();
        for step in 0..max_steps {
            let u_new = self.step_explicit(&u, dt, bc);
            let change = (&u_new - &u).abs().max();
            u = u_new;
            if change < tol {
                return (u, step + 1);
            }
        }
        (u, max_steps)
    }

    /// Step using backward (implicit) Euler: (I - α·dt·L) u^{n+1} = u^n + boundary terms.
    /// Uses Gauss-Seidel iteration for the linear solve.
    pub fn step_implicit(&self, u: &DVector<f64>, dt: f64, bc: &[f64; 2], gs_iters: usize) -> DVector<f64> {
        let n = self.grid.n;
        let dx = self.grid.dx;
        let dx2 = dx * dx;
        let r = self.alpha * dt / dx2;

        // We solve (I - r*L_modified) u_new = u + boundary terms
        // Diagonal: 1 + 2r, off-diagonal: -r
        let mut u_new = u.clone();
        for _ in 0..gs_iters {
            for i in 0..n {
                let mut sum = u[i];
                if i == 0 { sum += r * bc[0]; }
                if i == n - 1 { sum += r * bc[1]; }
                if i > 0 { sum += r * u_new[i - 1]; }
                if i < n - 1 { sum += r * u_new[i + 1]; }
                u_new[i] = sum / (1.0 + 2.0 * r);
            }
        }
        u_new
    }
}

/// 2D heat solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatSolver2D {
    pub grid: Grid2D,
    pub alpha: f64,
}

impl HeatSolver2D {
    pub fn new(grid: Grid2D, alpha: f64) -> Self {
        assert!(alpha > 0.0);
        Self { grid, alpha }
    }

    /// Maximum stable dt for explicit 2D: dt < 1/(2α(1/dx² + 1/dy²)).
    pub fn max_stable_dt(&self) -> f64 {
        1.0 / (2.0 * self.alpha * (1.0 / (self.grid.dx * self.grid.dx) + 1.0 / (self.grid.dy * self.grid.dy)))
    }

    /// One explicit Euler step on a flattened 2D grid (row-major, ny rows of nx).
    pub fn step_explicit(&self, u: &DVector<f64>, dt: f64, bc_val: f64) -> DVector<f64> {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let dx2 = self.grid.dx * self.grid.dx;
        let dy2 = self.grid.dy * self.grid.dy;
        let mut u_new = u.clone();

        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                let left = if i > 0 { u[idx - 1] } else { bc_val };
                let right = if i < nx - 1 { u[idx + 1] } else { bc_val };
                let below = if j > 0 { u[idx - nx] } else { bc_val };
                let above = if j < ny - 1 { u[idx + nx] } else { bc_val };

                let lap = (left - 2.0 * u[idx] + right) / dx2
                        + (below - 2.0 * u[idx] + above) / dy2;
                u_new[idx] = u[idx] + self.alpha * dt * lap;
            }
        }
        u_new
    }
}
