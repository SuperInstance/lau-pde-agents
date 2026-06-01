//! # lau-pde-agents
//!
//! Partial differential equations governing agent dynamics.
//!
//! Agent beliefs evolve in continuous space and time according to well-studied PDEs.
//! This crate provides numerical solvers for the key equations that describe how agent
//! populations distribute, oscillate, reach equilibrium, and interact.
//!
//! ## Equations
//!
//! | Equation | Physics | Agent Interpretation |
//! |----------|---------|---------------------|
//! | Heat | Diffusion | Beliefs spread and smooth out |
//! | Wave | Oscillation | Beliefs oscillate over time |
//! | Laplace | Equilibrium | Steady-state belief distribution |
//! | Poisson | Forced equilibrium | Beliefs under external influence |
//! | Schrödinger | Quantum dynamics | Probabilistic agent state evolution |
//! | Reaction-diffusion | Pattern formation | Belief creation and destruction |
//! | Transport | Advection | Directed belief propagation |
//! | Burgers | Nonlinear advection | Belief steepening and shocks |

pub mod heat;
pub mod wave;
pub mod laplace;
pub mod poisson;
pub mod schrodinger;
pub mod reaction_diffusion;
pub mod transport;
pub mod burgers;
pub mod maximum_principle;
pub mod energy;
pub mod application;
#[cfg(test)]
mod tests;

pub use heat::HeatSolver;
pub use wave::WaveSolver;
pub use laplace::LaplaceSolver;
pub use poisson::PoissonSolver;
pub use schrodinger::SchrodingerSolver;
pub use reaction_diffusion::ReactionDiffusionSolver;
pub use transport::TransportSolver;
pub use burgers::BurgersSolver;

/// Re-export nalgebra types for convenience.
pub use nalgebra::{DMatrix, DVector, ComplexField, RealField};

/// Grid discretization of a 1D domain [a, b] with N interior points.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Grid1D {
    /// Left boundary.
    pub a: f64,
    /// Right boundary.
    pub b: f64,
    /// Number of interior grid points.
    pub n: usize,
    /// Grid spacing dx = (b - a) / (n + 1).
    pub dx: f64,
}

impl Grid1D {
    /// Create a new 1D grid.
    pub fn new(a: f64, b: f64, n: usize) -> Self {
        assert!(b > a, "b must be greater than a");
        assert!(n >= 2, "need at least 2 interior points");
        let dx = (b - a) / (n + 1) as f64;
        Self { a, b, n, dx }
    }

    /// Get all interior grid points.
    pub fn points(&self) -> Vec<f64> {
        (0..self.n).map(|i| self.a + (i + 1) as f64 * self.dx).collect()
    }

    /// Get the i-th interior point.
    pub fn x(&self, i: usize) -> f64 {
        self.a + (i + 1) as f64 * self.dx
    }
}

/// Grid discretization of a 2D domain [ax, bx] × [ay, by] with nx × ny interior points.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Grid2D {
    pub ax: f64, pub bx: f64,
    pub ay: f64, pub by: f64,
    pub nx: usize, pub ny: usize,
    pub dx: f64, pub dy: f64,
}

impl Grid2D {
    pub fn new(ax: f64, bx: f64, ay: f64, by: f64, nx: usize, ny: usize) -> Self {
        assert!(bx > ax && by > ay, "domain must have positive extent");
        assert!(nx >= 2 && ny >= 2, "need at least 2 interior points per dimension");
        let dx = (bx - ax) / (nx + 1) as f64;
        let dy = (by - ay) / (ny + 1) as f64;
        Self { ax, bx, ay, by, nx, ny, dx, dy }
    }
}

/// Build the 1D finite-difference Laplacian (tridiagonal) matrix for interior points.
pub fn laplacian_1d(grid: &Grid1D) -> DMatrix<f64> {
    let n = grid.n;
    let dx2 = grid.dx * grid.dx;
    let mut mat = DMatrix::zeros(n, n);
    for i in 0..n {
        mat[(i, i)] = -2.0 / dx2;
        if i > 0 { mat[(i, i - 1)] = 1.0 / dx2; }
        if i < n - 1 { mat[(i, i + 1)] = 1.0 / dx2; }
    }
    mat
}
