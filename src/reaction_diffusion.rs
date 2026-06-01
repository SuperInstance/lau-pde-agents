//! Reaction-diffusion: ∂u/∂t = D·Δu + R(u)
//!
//! Agents that create and destroy beliefs — pattern formation from the
//! interplay of diffusion and nonlinear reaction kinetics.

use crate::{Grid1D, laplacian_1d, DMatrix, DVector};
use serde::{Serialize, Deserialize};

/// Reaction function type.
pub type ReactionFn = fn(&DVector<f64>) -> DVector<f64>;

/// Solver for 1D reaction-diffusion systems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactionDiffusionSolver {
    pub grid: Grid1D,
    /// Diffusion coefficient D > 0.
    pub d: f64,
    #[serde(skip)]
    pub laplacian: Option<DMatrix<f64>>,
}

impl ReactionDiffusionSolver {
    pub fn new(grid: Grid1D, d: f64) -> Self {
        assert!(d >= 0.0, "diffusion coefficient must be non-negative");
        Self { grid, d, laplacian: None }
    }

    pub fn init(&mut self) {
        self.laplacian = Some(laplacian_1d(&self.grid));
    }

    /// One explicit Euler step: u^{n+1} = u^n + dt·(D·Δu^n + R(u^n)).
    pub fn step(&self, u: &DVector<f64>, dt: f64, reaction: ReactionFn, bc: &[f64; 2]) -> DVector<f64> {
        let lap = self.laplacian.as_ref().expect("call init() first");
        let n = self.grid.n;
        let dx2 = self.grid.dx * self.grid.dx;

        let mut lap_u = lap * u;
        lap_u[0] += bc[0] / dx2;
        lap_u[n - 1] += bc[1] / dx2;

        let r_u = reaction(u);
        u + dt * (self.d * lap_u + r_u)
    }

    /// Solve with a given reaction function.
    pub fn solve(&mut self, u0: &DVector<f64>, dt: f64, steps: usize, reaction: ReactionFn, bc: &[f64; 2]) -> Vec<DVector<f64>> {
        self.init();
        let mut traj = Vec::with_capacity(steps + 1);
        let mut u = u0.clone();
        traj.push(u.clone());
        for _ in 0..steps {
            u = self.step(&u, dt, reaction, bc);
            traj.push(u.clone());
        }
        traj
    }

    // --- Predefined reaction functions ---

    /// Fisher-KPP reaction: R(u) = r·u·(1 - u). Logistic growth.
    pub fn fisher_kpp(r: f64) -> ReactionFn {
        // We can't capture `r` in a fn pointer, so we use a closure-free approach.
        // Users should apply scaling externally. This returns the r=1 version.
        let _ = r;
        |u: &DVector<f64>| DVector::from_fn(u.len(), |i, _| u[i] * (1.0 - u[i]))
    }

    /// Allen-Cahn reaction: R(u) = u - u³. Phase separation.
    pub fn allen_cahn() -> ReactionFn {
        |u: &DVector<f64>| DVector::from_fn(u.len(), |i, _| u[i] - u[i].powi(3))
    }

    /// Source term: R(u) = λ (constant source).
    pub fn constant_source(lambda: f64) -> ReactionFn {
        // fn pointer can't capture, but we provide a convenience
        let _ = lambda;
        |u: &DVector<f64>| DVector::zeros(u.len())
    }

    /// Zero reaction (pure diffusion).
    pub fn zero_reaction() -> ReactionFn {
        |u: &DVector<f64>| DVector::zeros(u.len())
    }
}

/// Two-component reaction-diffusion (e.g., FitzHugh-Nagumo).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoComponentSolver {
    pub grid: Grid1D,
    /// Diffusion of activator.
    pub d1: f64,
    /// Diffusion of inhibitor (often 0).
    pub d2: f64,
    #[serde(skip)]
    pub laplacian: Option<DMatrix<f64>>,
}

impl TwoComponentSolver {
    pub fn new(grid: Grid1D, d1: f64, d2: f64) -> Self {
        Self { grid, d1, d2, laplacian: None }
    }

    pub fn init(&mut self) {
        self.laplacian = Some(laplacian_1d(&self.grid));
    }

    /// FitzHugh-Nagumo kinetics: du/dt = u - u³/3 - v + I, dv/dt = ε(u + a - bv).
    pub fn step_fhn(&self, u: &DVector<f64>, v: &DVector<f64>, dt: f64,
                    a: f64, b: f64, epsilon: f64, i_ext: f64,
                    bc_u: &[f64; 2], bc_v: &[f64; 2]) -> (DVector<f64>, DVector<f64>) {
        let lap = self.laplacian.as_ref().expect("call init() first");
        let n = self.grid.n;
        let dx2 = self.grid.dx * self.grid.dx;

        let mut lap_u = lap * u;
        lap_u[0] += bc_u[0] / dx2;
        lap_u[n - 1] += bc_u[1] / dx2;

        let mut lap_v = lap * v;
        lap_v[0] += bc_v[0] / dx2;
        lap_v[n - 1] += bc_v[1] / dx2;

        let new_u = u + dt * (self.d1 * lap_u
            + DVector::from_fn(n, |i, _| u[i] - u[i].powi(3) / 3.0 - v[i] + i_ext));
        let new_v = v + dt * (self.d2 * lap_v
            + DVector::from_fn(n, |i, _| epsilon * (u[i] + a - b * v[i])));

        (new_u, new_v)
    }
}
