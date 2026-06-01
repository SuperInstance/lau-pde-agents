//! Schrödinger equation: iℏ ∂ψ/∂t = Hψ
//!
//! Quantum agent dynamics — agents in superposition of states evolving unitarily.
//! We use a real-valued split-step representation: ψ = p + iq.

use crate::{Grid1D, laplacian_1d, DMatrix, DVector};
use serde::{Serialize, Deserialize};

/// Solver for the 1D Schrödinger equation (free particle or with potential).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchrodingerSolver {
    pub grid: Grid1D,
    /// ℏ/(2m), controls dispersion.
    pub coeff: f64,
    /// Optional potential V(x) at each grid point.
    pub potential: Option<DVector<f64>>,
    #[serde(skip)]
    pub laplacian: Option<DMatrix<f64>>,
}

impl SchrodingerSolver {
    /// Create a free-particle solver.
    pub fn new(grid: Grid1D, coeff: f64) -> Self {
        Self { grid, coeff, potential: None, laplacian: None }
    }

    /// Create a solver with potential V(x).
    pub fn with_potential(grid: Grid1D, coeff: f64, potential: DVector<f64>) -> Self {
        assert_eq!(potential.len(), grid.n);
        Self { grid, coeff, potential: Some(potential), laplacian: None }
    }

    pub fn init(&mut self) {
        self.laplacian = Some(laplacian_1d(&self.grid));
    }

    /// One step using split-operator method: ψ = p + iq (real/imaginary parts).
    /// Returns (p_new, q_new).
    pub fn step(&self, p: &DVector<f64>, q: &DVector<f64>, dt: f64) -> (DVector<f64>, DVector<f64>) {
        let lap = self.laplacian.as_ref().expect("call init() first");
        let c = self.coeff;

        // H(p+iq) = -c·Δp + V·p + i(-c·Δq + V·q)
        // dp/dt = c·Δq - V·q,  dq/dt = -c·Δp + V·p
        let lap_p = lap * p;
        let lap_q = lap * q;

        let (dp, dq) = if let Some(ref v) = self.potential {
            (c * lap_q - v.component_mul(q), -c * lap_p + v.component_mul(p))
        } else {
            (c * &lap_q, -c * &lap_p)
        };

        let p_new = p + dp.scale(dt);
        let q_new = q + dq.scale(dt);
        (p_new, q_new)
    }

    /// Compute probability density |ψ|² = p² + q².
    pub fn probability_density(&self, p: &DVector<f64>, q: &DVector<f64>) -> DVector<f64> {
        DVector::from_fn(p.len(), |i, _| p[i] * p[i] + q[i] * q[i])
    }

    /// Compute total probability (integral of |ψ|²).
    pub fn total_probability(&self, p: &DVector<f64>, q: &DVector<f64>) -> f64 {
        let density = self.probability_density(p, q);
        density.iter().sum::<f64>() * self.grid.dx
    }

    /// Compute ⟨x⟩ = ∫x|ψ|² dx.
    pub fn expectation_x(&self, p: &DVector<f64>, q: &DVector<f64>) -> f64 {
        let density = self.probability_density(p, q);
        let mut sum = 0.0;
        for i in 0..self.grid.n {
            sum += self.grid.x(i) * density[i];
        }
        sum * self.grid.dx
    }

    /// Run simulation, returning trajectory of (p, q) pairs.
    pub fn solve(&mut self, p0: &DVector<f64>, q0: &DVector<f64>, dt: f64, steps: usize) -> Vec<(DVector<f64>, DVector<f64>)> {
        self.init();
        let mut traj = Vec::with_capacity(steps + 1);
        let mut p = p0.clone();
        let mut q = q0.clone();
        traj.push((p.clone(), q.clone()));
        for _ in 0..steps {
            let (p_new, q_new) = self.step(&p, &q, dt);
            p = p_new;
            q = q_new;
            traj.push((p.clone(), q.clone()));
        }
        traj
    }
}
