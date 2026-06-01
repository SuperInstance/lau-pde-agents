//! Wave equation: ∂²u/∂t² = c²·Δu
//!
//! Models oscillatory agent behavior — beliefs that propagate and reflect,
//! like waves on a string or membrane.

use crate::{Grid1D, laplacian_1d, DMatrix, DVector};
use serde::{Serialize, Deserialize};

/// Solver for the 1D wave equation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveSolver {
    pub grid: Grid1D,
    /// Wave speed c > 0.
    pub c: f64,
    #[serde(skip)]
    pub laplacian: Option<DMatrix<f64>>,
}

impl WaveSolver {
    pub fn new(grid: Grid1D, c: f64) -> Self {
        assert!(c > 0.0, "wave speed must be positive");
        Self { grid, c, laplacian: None }
    }

    pub fn init(&mut self) {
        self.laplacian = Some(laplacian_1d(&self.grid));
    }

    /// CFL stability condition: dt < dx/c.
    pub fn max_stable_dt(&self) -> f64 {
        self.grid.dx / self.c
    }

    /// One step using the leapfrog (Verlet) scheme.
    /// `u_prev`, `u_curr` are interior values. `bc` = [left, right] boundary.
    pub fn step(&self, u_prev: &DVector<f64>, u_curr: &DVector<f64>, dt: f64, bc: &[f64; 2]) -> DVector<f64> {
        let lap = self.laplacian.as_ref().expect("call init() first");
        let n = self.grid.n;
        let dx2 = self.grid.dx * self.grid.dx;

        let mut rhs = lap * u_curr;
        rhs[0] += bc[0] / dx2;
        rhs[n - 1] += bc[1] / dx2;

        // u_next = 2*u_curr - u_prev + (c*dt)² * Δu
        2.0 * u_curr - u_prev + (self.c * dt).powi(2) * rhs
    }

    /// Solve wave equation with initial displacement `u0` and initial velocity `v0`.
    pub fn solve(&mut self, u0: &DVector<f64>, v0: &DVector<f64>, dt: f64, steps: usize, bc: &[f64; 2]) -> Vec<DVector<f64>> {
        self.init();
        let mut traj = Vec::with_capacity(steps + 1);
        let mut u_prev = u0.clone();
        let mut u_curr = u0 + v0.scale(dt); // first-order Taylor for u^1
        traj.push(u_prev.clone());
        traj.push(u_curr.clone());
        for _ in 1..steps {
            let u_next = self.step(&u_prev, &u_curr, dt, bc);
            traj.push(u_next.clone());
            u_prev = u_curr;
            u_curr = u_next;
        }
        traj
    }

    /// Compute total energy: E = ½∫(∂u/∂t)² dx + ½c²∫(∂u/∂x)² dx.
    pub fn energy(&self, u_prev: &DVector<f64>, u_curr: &DVector<f64>, dt: f64) -> f64 {
        let dx = self.grid.dx;
        let ut = (u_curr - u_prev).scale(1.0 / dt);
        let kinetic = 0.5 * ut.iter().map(|x| x * x).sum::<f64>() * dx;

        let mut grad_sq = 0.0;
        for i in 0..self.grid.n - 1 {
            let diff = u_curr[i + 1] - u_curr[i];
            grad_sq += diff * diff;
        }
        let potential = 0.5 * self.c * self.c * grad_sq / dx;

        kinetic + potential
    }
}
