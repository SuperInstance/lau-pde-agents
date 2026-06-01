//! Application: PDEs governing agent belief dynamics.
//!
//! Maps each PDE to a concrete agent interpretation and provides
//! composite simulation scenarios.

use crate::{Grid1D, DVector};
use crate::heat::HeatSolver;
use crate::wave::WaveSolver;
use crate::laplace::LaplaceSolver;
use crate::poisson::PoissonSolver;
use crate::reaction_diffusion::ReactionDiffusionSolver;
use crate::transport::TransportSolver;
use crate::burgers::BurgersSolver;

/// Belief profile type.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum BeliefProfile {
    /// Uniform belief: u(x) = constant.
    Uniform { value: f64 },
    /// Gaussian belief: u(x) = A·exp(-(x-μ)²/(2σ²)).
    Gaussian { amplitude: f64, center: f64, width: f64 },
    /// Step function: belief jumps at x = x0.
    Step { left: f64, right: f64, x0: f64 },
    /// Sinusoidal: u(x) = A·sin(kπx/L).
    Sinusoidal { amplitude: f64, mode: usize },
}

impl BeliefProfile {
    /// Evaluate on a grid.
    pub fn evaluate(&self, grid: &Grid1D) -> DVector<f64> {
        match self {
            BeliefProfile::Uniform { value } => DVector::from_element(grid.n, *value),
            BeliefProfile::Gaussian { amplitude, center, width } => {
                DVector::from_fn(grid.n, |i, _| {
                    let x = grid.x(i);
                    amplitude * (-(x - center).powi(2) / (2.0 * width * width)).exp()
                })
            }
            BeliefProfile::Step { left, right, x0 } => {
                DVector::from_fn(grid.n, |i, _| {
                    if grid.x(i) < *x0 { *left } else { *right }
                })
            }
            BeliefProfile::Sinusoidal { amplitude, mode } => {
                let l = grid.b - grid.a;
                DVector::from_fn(grid.n, |i, _| {
                    let x = grid.x(i) - grid.a;
                    amplitude * (*mode as f64 * std::f64::consts::PI * x / l).sin()
                })
            }
        }
    }
}

/// Scenario: belief diffusion over time.
pub struct BeliefDiffusion {
    pub solver: HeatSolver,
}

impl BeliefDiffusion {
    pub fn new(grid: Grid1D, diffusion_rate: f64) -> Self {
        Self { solver: HeatSolver::new(grid, diffusion_rate) }
    }

    /// Run belief diffusion simulation.
    pub fn run(&mut self, initial: &DVector<f64>, dt: f64, steps: usize, bc: &[f64; 2]) -> Vec<DVector<f64>> {
        self.solver.solve_explicit(initial, dt, steps, bc)
    }
}

/// Scenario: belief oscillation (wave).
pub struct BeliefOscillation {
    pub solver: WaveSolver,
}

impl BeliefOscillation {
    pub fn new(grid: Grid1D, speed: f64) -> Self {
        Self { solver: WaveSolver::new(grid, speed) }
    }

    pub fn run(&mut self, displacement: &DVector<f64>, velocity: &DVector<f64>, dt: f64, steps: usize, bc: &[f64; 2]) -> Vec<DVector<f64>> {
        self.solver.solve(displacement, velocity, dt, steps, bc)
    }
}

/// Scenario: belief equilibrium (Poisson).
pub struct BeliefEquilibrium {
    pub solver: PoissonSolver,
}

impl BeliefEquilibrium {
    pub fn new(grid: Grid1D) -> Self {
        Self { solver: PoissonSolver::new(grid) }
    }

    pub fn run(&self, forcing: &DVector<f64>, bc_left: f64, bc_right: f64, tol: f64, max_iters: usize) -> (DVector<f64>, usize) {
        self.solver.solve(forcing, bc_left, bc_right, tol, max_iters)
    }
}

/// Scenario: belief propagation (transport).
pub struct BeliefPropagation {
    pub solver: TransportSolver,
}

impl BeliefPropagation {
    pub fn new(grid: Grid1D, velocity: f64) -> Self {
        Self { solver: TransportSolver::new(grid, velocity) }
    }

    pub fn run(&self, initial: &DVector<f64>, dt: f64, steps: usize) -> Vec<DVector<f64>> {
        self.solver.solve_upwind(initial, dt, steps)
    }
}

/// Scenario: belief pattern formation (reaction-diffusion).
pub struct BeliefPatternFormation {
    pub solver: ReactionDiffusionSolver,
}

impl BeliefPatternFormation {
    pub fn new(grid: Grid1D, diffusion: f64) -> Self {
        Self { solver: ReactionDiffusionSolver::new(grid, diffusion) }
    }

    pub fn run(&mut self, initial: &DVector<f64>, dt: f64, steps: usize, bc: &[f64; 2]) -> Vec<DVector<f64>> {
        self.solver.solve(initial, dt, steps, ReactionDiffusionSolver::allen_cahn(), bc)
    }
}

/// Scenario: nonlinear belief dynamics (Burgers).
pub struct NonlinearBelief {
    pub solver: BurgersSolver,
}

impl NonlinearBelief {
    pub fn new(grid: Grid1D, viscosity: f64) -> Self {
        Self { solver: BurgersSolver::new(grid, viscosity) }
    }

    pub fn run(&self, initial: &DVector<f64>, dt: f64, steps: usize) -> Vec<DVector<f64>> {
        self.solver.solve(initial, dt, steps)
    }
}

/// Compute belief consensus (mean) at each time step.
pub fn belief_consensus(trajectory: &[DVector<f64>], grid: &Grid1D) -> Vec<f64> {
    trajectory.iter().map(|u| {
        u.iter().sum::<f64>() * grid.dx / (grid.b - grid.a)
    }).collect()
}

/// Compute belief variance at each time step.
pub fn belief_variance(trajectory: &[DVector<f64>], grid: &Grid1D) -> Vec<f64> {
    trajectory.iter().map(|u| {
        let mean = u.iter().sum::<f64>() * grid.dx / (grid.b - grid.a);
        u.iter().map(|x| (x - mean).powi(2)).sum::<f64>() * grid.dx / (grid.b - grid.a)
    }).collect()
}

/// Detect if consensus is reached (variance < threshold).
pub fn is_consensus_reached(trajectory: &[DVector<f64>], grid: &Grid1D, threshold: f64) -> bool {
    if trajectory.is_empty() { return false; }
    belief_variance(trajectory, grid).last().map_or(false, |v| *v < threshold)
}
