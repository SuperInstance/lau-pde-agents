# lau-pde-agents

Partial differential equations governing agent dynamics.

Agent beliefs evolve in continuous space and time. This crate provides numerical solvers for the key PDEs describing how agent populations distribute, oscillate, reach equilibrium, and interact.

## Equations

| Equation | Formulation | Agent Interpretation |
|----------|-------------|---------------------|
| Heat | ∂u/∂t = αΔu | Belief diffusion — beliefs spread and smooth out |
| Wave | ∂²u/∂t² = c²Δu | Belief oscillation — beliefs propagate and reflect |
| Laplace | Δu = 0 | Agent equilibrium — steady-state belief distribution |
| Poisson | Δu = f | Forced equilibrium — beliefs under external influence |
| Schrödinger | iℏ∂ψ/∂t = Hψ | Quantum agent dynamics — probabilistic state evolution |
| Reaction-diffusion | ∂u/∂t = DΔu + R(u) | Belief creation/destruction — pattern formation |
| Transport | ∂u/∂t + v·∇u = 0 | Belief propagation — directed advection |
| Burgers | ∂u/∂t + u∂u/∂x = ν∂²u/∂x² | Nonlinear dynamics — steepening and shocks |

## Theoretical Results

- **Maximum principle**: Solutions of elliptic/parabolic PDEs achieve extrema on boundary
- **Energy estimates**: Control solutions via initial energy
- **Mean value property**: Harmonic functions satisfy averaging
- **Poincaré inequality**: Bounds on zero-mean functions
- **Sobolev embedding**: L∞ bounds from H¹ regularity

## Quick Start

```rust
use lau_pde_agents::{Grid1D, heat::HeatSolver, nalgebra::DVector};

let grid = Grid1D::new(0.0, 1.0, 50);
let mut solver = HeatSolver::new(grid, 0.01);

let u0 = DVector::from_fn(50, |i, _| (i as f64 / 50.0 * std::f64::consts::PI).sin());
let dt = solver.max_stable_dt() * 0.5;
let trajectory = solver.solve_explicit(&u0, dt, 1000, &[0.0, 0.0]);
```

## Features

- `serde` serialization for all solvers and grids
- `nalgebra` for linear algebra operations
- Explicit and implicit time-stepping schemes
- 1D and 2D solvers
- 87+ tests covering correctness, stability, and theoretical properties

## License

MIT
