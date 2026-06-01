# lau-pde-agents

**Partial differential equations for agent dynamics** — numerical solvers for the PDEs that govern how agent beliefs propagate, diffuse, oscillate, and reach equilibrium.

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-87-green.svg)](src/tests.rs)

---

## Why PDEs for Agents?

Imagine a population of agents whose beliefs are defined over a continuous domain — physical space, an opinion axis, or a parameter space. Each PDE in this crate describes a different way those beliefs evolve:

| Equation | PDE | Agent Interpretation |
|---|---|---|
| **Heat** | ∂u/∂t = α Δu | Beliefs **diffuse** and smooth out over time |
| **Wave** | ∂²u/∂t² = c² Δu | Beliefs **oscillate** — propagate and reflect |
| **Laplace** | Δu = 0 | **Equilibrium** — steady-state belief distribution |
| **Poisson** | Δu = f | **Forced equilibrium** — beliefs under external influence |
| **Schrödinger** | iℏ ∂ψ/∂t = Hψ | **Quantum agents** — probabilistic state evolution |
| **Reaction-diffusion** | ∂u/∂t = D Δu + R(u) | **Pattern formation** — beliefs created and destroyed |
| **Transport** | ∂u/∂t + v · ∂u/∂x = 0 | **Advection** — directed belief propagation |
| **Burgers** | ∂u/∂t + u · ∂u/∂x = ν ∂²u/∂x² | **Nonlinear dynamics** — belief steepening and shocks |

This crate provides clean, well-tested Rust implementations of all eight equations with multiple numerical methods, stability guarantees, and an agent-oriented API.

---

## Quick Start

```toml
# Cargo.toml
[dependencies]
lau-pde-agents = "0.1"
```

```rust
use lau_pde_agents::{Grid1D, HeatSolver};
use lau_pde_agents::nalgebra::DVector;

// Set up a spatial grid on [0, 1] with 50 interior points
let grid = Grid1D::new(0.0, 1.0, 50);

// Create a heat/diffusion solver with α = 0.01
let mut solver = HeatSolver::new(grid.clone(), 0.01);

// Initial condition: Gaussian bump of belief
let u0 = DVector::from_fn(50, |i, _| {
    let x = grid.x(i);
    (-((x - 0.5).powi(2)) / 0.01).exp()
});

// Solve with zero boundary conditions
let trajectory = solver.solve_explicit(&u0, 0.001, 1000, &[0.0, 0.0]);

println!("Initial max belief: {:.4}", u0.max());
println!("Final max belief:   {:.4}", trajectory.last().unwrap().max());
```

---

## Architecture

### Grid System

All solvers operate on discretized domains:

- **`Grid1D`** — 1D domain [a, b] with N interior points (uniform spacing)
- **`Grid2D`** — 2D rectangular domain [aₓ, bₓ] × [aᵧ, bᵧ]

Grids store the spacing `dx` (and `dy`) for you, and provide methods to evaluate interior point locations.

### Finite Difference Laplacian

The core building block is `laplacian_1d()`, which constructs the standard tridiagonal second-difference matrix:

```
L = (1/dx²) * [ -2   1   0  ...  0  ]
               [  1  -2   1  ...  0  ]
               [  0   1  -2  ...  0  ]
               [ ...                ... ]
               [  0  ...  1  -2   1  ]
               [  0  ...  0   1  -2  ]
```

### Solver Design

Each solver follows the same pattern:

```rust
// 1. Create grid
let grid = Grid1D::new(0.0, 1.0, 50);

// 2. Create solver
let mut solver = SomeSolver::new(grid, params);

// 3. Initialize (builds matrices)
solver.init();

// 4. Step or solve
let result = solver.solve(&initial_data, dt, steps, &boundary_conditions);
```

---

## Solvers in Detail

### Heat Equation (`heat`)

**∂u/∂t = α Δu** — the quintessential diffusion equation.

```rust
let mut solver = HeatSolver::new(grid, 0.1);
let trajectory = solver.solve_explicit(&u0, dt, steps, &[0.0, 0.0]);
```

Methods:
- **Forward Euler (explicit)** — `step_explicit()`, `solve_explicit()`
- **Backward Euler (implicit)** — `step_implicit()` with Gauss-Seidel inner solve
- **Steady-state solver** — `solve_to_steady()` iterates until convergence

Stability: `dt < dx² / (2α)` for explicit Euler. Use `max_stable_dt()` to check.

Also includes **`HeatSolver2D`** for 2D diffusion problems.

### Wave Equation (`wave`)

**∂²u/∂t² = c² Δu** — oscillatory dynamics with energy conservation.

```rust
let mut solver = WaveSolver::new(grid, 1.0);
let trajectory = solver.solve(&u0, &v0, dt, steps, &[0.0, 0.0]);
let energy = solver.energy(&u_prev, &u_curr, dt);
```

Uses the **leapfrog (Störmer-Verlet)** scheme — symplectic, second-order, and energy-conserving for CFL-stable time steps (`dt < dx/c`).

### Laplace Equation (`laplace`)

**Δu = 0** — harmonic equilibrium with no forcing.

```rust
let solver = LaplaceSolver::new(grid);
let u = solver.solve(0.0, 1.0); // Dirichlet BCs
let (u_jacobi, iters) = solver.solve_jacobi(0.0, 1.0, 1e-10, 10000);
```

Methods:
- **Analytic** — exact solution (linear interpolation in 1D)
- **Jacobi iteration** — parallelizable, guaranteed convergence
- **Gauss-Seidel** — faster sequential convergence

Includes **`LaplaceSolver2D`** for rectangular domains.

### Poisson Equation (`poisson`)

**Δu = f** — equilibrium with a forcing term.

```rust
let solver = PoissonSolver::new(grid);
let f = DVector::from_element(n, -1.0); // uniform source
let (u, iters) = solver.solve(&f, 0.0, 0.0, 1e-10, 10000);
```

Supports:
- **Dirichlet BCs** — fixed values at boundaries
- **Robin (mixed) BCs** — `α·u + β·u' = γ` via ghost-point method
- **Gauss-Seidel** — `solve_gs()` for faster convergence

### Schrödinger Equation (`schrodinger`)

**iℏ ∂ψ/∂t = Hψ** — quantum agent state evolution.

```rust
let mut solver = SchrodingerSolver::new(grid, 0.5);
let (p_traj, q_traj) = solver.solve(&p0, &q0, dt, steps);
let density = solver.probability_density(&p, &q);
let total_prob = solver.total_probability(&p, &q);
```

The wavefunction is split into real/imaginary parts: `ψ = p + iq`. Supports optional potential `V(x)` for non-free particles. Key observables:
- `probability_density()` — |ψ|²
- `total_probability()` — ∫|ψ|² dx (conserved)
- `expectation_x()` — ⟨x⟩

### Reaction-Diffusion (`reaction_diffusion`)

**∂u/∂t = D Δu + R(u)** — pattern formation from the interplay of diffusion and reaction.

```rust
let mut solver = ReactionDiffusionSolver::new(grid, 0.01);
let trajectory = solver.solve(&u0, dt, steps, ReactionDiffusionSolver::allen_cahn(), &[0.0, 0.0]);
```

Built-in reaction kinetics:
- **Fisher-KPP** — `R(u) = u(1-u)` — logistic growth, traveling wave solutions
- **Allen-Cahn** — `R(u) = u - u³` — phase separation, interface dynamics
- **Constant source** — uniform forcing
- **Zero reaction** — pure diffusion fallback

Also includes **`TwoComponentSolver`** for systems like **FitzHugh-Nagumo** (activator-inhibitor dynamics).

### Transport Equation (`transport`)

**∂u/∂t + v · ∂u/∂x = 0** — pure advection.

```rust
let solver = TransportSolver::new(grid, 1.0);
let trajectory = solver.solve_upwind(&u0, dt, steps);
```

Three numerical schemes:
- **Upwind** — first-order, stable, diffusive
- **Lax-Friedrichs** — first-order, adds numerical viscosity
- **Lax-Wendroff** — second-order, less diffusive but can oscillate

Use `courant(dt)` to check the CFL number (should be ≤ 1 for stability).

### Burgers' Equation (`burgers`)

**∂u/∂t + u · ∂u/∂x = ν ∂²u/∂x²** — the canonical nonlinear PDE.

```rust
let solver = BurgersSolver::new(grid, 0.01);
let trajectory = solver.solve(&u0, dt, steps);
let re = solver.reynolds(&u);
```

- `energy()` — ∫u² dx (L² norm squared)
- `total_variation()` — TV(u), monitors shock formation
- `reynolds()` — Re = U·L/ν, high Re → shock-dominated

Methods: upwind + central diffusion, Lax-Friedrichs + diffusion.

---

## Theoretical Foundations

### Maximum Principle (`maximum_principle`)

Verifies fundamental PDE properties:

| Function | Property |
|---|---|
| `verify_laplace_max_principle` | Solution extrema lie on boundary |
| `verify_strong_max_principle` | Interior max implies constant solution |
| `verify_parabolic_max_principle` | Heat equation respects initial/boundary extrema |
| `verify_comparison_principle` | If u₀ ≥ v₀ then u(t) ≥ v(t) |
| `verify_mean_value_property` | Harmonic functions satisfy discrete mean value |
| `harnack_ratio` | Harnack inequality bound for positive solutions |
| `gradient_bound` | Gradient controlled by boundary data |

### Energy Estimates (`energy`)

Sobolev space tools for solution control:

| Function | What It Measures |
|---|---|
| `l2_norm` | Standard L² norm |
| `h1_norm` / `h1_seminorm` | Sobolev H¹ norm and seminorm |
| `heat_energy` | ½∫u² dx — non-increasing for heat equation |
| `heat_dissipation` | dE/dt = -α∫|∇u|² dx ≤ 0 |
| `wave_energy` | Kinetic + potential — conserved for waves |
| `verify_poincare` | Poincaré inequality: ∫u² ≤ C∫|∇u|² |
| `verify_sobolev_bound` | L∞ ≤ C·H¹ embedding |
| `verify_exponential_decay` | ‖u(t)‖ ≤ ‖u(0)‖·exp(-λ₁t) |

---

## Application Layer (`application`)

Maps each PDE to concrete agent scenarios with ready-to-use types:

```rust
use lau_pde_agents::application::*;

// Define initial belief profiles
let gaussian = BeliefProfile::Gaussian {
    amplitude: 1.0, center: 0.5, width: 0.1
};
let step = BeliefProfile::Step {
    left: 0.0, right: 1.0, x0: 0.5
};
let sinusoidal = BeliefProfile::Sinusoidal {
    amplitude: 1.0, mode: 3
};

let u0 = gaussian.evaluate(&grid);
```

**Scenarios:**

| Scenario | PDE | Use Case |
|---|---|---|
| `BeliefDiffusion` | Heat | Opinions smooth out in a population |
| `BeliefOscillation` | Wave | Polarization cycles |
| `BeliefEquilibrium` | Poisson | Beliefs under external messaging |
| `BeliefPropagation` | Transport | News traveling through a network |
| `BeliefPatternFormation` | Reaction-diffusion | Emergence of opinion clusters |
| `NonlinearBelief` | Burgers | Shock formation in belief space |

**Analysis utilities:**

- `belief_consensus(trajectory, grid)` — mean belief at each timestep
- `belief_variance(trajectory, grid)` — variance over time
- `is_consensus_reached(trajectory, grid, threshold)` — consensus detection

---

## Testing

87 tests covering every solver, numerical method, and theoretical property:

```bash
cargo test
```

Tests verify:
- **Convergence** — numerical solutions approach analytic answers
- **Stability** — solutions remain bounded under CFL conditions
- **Conservation** — wave energy, Schrödinger probability
- **Decay** — heat energy decreases monotonically
- **Maximum principle** — all variants pass
- **Poincaré / Sobolev inequalities** — verified numerically
- **Boundary conditions** — Dirichlet, Robin, and zero BCs
- **Agent scenarios** — belief diffusion, consensus, pattern formation

---

## Dependencies

| Crate | Purpose |
|---|---|
| `nalgebra` | Linear algebra (matrices, vectors) with serde support |
| `serde` / `serde_json` | Serialization of solvers, grids, and trajectories |
| `approx` (dev) | Floating-point comparison in tests |

---

## License

MIT
