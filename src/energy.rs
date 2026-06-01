//! Energy estimates: control solutions via initial energy.
//!
//! For PDEs governing agent dynamics, energy methods provide bounds on solution
//! growth, stability, and decay — essential guarantees for agent systems.

use crate::{Grid1D, DVector};

/// L² norm: √(∫u² dx).
pub fn l2_norm(u: &DVector<f64>, dx: f64) -> f64 {
    (u.iter().map(|x| x * x).sum::<f64>() * dx).sqrt()
}

/// L² inner product: ∫u·v dx.
pub fn l2_inner(u: &DVector<f64>, v: &DVector<f64>, dx: f64) -> f64 {
    u.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<f64>() * dx
}

/// H¹ seminorm: √(∫|∇u|² dx).
pub fn h1_seminorm(u: &DVector<f64>, dx: f64) -> f64 {
    let grad_sq: f64 = (0..u.len() - 1)
        .map(|i| {
            let diff = u[i + 1] - u[i];
            diff * diff
        })
        .sum::<f64>() / dx;
    grad_sq.sqrt()
}

/// H¹ norm: √(∫u² + |∇u|² dx).
pub fn h1_norm(u: &DVector<f64>, dx: f64) -> f64 {
    let l2_sq = u.iter().map(|x| x * x).sum::<f64>() * dx;
    let grad_sq = (0..u.len() - 1)
        .map(|i| (u[i + 1] - u[i]).powi(2))
        .sum::<f64>() / dx;
    (l2_sq + grad_sq).sqrt()
}

/// Heat energy: E(t) = ½∫u² dx. Should be non-increasing for heat equation with zero BCs.
pub fn heat_energy(u: &DVector<f64>, dx: f64) -> f64 {
    0.5 * u.iter().map(|x| x * x).sum::<f64>() * dx
}

/// Heat energy dissipation rate: dE/dt = -α∫|∇u|² dx ≤ 0.
pub fn heat_dissipation(u: &DVector<f64>, dx: f64, alpha: f64) -> f64 {
    -alpha * (0..u.len() - 1)
        .map(|i| (u[i + 1] - u[i]).powi(2))
        .sum::<f64>() / dx
}

/// Wave energy: E = ½∫(∂u/∂t)² dx + ½c²∫(∂u/∂x)² dx. Should be conserved.
pub fn wave_energy(u_prev: &DVector<f64>, u_curr: &DVector<f64>, dx: f64, dt: f64, c: f64) -> f64 {
    let ut = (u_curr - u_prev).scale(1.0 / dt);
    let kinetic = 0.5 * ut.iter().map(|x| x * x).sum::<f64>() * dx;
    let potential = 0.5 * c * c * (0..u_curr.len() - 1)
        .map(|i| (u_curr[i + 1] - u_curr[i]).powi(2))
        .sum::<f64>() / dx;
    kinetic + potential
}

/// Verify heat energy decay: E(t) ≤ E(0) for all t.
pub fn verify_heat_energy_decay(trajectory: &[DVector<f64>], dx: f64) -> bool {
    let e0 = heat_energy(&trajectory[0], dx);
    for u in &trajectory[1..] {
        if heat_energy(u, dx) > e0 + 1e-10 {
            return false;
        }
    }
    true
}

/// Verify wave energy conservation: |E(t) - E(0)| ≤ tolerance.
pub fn verify_wave_energy_conservation(
    trajectory: &[DVector<f64>], dx: f64, dt: f64, c: f64, tol: f64
) -> bool {
    if trajectory.len() < 2 { return true; }
    let e0 = wave_energy(&trajectory[0], &trajectory[1], dx, dt, c);
    for k in 1..trajectory.len() - 1 {
        let e = wave_energy(&trajectory[k], &trajectory[k + 1], dx, dt, c);
        if (e - e0).abs() > tol {
            return false;
        }
    }
    true
}

/// Sobolev embedding bound: L∞ ≤ C·H¹ (in 1D, C = 1).
pub fn verify_sobolev_bound(u: &DVector<f64>, dx: f64) -> bool {
    let linf = u.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
    let h1 = h1_norm(u, dx);
    linf <= h1 + 1e-10
}

/// Poincaré inequality: ∫u² ≤ C·∫|∇u|² for zero-mean functions.
pub fn verify_poincare(u: &DVector<f64>, dx: f64) -> bool {
    let mean = u.iter().sum::<f64>() / u.len() as f64;
    let u_centered = DVector::from_fn(u.len(), |i, _| u[i] - mean);
    let l2_sq = u_centered.iter().map(|x| x * x).sum::<f64>() * dx;
    let grad_sq = (0..u.len() - 1)
        .map(|i| (u[i + 1] - u[i]).powi(2))
        .sum::<f64>() / dx;
    // Poincaré constant for [0,L] is L²/π²
    let domain_length = (u.len() as f64 + 1.0) * dx;
    let c_poincare = domain_length * domain_length / (std::f64::consts::PI * std::f64::consts::PI);
    l2_sq <= c_poincare * grad_sq + 1e-10 || grad_sq < 1e-15
}

/// Exponential decay estimate for heat equation: ||u(t)|| ≤ ||u(0)||·exp(-λ₁t).
/// λ₁ = π²/L² is the principal eigenvalue of -Δ on [0,L] with zero BCs.
pub fn verify_exponential_decay(
    trajectory: &[DVector<f64>], dx: f64, dt: f64, alpha: f64
) -> bool {
    let domain_length = (trajectory[0].len() as f64 + 1.0) * dx;
    let lambda1 = std::f64::consts::PI * std::f64::consts::PI / (domain_length * domain_length);
    let e0 = heat_energy(&trajectory[0], dx);

    for (k, u) in trajectory.iter().enumerate() {
        let t = k as f64 * dt;
        let bound = e0 * (-2.0 * alpha * lambda1 * t).exp();
        if heat_energy(u, dx) > bound + 1e-10 {
            return false;
        }
    }
    true
}
