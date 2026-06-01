//! Maximum principle: solutions of elliptic/parabolic PDEs achieve extrema on boundary.
//!
//! The maximum principle is fundamental to PDE theory and has deep implications
//! for agent dynamics: no agent belief can exceed the range of boundary beliefs
//! in equilibrium, and diffusion can only smooth, not amplify.

use crate::{Grid1D, DVector};

/// Verify the discrete maximum principle for a Laplace solution.
/// Returns true if max(u) ≤ max(bc) and min(u) ≥ min(bc).
pub fn verify_laplace_max_principle(u: &DVector<f64>, bc_left: f64, bc_right: f64) -> bool {
    let u_max = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let u_min = u.iter().cloned().fold(f64::INFINITY, f64::min);
    let bc_max = bc_left.max(bc_right);
    let bc_min = bc_left.min(bc_right);
    // Allow small numerical tolerance
    u_max <= bc_max + 1e-10 && u_min >= bc_min - 1e-10
}

/// Verify the strong maximum principle: if u achieves its maximum at an interior
/// point, then u is constant. For discrete: check that interior max < boundary max.
pub fn verify_strong_max_principle(u: &DVector<f64>, bc_left: f64, bc_right: f64) -> bool {
    let u_max = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bc_max = bc_left.max(bc_right);
    // Interior maximum must be strictly less than boundary maximum
    // (unless u is constant)
    let tol = 1e-10;
    if (u_max - bc_max).abs() < tol {
        // u is approximately constant — check all values are nearly equal
        let u_min = u.iter().cloned().fold(f64::INFINITY, f64::min);
        (u_max - u_min).abs() < tol
    } else {
        u_max < bc_max + tol
    }
}

/// Verify the parabolic maximum principle for heat equation solutions.
/// At each time step, u should be bounded by the initial and boundary extrema.
pub fn verify_parabolic_max_principle(
    trajectory: &[DVector<f64>],
    bc_left: f64,
    bc_right: f64,
    u0: &DVector<f64>,
) -> bool {
    let mut global_max = u0.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        .max(bc_left).max(bc_right);
    let mut global_min = u0.iter().cloned().fold(f64::INFINITY, f64::min)
        .min(bc_left).min(bc_right);
    for u in trajectory {
        let u_max = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let u_min = u.iter().cloned().fold(f64::INFINITY, f64::min);
        if u_max > global_max + 1e-10 || u_min < global_min - 1e-10 {
            return false;
        }
    }
    true
}

/// Compute the Harnack inequality ratio: max(u)/min(u) for positive solutions.
/// For Laplace on an interval, this should be bounded.
pub fn harnack_ratio(u: &DVector<f64>) -> f64 {
    let u_max = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let u_min = u.iter().cloned().filter(|&x| x > 0.0).fold(f64::INFINITY, f64::min);
    if u_min > 1e-15 { u_max / u_min } else { f64::INFINITY }
}

/// Check that a solution satisfies a comparison principle:
/// if u0 ≥ v0 and same BCs, then u(t) ≥ v(t) for all t.
pub fn verify_comparison_principle(
    u_traj: &[DVector<f64>],
    v_traj: &[DVector<f64>],
) -> bool {
    let min_len = u_traj.len().min(v_traj.len());
    for k in 0..min_len {
        for i in 0..u_traj[k].len().min(v_traj[k].len()) {
            if u_traj[k][i] < v_traj[k][i] - 1e-10 {
                return false;
            }
        }
    }
    true
}

/// Mean value property: for harmonic functions, the value at a point equals
/// the average of neighboring values.
pub fn verify_mean_value_property(u: &DVector<f64>, bc_left: f64, bc_right: f64, tol: f64) -> bool {
    let n = u.len();
    for i in 0..n {
        let left = if i == 0 { bc_left } else { u[i - 1] };
        let right = if i == n - 1 { bc_right } else { u[i + 1] };
        if (u[i] - 0.5 * (left + right)).abs() > tol {
            return false;
        }
    }
    true
}

/// Gradient bound: for solutions of the Laplace equation, |∇u| is bounded
/// by the boundary data range divided by domain length.
pub fn gradient_bound(u: &DVector<f64>, dx: f64, bc_left: f64, bc_right: f64) -> bool {
    let range = (bc_right - bc_left).abs();
    let domain_length = (u.len() as f64 + 1.0) * dx;
    for i in 0..u.len() - 1 {
        let grad = (u[i + 1] - u[i]) / dx;
        if grad.abs() > range / domain_length + 1e-10 {
            return false;
        }
    }
    true
}
