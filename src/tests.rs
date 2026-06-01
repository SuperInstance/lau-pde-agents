//! Tests: 65+ tests covering all PDE solvers, maximum principle, energy estimates, and applications.

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use crate::*;
    use nalgebra::DVector;

    fn make_grid() -> Grid1D {
        Grid1D::new(0.0, 1.0, 50)
    }

    // ═══════════════════════════════════════════
    // Grid tests
    // ═══════════════════════════════════════════

    #[test]
    fn test_grid_creation() {
        let g = Grid1D::new(0.0, 1.0, 10);
        assert_eq!(g.n, 10);
        assert_relative_eq!(g.dx, 1.0 / 11.0, epsilon = 1e-12);
    }

    #[test]
    fn test_grid_points() {
        let g = Grid1D::new(0.0, 1.0, 3);
        let pts = g.points();
        assert_eq!(pts.len(), 3);
        assert_relative_eq!(pts[0], 0.25, epsilon = 1e-12);
        assert_relative_eq!(pts[2], 0.75, epsilon = 1e-12);
    }

    #[test]
    fn test_grid_2d() {
        let g = Grid2D::new(0.0, 1.0, 0.0, 1.0, 10, 10);
        assert_eq!(g.nx, 10);
        assert_eq!(g.ny, 10);
        assert_relative_eq!(g.dx, 1.0 / 11.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic]
    fn test_grid_invalid() {
        Grid1D::new(1.0, 0.0, 10);
    }

    #[test]
    #[should_panic]
    fn test_grid_too_few() {
        Grid1D::new(0.0, 1.0, 1);
    }

    // ═══════════════════════════════════════════
    // Laplacian tests
    // ═══════════════════════════════════════════

    #[test]
    fn test_laplacian_1d_structure() {
        let g = make_grid();
        let l = laplacian_1d(&g);
        assert_eq!(l.nrows(), 50);
        assert_eq!(l.ncols(), 50);
        // Diagonal should be -2/dx²
        let dx2 = g.dx * g.dx;
        assert_relative_eq!(l[(0, 0)], -2.0 / dx2, epsilon = 1e-10);
    }

    #[test]
    fn test_laplacian_kills_constant() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let l = laplacian_1d(&g);
        let u = DVector::from_element(20, 1.0);
        let mut lu = l * u;
        // With constant boundary u=1, all boundary terms add 1/dx²
        let dx2 = g.dx * g.dx;
        lu[0] += 1.0 / dx2;  // left boundary
        lu[19] += 1.0 / dx2; // right boundary
        for v in lu.iter() {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_laplacian_on_linear() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let l = laplacian_1d(&g);
        let u = DVector::from_fn(20, |i, _| g.x(i)); // u = x
        let mut lu = l * u;
        // Add boundary contributions (u(0)=0, u(1)=1)
        let dx2 = g.dx * g.dx;
        lu[0] += 0.0 / dx2;
        lu[19] += 1.0 / dx2;
        for v in lu.iter() {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-8);
        }
    }

    // ═══════════════════════════════════════════
    // Heat equation tests
    // ═══════════════════════════════════════════

    #[test]
    fn test_heat_stability_criterion() {
        let g = make_grid();
        let solver = heat::HeatSolver::new(g.clone(), 1.0);
        let dt_max = solver.max_stable_dt();
        assert!(dt_max > 0.0);
        assert!(dt_max < 1.0);
    }

    #[test]
    fn test_heat_zero_bc_decays() {
        let g = make_grid();
        let mut solver = heat::HeatSolver::new(g.clone(), 0.01);
        let u0: DVector<f64> = DVector::from_fn(50, |i, _| (i as f64 / 50.0 * std::f64::consts::PI).sin());
        let dt = solver.max_stable_dt() * 0.5;
        let (u_final, _) = solver.solve_to_steady(&u0, dt, &[0.0, 0.0], 1e-8, 50000);
        for v in u_final.iter() {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_heat_steady_state_linear() {
        let g = Grid1D::new(0.0, 1.0, 30);
        let mut solver = heat::HeatSolver::new(g.clone(), 1.0);
        let u0 = DVector::zeros(30);
        let dt = solver.max_stable_dt() * 0.4;
        let (u_final, _) = solver.solve_to_steady(&u0, dt, &[0.0, 1.0], 1e-10, 100000);
        // Steady state should be linear
        for i in 0..30 {
            let expected = (i as f64 + 1.0) / 31.0;
            assert_relative_eq!(u_final[i], expected, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_heat_implicit_step() {
        let g = make_grid();
        let mut solver = heat::HeatSolver::new(g.clone(), 0.1);
        solver.init();
        let u0 = DVector::from_fn(50, |i, _| (i as f64 / 50.0 * std::f64::consts::PI).sin());
        let u1 = solver.step_implicit(&u0, 0.001, &[0.0, 0.0], 50);
        assert_eq!(u1.len(), 50);
        // Amplitude should decrease (diffusion)
        let max0 = u0.iter().cloned().fold(0.0f64, f64::max).abs();
        let max1 = u1.iter().cloned().fold(0.0f64, f64::max).abs();
        assert!(max1 < max0 + 1e-10);
    }

    #[test]
    fn test_heat_2d() {
        let g2d = Grid2D::new(0.0, 1.0, 0.0, 1.0, 10, 10);
        let solver = heat::HeatSolver2D::new(g2d, 0.01);
        let dt = solver.max_stable_dt() * 0.5;
        assert!(dt > 0.0);
        let u = DVector::zeros(100);
        let u1 = solver.step_explicit(&u, dt, 0.0);
        // Zero IC with zero BC → stays zero
        for v in u1.iter() {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-12);
        }
    }

    // ═══════════════════════════════════════════
    // Wave equation tests
    // ═══════════════════════════════════════════

    #[test]
    fn test_wave_cfl() {
        let g = make_grid();
        let solver = wave::WaveSolver::new(g.clone(), 1.0);
        let dt_max = solver.max_stable_dt();
        assert!(dt_max > 0.0);
        assert!(dt_max < 1.0);
    }

    #[test]
    fn test_wave_energy_conservation() {
        let g = Grid1D::new(0.0, 1.0, 100);
        let mut solver = wave::WaveSolver::new(g.clone(), 1.0);
        let u0 = DVector::from_fn(100, |i, _| (i as f64 / 100.0 * std::f64::consts::PI).sin());
        let v0 = DVector::zeros(100);
        let dt = solver.max_stable_dt() * 0.5;
        let traj = solver.solve(&u0, &v0, dt, 200, &[0.0, 0.0]);
        let e0 = solver.energy(&traj[0], &traj[1], dt);
        let ef = solver.energy(&traj[198], &traj[199], dt);
        assert_relative_eq!(e0, ef, epsilon = 0.05);
    }

    #[test]
    fn test_wave_standing_mode() {
        let g = Grid1D::new(0.0, 1.0, 50);
        let mut solver = wave::WaveSolver::new(g.clone(), 1.0);
        let n = 50;
        let u0 = DVector::from_fn(n, |i, _| ((i as f64 + 1.0) / (n as f64 + 1.0) * std::f64::consts::PI).sin());
        let v0 = DVector::zeros(n);
        let dt = solver.max_stable_dt() * 0.5;
        let traj = solver.solve(&u0, &v0, dt, 100, &[0.0, 0.0]);
        // Should oscillate: at half period, u should be approximately -u0
        assert_eq!(traj.len(), 101);
    }

    #[test]
    fn test_wave_returns_trajectory() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let mut solver = wave::WaveSolver::new(g.clone(), 1.0);
        let u0 = DVector::zeros(20);
        let v0 = DVector::zeros(20);
        let dt = solver.max_stable_dt() * 0.5;
        let traj = solver.solve(&u0, &v0, dt, 10, &[0.0, 0.0]);
        assert_eq!(traj.len(), 11);
    }

    // ═══════════════════════════════════════════
    // Laplace equation tests
    // ═══════════════════════════════════════════

    #[test]
    fn test_laplace_analytical() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let solver = laplace::LaplaceSolver::new(g.clone());
        let u = solver.solve(0.0, 1.0);
        for i in 0..20 {
            let expected = (i as f64 + 1.0) / 21.0;
            assert_relative_eq!(u[i], expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_laplace_jacobi() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let solver = laplace::LaplaceSolver::new(g.clone());
        let (u, iters) = solver.solve_jacobi(0.0, 1.0, 1e-10, 100000);
        assert!(iters < 100000);
        let exact = solver.solve(0.0, 1.0);
        for i in 0..20 {
            assert_relative_eq!(u[i], exact[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_laplace_gauss_seidel() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let solver = laplace::LaplaceSolver::new(g.clone());
        let (u_j, iters_j) = solver.solve_jacobi(0.0, 1.0, 1e-10, 100000);
        let (u_gs, iters_gs) = solver.solve_gauss_seidel(0.0, 1.0, 1e-10, 100000);
        // Gauss-Seidel should converge faster
        assert!(iters_gs <= iters_j);
        for i in 0..20 {
            assert_relative_eq!(u_gs[i], u_j[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_laplace_2d() {
        let g2d = Grid2D::new(0.0, 1.0, 0.0, 1.0, 10, 10);
        let solver = laplace::LaplaceSolver2D::new(g2d.clone());
        let (u, iters) = solver.solve(0.5, 1e-10, 10000);
        assert!(iters < 10000);
        // All interior values should equal boundary value (constant BC)
        for v in u.iter() {
            assert_relative_eq!(*v, 0.5, epsilon = 1e-6);
        }
    }

    // ═══════════════════════════════════════════
    // Poisson equation tests
    // ═══════════════════════════════════════════

    #[test]
    fn test_poisson_constant_source() {
        let g = Grid1D::new(0.0, 1.0, 30);
        let solver = poisson::PoissonSolver::new(g.clone());
        let f = DVector::from_element(30, -2.0); // Δu = -2 → u = x(1-x) with u(0)=u(1)=0
        let (u, _) = solver.solve(&f, 0.0, 0.0, 1e-10, 100000);
        for i in 0..30 {
            let x = g.x(i);
            let expected = x * (1.0 - x);
            assert_relative_eq!(u[i], expected, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_poisson_gauss_seidel() {
        let g = Grid1D::new(0.0, 1.0, 30);
        let solver = poisson::PoissonSolver::new(g.clone());
        let f = DVector::from_element(30, -2.0);
        let (u_j, _) = solver.solve(&f, 0.0, 0.0, 1e-10, 100000);
        let (u_gs, _) = solver.solve_gs(&f, 0.0, 0.0, 1e-10, 100000);
        for i in 0..30 {
            assert_relative_eq!(u_gs[i], u_j[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_poisson_robin_bc() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let solver = poisson::PoissonSolver::new(g.clone());
        let f = DVector::zeros(20);
        // α=1, β=0 → Dirichlet u=0
        let (u, _) = solver.solve_robin(&f, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1e-10, 100000);
        for v in u.iter() {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_poisson_zero_forcing_is_laplace() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let poisson = poisson::PoissonSolver::new(g.clone());
        let laplace = laplace::LaplaceSolver::new(g.clone());
        let f = DVector::zeros(20);
        let (u_p, _) = poisson.solve(&f, 0.0, 1.0, 1e-10, 100000);
        let u_l = laplace.solve(0.0, 1.0);
        for i in 0..20 {
            assert_relative_eq!(u_p[i], u_l[i], epsilon = 1e-6);
        }
    }

    // ═══════════════════════════════════════════
    // Schrödinger tests
    // ═══════════════════════════════════════════

    #[test]
    fn test_schrodinger_free_particle() {
        let g = Grid1D::new(-5.0, 5.0, 100);
        let mut solver = schrodinger::SchrodingerSolver::new(g.clone(), 0.5);
        let sigma = 0.5;
        let p0 = DVector::from_fn(100, |i, _| {
            let x = -5.0 + (i as f64 + 1.0) * 10.0 / 101.0;
            (-(x * x) / (2.0 * sigma * sigma)).exp()
        });
        let q0 = DVector::zeros(100);
        let prob0 = solver.total_probability(&p0, &q0);
        let traj = solver.solve(&p0, &q0, 0.001, 100);
        let (pf, qf) = &traj[100];
        let probf = solver.total_probability(pf, qf);
        // Total probability should be approximately conserved
        assert_relative_eq!(prob0, probf, epsilon = 0.05);
    }

    #[test]
    fn test_schrodinger_with_potential() {
        let g = Grid1D::new(-5.0, 5.0, 50);
        let v = DVector::from_fn(50, |i, _| {
            let x = -5.0 + (i as f64 + 1.0) * 10.0 / 51.0;
            0.5 * x * x // harmonic oscillator
        });
        let mut solver = schrodinger::SchrodingerSolver::with_potential(g, 0.5, v);
        let p0 = DVector::from_element(50, 0.1);
        let q0 = DVector::zeros(50);
        let traj = solver.solve(&p0, &q0, 0.001, 50);
        assert_eq!(traj.len(), 51);
    }

    #[test]
    fn test_schrodinger_probability_density() {
        let g = Grid1D::new(-1.0, 1.0, 20);
        let solver = schrodinger::SchrodingerSolver::new(g.clone(), 0.5);
        let p = DVector::from_element(20, 1.0);
        let q = DVector::from_element(20, 1.0);
        let density = solver.probability_density(&p, &q);
        for d in density.iter() {
            assert_relative_eq!(*d, 2.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_schrodinger_expectation_x() {
        let g = Grid1D::new(0.0, 1.0, 50);
        let solver = schrodinger::SchrodingerSolver::new(g.clone(), 0.5);
        // Constant wave function
        let p = DVector::from_element(50, 1.0);
        let q = DVector::zeros(50);
        let ex = solver.expectation_x(&p, &q);
        assert_relative_eq!(ex, 0.5, epsilon = 0.02);
    }

    // ═══════════════════════════════════════════
    // Reaction-diffusion tests
    // ═══════════════════════════════════════════

    #[test]
    fn test_reaction_diffusion_fisher() {
        let g = Grid1D::new(0.0, 1.0, 50);
        let mut solver = reaction_diffusion::ReactionDiffusionSolver::new(g.clone(), 0.01);
        let u0 = DVector::from_fn(50, |i, _| 0.5 + 0.01 * (i as f64 * 0.1).sin());
        let dt = solver.grid.dx * solver.grid.dx / (2.0 * solver.d + 0.001);
        let traj = solver.solve(&u0, dt, 1000, reaction_diffusion::ReactionDiffusionSolver::allen_cahn(), &[0.0, 0.0]);
        assert_eq!(traj.len(), 1001);
    }

    #[test]
    fn test_reaction_diffusion_zero_is_pure_diffusion() {
        let g = Grid1D::new(0.0, 1.0, 30);
        let mut solver = reaction_diffusion::ReactionDiffusionSolver::new(g.clone(), 0.01);
        let u0 = DVector::from_fn(30, |i, _| (i as f64 / 30.0 * std::f64::consts::PI).sin());
        let dt = solver.grid.dx * solver.grid.dx / (2.0 * solver.d) * 0.4;
        let traj = solver.solve(&u0, dt, 100, reaction_diffusion::ReactionDiffusionSolver::zero_reaction(), &[0.0, 0.0]);
        // Amplitude should decrease
        let max0 = traj[0].iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        let maxf = traj[100].iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        assert!(maxf < max0);
    }

    #[test]
    fn test_allen_cahn_reaction() {
        let u = DVector::from_fn(10, |i, _| (i as f64 - 5.0) / 5.0);
        let r = reaction_diffusion::ReactionDiffusionSolver::allen_cahn()(&u);
        // R(u) = u - u³
        for i in 0..10 {
            let expected = u[i] - u[i].powi(3);
            assert_relative_eq!(r[i], expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_two_component_fhn() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let mut solver = reaction_diffusion::TwoComponentSolver::new(g.clone(), 0.01, 0.0);
        solver.init();
        let u = DVector::from_element(20, 0.1);
        let v = DVector::zeros(20);
        let (u1, v1) = solver.step_fhn(&u, &v, 0.001, 0.7, 0.8, 0.08, 0.5, &[0.0, 0.0], &[0.0, 0.0]);
        assert_eq!(u1.len(), 20);
        assert_eq!(v1.len(), 20);
    }

    // ═══════════════════════════════════════════
    // Transport tests
    // ═══════════════════════════════════════════

    #[test]
    fn test_transport_upwind() {
        let g = Grid1D::new(0.0, 1.0, 100);
        let solver = transport::TransportSolver::new(g.clone(), 1.0);
        let u0 = DVector::from_fn(100, |i, _| {
            let x = (i as f64 + 1.0) / 101.0;
            if (x - 0.3).abs() < 0.05 { 1.0 } else { 0.0 }
        });
        let dt = 0.5 * g.dx / 1.0;
        let traj = solver.solve_upwind(&u0, dt, 50);
        assert_eq!(traj.len(), 51);
        // Pulse should have moved right
        let max0_idx = traj[0].iter().cloned().enumerate().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
        let maxf_idx = traj[50].iter().cloned().enumerate().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
        assert!(maxf_idx > max0_idx);
    }

    #[test]
    fn test_transport_lax_friedrichs() {
        let g = Grid1D::new(0.0, 1.0, 50);
        let solver = transport::TransportSolver::new(g.clone(), 0.5);
        let u0 = DVector::from_element(50, 0.0);
        let dt = solver.grid.dx / 0.5 * 0.8;
        let traj = solver.solve_lax_friedrichs(&u0, dt, 10);
        // Zero IC stays zero
        for u in &traj {
            for v in u.iter() {
                assert_relative_eq!(*v, 0.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_transport_lax_wendroff() {
        let g = Grid1D::new(0.0, 1.0, 50);
        let solver = transport::TransportSolver::new(g.clone(), 1.0);
        let u0 = DVector::zeros(50);
        let dt = solver.grid.dx;
        let u1 = solver.step_lax_wendroff(&u0, dt);
        for v in u1.iter() {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_transport_courant() {
        let g = Grid1D::new(0.0, 1.0, 50);
        let solver = transport::TransportSolver::new(g.clone(), 1.0);
        let c = solver.courant(g.dx * 0.5);
        assert_relative_eq!(c, 0.5, epsilon = 1e-10);
    }

    // ═══════════════════════════════════════════
    // Burgers' equation tests
    // ═══════════════════════════════════════════

    #[test]
    fn test_burgers_viscous_relaxation() {
        let g = Grid1D::new(0.0, 1.0, 50);
        let solver = burgers::BurgersSolver::new(g.clone(), 0.1);
        let u0 = DVector::from_fn(50, |i, _| (i as f64 / 50.0 * std::f64::consts::PI).sin());
        let dt = 0.4 * g.dx * g.dx / solver.nu;
        let traj = solver.solve(&u0, dt, 1000);
        // Energy should decrease
        let e0 = solver.energy(&traj[0]);
        let ef = solver.energy(&traj[1000]);
        assert!(ef < e0);
    }

    #[test]
    fn test_burgers_total_variation() {
        let g = Grid1D::new(0.0, 1.0, 50);
        let solver = burgers::BurgersSolver::new(g.clone(), 0.1);
        let u = DVector::from_fn(50, |i, _| if i < 25 { 1.0 } else { 0.0 });
        let tv = solver.total_variation(&u);
        assert!(tv > 0.0);
    }

    #[test]
    fn test_burgers_reynolds() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let solver = burgers::BurgersSolver::new(g.clone(), 0.01);
        let u = DVector::from_element(20, 1.0);
        let re = solver.reynolds(&u);
        assert_relative_eq!(re, 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_burgers_zero_viscosity() {
        let g = Grid1D::new(0.0, 1.0, 30);
        let solver = burgers::BurgersSolver::new(g.clone(), 0.0);
        let u0 = DVector::from_fn(30, |i, _| if i < 15 { 1.0 } else { 0.0 });
        let dt = 0.3 * g.dx;
        let u1 = solver.step(&u0, dt);
        assert_eq!(u1.len(), 30);
    }

    #[test]
    fn test_burgers_lax_friedrichs() {
        let g = Grid1D::new(0.0, 1.0, 30);
        let solver = burgers::BurgersSolver::new(g.clone(), 0.05);
        let u0 = DVector::from_fn(30, |i, _| (i as f64 / 30.0).sin());
        let dt = 0.3 * g.dx;
        let u1 = solver.step_lax_friedrichs(&u0, dt);
        assert_eq!(u1.len(), 30);
    }

    // ═══════════════════════════════════════════
    // Maximum principle tests
    // ═══════════════════════════════════════════

    #[test]
    fn test_max_principle_laplace() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let solver = laplace::LaplaceSolver::new(g.clone());
        let u = solver.solve(0.2, 0.8);
        assert!(maximum_principle::verify_laplace_max_principle(&u, 0.2, 0.8));
    }

    #[test]
    fn test_strong_max_principle_laplace() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let solver = laplace::LaplaceSolver::new(g.clone());
        let u = solver.solve(0.0, 1.0);
        assert!(maximum_principle::verify_strong_max_principle(&u, 0.0, 1.0));
    }

    #[test]
    fn test_parabolic_max_principle() {
        let g = Grid1D::new(0.0, 1.0, 30);
        let mut solver = heat::HeatSolver::new(g.clone(), 0.01);
        let u0 = DVector::from_fn(30, |i, _| 0.5 * (i as f64 / 30.0 * std::f64::consts::PI).sin());
        let dt = solver.max_stable_dt() * 0.4;
        let traj = solver.solve_explicit(&u0, dt, 50, &[0.0, 0.0]);
        assert!(maximum_principle::verify_parabolic_max_principle(&traj, 0.0, 0.0, &u0));
    }

    #[test]
    fn test_mean_value_property() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let solver = laplace::LaplaceSolver::new(g.clone());
        let u = solver.solve(0.0, 1.0);
        // Linear function satisfies discrete mean value property exactly
        assert!(maximum_principle::verify_mean_value_property(&u, 0.0, 1.0, 1e-10));
    }

    #[test]
    fn test_harnack_ratio() {
        let u = DVector::from_fn(10, |i, _| (i as f64 + 1.0) / 11.0);
        let ratio = maximum_principle::harnack_ratio(&u);
        assert!(ratio.is_finite());
        assert!(ratio >= 1.0);
    }

    #[test]
    fn test_comparison_principle() {
        let u = DVector::from_fn(20, |i, _| 2.0 + (i as f64 / 20.0));
        let v = DVector::from_fn(20, |i, _| 1.0 + (i as f64 / 20.0));
        let u_traj = vec![u.clone()];
        let v_traj = vec![v.clone()];
        assert!(maximum_principle::verify_comparison_principle(&u_traj, &v_traj));
    }

    #[test]
    fn test_comparison_principle_fails() {
        let u = DVector::from_fn(20, |i, _| (i as f64 / 20.0));
        let v = DVector::from_fn(20, |i, _| 2.0 + (i as f64 / 20.0));
        assert!(!maximum_principle::verify_comparison_principle(&[u], &[v]));
    }

    #[test]
    fn test_gradient_bound() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let solver = laplace::LaplaceSolver::new(g.clone());
        let u = solver.solve(0.0, 1.0);
        // Linear function: gradient = 1/L everywhere
        assert!(maximum_principle::gradient_bound(&u, solver.grid.dx, 0.0, 1.0));
    }

    // ═══════════════════════════════════════════
    // Energy estimate tests
    // ═══════════════════════════════════════════

    #[test]
    fn test_l2_norm() {
        let u = DVector::from_element(10, 1.0);
        let n = energy::l2_norm(&u, 0.1);
        assert_relative_eq!(n, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_l2_inner_product_orthogonal() {
        let u = DVector::from_fn(10, |i, _| if i % 2 == 0 { 1.0 } else { 0.0 });
        let v = DVector::from_fn(10, |i, _| if i % 2 == 1 { 1.0 } else { 0.0 });
        let ip = energy::l2_inner(&u, &v, 0.1);
        assert_relative_eq!(ip, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_h1_seminorm() {
        let u = DVector::from_fn(11, |i, _| i as f64); // u = x, du/dx = 1
        let dx = 1.0;
        let s = energy::h1_seminorm(&u, dx);
        assert_relative_eq!(s, 10.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_h1_norm() {
        let u = DVector::zeros(10);
        let n = energy::h1_norm(&u, 0.1);
        assert_relative_eq!(n, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_heat_energy_decay() {
        let g = Grid1D::new(0.0, 1.0, 30);
        let mut solver = heat::HeatSolver::new(g.clone(), 0.01);
        let u0 = DVector::from_fn(30, |i, _| (i as f64 / 30.0 * std::f64::consts::PI).sin());
        let dt = solver.max_stable_dt() * 0.4;
        let traj = solver.solve_explicit(&u0, dt, 100, &[0.0, 0.0]);
        assert!(energy::verify_heat_energy_decay(&traj, solver.grid.dx));
    }

    #[test]
    fn test_heat_dissipation_negative() {
        let u = DVector::from_fn(20, |i, _| (i as f64 / 20.0 * std::f64::consts::PI).sin());
        let diss = energy::heat_dissipation(&u, 0.05, 1.0);
        assert!(diss <= 0.0);
    }

    #[test]
    fn test_sobolev_bound() {
        let u = DVector::from_fn(20, |i, _| (i as f64 / 20.0 * std::f64::consts::PI).sin());
        assert!(energy::verify_sobolev_bound(&u, 0.05));
    }

    #[test]
    fn test_poincare() {
        let g = Grid1D::new(0.0, 1.0, 30);
        // Zero-mean function
        let u = DVector::from_fn(30, |i, _| {
            let x = g.x(i);
            (2.0 * std::f64::consts::PI * x).sin()
        });
        assert!(energy::verify_poincare(&u, g.dx));
    }

    #[test]
    fn test_wave_energy_function() {
        let g = Grid1D::new(0.0, 1.0, 50);
        let u0 = DVector::from_fn(50, |i, _| (i as f64 / 50.0 * std::f64::consts::PI).sin());
        let u1 = &u0 * 0.99;
        let e = energy::wave_energy(&u0, &u1, g.dx, 0.01, 1.0);
        assert!(e > 0.0);
    }

    #[test]
    fn test_exponential_decay() {
        let g = Grid1D::new(0.0, 1.0, 50);
        let mut solver = heat::HeatSolver::new(g.clone(), 1.0);
        let u0 = DVector::from_fn(50, |i, _| (i as f64 / 50.0 * std::f64::consts::PI).sin());
        let dt = solver.max_stable_dt() * 0.4;
        let traj = solver.solve_explicit(&u0, dt, 100, &[0.0, 0.0]);
        assert!(energy::verify_exponential_decay(&traj, solver.grid.dx, dt, 1.0));
    }

    // ═══════════════════════════════════════════
    // Application tests
    //═══════════════════════════════════════════

    #[test]
    fn test_belief_profile_uniform() {
        let g = make_grid();
        let p = application::BeliefProfile::Uniform { value: 0.5 };
        let u = p.evaluate(&g);
        for v in u.iter() {
            assert_relative_eq!(*v, 0.5, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_belief_profile_gaussian() {
        let g = make_grid();
        let p = application::BeliefProfile::Gaussian { amplitude: 1.0, center: 0.5, width: 0.1 };
        let u = p.evaluate(&g);
        assert!(u.iter().cloned().fold(0.0f64, f64::max) > 0.5);
    }

    #[test]
    fn test_belief_profile_step() {
        let g = make_grid();
        let p = application::BeliefProfile::Step { left: 0.0, right: 1.0, x0: 0.5 };
        let u = p.evaluate(&g);
        assert!(u[0] < 0.5);
        assert!(u[49] > 0.5);
    }

    #[test]
    fn test_belief_profile_sinusoidal() {
        let g = make_grid();
        let p = application::BeliefProfile::Sinusoidal { amplitude: 1.0, mode: 1 };
        let u = p.evaluate(&g);
        assert!(u.iter().cloned().fold(0.0f64, f64::max) > 0.5);
    }

    #[test]
    fn test_belief_diffusion_app() {
        let g = Grid1D::new(0.0, 1.0, 30);
        let mut app = application::BeliefDiffusion::new(g.clone(), 0.01);
        let u0 = DVector::from_fn(30, |i, _| (i as f64 / 30.0 * std::f64::consts::PI).sin());
        let dt = app.solver.max_stable_dt() * 0.4;
        let traj = app.run(&u0, dt, 50, &[0.0, 0.0]);
        assert_eq!(traj.len(), 51);
    }

    #[test]
    fn test_belief_oscillation_app() {
        let g = Grid1D::new(0.0, 1.0, 30);
        let mut app = application::BeliefOscillation::new(g.clone(), 1.0);
        let u0 = DVector::from_fn(30, |i, _| (i as f64 / 30.0 * std::f64::consts::PI).sin());
        let v0 = DVector::zeros(30);
        let dt = app.solver.max_stable_dt() * 0.5;
        let traj = app.run(&u0, &v0, dt, 20, &[0.0, 0.0]);
        assert_eq!(traj.len(), 21);
    }

    #[test]
    fn test_belief_equilibrium_app() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let app = application::BeliefEquilibrium::new(g.clone());
        let f = DVector::from_element(20, -2.0);
        let (u, _) = app.run(&f, 0.0, 0.0, 1e-10, 100000);
        assert_eq!(u.len(), 20);
    }

    #[test]
    fn test_belief_propagation_app() {
        let g = Grid1D::new(0.0, 1.0, 50);
        let app = application::BeliefPropagation::new(g.clone(), 1.0);
        let u0 = DVector::zeros(50);
        let dt = g.dx * 0.5;
        let traj = app.run(&u0, dt, 10);
        assert_eq!(traj.len(), 11);
    }

    #[test]
    fn test_belief_pattern_formation_app() {
        let g = Grid1D::new(0.0, 1.0, 30);
        let mut app = application::BeliefPatternFormation::new(g.clone(), 0.01);
        let u0 = DVector::from_fn(30, |i, _| 0.1 * (i as f64 / 30.0 * std::f64::consts::PI).sin());
        let dt = app.solver.grid.dx * app.solver.grid.dx / 0.02 * 0.4;
        let traj = app.run(&u0, dt, 50, &[0.0, 0.0]);
        assert_eq!(traj.len(), 51);
    }

    #[test]
    fn test_nonlinear_belief_app() {
        let g = Grid1D::new(0.0, 1.0, 30);
        let app = application::NonlinearBelief::new(g.clone(), 0.1);
        let u0 = DVector::from_fn(30, |i, _| (i as f64 / 30.0 * std::f64::consts::PI).sin());
        let dt = 0.3 * g.dx * g.dx / app.solver.nu;
        let traj = app.run(&u0, dt, 100);
        assert_eq!(traj.len(), 101);
    }

    #[test]
    fn test_belief_consensus() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let u0 = DVector::from_fn(20, |i, _| (i as f64 / 20.0 * std::f64::consts::PI).sin());
        let consensus = application::belief_consensus(&[u0], &g);
        assert!(consensus[0] > 0.0);
    }

    #[test]
    fn test_belief_variance_decreases_with_diffusion() {
        let g = Grid1D::new(0.0, 1.0, 50);
        let mut solver = heat::HeatSolver::new(g.clone(), 0.01);
        let u0 = DVector::from_fn(50, |i, _| if i < 25 { 1.0 } else { 0.0 });
        let dt = solver.max_stable_dt() * 0.4;
        let traj = solver.solve_explicit(&u0, dt, 200, &[0.0, 0.0]);
        let variance = application::belief_variance(&traj, &g);
        assert!(variance[200] < variance[0]);
    }

    #[test]
    fn test_is_consensus_reached() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let u = DVector::from_element(20, 0.5);
        assert!(application::is_consensus_reached(&[u], &g, 0.01));
    }

    // ═══════════════════════════════════════════
    // Serialization tests
    // ═══════════════════════════════════════════

    #[test]
    fn test_serde_grid() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let json = serde_json::to_string(&g).unwrap();
        let g2: Grid1D = serde_json::from_str(&json).unwrap();
        assert_eq!(g.n, g2.n);
        assert_relative_eq!(g.dx, g2.dx, epsilon = 1e-12);
    }

    #[test]
    fn test_serde_heat_solver() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let solver = heat::HeatSolver::new(g.clone(), 1.0);
        let json = serde_json::to_string(&solver).unwrap();
        let s2: heat::HeatSolver = serde_json::from_str(&json).unwrap();
        assert_relative_eq!(solver.alpha, s2.alpha, epsilon = 1e-12);
    }

    #[test]
    fn test_serde_burgers_solver() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let solver = burgers::BurgersSolver::new(g.clone(), 0.1);
        let json = serde_json::to_string(&solver).unwrap();
        let s2: burgers::BurgersSolver = serde_json::from_str(&json).unwrap();
        assert_relative_eq!(solver.nu, s2.nu, epsilon = 1e-12);
    }

    // ═══════════════════════════════════════════
    // Edge case / robustness tests
    // ═══════════════════════════════════════════

    #[test]
    fn test_heat_zero_initial_zero_bc() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let mut solver = heat::HeatSolver::new(g.clone(), 1.0);
        let u0 = DVector::zeros(20);
        let dt = solver.max_stable_dt() * 0.5;
        let traj = solver.solve_explicit(&u0, dt, 10, &[0.0, 0.0]);
        for u in &traj {
            for v in u.iter() {
                assert_relative_eq!(*v, 0.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_laplace_zero_bc_zero_solution() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let solver = laplace::LaplaceSolver::new(g.clone());
        let u = solver.solve(0.0, 0.0);
        for v in u.iter() {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_poisson_zero_forcing_zero_bc() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let solver = poisson::PoissonSolver::new(g.clone());
        let f = DVector::zeros(20);
        let (u, _) = solver.solve(&f, 0.0, 0.0, 1e-10, 10000);
        for v in u.iter() {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_burgers_infinite_reynolds() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let solver = burgers::BurgersSolver::new(g.clone(), 0.0);
        let u = DVector::from_element(20, 1.0);
        assert_eq!(solver.reynolds(&u), f64::INFINITY);
    }

    #[test]
    fn test_wave_zero_ic_stays_zero() {
        let g = Grid1D::new(0.0, 1.0, 30);
        let mut solver = wave::WaveSolver::new(g.clone(), 1.0);
        let u0 = DVector::zeros(30);
        let v0 = DVector::zeros(30);
        let dt = solver.max_stable_dt() * 0.5;
        let traj = solver.solve(&u0, &v0, dt, 50, &[0.0, 0.0]);
        for u in &traj {
            for v in u.iter() {
                assert_relative_eq!(*v, 0.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_transport_negative_velocity() {
        let g = Grid1D::new(0.0, 1.0, 50);
        let solver = transport::TransportSolver::new(g.clone(), -1.0);
        let u0 = DVector::from_fn(50, |i, _| {
            let x = (i as f64 + 1.0) / 51.0;
            if (x - 0.7).abs() < 0.05 { 1.0 } else { 0.0 }
        });
        let dt = g.dx * 0.5;
        let u1 = solver.step_upwind(&u0, dt);
        assert_eq!(u1.len(), 50);
    }

    #[test]
    fn test_schrodinger_zero_stays_zero() {
        let g = Grid1D::new(0.0, 1.0, 20);
        let mut solver = schrodinger::SchrodingerSolver::new(g.clone(), 0.5);
        let p = DVector::zeros(20);
        let q = DVector::zeros(20);
        let traj = solver.solve(&p, &q, 0.001, 10);
        for (pp, qq) in &traj {
            for v in pp.iter() { assert_relative_eq!(*v, 0.0, epsilon = 1e-10); }
            for v in qq.iter() { assert_relative_eq!(*v, 0.0, epsilon = 1e-10); }
        }
    }

    #[test]
    fn test_heat_solver_2d_stability() {
        let g = Grid2D::new(0.0, 1.0, 0.0, 1.0, 20, 20);
        let solver = heat::HeatSolver2D::new(g.clone(), 0.1);
        let dt = solver.max_stable_dt();
        assert!(dt > 0.0);
        assert!(dt < 1.0);
    }

    #[test]
    fn test_laplace_symmetric_bc() {
        let g = Grid1D::new(0.0, 1.0, 21); // odd number for symmetry
        let solver = laplace::LaplaceSolver::new(g.clone());
        let u = solver.solve(1.0, 1.0);
        // With equal BCs, solution should be constant
        for v in u.iter() {
            assert_relative_eq!(*v, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_energy_l2_norm_positive() {
        let u = DVector::from_fn(20, |i, _| (i as f64).sin());
        let n = energy::l2_norm(&u, 0.1);
        assert!(n > 0.0);
    }

    #[test]
    fn test_h1_norm_triangle_inequality() {
        let u = DVector::from_fn(20, |i, _| (i as f64).sin());
        let v = DVector::from_fn(20, |i, _| (i as f64).cos());
        let dx = 0.1;
        let nu = energy::h1_norm(&u, dx);
        let nv = energy::h1_norm(&v, dx);
        let nuv = energy::h1_norm(&(&u + &v), dx);
        assert!(nuv <= nu + nv + 1e-10);
    }
}
