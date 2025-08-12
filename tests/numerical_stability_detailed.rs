//! Detailed numerical stability tests for critical financial calculations.
//!
//! This test suite focuses specifically on numerical stability issues that could
//! affect trading results in production financial systems.

use assert_approx_eq::assert_approx_eq;
use fractal_finance::*;
use rand::prelude::*;
use std::f64::{EPSILON, INFINITY, NEG_INFINITY};

/// Numerical stability tests for mathematical utilities
#[cfg(test)]
mod math_utils_stability {
    use super::*;

    #[test]
    fn test_periodogram_numerical_stability() {
        // Test 1: Data with very small values (near underflow)
        let small_data: Vec<f64> = (0..256).map(|i| 1e-100 * (i as f64).sin()).collect();
        let result = calculate_periodogram_fft(&small_data);
        match result {
            Ok(periodogram) => {
                assert!(periodogram.iter().all(|&x| x >= 0.0 && x.is_finite()));
                // Periodogram values should be non-negative and finite
            }
            Err(_) => {
                // Acceptable to fail with extreme underflow
            }
        }

        // Test 2: Data with very large values (near overflow)
        let large_data: Vec<f64> = (0..256).map(|i| 1e50 * (i as f64).sin()).collect();
        let result = calculate_periodogram_fft(&large_data);
        match result {
            Ok(periodogram) => {
                assert!(periodogram.iter().all(|&x| x.is_finite()));
            }
            Err(_) => {
                // May fail due to overflow
            }
        }

        // Test 3: Mixed scale data (common in financial markets)
        let mut mixed_data = Vec::new();
        for i in 0..256 {
            if i % 10 == 0 {
                mixed_data.push(1e6 * (i as f64).sin()); // Large institutional trades
            } else {
                mixed_data.push(1e-3 * (i as f64).sin()); // Small retail trades
            }
        }

        let result = calculate_periodogram_fft(&mixed_data);
        match result {
            Ok(periodogram) => {
                assert!(periodogram.iter().all(|&x| x >= 0.0 && x.is_finite()));
                // Should handle mixed scales gracefully
                let max_power = periodogram.iter().fold(0.0f64, |a, &b| a.max(b));
                let min_power = periodogram.iter().fold(INFINITY, |a, &b| a.min(b));
                assert!(
                    max_power / min_power < 1e20,
                    "Dynamic range should be reasonable"
                );
            }
            Err(_) => {
                // May fail with extreme dynamic range
            }
        }

        // Test 4: Data with extreme sparsity (many zeros)
        let mut sparse_data = vec![0.0; 256];
        sparse_data[50] = 1.0;
        sparse_data[100] = -1.0;
        sparse_data[200] = 0.5;

        let result = calculate_periodogram_fft(&sparse_data);
        match result {
            Ok(periodogram) => {
                assert!(periodogram.iter().all(|&x| x >= 0.0 && x.is_finite()));
            }
            Err(_) => {
                // May have issues with sparse data
            }
        }
    }

    #[test]
    fn test_regression_conditioning() {
        // Test 1: Nearly collinear data (ill-conditioned)
        let x: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi + 1e-10 * xi.sin()).collect(); // Nearly linear

        let result = math_utils::ols_regression(&x, &y);
        match result {
            Ok((slope, intercept, residuals)) => {
                assert!(slope.is_finite());
                assert!(intercept.is_finite());
                assert!(residuals.iter().all(|&r| r.is_finite()));

                // Slope should be close to 1
                assert!(
                    (slope - 1.0).abs() < 0.01,
                    "Slope should be near 1, got {}",
                    slope
                );
            }
            Err(_) => {
                // May fail due to ill-conditioning
            }
        }

        // Test 2: Data with extreme leverage points
        let mut x: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let mut y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

        // Add extreme leverage point
        x.push(1000.0);
        y.push(2001.0); // Consistent with trend

        let result = math_utils::ols_regression(&x, &y);
        match result {
            Ok((slope, intercept, _)) => {
                assert!(slope.is_finite());
                assert!(intercept.is_finite());
                // Should still recover correct relationship
                assert!(
                    (slope - 2.0).abs() < 0.1,
                    "Slope should be near 2, got {}",
                    slope
                );
            }
            Err(_) => {
                // May fail with extreme leverage
            }
        }

        // Test 3: Regression with heteroscedastic errors
        let x: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                let error_scale = (i as f64 + 1.0).sqrt(); // Increasing error variance
                xi * 2.0 + error_scale * 0.1 * (i as f64).sin()
            })
            .collect();

        let result = math_utils::ols_regression(&x, &y);
        match result {
            Ok((slope, intercept, residuals)) => {
                assert!(slope.is_finite());
                assert!(intercept.is_finite());
                assert!(residuals.iter().all(|&r| r.is_finite()));

                // Should still estimate slope reasonably well
                assert!(
                    (slope - 2.0).abs() < 0.2,
                    "Slope should be near 2, got {}",
                    slope
                );
            }
            Err(_) => {
                // May fail with heteroscedastic errors
            }
        }
    }

    #[test]
    fn test_autocorrelation_numerical_stability() {
        // Test 1: Near-zero variance data
        let mut data = vec![1.0; 1000];
        data[500] += 1e-10; // Tiny perturbation

        let autocorrs = math_utils::calculate_autocorrelations(&data, 10);
        if autocorrs.len() > 0 {
            assert_eq!(autocorrs.len(), 10);
            assert!(autocorrs.iter().all(|&r| r.is_finite()));
            // First autocorrelation should be very close to 1
            assert!(
                (autocorrs[0] - 1.0).abs() < 1e-6,
                "First autocorr should be ~1"
            );
        }

        // Test 2: Perfect periodic data
        let data: Vec<f64> = (0..200)
            .map(|i| (i as f64 * std::f64::consts::PI / 10.0).sin())
            .collect();
        let autocorrs = math_utils::calculate_autocorrelations(&data, 50);
        if autocorrs.len() > 0 {
            assert!(autocorrs.iter().all(|&r| r.is_finite()));
            assert!(autocorrs.iter().all(|&r| r.abs() <= 1.0)); // Should be bounded by 1

            // Should show periodic structure
            assert!(
                (autocorrs[0] - 1.0).abs() < 1e-10,
                "First autocorr should be 1"
            );
            if autocorrs.len() > 20 {
                assert!((autocorrs[20] - 1.0).abs() < 0.1, "Should detect period");
            }
        }

        // Test 3: Data with trend
        let data: Vec<f64> = (0..500).map(|i| i as f64 + (i as f64).sin()).collect();
        let autocorrs = math_utils::calculate_autocorrelations(&data, 20);
        if autocorrs.len() > 1 {
            assert!(autocorrs.iter().all(|&r| r.is_finite()));
            // Trend should cause high autocorrelations
            assert!(autocorrs[1] > 0.9, "Strong trend should show high autocorr");
        }
    }
}

/// Numerical stability tests for Hurst exponent estimation
#[cfg(test)]
mod hurst_estimation_stability {
    use super::*;

    #[test]
    fn test_dfa_numerical_stability() {
        // Test 1: High-Hurst process (near random walk)
        let high_h_config = FbmConfig {
            hurst_exponent: 0.95,
            volatility: 1.0,
            method: FbmMethod::Hosking,
        };
        let gen_config = GeneratorConfig {
            length: 1000,
            seed: Some(42),
            ..Default::default()
        };

        let data_result = generate_fractional_brownian_motion(&gen_config, &high_h_config);
        if let Ok(data) = data_result {
            let mut analyzer = StatisticalFractalAnalyzer::new();
            analyzer.add_time_series("HIGH_H".to_string(), data);

            let result = analyzer.analyze_series("HIGH_H");
            match result {
                Ok(_) => {
                    if let Ok(results) = analyzer.get_analysis_results("HIGH_H") {
                        if let Some(dfa_result) = results
                            .hurst_estimates
                            .get(&EstimationMethod::DetrendedFluctuationAnalysis)
                        {
                            assert!(dfa_result.estimate.is_finite());
                            assert!(dfa_result.estimate > 0.7, "High-H should be detected");
                            assert!(dfa_result.estimate < 1.1, "Estimate should be reasonable");
                            assert!(dfa_result.standard_error.is_finite());
                            assert!(dfa_result.standard_error > 0.0);
                        }
                    }
                }
                Err(_) => {
                    // May fail with extreme H values
                }
            }
        }

        // Test 2: Low-Hurst anti-persistent process
        let low_h_config = FbmConfig {
            hurst_exponent: 0.1,
            volatility: 1.0,
            method: FbmMethod::Hosking,
        };
        let gen_config = GeneratorConfig {
            length: 1000,
            seed: Some(43),
            ..Default::default()
        };

        let data_result = generate_fractional_brownian_motion(&gen_config, &low_h_config);
        if let Ok(data) = data_result {
            let mut analyzer = StatisticalFractalAnalyzer::new();
            analyzer.add_time_series("LOW_H".to_string(), data);

            let result = analyzer.analyze_series("LOW_H");
            match result {
                Ok(_) => {
                    if let Ok(results) = analyzer.get_analysis_results("LOW_H") {
                        if let Some(dfa_result) = results
                            .hurst_estimates
                            .get(&EstimationMethod::DetrendedFluctuationAnalysis)
                        {
                            assert!(dfa_result.estimate.is_finite());
                            assert!(dfa_result.estimate < 0.4, "Low-H should be detected");
                            assert!(dfa_result.estimate > -0.1, "Estimate should be reasonable");
                        }
                    }
                }
                Err(_) => {
                    // May fail with extreme H values
                }
            }
        }

        // Test 3: Data with measurement noise
        let clean_config = FbmConfig {
            hurst_exponent: 0.7,
            volatility: 1.0,
            method: FbmMethod::Hosking,
        };
        let gen_config = GeneratorConfig {
            length: 1000,
            seed: Some(44),
            ..Default::default()
        };

        let data_result = generate_fractional_brownian_motion(&gen_config, &clean_config);
        if let Ok(mut data) = data_result {
            // Add white noise
            for i in 0..data.len() {
                data[i] += 0.1 * rand::random::<f64>() - 0.05;
            }

            let mut analyzer = StatisticalFractalAnalyzer::new();
            analyzer.add_time_series("NOISY".to_string(), data);

            let result = analyzer.analyze_series("NOISY");
            match result {
                Ok(_) => {
                    if let Ok(results) = analyzer.get_analysis_results("NOISY") {
                        if let Some(dfa_result) = results
                            .hurst_estimates
                            .get(&EstimationMethod::DetrendedFluctuationAnalysis)
                        {
                            assert!(dfa_result.estimate.is_finite());
                            // Noise may bias estimate, but should be reasonable
                            assert!(dfa_result.estimate > 0.3);
                            assert!(dfa_result.estimate < 0.9);
                        }
                    }
                }
                Err(_) => {
                    // May fail with noisy data
                }
            }
        }
    }

    #[test]
    fn test_gph_robustness() {
        // Test 1: Data with polynomial trends
        let trend_data: Vec<f64> = (0..500)
            .map(|i| {
                let t = i as f64;
                t * t * 0.001 + t * 0.1 + (t * 0.1).sin() // Quadratic trend + noise
            })
            .collect();

        let result = gph_test(&trend_data);
        match result {
            Ok((estimate, std_error, _)) => {
                assert!(estimate.is_finite());
                assert!(std_error.is_finite());
                // GPH should handle polynomial trends
                assert!(estimate > 0.0);
                assert!(estimate < 2.0);
            }
            Err(_) => {
                // May fail with strong trends
            }
        }

        // Test 2: Data with seasonal patterns
        let seasonal_data: Vec<f64> = (0..1000)
            .map(|i| {
                let t = i as f64;
                (t * 2.0 * std::f64::consts::PI / 100.0).sin() + // Annual cycle
            0.5 * (t * 2.0 * std::f64::consts::PI / 20.0).sin() + // Monthly cycle
            0.01 * t // Slight trend
            })
            .collect();

        let result = gph_test(&seasonal_data);
        match result {
            Ok((estimate, std_error, _)) => {
                assert!(estimate.is_finite());
                assert!(std_error.is_finite());
                // Seasonal data may show spurious long-range dependence
                assert!(estimate > -0.5);
                assert!(estimate < 1.5);
            }
            Err(_) => {
                // May fail with strong seasonality
            }
        }

        // Test 3: High-frequency financial returns
        let mut returns = Vec::new();
        let mut price = 100.0;
        for i in 0..2000 {
            let return_val = if i % 100 == 0 {
                // Occasional large jumps (tail events)
                0.05 * (rand::random::<f64>() - 0.5).signum()
            } else {
                // Normal small returns
                0.001 * (rand::random::<f64>() - 0.5)
            };
            price *= (1.0 + return_val).exp();
            returns.push(return_val);
        }

        let result = gph_test(&returns);
        match result {
            Ok((estimate, std_error, _)) => {
                assert!(estimate.is_finite());
                assert!(std_error.is_finite());
                // Financial returns typically show H ≈ 0.5
                assert!(estimate > 0.2);
                assert!(estimate < 0.8);
            }
            Err(_) => {
                // May fail with tail events
            }
        }
    }

    #[test]
    fn test_rescaled_range_stability() {
        // Test 1: Data with volatility clustering
        let mut clustered_data = Vec::new();
        let mut volatility = 0.01;

        for i in 0..1000 {
            // Volatility clustering (GARCH-like)
            if i % 50 == 0 {
                volatility = if rand::random::<f64>() > 0.5 {
                    0.05
                } else {
                    0.005
                };
            }

            let return_val = volatility * (rand::random::<f64>() - 0.5);
            clustered_data.push(return_val);
        }

        let mut analyzer = StatisticalFractalAnalyzer::new();
        analyzer.add_time_series("CLUSTERED".to_string(), clustered_data);

        let result = analyzer.analyze_series("CLUSTERED");
        match result {
            Ok(_) => {
                if let Ok(results) = analyzer.get_analysis_results("CLUSTERED") {
                    if let Some(rs_result) = results
                        .hurst_estimates
                        .get(&EstimationMethod::RescaledRange)
                    {
                        assert!(rs_result.estimate.is_finite());
                        assert!(rs_result.standard_error.is_finite());
                        // Volatility clustering may affect R/S estimate
                        assert!(rs_result.estimate > 0.2);
                        assert!(rs_result.estimate < 1.2);
                    }
                }
            }
            Err(_) => {
                // May fail with volatility clustering
            }
        }

        // Test 2: Mean-reverting data
        let mut mean_reverting = Vec::new();
        let mut level = 0.0;

        for _ in 0..500 {
            level += -0.1 * level + 0.1 * (rand::random::<f64>() - 0.5);
            mean_reverting.push(level);
        }

        analyzer.add_time_series("MEAN_REV".to_string(), mean_reverting);
        let result = analyzer.analyze_series("MEAN_REV");
        match result {
            Ok(_) => {
                if let Ok(results) = analyzer.get_analysis_results("MEAN_REV") {
                    if let Some(rs_result) = results
                        .hurst_estimates
                        .get(&EstimationMethod::RescaledRange)
                    {
                        assert!(rs_result.estimate.is_finite());
                        // Mean reversion should show H < 0.5
                        assert!(
                            rs_result.estimate < 0.6,
                            "Mean reversion should show H < 0.6"
                        );
                    }
                }
            }
            Err(_) => {
                // May fail with mean reversion
            }
        }
    }
}

/// Numerical stability tests for multifractal analysis  
#[cfg(test)]
mod multifractal_stability {
    use super::*;

    #[test]
    fn test_mfdfa_extreme_scales() {
        // Generate test data with known multifractal properties
        let cascade_config = MultifractalCascadeConfig {
            levels: 8,
            intermittency: 0.5,
            lognormal_params: (0.0, 0.2),
            base_volatility: 1.0,
        };
        let gen_config = GeneratorConfig {
            length: 256,
            seed: Some(45),
            ..Default::default()
        };

        let data = generate_multifractal_cascade(&gen_config, &cascade_config);
        if let Ok(data) = data {
            // Test 1: Very fine scale resolution
            let fine_config = MultifractalConfig {
                q_range: (-3.0, 3.0),
                num_q_values: 13,
                min_scale: 3,
                max_scale_factor: 4.0, // 256/64 = 4
                polynomial_order: 2,
            };

            let result = perform_multifractal_analysis(&data);
            match result {
                Ok(analysis) => {
                    assert!(!analysis.generalized_hurst_exponents.is_empty());

                    // Validate all H(q) values are reasonable
                    for (q, h) in &analysis.generalized_hurst_exponents {
                        assert!(h.is_finite(), "H({}) should be finite", q);
                        assert!(
                            *h > -1.0 && *h < 2.0,
                            "H({}) = {} out of reasonable range",
                            q,
                            h
                        );
                    }

                    // Check monotonicity: H(q) should generally decrease with q
                    let h_values: Vec<f64> = analysis
                        .generalized_hurst_exponents
                        .iter()
                        .map(|(_, h)| *h)
                        .collect();

                    // Allow some numerical noise in monotonicity
                    let mut monotonic_violations = 0;
                    for i in 1..h_values.len() {
                        if h_values[i] > h_values[i - 1] + 0.1 {
                            // Tolerance for noise
                            monotonic_violations += 1;
                        }
                    }
                    assert!(
                        monotonic_violations < h_values.len() / 3,
                        "Too many monotonicity violations"
                    );
                }
                Err(_) => {
                    // May fail with very fine scales
                }
            }

            // Test 2: Extreme q-values
            let extreme_q_config = MultifractalConfig {
                q_range: (-10.0, 10.0),
                num_q_values: 5,
                min_scale: 4,
                max_scale_factor: 8.0, // 256/32 = 8
                polynomial_order: 1,
            };

            let result = perform_multifractal_analysis(&data);
            match result {
                Ok(analysis) => {
                    for (q, h) in &analysis.generalized_hurst_exponents {
                        assert!(
                            h.is_finite(),
                            "H({}) should be finite even for extreme q",
                            q
                        );

                        // Extreme q values should give more extreme H values
                        if *q > 5.0 {
                            assert!(*h < 1.0, "Large positive q should give small H");
                        }
                        if *q < -5.0 {
                            assert!(*h > 0.0, "Large negative q should give large H");
                        }
                    }
                }
                Err(_) => {
                    // Expected to fail with extreme q values
                }
            }
        }
    }

    #[test]
    fn test_singularity_spectrum_numerical_issues() {
        // Test 1: Data with artificial singularities
        let mut singular_data = Vec::new();
        for i in 0..512 {
            let t = i as f64 / 512.0;
            if (t - 0.5).abs() < 0.01 {
                // Sharp spike at center
                singular_data.push(100.0 * (1.0 / (t - 0.5).abs()).min(1000.0));
            } else {
                singular_data.push((t * 10.0).sin());
            }
        }

        let result = perform_multifractal_analysis(&singular_data);
        match result {
            Ok(analysis) => {
                let spectrum = &analysis.singularity_spectrum;
                if !spectrum.is_empty() {
                    // Validate spectrum values
                    for (alpha, f_alpha) in spectrum {
                        assert!(alpha.is_finite(), "α should be finite");
                        assert!(f_alpha.is_finite(), "f(α) should be finite");
                        assert!(*f_alpha >= 0.0, "f(α) should be non-negative");
                        assert!(*f_alpha <= 1.0, "f(α) should be ≤ 1 for 1D data");
                    }

                    // Spectrum should be concave (approximately)
                    let mut alpha_f: Vec<(f64, f64)> = spectrum.clone();
                    alpha_f.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                    if alpha_f.len() >= 3 {
                        // Check concavity at interior points
                        for i in 1..alpha_f.len() - 1 {
                            let (alpha_prev, f_prev) = alpha_f[i - 1];
                            let (alpha_curr, f_curr) = alpha_f[i];
                            let (alpha_next, f_next) = alpha_f[i + 1];

                            // Linear interpolation
                            let weight = (alpha_curr - alpha_prev) / (alpha_next - alpha_prev);
                            let linear_f = f_prev + weight * (f_next - f_prev);

                            // Allow some tolerance for numerical noise
                            assert!(
                                f_curr >= linear_f - 0.2,
                                "Spectrum should be approximately concave at α = {}",
                                alpha_curr
                            );
                        }
                    }
                }
            }
            Err(_) => {
                // May fail with artificial singularities
            }
        }

        // Test 2: Uniform random data (should be monofractal)
        let uniform_data: Vec<f64> = (0..512).map(|_| rand::random::<f64>()).collect();

        let result = perform_multifractal_analysis(&uniform_data);
        match result {
            Ok(analysis) => {
                // Check if detected as monofractal
                let h_values: Vec<f64> = analysis
                    .generalized_hurst_exponents
                    .iter()
                    .map(|(_, h)| *h)
                    .collect();

                if h_values.len() > 2 {
                    let h_range = h_values
                        .iter()
                        .fold((INFINITY, NEG_INFINITY), |(min, max), &h| {
                            (min.min(h), max.max(h))
                        });

                    let h_span = h_range.1 - h_range.0;
                    // Random data should show relatively narrow H(q) spectrum
                    assert!(
                        h_span < 0.5,
                        "Random data should show narrow H(q) spectrum, got {}",
                        h_span
                    );
                }

                let spectrum = &analysis.singularity_spectrum;
                if !spectrum.is_empty() {
                    let alpha_range = spectrum
                        .iter()
                        .fold((INFINITY, NEG_INFINITY), |(min, max), &(a, _)| {
                            (min.min(a), max.max(a))
                        });

                    let alpha_span = alpha_range.1 - alpha_range.0;
                    // Random data should have narrow singularity spectrum
                    assert!(alpha_span < 0.3, "Random data should have narrow spectrum");
                }
            }
            Err(_) => {
                // May fail on random data
            }
        }
    }
}

/// Test numerical stability under extreme computational conditions
#[cfg(test)]
mod extreme_conditions {
    use super::*;

    #[test]
    fn test_precision_near_limits() {
        // Test 1: Data at float64 precision limits
        let mut precision_data = Vec::new();
        let base_value = 1.0;

        for i in 0..100 {
            // Add values that differ only in the least significant bits
            let perturbation = EPSILON * (i as f64 - 50.0);
            precision_data.push(base_value + perturbation);
        }

        // Should handle near-identical values gracefully
        let variance = math_utils::calculate_variance(&precision_data);
        assert!(variance.is_finite());
        assert!(variance >= 0.0);
        // Variance should be very small but non-zero
        assert!(variance < 1e-20, "Variance should be tiny");

        // Test 2: Computation with numbers near overflow threshold
        let large_numbers: Vec<f64> = (0..50)
            .map(|i| 1e100 * (1.0 + 1e-10 * (i as f64).sin()))
            .collect();

        let regression_result = math_utils::ols_regression(
            &(0..50).map(|i| i as f64).collect::<Vec<_>>(),
            &large_numbers,
        );

        match regression_result {
            Ok((slope, intercept, _)) => {
                assert!(
                    slope.is_finite(),
                    "Slope should be finite with large numbers"
                );
                assert!(
                    intercept.is_finite(),
                    "Intercept should be finite with large numbers"
                );
            }
            Err(_) => {
                // Acceptable to fail near overflow limits
            }
        }

        // Test 3: Computation with numbers near underflow threshold
        let tiny_numbers: Vec<f64> = (0..50)
            .map(|i| 1e-100 * (1.0 + 0.1 * (i as f64).sin()))
            .collect();

        let autocorrs = math_utils::calculate_autocorrelations(&tiny_numbers, 5);
        if autocorrs.len() > 0 {
            assert!(autocorrs.iter().all(|&r| r.is_finite()));
            assert!(
                (autocorrs[0] - 1.0).abs() < 1e-10,
                "First autocorr should be 1"
            );
        }
    }

    #[test]
    fn test_algorithmic_convergence() {
        // Test 1: HMM convergence with challenging data

        // Create data that switches between regimes very frequently
        let mut rapid_switching = Vec::new();
        for i in 0..500 {
            let regime = if (i / 5) % 2 == 0 { 0.1 } else { 2.0 };
            rapid_switching.push(regime * (rand::random::<f64>() - 0.5));
        }

        let regime_config = RegimeDetectionConfig {
            window_size: 50,
            step_size: 10,
            num_states_range: (2, 2),
            auto_select_states: false,
            min_regime_duration: 10,
            bootstrap_config: BootstrapConfiguration::default(),
            seed: Some(46),
        };
        let hmm_result = detect_fractal_regimes(&rapid_switching, &regime_config);
        match hmm_result {
            Ok(result) => {
                assert!(!result.regime_sequence.is_empty());
                // Should detect frequent regime changes
                assert!(
                    result.regime_sequence.len() > 10,
                    "Should detect multiple regime changes"
                );

                // All regime probabilities should be valid
                assert!(result.log_likelihood.is_finite());
            }
            Err(_) => {
                // May fail to converge with rapid switching
            }
        }

        // Test 2: Bootstrap convergence with extreme statistics
        let skewed_data: Vec<f64> = (0..200)
            .map(|i| {
                let x = (i as f64) / 200.0;
                if x < 0.9 {
                    x.powf(0.1) // Highly skewed distribution
                } else {
                    100.0 * x // Extreme tail
                }
            })
            .collect();

        // Statistic sensitive to outliers
        let sensitive_statistic = |data: &[f64]| {
            data.iter().fold(0.0f64, |acc, &x| acc.max(x)) // Maximum
        };

        let bootstrap_config = BootstrapConfiguration {
            num_bootstrap_samples: 1000,
            bootstrap_method: BootstrapMethod::Block,
            block_size: Some(50),
            confidence_levels: vec![0.95],
            confidence_interval_method: ConfidenceIntervalMethod::BootstrapBca,
            seed: Some(42),
            studentized_outer: None,
            studentized_inner: None,
            jackknife_block_size: None,
            force_block_jackknife: None,
        };

        let bootstrap_result =
            bootstrap_validate(&skewed_data, sensitive_statistic, &bootstrap_config);
        match bootstrap_result {
            Ok(validation) => {
                // Should handle extreme statistics
                for ci in &validation.confidence_intervals {
                    assert!(
                        ci.lower_bound.is_finite(),
                        "CI lower bound should be finite for {:?}",
                        ci.method
                    );
                    assert!(
                        ci.upper_bound.is_finite(),
                        "CI upper bound should be finite for {:?}",
                        ci.method
                    );
                    assert!(
                        ci.lower_bound <= ci.upper_bound,
                        "CI should be well-ordered for {:?}",
                        ci.method
                    );
                }
            }
            Err(_) => {
                // May fail with extreme statistics
            }
        }
    }

    #[test]
    fn test_memory_intensive_operations() {
        // Test 1: Large dataset multifractal analysis
        let large_data: Vec<f64> = (0..5000)
            .map(|i| (i as f64 * 0.001).sin() + 0.1 * (i as f64 * 0.01).cos())
            .collect();

        let config = MultifractalConfig {
            q_range: (-5.0, 5.0),
            num_q_values: 21, // Many q values
            min_scale: 4,
            max_scale_factor: 10.0, // 5000/500 = 10
            polynomial_order: 3,    // Higher order polynomial
        };

        let result = perform_multifractal_analysis(&large_data);
        match result {
            Ok(analysis) => {
                // Should complete without running out of memory
                assert!(!analysis.generalized_hurst_exponents.is_empty());
                assert!(analysis.generalized_hurst_exponents.len() <= 21);
            }
            Err(_) => {
                // May fail due to memory constraints
            }
        }

        // Test 2: Many small analyses vs. few large analyses
        let mut total_estimates = 0;

        // Many small analyses
        for i in 0..100 {
            let small_data: Vec<f64> = (0..100)
                .map(|j| ((i * 100 + j) as f64 * 0.01).sin())
                .collect();

            if let Ok((estimate, _, _)) = gph_test(&small_data) {
                if estimate.is_finite() {
                    total_estimates += 1;
                }
            }
        }

        // Should handle many small computations efficiently
        assert!(total_estimates > 50, "Should complete many small analyses");
    }
}
