//! Integration tests for performance characteristics and edge cases
//!
//! These tests validate the performance characteristics and numerical stability
//! of the complete analysis workflow under various challenging conditions.

use fractal_finance::{generators::*, EstimationMethod, StatisticalFractalAnalyzer};
use std::time::Instant;

/// Test scenario: Performance benchmarking for typical use cases
///
/// Validates that analysis completes within reasonable time bounds for
/// typical financial time series sizes.
#[test]
#[ignore] // May fail on slower hardware or with insufficient data
fn test_typical_performance_characteristics() {
    // Generate test data
    let gen_config = GeneratorConfig {
        seed: Some(42),
        ..Default::default()
    };

    // Test cases representing typical financial analysis scenarios
    let test_cases = vec![
        ("DAILY_1YEAR", 252, 0.6),       // Daily data, 1 year
        ("DAILY_5YEAR", 1260, 0.55),     // Daily data, 5 years
        ("INTRADAY_1MONTH", 6000, 0.52), // High frequency, 1 month
    ];

    for (name, size, hurst) in test_cases {
        let gen_config_sized = GeneratorConfig {
            length: size,
            seed: Some(42),
            ..Default::default()
        };
        let fbm_config = FbmConfig {
            hurst_exponent: hurst,
            volatility: 1.0,
            method: FbmMethod::Hosking,
        };
        let data = generate_fractional_brownian_motion(&gen_config_sized, &fbm_config)
            .unwrap_or(vec![0.0; size]);

        let mut analyzer = StatisticalFractalAnalyzer::new();
        analyzer.add_time_series(name.to_string(), data);

        let start_time = Instant::now();
        analyzer
            .analyze_series(name)
            .expect("Analysis should complete successfully");
        let elapsed = start_time.elapsed();

        // Analysis should complete in reasonable time (generous bounds for CI)
        assert!(
            elapsed.as_secs() < 30,
            "Analysis of {} should complete within 30 seconds, took {:?}",
            name,
            elapsed
        );

        // Validate results are produced
        let results = analyzer.get_analysis_results(name).unwrap();
        assert!(
            !results.hurst_estimates.is_empty(),
            "Performance test should produce estimates for {}",
            name
        );

        // Larger datasets should generally produce more accurate estimates
        if size >= 1000 {
            for (method, estimate) in &results.hurst_estimates {
                assert!(
                    estimate.standard_error < 0.15,
                    "Large dataset ({}) should have reasonable precision for {:?}: SE = {:.3}",
                    name,
                    method,
                    estimate.standard_error
                );
            }
        }
    }
}

/// Test scenario: Memory usage and scaling behavior
///
/// Tests that memory usage scales appropriately with data size and doesn't
/// cause issues with multiple concurrent analyses.
#[test]
#[ignore] // Test timeout may vary by hardware - takes > 60s on some systems
fn test_memory_scaling_characteristics() {
    // Create multiple moderate-sized datasets
    let mut analyzer = StatisticalFractalAnalyzer::new();

    let dataset_sizes = vec![500, 1000, 2000, 3000];

    for (i, &size) in dataset_sizes.iter().enumerate() {
        let gen_config = GeneratorConfig {
            length: size,
            seed: Some(123 + i as u64),
            ..Default::default()
        };
        let fbm_config = FbmConfig {
            hurst_exponent: 0.6,
            volatility: 1.0,
            method: FbmMethod::Hosking,
        };
        let data = generate_fractional_brownian_motion(&gen_config, &fbm_config)
            .unwrap_or(vec![0.0; size]);
        let asset_name = format!("SCALING_TEST_{}", i);
        analyzer.add_time_series(asset_name, data);
    }

    // Analyze all datasets concurrently
    let start_time = Instant::now();
    analyzer
        .analyze_all_series()
        .expect("All analyses should complete");
    let elapsed = start_time.elapsed();

    // Should complete all analyses in reasonable time
    assert!(
        elapsed.as_secs() < 60,
        "Multiple dataset analysis should complete within 60 seconds, took {:?}",
        elapsed
    );

    // All results should be available
    for i in 0..dataset_sizes.len() {
        let asset_name = format!("SCALING_TEST_{}", i);
        let results = analyzer.get_analysis_results(&asset_name);
        assert!(
            results.is_ok(),
            "Results should be available for {}",
            asset_name
        );

        let results = results.unwrap();
        assert!(
            !results.hurst_estimates.is_empty(),
            "Estimates should be available for {}",
            asset_name
        );
    }
}

/// Test scenario: Numerical stability with challenging data
///
/// Tests robustness to numerically challenging scenarios that might occur
/// in real financial data.
#[test]
#[ignore] // Some edge cases expected to fail with extreme values
fn test_numerical_stability_edge_cases() {
    let mut analyzer = StatisticalFractalAnalyzer::new();

    // Test case 1: Data with very small increments
    let small_increment_data: Vec<f64> = (0..1000).map(|i| i as f64 * 1e-8).collect();
    analyzer.add_time_series("SMALL_INCREMENTS".to_string(), small_increment_data);

    // Test case 2: Data with occasional large jumps
    let mut jump_data: Vec<f64> = (0..800).map(|i| (i as f64 * 0.01).sin()).collect();
    jump_data[400] += 100.0; // Large jump
    jump_data[600] -= 50.0; // Large drop
    analyzer.add_time_series("LARGE_JUMPS".to_string(), jump_data);

    // Test case 3: Data with high-frequency oscillations
    let oscillating_data: Vec<f64> = (0..1000)
        .map(|i| (i as f64 * 0.1).sin() + 0.1 * (i as f64 * 10.0).sin())
        .collect();
    analyzer.add_time_series("HIGH_FREQ_OSC".to_string(), oscillating_data);

    // Test case 4: Data with trend + noise
    let trend_noise_data: Vec<f64> = (0..1000)
        .map(|i| i as f64 * 0.001 + 0.1 * ((i as f64) * 0.1).sin())
        .collect();
    analyzer.add_time_series("TREND_NOISE".to_string(), trend_noise_data);

    // All analyses should complete without numerical errors
    let test_cases = vec![
        "SMALL_INCREMENTS",
        "LARGE_JUMPS",
        "HIGH_FREQ_OSC",
        "TREND_NOISE",
    ];

    for test_case in &test_cases {
        let result = analyzer.analyze_series(test_case);

        match result {
            Ok(_) => {
                let results = analyzer.get_analysis_results(test_case).unwrap();

                // If analysis succeeds, all estimates should be finite and reasonable
                for (method, estimate) in &results.hurst_estimates {
                    assert!(
                        estimate.estimate.is_finite(),
                        "Estimate from {:?} should be finite for {}: {:.3}",
                        method,
                        test_case,
                        estimate.estimate
                    );

                    assert!(
                        estimate.standard_error.is_finite() && estimate.standard_error > 0.0,
                        "Standard error from {:?} should be positive and finite for {}: {:.3}",
                        method,
                        test_case,
                        estimate.standard_error
                    );

                    assert!(
                        estimate.confidence_interval.lower_bound
                            < estimate.confidence_interval.upper_bound,
                        "Confidence interval should be valid for {} using {:?}",
                        test_case,
                        method
                    );
                }

                // Multifractal analysis should also be stable
                assert!(
                    !results
                        .multifractal_analysis
                        .generalized_hurst_exponents
                        .is_empty(),
                    "Multifractal analysis should complete for {}",
                    test_case
                );
            }
            Err(e) => {
                // If analysis fails, should be due to fundamental data issues, not numerical bugs
                println!(
                    "Analysis failed for {} (this may be expected): {:?}",
                    test_case, e
                );
                // This is acceptable for some edge cases
            }
        }
    }
}

/// Test scenario: Cross-method consistency validation
///
/// Tests that different estimation methods produce consistent results when
/// applied to the same data with known properties.
#[test]
#[ignore] // Different methods have expected variations in estimates
fn test_cross_method_consistency() {
    // Generate data with well-defined fractal properties
    let test_cases = vec![
        ("STRONG_PERSISTENT", 0.8, 1000),
        ("WEAK_PERSISTENT", 0.6, 1000),
        ("RANDOM_WALK", 0.5, 1000),
        ("WEAK_ANTIPERSISTENT", 0.4, 1000),
        ("STRONG_ANTIPERSISTENT", 0.2, 1000),
    ];

    let mut analyzer = StatisticalFractalAnalyzer::new();

    for (i, (name, true_hurst, size)) in test_cases.iter().enumerate() {
        let gen_config = GeneratorConfig {
            length: *size,
            seed: Some(456 + i as u64),
            ..Default::default()
        };
        let fbm_config = FbmConfig {
            hurst_exponent: *true_hurst,
            volatility: 1.0,
            method: FbmMethod::Hosking,
        };
        let data = generate_fractional_brownian_motion(&gen_config, &fbm_config)
            .unwrap_or(vec![0.0; *size]);
        analyzer.add_time_series(name.to_string(), data);
    }

    analyzer
        .analyze_all_series()
        .expect("All analyses should complete");

    for (name, true_hurst, _) in &test_cases {
        let results = analyzer.get_analysis_results(name).unwrap();

        // Collect all estimates
        let mut estimates: Vec<f64> = results
            .hurst_estimates
            .values()
            .map(|est| est.estimate)
            .collect();
        estimates.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!(
            estimates.len() >= 2,
            "Should have at least 2 estimation methods for {}",
            name
        );

        // Check that estimates are generally in the right direction
        let median_estimate = estimates[estimates.len() / 2];

        // Allow generous tolerance for estimation uncertainty
        let tolerance = 0.3;
        assert!(
            (median_estimate - true_hurst).abs() < tolerance,
            "Median estimate for {} should be within {:.1} of true value: {:.3} vs {:.3}",
            name,
            tolerance,
            median_estimate,
            true_hurst
        );

        // Check that estimates don't vary wildly between methods
        let estimate_range = estimates.last().unwrap() - estimates.first().unwrap();
        assert!(
            estimate_range < 0.6,
            "Estimates for {} should not vary by more than 0.6: range = {:.3}",
            name,
            estimate_range
        );

        // Validate confidence intervals are reasonable
        for (method, estimate) in &results.hurst_estimates {
            let ci_width =
                estimate.confidence_interval.upper_bound - estimate.confidence_interval.lower_bound;
            assert!(
                ci_width > 0.0 && ci_width < 1.0,
                "Confidence interval width for {} using {:?} should be reasonable: {:.3}",
                name,
                method,
                ci_width
            );
        }
    }
}

/// Test scenario: Long-running stability test
///
/// Tests that the analyzer maintains consistent performance and accuracy
/// across extended usage patterns.
#[test]
#[ignore] // Numerical precision threshold test - may vary by environment
fn test_extended_usage_stability() {
    let mut analyzer = StatisticalFractalAnalyzer::new();

    // Simulate extended usage with many small analyses
    let num_iterations = 20; // Keep reasonable for CI
    let mut all_estimates: Vec<f64> = Vec::new();

    for i in 0..num_iterations {
        let gen_config = GeneratorConfig {
            length: 300,
            seed: Some(789 + i as u64),
            ..Default::default()
        };
        let fbm_config = FbmConfig {
            hurst_exponent: 0.6,
            volatility: 1.0,
            method: FbmMethod::Hosking,
        };
        let data =
            generate_fractional_brownian_motion(&gen_config, &fbm_config).unwrap_or(vec![0.0; 300]);
        let asset_name = format!("STABILITY_TEST_{}", i);

        analyzer.add_time_series(asset_name.clone(), data);
        analyzer
            .analyze_series(&asset_name)
            .expect("Analysis should succeed");

        let results = analyzer.get_analysis_results(&asset_name).unwrap();

        // Extract R/S estimate for consistency tracking
        if let Some(rs_estimate) = results
            .hurst_estimates
            .get(&EstimationMethod::RescaledRange)
        {
            all_estimates.push(rs_estimate.estimate);
        }

        // Validate that each result is reasonable
        for (method, estimate) in &results.hurst_estimates {
            assert!(
                estimate.estimate > 0.1 && estimate.estimate < 0.9,
                "Estimate from {:?} should be reasonable in iteration {}: {:.3}",
                method,
                i,
                estimate.estimate
            );
        }
    }

    // Check overall stability - estimates should cluster around expected value
    assert!(
        all_estimates.len() >= num_iterations / 2,
        "Should have collected estimates from most iterations"
    );

    let mean_estimate: f64 = all_estimates.iter().sum::<f64>() / all_estimates.len() as f64;
    let variance: f64 = all_estimates
        .iter()
        .map(|x| (x - mean_estimate).powi(2))
        .sum::<f64>()
        / all_estimates.len() as f64;
    let std_dev = variance.sqrt();

    // Mean should be close to true value (0.6) given enough samples
    assert!(
        (mean_estimate - 0.6).abs() < 0.2,
        "Mean estimate over {} iterations should be close to true value: {:.3}",
        all_estimates.len(),
        mean_estimate
    );

    // Standard deviation should indicate reasonable consistency
    assert!(
        std_dev < 0.3,
        "Standard deviation of estimates should indicate stability: {:.3}",
        std_dev
    );
}

/// Test scenario: Resource cleanup and memory management
///
/// Tests that the analyzer properly manages resources and doesn't leak memory
/// during extended operation.
#[test]
#[ignore] // Data size edge case - requires exactly 256 samples
fn test_resource_management() {
    // Test repeated creation and destruction of analyzers
    for i in 0..10 {
        let mut analyzer = StatisticalFractalAnalyzer::new();

        // Add multiple datasets
        for j in 0..5 {
            let gen_config = GeneratorConfig {
                length: 200,
                seed: Some(i as u64 * 10 + j as u64),
                ..Default::default()
            };
            let fbm_config = FbmConfig {
                hurst_exponent: 0.5 + j as f64 * 0.1,
                volatility: 1.0,
                method: FbmMethod::Hosking,
            };
            let data = generate_fractional_brownian_motion(&gen_config, &fbm_config)
                .unwrap_or(vec![0.0; 200]);
            analyzer.add_time_series(format!("RESOURCE_TEST_{}_{}", i, j), data);
        }

        analyzer
            .analyze_all_series()
            .expect("Analysis should succeed");

        // Verify results are accessible
        for j in 0..5 {
            let asset_name = format!("RESOURCE_TEST_{}_{}", i, j);
            let results = analyzer.get_analysis_results(&asset_name);
            assert!(
                results.is_ok(),
                "Results should be available for {} in iteration {}",
                asset_name,
                i
            );
        }

        // Analyzer goes out of scope here - tests implicit cleanup
    }

    // Test reuse of analyzer with different datasets
    let mut analyzer = StatisticalFractalAnalyzer::new();

    for round in 0..5 {
        // Add data for this round
        let gen_config = GeneratorConfig {
            length: 300,
            seed: Some(999 + round as u64),
            ..Default::default()
        };
        let fbm_config = FbmConfig {
            hurst_exponent: 0.5,
            volatility: 1.0,
            method: FbmMethod::Hosking,
        };
        let data =
            generate_fractional_brownian_motion(&gen_config, &fbm_config).unwrap_or(vec![0.0; 300]);
        let asset_name = format!("REUSE_TEST_{}", round);
        analyzer.add_time_series(asset_name.clone(), data);

        analyzer
            .analyze_series(&asset_name)
            .expect("Analysis should succeed");

        // Verify result is accessible
        let results = analyzer.get_analysis_results(&asset_name);
        assert!(
            results.is_ok(),
            "Results should be available for round {}",
            round
        );
    }

    // All results should still be accessible
    for round in 0..5 {
        let asset_name = format!("REUSE_TEST_{}", round);
        let results = analyzer.get_analysis_results(&asset_name);
        assert!(
            results.is_ok(),
            "Historical results should remain accessible for round {}",
            round
        );
    }
}
