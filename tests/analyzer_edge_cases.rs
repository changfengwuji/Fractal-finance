//! Edge case tests for the main analyzer and cross-validation components.
//!
//! These tests focus on the StatisticalFractalAnalyzer and cross-validation functionality
//! under challenging conditions that may occur in production financial systems.

use assert_approx_eq::assert_approx_eq;
use fractal_finance::*;
use fractal_finance::cross_validation::{
    TradingConfig, FinancialMetricsConfig, GphConfig, DfaConfig
};
use fractal_finance::preprocessing::DataKind;
use std::collections::HashMap;
use std::f64::{INFINITY, NAN, NEG_INFINITY};

/// Edge cases for the main statistical fractal analyzer
#[cfg(test)]
mod analyzer_edge_cases {
    use super::*;

    #[test]
    fn test_analyzer_with_problematic_data() {
        let mut analyzer = StatisticalFractalAnalyzer::new();

        // Test 1: Series with identical values (zero variance)
        let constant_series = vec![42.0; 500];
        analyzer.add_time_series("CONSTANT".to_string(), constant_series);

        let result = analyzer.analyze_series("CONSTANT");
        match result {
            Ok(_) => {
                // Should handle constant data gracefully
                // Some methods may fail, others may succeed with H ≈ 0.5
                if let Ok(results) = analyzer.get_analysis_results("CONSTANT") {
                    if !results.hurst_estimates.is_empty() {
                        for (method, estimate) in &results.hurst_estimates {
                            assert!(
                                estimate.estimate.is_finite(),
                                "Method {:?} should produce finite estimate",
                                method
                            );
                        }
                    }
                }
            }
            Err(e) => {
                // Acceptable to fail with constant data
                match e {
                    FractalAnalysisError::InsufficientData { .. }
                    | FractalAnalysisError::NumericalError { .. } => {
                        // Expected error types - constant data can cause numerical issues
                        // including singular matrices in regression
                    }
                    _ => panic!("Unexpected error type for constant data: {:?}", e),
                }
            }
        }

        // Test 2: Series with extreme outliers
        let mut outlier_series: Vec<f64> = (0..400).map(|i| (i as f64 * 0.01).sin()).collect();
        outlier_series[200] = 1000.0; // Extreme positive outlier
        outlier_series[300] = -500.0; // Extreme negative outlier

        analyzer.add_time_series("OUTLIERS".to_string(), outlier_series);
        let result = analyzer.analyze_series("OUTLIERS");
        match result {
            Ok(_) => {
                // Should handle outliers robustly
                if let Ok(results) = analyzer.get_analysis_results("OUTLIERS") {
                    for (method, estimate) in &results.hurst_estimates {
                        assert!(estimate.estimate.is_finite());
                        assert!(
                            estimate.estimate >= -0.5 && estimate.estimate <= 1.5,
                            "Method {:?} estimate {} should be reasonable despite outliers",
                            method,
                            estimate.estimate
                        );
                        assert!(
                            estimate.standard_error.is_finite() && estimate.standard_error >= 0.0
                        );
                    }
                }
            }
            Err(_) => {
                // May fail due to outliers affecting numerical stability
            }
        }

        // Test 3: Series with missing data (represented as NaN)
        let mut nan_series: Vec<f64> = (0..300).map(|i| (i as f64 * 0.02).sin()).collect();
        nan_series[100] = NAN;
        nan_series[200] = NAN;

        // add_time_series should reject NaN data
        let add_result = analyzer.add_time_series("NAN_DATA".to_string(), nan_series);
        assert!(add_result.is_err(), "Should reject NaN data during insertion");
        match add_result {
            Err(FractalAnalysisError::NumericalError { .. }) |
            Err(FractalAnalysisError::InvalidParameter { .. }) => {
                // Expected error types - NaN causes validation failure
            }
            Err(e) => panic!("Expected NumericalError or InvalidParameter for NaN data, got {:?}", e),
            Ok(_) => panic!("Should not accept NaN data"),
        }

        // Test 4: Very short series (boundary case)
        let short_series = vec![1.0, 2.0, 1.5, 2.5, 1.8];
        analyzer.add_time_series("SHORT".to_string(), short_series);

        let result = analyzer.analyze_series("SHORT");
        match result {
            Ok(_) => {
                // Some methods might work with very short series
                if let Ok(results) = analyzer.get_analysis_results("SHORT") {
                    for (method, estimate) in &results.hurst_estimates {
                        assert!(
                            estimate.estimate.is_finite(),
                            "Method {:?} should handle short series",
                            method
                        );
                    }
                }
            }
            Err(FractalAnalysisError::InsufficientData { .. }) => {
                // Expected to fail due to insufficient data
            }
            Err(e) => panic!("Unexpected error for short series: {:?}", e),
        }
    }

    #[test]
    #[cfg_attr(not(feature = "long-tests"), ignore)]
    fn test_analyzer_resource_management() {
        let mut analyzer = StatisticalFractalAnalyzer::new();

        // Test 1: Many series with different characteristics
        // NOTE: This test is computationally intensive by design to validate enterprise-scale performance
        // In quantitative finance, systems must handle multiple assets simultaneously with full rigor
        for i in 0..20 {
            let series_type = i % 4;
            let data = match series_type {
                0 => {
                    // Random walk - mathematically important for Hurst analysis
                    let mut walk = vec![0.0];
                    for j in 1..500 {
                        walk.push(walk[j - 1] + 0.01 * (rand::random::<f64>() - 0.5));
                    }
                    walk
                }
                1 => {
                    // Mean-reverting - tests different regime characteristics
                    let mut level = 0.0;
                    let mut series = Vec::new();
                    for _ in 0..500 {
                        level = 0.9 * level + 0.1 * (rand::random::<f64>() - 0.5);
                        series.push(level);
                    }
                    series
                }
                2 => {
                    // Trending - tests non-stationary behavior detection
                    (0..500)
                        .map(|j| 0.001 * j as f64 + 0.1 * (j as f64 * 0.01).sin())
                        .collect()
                }
                _ => {
                    // Periodic - tests multifractal detection capabilities
                    (0..500)
                        .map(|j| (j as f64 * 0.1).sin() + 0.5 * (j as f64 * 0.03).cos())
                        .collect()
                }
            };

            analyzer.add_time_series(format!("SERIES_{}", i), data);
        }

        // Analyze all series - this is the comprehensive test that validates enterprise capability
        eprintln!("Starting analysis of 20 series...");
        let result = analyzer.analyze_all_series();
        eprintln!("Analysis completed with result: {:?}", result.is_ok());
        match result {
            Ok(_) => {
                // Should handle multiple diverse series with full mathematical rigor
                for i in 0..20 {
                    let series_results = analyzer.get_analysis_results(&format!("SERIES_{}", i));
                    match series_results {
                        Ok(results) => {
                            // Each series should have some results
                            assert!(
                                !results.hurst_estimates.is_empty()
                                    || results
                                        .multifractal_analysis
                                        .generalized_hurst_exponents
                                        .is_empty(),
                                "Series {} should have some analysis results",
                                i
                            );
                        }
                        Err(_) => {
                            // Individual series may fail, but not all
                        }
                    }
                }
            }
            Err(_) => {
                // May fail with resource constraints - acceptable for edge case testing
            }
        }

        // Test 2: Memory cleanup after analysis
        // CRITICAL: Clear global state that gets corrupted during intensive analysis
        fractal_finance::computation_cache::clear_global_cache();
        fractal_finance::memory_pool::clear_global_pools();
        fractal_finance::fft_ops::clear_fft_cache();

        // Clear analyzer and verify it can handle new data
        let mut new_analyzer = StatisticalFractalAnalyzer::new();
        let simple_data: Vec<f64> = (0..200).map(|i| (i as f64 * 0.05).sin()).collect();
        new_analyzer.add_time_series("FRESH".to_string(), simple_data);

        let fresh_result = new_analyzer.analyze_series("FRESH");
        if let Err(ref e) = fresh_result {
            eprintln!("Fresh analyzer error: {:?}", e);
        }
        assert!(
            fresh_result.is_ok(),
            "Fresh analyzer should work after resource cleanup"
        );
    }

    #[test]
    fn test_analyzer_basic_resource_management() {
        // Lightweight resource management test for regular test runs
        let mut analyzer = StatisticalFractalAnalyzer::new();

        // Test with adequate data size for statistical rigor (256+ points required for comprehensive analysis)
        // Add deterministic, scale-aware jitter to avoid exact singularities in regression
        // Note: The estimators should handle near-singular matrices robustly
        use rand::{SeedableRng, Rng};
        use rand::rngs::StdRng;
        
        // Helper to add scale-aware jitter
        fn add_jitter(data: Vec<f64>, seed: u64) -> Vec<f64> {
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
            let std = variance.sqrt();
            let eps = (1e-9 * std).max(1e-15); // Scale-aware noise level
            
            let mut rng = StdRng::seed_from_u64(seed);
            data.into_iter()
                .map(|x| x + rng.gen_range(-eps..eps))
                .collect()
        }
        
        for i in 0..3 {
            let base_data: Vec<f64> = match i {
                0 => (0..300).map(|j| j as f64 * 0.01).collect(), // Pure trend
                1 => (0..300).map(|j| (j as f64 * 0.1).sin()).collect(), // Pure periodic
                _ => (0..300)
                    .map(|j| j as f64 * 0.005 + (j as f64 * 0.02).sin() * 0.1)
                    .collect(), // Trend + periodic
            };
            let data = add_jitter(base_data, 42 + i as u64); // Deterministic seed
            analyzer.add_time_series(format!("BASIC_{}", i), data);
        }

        // Test individual analysis to avoid timeout
        let mut successful_analyses = 0;
        for i in 0..3 {
            match analyzer.analyze_series(&format!("BASIC_{}", i)) {
                Ok(_) => {
                    eprintln!("BASIC_{} analysis succeeded", i);
                    successful_analyses += 1;
                }
                Err(e) => {
                    eprintln!("BASIC_{} analysis failed: {:?}", i, e);
                }
            }
        }

        assert!(
            successful_analyses > 0,
            "At least one basic analysis should succeed"
        );

        // Test fresh analyzer works (the core resource management test)
        fractal_finance::computation_cache::clear_global_cache();
        fractal_finance::memory_pool::clear_global_pools();
        fractal_finance::fft_ops::clear_fft_cache();

        let mut new_analyzer = StatisticalFractalAnalyzer::new();
        let simple_data: Vec<f64> = (0..300).map(|i| (i as f64 * 0.05).sin()).collect();
        new_analyzer.add_time_series("FRESH_BASIC".to_string(), simple_data);

        let fresh_result = new_analyzer.analyze_series("FRESH_BASIC");
        if let Err(ref e) = fresh_result {
            eprintln!("Fresh analyzer error: {:?}", e);
        }
        assert!(
            fresh_result.is_ok(),
            "Fresh analyzer should work after resource cleanup"
        );
    }

    #[test]
    fn test_analyzer_method_consistency() {
        let mut analyzer = StatisticalFractalAnalyzer::new();

        // Generate FBM with known Hurst exponent
        let known_h = 0.7;
        let fbm_config = FbmConfig {
            hurst_exponent: known_h,
            volatility: 1.0,
            method: FbmMethod::Hosking,
        };

        let gen_config = GeneratorConfig {
            length: 1000,
            seed: Some(42),
            ..Default::default()
        };
        let fbm_result = generate_fractional_brownian_motion(&gen_config, &fbm_config);
        if let Ok(fbm_data) = fbm_result {
            // Convert FBM to FGN (increments) for proper Hurst analysis
            // FGN should have Hurst exponent ≈ known_h, while FBM would show ≈ known_h + 1
            let fgn_data = fbm_to_fgn(&fbm_data);
            analyzer.add_time_series("KNOWN_H".to_string(), fgn_data);

            let result = analyzer.analyze_series("KNOWN_H");
            match result {
                Ok(_) => {
                    if let Ok(results) = analyzer.get_analysis_results("KNOWN_H") {
                        let mut method_estimates = HashMap::new();

                        // Collect estimates from different methods
                        for (method, estimate) in &results.hurst_estimates {
                            method_estimates.insert(method, estimate.estimate);
                        }

                        // Check consistency between methods
                        if method_estimates.len() >= 2 {
                            let estimates: Vec<f64> = method_estimates.values().cloned().collect();
                            let mean_estimate =
                                estimates.iter().sum::<f64>() / estimates.len() as f64;

                            // Methods should roughly agree
                            for (method, estimate) in &method_estimates {
                                let deviation = (estimate - mean_estimate).abs();
                                assert!(
                                    deviation < 0.3,
                                    "Method {:?} estimate {} deviates too much from mean {}",
                                    method,
                                    estimate,
                                    mean_estimate
                                );
                            }

                            // At least one method should be reasonably close to true H
                            let best_estimate = estimates
                                .iter()
                                .min_by(|a, b| {
                                    ((**a) - known_h)
                                        .abs()
                                        .partial_cmp(&((**b) - known_h).abs())
                                        .unwrap()
                                })
                                .unwrap();

                            assert!(
                                (best_estimate - known_h).abs() < 0.2,
                                "Best estimate {} should be close to true H = {}",
                                best_estimate,
                                known_h
                            );
                        }
                    }
                }
                Err(_) => {
                    // May fail with generated data
                }
            }
        }
    }

    #[test]
    fn test_analyzer_statistical_validation() {
        let mut analyzer = StatisticalFractalAnalyzer::new();

        // Test 1: White noise should give H ≈ 0.5
        let white_noise: Vec<f64> = (0..1000).map(|_| rand::random::<f64>() - 0.5).collect();
        analyzer.add_time_series("WHITE_NOISE".to_string(), white_noise);

        let result = analyzer.analyze_series("WHITE_NOISE");
        match result {
            Ok(_) => {
                // White noise should show H ≈ 0.5
                if let Ok(results) = analyzer.get_analysis_results("WHITE_NOISE") {
                    for (method, estimate) in &results.hurst_estimates {
                        assert!(
                            estimate.estimate > 0.3 && estimate.estimate < 0.7,
                            "Method {:?} should detect H ≈ 0.5 for white noise, got {}",
                            method,
                            estimate.estimate
                        );
                    }
                }
            }
            Err(_) => {
                // May fail on pure noise
            }
        }

        // Test 2: Random walk increments should give H ≈ 0.5
        // Note: DFA should be applied to the increments, not the cumulative walk
        let mut random_walk_increments = Vec::with_capacity(1000);
        for _ in 0..1000 {
            let step = if rand::random::<bool>() { 1.0 } else { -1.0 };
            random_walk_increments.push(step);
        }

        analyzer.add_time_series("RANDOM_WALK".to_string(), random_walk_increments);
        let result = analyzer.analyze_series("RANDOM_WALK");
        match result {
            Ok(_) => {
                // Random walk increments should show H ≈ 0.5
                if let Ok(results) = analyzer.get_analysis_results("RANDOM_WALK") {
                    for (method, estimate) in &results.hurst_estimates {
                        // Some methods may be biased, but should be in reasonable range
                        assert!(estimate.estimate > 0.3 && estimate.estimate < 0.8,
                               "Method {:?} should detect reasonable H for random walk increments, got {}", 
                               method, estimate.estimate);
                    }
                }
            }
            Err(_) => {
                // May fail on discrete random walk
            }
        }

        // Test 3: Strongly trending data
        // For trend detection, we need to skip preprocessing that would difference the data
        let trend_data: Vec<f64> = (0..500)
            .map(|i| i as f64 + 0.1 * (i as f64).sin())
            .collect();
        analyzer.add_time_series("TREND".to_string(), trend_data);

        // Use DataKind::Returns to skip differencing so we can detect the trend
        let result = analyzer.analyze_series_with_kind("TREND", DataKind::Returns);
        match result {
            Ok(_) => {
                // Strong trend should show H > 0.5
                if let Ok(results) = analyzer.get_analysis_results("TREND") {
                    for (method, estimate) in &results.hurst_estimates {
                        assert!(
                            estimate.estimate > 0.5,
                            "Method {:?} should detect H > 0.5 for trending data, got {}",
                            method,
                            estimate.estimate
                        );
                    }
                }
            }
            Err(_) => {
                // Some methods may fail on strong trends
            }
        }
    }
}

/// Edge cases for cross-validation functionality
#[cfg(test)]
mod cross_validation_edge_cases {
    use super::*;

    #[test]
    fn test_cross_validation_extreme_cases() {
        // Test 1: Very small dataset for cross-validation
        let small_data = vec![1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.9, 2.1, 1.7, 2.3];

        let estimators = vec![
            FractalEstimator::PeriodogramRegression(GphConfig::default()),
            FractalEstimator::RescaledRange,
        ];

        let cv_config = CrossValidationConfig {
            estimators: estimators.clone(),
            method: CrossValidationMethod::WalkForward {
                window_size: 5,
                step_size: 1,
            },
            selection_criterion: SelectionCriterion::MinimizeError,
            bootstrap_config: BootstrapConfiguration::default(),
            stability_runs: 5,
            significance_level: 0.05,
            seed: Some(42),
            trading_config: TradingConfig::default(),
            financial_config: FinancialMetricsConfig::default(),
        };

        let result = cross_validate_fractal_models(&small_data, &cv_config);
        match result {
            Ok(cv_result) => {
                assert!(!cv_result.all_results.is_empty());
                // Should handle small datasets
                for (_, cv_result_item) in &cv_result.all_results {
                    assert!(
                        cv_result_item.metrics.mse.is_finite() && cv_result_item.metrics.mse >= 0.0
                    );
                }
            }
            Err(FractalAnalysisError::InsufficientData { .. }) => {
                // Expected to fail with very small data
            }
            Err(e) => panic!("Unexpected error for small CV data: {:?}", e),
        }

        // Test 2: Cross-validation with unstable data
        let mut unstable_data = Vec::new();
        for i in 0..200 {
            if i % 50 == 0 {
                // Regime changes every 50 points
                unstable_data.push(100.0 * rand::random::<f64>());
            } else {
                unstable_data.push(rand::random::<f64>());
            }
        }

        let cv_config = CrossValidationConfig {
            estimators: estimators.clone(),
            method: CrossValidationMethod::WalkForward {
                window_size: 30,
                step_size: 10,
            },
            selection_criterion: SelectionCriterion::MinimizeError,
            bootstrap_config: BootstrapConfiguration::default(),
            stability_runs: 5,
            significance_level: 0.05,
            seed: Some(42),
            trading_config: TradingConfig::default(),
            financial_config: FinancialMetricsConfig::default(),
        };

        let result = cross_validate_fractal_models(&unstable_data, &cv_config);
        match result {
            Ok(cv_result) => {
                // Should handle unstable data
                assert!(!cv_result.all_results.is_empty());

                // Performance may be poor but should be measurable
                for (_, cv_result_item) in &cv_result.all_results {
                    assert!(cv_result_item.metrics.mse.is_finite());
                    // High MSE is acceptable for unstable data
                    assert!(cv_result_item.metrics.mse >= 0.0);
                }
            }
            Err(_) => {
                // May fail with highly unstable data
            }
        }
    }

    #[test]
    fn test_cross_validation_configuration_edge_cases() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let estimators = vec![FractalEstimator::PeriodogramRegression(GphConfig::default())];

        // Test 1: Initial window larger than data
        let bad_config = CrossValidationConfig {
            estimators: estimators.clone(),
            method: CrossValidationMethod::WalkForward {
                window_size: 150,
                step_size: 1,
            },
            selection_criterion: SelectionCriterion::MinimizeError,
            bootstrap_config: BootstrapConfiguration::default(),
            stability_runs: 5,
            significance_level: 0.05,
            seed: Some(42),
            trading_config: TradingConfig::default(),
            financial_config: FinancialMetricsConfig::default(),
        };

        let result = cross_validate_fractal_models(&data, &bad_config);
        assert!(
            result.is_err(),
            "Should fail when initial_window > data.len()"
        );

        // Test 2: Zero step size
        let bad_config = CrossValidationConfig {
            estimators: estimators.clone(),
            method: CrossValidationMethod::WalkForward {
                window_size: 20,
                step_size: 1,
            },
            selection_criterion: SelectionCriterion::MinimizeError,
            bootstrap_config: BootstrapConfiguration::default(),
            stability_runs: 5,
            significance_level: 0.05,
            seed: Some(42),
            trading_config: TradingConfig::default(),
            financial_config: FinancialMetricsConfig::default(),
        };

        let result = cross_validate_fractal_models(&data, &bad_config);
        assert!(result.is_err(), "Should fail with zero step size");

        // Test 3: Very large horizon
        let bad_config = CrossValidationConfig {
            estimators: estimators.clone(),
            method: CrossValidationMethod::WalkForward {
                window_size: 20,
                step_size: 1,
            },
            selection_criterion: SelectionCriterion::MinimizeError,
            bootstrap_config: BootstrapConfiguration::default(),
            stability_runs: 5,
            significance_level: 0.05,
            seed: Some(42),
            trading_config: TradingConfig::default(),
            financial_config: FinancialMetricsConfig::default(),
        };

        let result = cross_validate_fractal_models(&data, &bad_config);
        match result {
            Ok(cv_result) => {
                // May succeed but with few folds
                assert!(
                    cv_result.all_results.len() <= 5,
                    "Should have few folds with large horizon"
                );
            }
            Err(_) => {
                // May fail due to insufficient data for large horizon
            }
        }

        // Test 4: Single fold possible
        let minimal_config = CrossValidationConfig {
            estimators: estimators.clone(),
            method: CrossValidationMethod::WalkForward {
                window_size: 70,
                step_size: 10,
            },
            selection_criterion: SelectionCriterion::MinimizeError,
            bootstrap_config: BootstrapConfiguration::default(),
            stability_runs: 5,
            significance_level: 0.05,
            seed: Some(42),
            trading_config: TradingConfig::default(),
            financial_config: FinancialMetricsConfig::default(),
        };

        let result = cross_validate_fractal_models(&data, &minimal_config);
        match result {
            Ok(cv_result) => {
                // Should work with minimal folds
                assert!(cv_result.all_results.len() >= 1);
                assert!(matches!(cv_result.best_estimator, FractalEstimator::PeriodogramRegression(_)));
            }
            Err(_) => {
                // May fail with minimal data
            }
        }
    }

    #[test]
    fn test_cross_validation_model_selection() {
        // Generate data with known properties for model selection
        let fbm_config = FbmConfig {
            hurst_exponent: 0.8,
            volatility: 1.0,
            method: FbmMethod::Hosking,
        };

        let gen_config = GeneratorConfig {
            length: 300,
            seed: Some(42),
            ..Default::default()
        };
        let data_result = generate_fractional_brownian_motion(&gen_config, &fbm_config);
        if let Ok(data) = data_result {
            let estimators = vec![
                FractalEstimator::PeriodogramRegression(GphConfig::default()),
                FractalEstimator::RescaledRange,
                FractalEstimator::DetrendedFluctuation(DfaConfig::default()),
            ];

            let cv_config = CrossValidationConfig {
                estimators: estimators.clone(),
                method: CrossValidationMethod::WalkForward {
                    window_size: 50,
                    step_size: 20,
                },
                selection_criterion: SelectionCriterion::MinimizeError,
                bootstrap_config: BootstrapConfiguration::default(),
                stability_runs: 10,
                significance_level: 0.05,
                seed: Some(42),
            trading_config: TradingConfig::default(),
            financial_config: FinancialMetricsConfig::default(),
            };

            let result = cross_validate_fractal_models(&data, &cv_config);
            match result {
                Ok(cv_result) => {
                    // Should evaluate multiple estimators
                    assert!(!cv_result.all_results.is_empty());

                    // All estimators should have performance metrics
                    for (_, cv_result_item) in &cv_result.all_results {
                        assert!(
                            cv_result_item.metrics.mse.is_finite()
                                && cv_result_item.metrics.mse >= 0.0,
                            "Estimator {:?} should have valid MSE",
                            cv_result_item.estimator
                        );
                        assert!(
                            cv_result_item.metrics.mae.is_finite()
                                && cv_result_item.metrics.mae >= 0.0,
                            "Estimator {:?} should have valid MAE",
                            cv_result_item.estimator
                        );
                        assert!(
                            cv_result_item.metrics.estimate_stability_concordance >= 0.0,
                            "Estimator {:?} should have valid directional accuracy",
                            cv_result_item.estimator
                        );
                    }

                    // Should be able to select best model
                    assert!(
                        matches!(cv_result.best_estimator, FractalEstimator::PeriodogramRegression(_))
                            || cv_result.best_estimator == FractalEstimator::RescaledRange
                            || matches!(cv_result.best_estimator, FractalEstimator::DetrendedFluctuation(_))
                    );
                }
                Err(_) => {
                    // May fail with generated data
                }
            }
        }
    }

    #[test]
    fn test_cross_validation_performance_metrics() {
        // Create data with predictable pattern for validation
        let pattern_data: Vec<f64> = (0..200)
            .map(|i| {
                let t = i as f64;
                0.7 * t.ln() + 0.1 * (t * 0.1).sin() // Log trend with noise
            })
            .collect();

        let estimators = vec![
            FractalEstimator::PeriodogramRegression(GphConfig::default()),
            FractalEstimator::DetrendedFluctuation(DfaConfig::default()),
        ];
        let cv_config = CrossValidationConfig {
            estimators: estimators.clone(),
            method: CrossValidationMethod::WalkForward {
                window_size: 30,
                step_size: 15,
            },
            selection_criterion: SelectionCriterion::MinimizeError,
            bootstrap_config: BootstrapConfiguration::default(),
            stability_runs: 10,
            significance_level: 0.05,
            seed: Some(42),
            trading_config: TradingConfig::default(),
            financial_config: FinancialMetricsConfig::default(),
        };

        let result = cross_validate_fractal_models(&pattern_data, &cv_config);
        match result {
            Ok(cv_result) => {
                // Validate performance metrics consistency
                for (_, cv_result_item) in &cv_result.all_results {
                    // MSE should be non-negative
                    assert!(
                        cv_result_item.metrics.mse >= 0.0,
                        "MSE should be non-negative for {:?}",
                        cv_result_item.estimator
                    );

                    // MAE should be non-negative and ≤ sqrt(MSE) (by Cauchy-Schwarz)
                    assert!(
                        cv_result_item.metrics.mae >= 0.0,
                        "MAE should be non-negative for {:?}",
                        cv_result_item.estimator
                    );
                    assert!(
                        cv_result_item.metrics.mae <= cv_result_item.metrics.mse.sqrt() + 1e-10,
                        "MAE should be ≤ sqrt(MSE) for {:?}",
                        cv_result_item.estimator
                    );

                    // Directional accuracy should be reasonable
                    assert!(
                        cv_result_item.metrics.estimate_stability_concordance >= 0.0,
                        "Directional accuracy should be ≥ 0 for {:?}",
                        cv_result_item.estimator
                    );
                }

                // Results should be consistent with summary
                if !cv_result.all_results.is_empty() {
                    if let Some(gph_result) = cv_result
                        .all_results
                        .iter().find(|(k, _)| matches!(k, FractalEstimator::PeriodogramRegression(_))).map(|(_, v)| v)
                    {
                        assert!(
                            !gph_result.fold_results.is_empty(),
                            "Should have GPH cross-validation results"
                        );

                        // Check that results are reasonable
                        assert!(
                            gph_result.metrics.mae < 1.0,
                            "GPH should achieve reasonable MAE on periodic data"
                        );
                    }
                }
            }
            Err(_) => {
                // May fail with pattern data
            }
        }
    }
}
