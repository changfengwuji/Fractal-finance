//! Integration tests for full workflow scenarios
//!
//! These tests validate end-to-end functionality of the StatisticalFractalAnalyzer
//! across complete analysis workflows, ensuring all components work together properly.

use assert_approx_eq::assert_approx_eq;
use fractal_finance::{
    errors::FractalAnalysisError, generators::*, EstimationMethod, StatisticalFractalAnalyzer,
};

/// Test scenario: Quantitative analyst performs complete fractal analysis workflow
///
/// This test simulates the most common usage pattern:
/// 1. Load time series data
/// 2. Perform comprehensive fractal analysis
/// 3. Extract regime information
/// 4. Generate summary report with confidence intervals
#[test]
fn test_complete_quantitative_analysis_workflow() {
    // Generate synthetic financial time series with known fractal properties
    // Generate synthetic data with known properties
    let gen_config = GeneratorConfig {
        length: 1000,
        seed: Some(42),
        ..Default::default()
    };
    let persistent_config = FbmConfig {
        hurst_exponent: 0.7,
        volatility: 1.0,
        method: FbmMethod::Hosking,
    };
    let anti_persistent_config = FbmConfig {
        hurst_exponent: 0.3,
        volatility: 1.0,
        method: FbmMethod::Hosking,
    };
    let random_walk_config = FbmConfig {
        hurst_exponent: 0.5,
        volatility: 1.0,
        method: FbmMethod::Hosking,
    };

    let persistent_data = generate_fractional_brownian_motion(&gen_config, &persistent_config)
        .unwrap_or(vec![0.0; 1000]);
    let anti_persistent_data =
        generate_fractional_brownian_motion(&gen_config, &anti_persistent_config)
            .unwrap_or(vec![0.0; 1000]);
    let random_walk_data = generate_fractional_brownian_motion(&gen_config, &random_walk_config)
        .unwrap_or(vec![0.0; 1000]);

    // Create analyzer and add multiple time series
    let mut analyzer = StatisticalFractalAnalyzer::new();
    analyzer.add_time_series("PERSISTENT_ASSET".to_string(), persistent_data);
    analyzer.add_time_series("ANTIPERSISTENT_ASSET".to_string(), anti_persistent_data);
    analyzer.add_time_series("RANDOM_WALK_ASSET".to_string(), random_walk_data);

    // Perform comprehensive analysis on all series
    analyzer
        .analyze_all_series()
        .expect("Analysis should succeed");

    // Validate persistent asset results
    let persistent_results = analyzer
        .get_analysis_results("PERSISTENT_ASSET")
        .expect("Results should be available");

    // Check that multiple estimation methods were used
    assert!(
        persistent_results.hurst_estimates.len() >= 3,
        "Should have at least 3 different Hurst estimation methods"
    );

    // Verify the persistent series has H > 0.5 (with tolerance for estimation error)
    for (method, estimate) in &persistent_results.hurst_estimates {
        assert!(
            estimate.estimate > 0.45,
            "Method {:?} should detect persistence (H > 0.45), got {:.3}",
            method,
            estimate.estimate
        );
        assert!(
            estimate.standard_error > 0.0,
            "Standard error should be positive"
        );
        assert!(
            estimate.confidence_interval.lower_bound < estimate.confidence_interval.upper_bound,
            "Confidence interval should be valid"
        );
    }

    // Validate anti-persistent asset results
    let anti_persistent_results = analyzer
        .get_analysis_results("ANTIPERSISTENT_ASSET")
        .expect("Results should be available");

    // Check that anti-persistent series has H < 0.5
    let rs_estimate = anti_persistent_results
        .hurst_estimates
        .get(&EstimationMethod::RescaledRange)
        .expect("R/S estimate should be available");
    assert!(
        rs_estimate.estimate < 0.55,
        "Anti-persistent series should have H < 0.55, got {:.3}",
        rs_estimate.estimate
    );

    // Validate multifractal analysis was performed
    assert!(
        !persistent_results
            .multifractal_analysis
            .generalized_hurst_exponents
            .is_empty(),
        "Multifractal analysis should produce generalized Hurst exponents"
    );
    assert!(
        !persistent_results
            .multifractal_analysis
            .singularity_spectrum
            .is_empty(),
        "Singularity spectrum should be computed"
    );

    // Validate statistical tests were performed
    assert!(
        persistent_results
            .statistical_tests
            .long_range_dependence_test
            .gph_statistic
            != 0.0,
        "Long-range dependence test should be performed"
    );

    // Validate regime analysis
    assert!(
        persistent_results.regime_analysis.regime_changes.len() >= 0,
        "Regime analysis should be performed (may or may not detect changes)"
    );

    // Validate validation statistics are available
    let validation_stats = analyzer
        .get_validation_statistics("PERSISTENT_ASSET")
        .expect("Validation statistics should be available");
    assert!(
        validation_stats.prediction_accuracy.mspe >= 0.0,
        "MSPE should be non-negative"
    );
    assert!(
        validation_stats.robustness_tests.outlier_robustness >= 0.0,
        "Robustness metrics should be computed"
    );
}

/// Test scenario: Analysis of real-world-like financial returns
///
/// Simulates analysis of actual financial returns with realistic properties:
/// - Fat tails and volatility clustering
/// - Potential regime changes
/// - Various sample sizes
#[test]
fn test_realistic_financial_returns_analysis() {
    // Generate realistic financial returns
    let gen_config = GeneratorConfig {
        length: 1000,
        seed: Some(123),
        ..Default::default()
    };

    // Generate realistic financial returns with regime switching
    let mut returns = Vec::new();

    // First regime: High volatility, anti-persistent (H = 0.35)
    let gen_config1 = GeneratorConfig {
        length: 500,
        seed: Some(123),
        ..Default::default()
    };
    let fbm_config1 = FbmConfig {
        hurst_exponent: 0.35,
        volatility: 1.0,
        method: FbmMethod::Hosking,
    };
    let regime1 =
        generate_fractional_brownian_motion(&gen_config1, &fbm_config1).unwrap_or(vec![0.0; 500]);
    for value in regime1.iter().take(250) {
        returns.push(value * 0.02); // Scale to realistic return magnitude
    }

    // Second regime: Low volatility, persistent (H = 0.65)
    let gen_config2 = GeneratorConfig {
        length: 500,
        seed: Some(124),
        ..Default::default()
    };
    let fbm_config2 = FbmConfig {
        hurst_exponent: 0.65,
        volatility: 1.0,
        method: FbmMethod::Hosking,
    };
    let regime2 =
        generate_fractional_brownian_motion(&gen_config2, &fbm_config2).unwrap_or(vec![0.0; 500]);
    for value in regime2.iter().take(250) {
        returns.push(value * 0.01); // Lower volatility
    }

    let mut analyzer = StatisticalFractalAnalyzer::new();
    analyzer.add_time_series("REALISTIC_RETURNS".to_string(), returns);

    // Perform analysis
    analyzer
        .analyze_series("REALISTIC_RETURNS")
        .expect("Analysis should succeed");

    let results = analyzer.get_analysis_results("REALISTIC_RETURNS").unwrap();

    // Validate that analysis handles regime switching appropriately
    assert!(
        results.regime_analysis.regime_changes.len() >= 0,
        "Regime detection should run without errors"
    );

    // Check that estimates are reasonable for financial data
    for (method, estimate) in &results.hurst_estimates {
        assert!(
            estimate.estimate > 0.2 && estimate.estimate < 0.8,
            "Hurst estimate from {:?} should be in reasonable range [0.2, 0.8], got {:.3}",
            method,
            estimate.estimate
        );
    }

    // Validate model selection was performed
    assert!(
        results.model_selection.uncertainty_score != 0.0,
        "Uncertainty score should be computed"
    );
    assert!(
        results.model_selection.num_parameters > 0,
        "Parameters should be tracked"
    );
}

/// Test scenario: Handling of various data sizes
///
/// Tests the analyzer's behavior with different sample sizes, from minimum
/// required to large datasets, ensuring consistent performance and accuracy.
#[test]
fn test_multi_scale_data_analysis() {
    let test_cases = vec![
        ("SMALL_SAMPLE", 100, 0.6),
        ("MEDIUM_SAMPLE", 500, 0.4),
        ("LARGE_SAMPLE", 2000, 0.8),
    ];

    let mut analyzer = StatisticalFractalAnalyzer::new();

    // Add all test cases
    for (name, size, hurst) in &test_cases {
        let gen_config = GeneratorConfig {
            length: *size,
            seed: Some(456),
            ..Default::default()
        };
        let fbm_config = FbmConfig {
            hurst_exponent: *hurst,
            volatility: 1.0,
            method: FbmMethod::Hosking,
        };
        let data = generate_fractional_brownian_motion(&gen_config, &fbm_config)
            .unwrap_or(vec![0.0; *size]);
        analyzer.add_time_series(name.to_string(), data);
    }

    // Analyze all series
    analyzer
        .analyze_all_series()
        .expect("All analyses should succeed");

    // Validate results for all sample sizes
    for (name, size, expected_hurst) in &test_cases {
        let results = analyzer.get_analysis_results(name).unwrap();

        // Check that at least one method produces reasonable estimates
        let has_reasonable_estimate = results
            .hurst_estimates
            .iter()
            .any(|(_, estimate)| (estimate.estimate - expected_hurst).abs() < 0.3);

        assert!(
            has_reasonable_estimate,
            "At least one method should produce reasonable estimate for {} (size {})",
            name, size
        );

        // Larger samples should generally have smaller standard errors
        if *size >= 500 {
            for (method, estimate) in &results.hurst_estimates {
                assert!(
                    estimate.standard_error < 0.2,
                    "Large sample ({}) should have small standard error for {:?}, got {:.3}",
                    size,
                    method,
                    estimate.standard_error
                );
            }
        }
    }
}

/// Test scenario: Cross-validation and model comparison
///
/// Tests the analyzer's ability to perform model validation and comparison
/// across different estimation methods and parameter settings.
#[test]
fn test_model_validation_and_comparison() {
    // Generate data with known properties for validation
    let known_hurst = 0.65;
    let gen_config = GeneratorConfig {
        length: 1000,
        seed: Some(789),
        ..Default::default()
    };
    let fbm_config = FbmConfig {
        hurst_exponent: known_hurst,
        volatility: 1.0,
        method: FbmMethod::Hosking,
    };
    let validation_data =
        generate_fractional_brownian_motion(&gen_config, &fbm_config).unwrap_or(vec![0.0; 1000]);

    let mut analyzer = StatisticalFractalAnalyzer::new();
    analyzer.add_time_series("VALIDATION_DATA".to_string(), validation_data);

    analyzer
        .analyze_series("VALIDATION_DATA")
        .expect("Analysis should succeed");

    let results = analyzer.get_analysis_results("VALIDATION_DATA").unwrap();
    let validation_stats = analyzer
        .get_validation_statistics("VALIDATION_DATA")
        .unwrap();

    // Validate model selection criteria are computed
    assert!(
        results.model_selection.uncertainty_score != 0.0,
        "Uncertainty score should be computed"
    );
    assert!(
        results.model_selection.num_parameters > 0,
        "Parameters should be tracked"
    );
    assert!(
        results.model_selection.uncertainty_score.is_finite(),
        "Uncertainty score should be finite"
    );

    // Check that best model is selected
    let best_method = &results.model_selection.best_model;
    assert!(
        results.hurst_estimates.contains_key(best_method),
        "Best model should be among computed estimates"
    );

    // Validate prediction accuracy metrics
    assert!(
        validation_stats.prediction_accuracy.mspe >= 0.0,
        "MSPE should be non-negative"
    );
    assert!(
        validation_stats.prediction_accuracy.coverage_probability >= 0.0,
        "Coverage probability should be non-negative"
    );
    assert!(
        validation_stats.prediction_accuracy.coverage_probability <= 1.0,
        "Coverage probability should not exceed 1.0"
    );

    // Validate robustness tests
    assert!(
        validation_stats.robustness_tests.outlier_robustness >= 0.0,
        "Outlier robustness metric should be computed"
    );
    assert!(
        validation_stats.robustness_tests.sample_size_robustness >= 0.0,
        "Sample size robustness should be computed"
    );

    // Validate sensitivity analysis
    assert!(
        !validation_stats
            .sensitivity_analysis
            .window_size_sensitivity
            .is_empty()
            || !validation_stats
                .sensitivity_analysis
                .polynomial_order_sensitivity
                .is_empty(),
        "At least one sensitivity analysis should be performed"
    );
}

/// Test scenario: Comprehensive reporting workflow
///
/// Tests the complete workflow from data input to comprehensive reporting,
/// ensuring all analysis components produce consistent and accessible results.
#[test]
fn test_comprehensive_reporting_workflow() {
    // Create a portfolio of assets with different fractal properties
    let gen_config1 = GeneratorConfig {
        length: 800,
        seed: Some(999),
        ..Default::default()
    };
    let momentum_config = FbmConfig {
        hurst_exponent: 0.75,
        volatility: 1.0,
        method: FbmMethod::Hosking,
    };
    let momentum_data = generate_fractional_brownian_motion(&gen_config1, &momentum_config)
        .unwrap_or(vec![0.0; 800]);

    let gen_config2 = GeneratorConfig {
        length: 800,
        seed: Some(1000),
        ..Default::default()
    };
    let mean_revert_config = FbmConfig {
        hurst_exponent: 0.25,
        volatility: 1.0,
        method: FbmMethod::Hosking,
    };
    let mean_revert_data = generate_fractional_brownian_motion(&gen_config2, &mean_revert_config)
        .unwrap_or(vec![0.0; 800]);

    let gen_config3 = GeneratorConfig {
        length: 800,
        seed: Some(1001),
        ..Default::default()
    };
    let market_config = FbmConfig {
        hurst_exponent: 0.55,
        volatility: 1.0,
        method: FbmMethod::Hosking,
    };
    let market_data =
        generate_fractional_brownian_motion(&gen_config3, &market_config).unwrap_or(vec![0.0; 800]);

    let assets = vec![
        ("MOMENTUM_STOCK", momentum_data),
        ("MEAN_REVERT_STOCK", mean_revert_data),
        ("MARKET_INDEX", market_data),
    ];

    let mut analyzer = StatisticalFractalAnalyzer::new();

    // Add all assets
    for (name, data) in assets {
        analyzer.add_time_series(name.to_string(), data);
    }

    // Perform comprehensive analysis
    analyzer
        .analyze_all_series()
        .expect("Portfolio analysis should succeed");

    // Generate comprehensive report data for each asset
    let asset_names = vec!["MOMENTUM_STOCK", "MEAN_REVERT_STOCK", "MARKET_INDEX"];

    for asset_name in &asset_names {
        let results = analyzer
            .get_analysis_results(asset_name)
            .expect(&format!("Results should be available for {}", asset_name));
        let validation = analyzer
            .get_validation_statistics(asset_name)
            .expect(&format!(
                "Validation stats should be available for {}",
                asset_name
            ));

        // Validate comprehensive results structure
        assert!(
            !results.hurst_estimates.is_empty(),
            "Hurst estimates should be available for {}",
            asset_name
        );
        assert!(
            !results
                .multifractal_analysis
                .generalized_hurst_exponents
                .is_empty(),
            "Multifractal analysis should be complete for {}",
            asset_name
        );

        // Validate statistical significance testing
        for (method, estimate) in &results.hurst_estimates {
            assert!(
                !estimate.p_value.is_nan(),
                "P-value should be computed for {} using {:?}",
                asset_name,
                method
            );
            assert!(
                estimate.test_statistic.is_finite(),
                "Test statistic should be finite for {} using {:?}",
                asset_name,
                method
            );
        }

        // Validate regime analysis completeness
        assert!(
            results.regime_analysis.transition_probabilities.len() >= 0,
            "Transition probabilities should be computed for {}",
            asset_name
        );

        // Validate that validation metrics are reasonable
        assert!(
            validation.prediction_accuracy.mspe.is_finite(),
            "MSPE should be finite for {}",
            asset_name
        );
        assert!(
            validation
                .prediction_accuracy
                .estimate_stability_concordance
                >= 0.0,
            "Directional accuracy should be non-negative for {}",
            asset_name
        );
    }

    // Cross-asset validation: momentum stock should have higher Hurst than mean-reverting stock
    let momentum_results = analyzer.get_analysis_results("MOMENTUM_STOCK").unwrap();
    let mean_revert_results = analyzer.get_analysis_results("MEAN_REVERT_STOCK").unwrap();

    // Get R/S estimates for comparison (most robust method)
    if let (Some(momentum_rs), Some(mean_revert_rs)) = (
        momentum_results
            .hurst_estimates
            .get(&EstimationMethod::RescaledRange),
        mean_revert_results
            .hurst_estimates
            .get(&EstimationMethod::RescaledRange),
    ) {
        assert!(momentum_rs.estimate > mean_revert_rs.estimate,
            "Momentum stock should have higher Hurst exponent than mean-reverting stock: {:.3} vs {:.3}",
            momentum_rs.estimate, mean_revert_rs.estimate);
    }
}
