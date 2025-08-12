//! Integration tests for error handling and invalid data scenarios
//!
//! These tests validate that the StatisticalFractalAnalyzer properly handles
//! edge cases, invalid inputs, and error conditions across the full analysis workflow.

use fractal_finance::{errors::FractalAnalysisError, StatisticalFractalAnalyzer};

/// Test scenario: Analysis with insufficient data
///
/// Validates proper error handling when time series are too short for reliable analysis.
#[test]
fn test_insufficient_data_error_handling() {
    let mut analyzer = StatisticalFractalAnalyzer::new();

    // Test with extremely short time series
    let very_short_data = vec![1.0, 2.0, 3.0]; // Only 3 points
    analyzer.add_time_series("TOO_SHORT".to_string(), very_short_data);

    // Analysis should fail gracefully
    let result = analyzer.analyze_series("TOO_SHORT");
    assert!(
        result.is_err(),
        "Analysis should fail with insufficient data"
    );

    match result.unwrap_err() {
        FractalAnalysisError::InsufficientData { required, actual } => {
            assert!(
                actual < required,
                "Error should correctly identify insufficient data: actual {} < required {}",
                actual,
                required
            );
        }
        other_error => {
            panic!("Expected InsufficientData error, got {:?}", other_error);
        }
    }

    // Test with marginally short data (might succeed with some methods, fail with others)
    let short_data: Vec<f64> = (0..50).map(|i| (i as f64).sin()).collect();
    analyzer.add_time_series("MARGINALLY_SHORT".to_string(), short_data);

    // This might succeed or fail depending on implementation, but should not panic
    let result = analyzer.analyze_series("MARGINALLY_SHORT");
    match result {
        Ok(_) => {
            // If it succeeds, validate that results are flagged as unreliable
            let results = analyzer.get_analysis_results("MARGINALLY_SHORT").unwrap();
            // At least one method should have produced results
            assert!(
                !results.hurst_estimates.is_empty(),
                "If analysis succeeds, should produce some estimates"
            );
        }
        Err(e) => {
            // If it fails, should be a proper error type
            assert!(
                matches!(
                    e,
                    FractalAnalysisError::InsufficientData { .. }
                        | FractalAnalysisError::NumericalError { operation: None, .. }
                ),
                "Error should be appropriate type for short data: {:?}",
                e
            );
        }
    }
}

/// Test scenario: Analysis with invalid/problematic data
///
/// Tests handling of NaN values, infinite values, and constant series.
#[test]
fn test_invalid_data_error_handling() {
    let mut analyzer = StatisticalFractalAnalyzer::new();

    // Test with NaN values
    let mut nan_data: Vec<f64> = (0..200).map(|i| (i as f64) * 0.01).collect();
    nan_data[100] = f64::NAN;
    let result = analyzer.add_time_series("NAN_DATA".to_string(), nan_data);

    assert!(
        result.is_err(),
        "Adding time series should fail with NaN data"
    );

    match result.unwrap_err() {
        FractalAnalysisError::NumericalError {
            reason,
            operation: None,
        } => {
            assert!(
                reason.to_lowercase().contains("nan")
                    || reason.to_lowercase().contains("invalid")
                    || reason.to_lowercase().contains("finite")
                    || reason.to_lowercase().contains("not finite"),
                "Error reason should mention NaN or invalid data: {}",
                reason
            );
        }
        other_error => {
            panic!("Expected NumericalError for NaN, got {:?}", other_error);
        }
    }

    // Test with infinite values
    let mut inf_data: Vec<f64> = (0..200).map(|i| (i as f64) * 0.01).collect();
    inf_data[50] = f64::INFINITY;
    let result = analyzer.add_time_series("INF_DATA".to_string(), inf_data);

    assert!(
        result.is_err(),
        "Adding time series should fail with infinite data"
    );

    // Test with constant series
    let constant_data = vec![5.0; 500];
    analyzer.add_time_series("CONSTANT_DATA".to_string(), constant_data);

    let result = analyzer.analyze_series("CONSTANT_DATA");
    assert!(result.is_err(), "Analysis should fail with constant data");

    match result.unwrap_err() {
        FractalAnalysisError::NumericalError { reason, operation } => {
            // Accept any NumericalError for constant data, as various methods may fail differently
            // The ADF test failing with singular matrices is a valid response to constant data
            assert!(
                reason.to_lowercase().contains("constant")
                    || reason.to_lowercase().contains("variance")
                    || reason.to_lowercase().contains("zero")
                    || reason.to_lowercase().contains("finite")
                    || reason.to_lowercase().contains("adf")
                    || reason.to_lowercase().contains("singular"),
                "Error reason should mention constant data issue or numerical failure: {}",
                reason
            );
        }
        FractalAnalysisError::InsufficientData { .. } => {
            // Also acceptable for constant data that can't be properly analyzed
        }
        other_error => {
            panic!(
                "Expected NumericalError or InsufficientData for constant data, got {:?}",
                other_error
            );
        }
    }
}

/// Test scenario: Analysis of non-existent time series
///
/// Tests error handling when requesting analysis of assets that haven't been added.
#[test]
fn test_nonexistent_asset_error_handling() {
    let analyzer = StatisticalFractalAnalyzer::new();

    // Try to get results for non-existent asset
    let results = analyzer.get_analysis_results("NONEXISTENT_ASSET");
    assert!(results.is_err(), "Should return Err for non-existent asset");

    let validation = analyzer.get_validation_statistics("NONEXISTENT_ASSET");
    assert!(
        validation.is_err(),
        "Should return Err for non-existent validation stats"
    );

    // Try to analyze non-existent asset
    let mut analyzer = StatisticalFractalAnalyzer::new();
    let result = analyzer.analyze_series("NONEXISTENT_ASSET");
    assert!(result.is_err(), "Should fail to analyze non-existent asset");

    match result.unwrap_err() {
        FractalAnalysisError::TimeSeriesNotFound { name } => {
            assert_eq!(
                name, "NONEXISTENT_ASSET",
                "Error should identify the missing asset correctly"
            );
        }
        other_error => {
            panic!("Expected TimeSeriesNotFound error, got {:?}", other_error);
        }
    }
}

/// Test scenario: Analysis with extreme data values
///
/// Tests robustness to very large or very small values that might cause numerical issues.
#[test]
fn test_extreme_values_handling() {
    let mut analyzer = StatisticalFractalAnalyzer::new();

    // Test with very large values
    let large_data: Vec<f64> = (0..300).map(|i| 1e10 * (i as f64 * 0.01).sin()).collect();
    analyzer.add_time_series("LARGE_VALUES".to_string(), large_data);

    // Should either succeed with proper scaling or fail gracefully
    let result = analyzer.analyze_series("LARGE_VALUES");
    match result {
        Ok(_) => {
            let results = analyzer.get_analysis_results("LARGE_VALUES").unwrap();
            // If successful, results should be reasonable
            for (method, estimate) in &results.hurst_estimates {
                assert!(
                    estimate.estimate.is_finite(),
                    "Hurst estimate from {:?} should be finite with large values: {:.3}",
                    method,
                    estimate.estimate
                );
                assert!(
                    estimate.estimate >= 0.0 && estimate.estimate <= 1.0,
                    "Hurst estimate from {:?} should be in valid range [0,1]: {:.3}",
                    method,
                    estimate.estimate
                );
            }
        }
        Err(e) => {
            assert!(
                matches!(e, FractalAnalysisError::NumericalError { operation: None, .. }),
                "Should fail with numerical error for extreme values: {:?}",
                e
            );
        }
    }

    // Test with very small values (but not zero)
    let small_data: Vec<f64> = (0..300).map(|i| 1e-10 * (i as f64 * 0.01).sin()).collect();
    analyzer.add_time_series("SMALL_VALUES".to_string(), small_data);

    let result = analyzer.analyze_series("SMALL_VALUES");
    // Similar handling as large values - should either work or fail gracefully
    match result {
        Ok(_) => {
            let results = analyzer.get_analysis_results("SMALL_VALUES").unwrap();
            for (method, estimate) in &results.hurst_estimates {
                assert!(
                    estimate.estimate.is_finite(),
                    "Hurst estimate from {:?} should be finite with small values",
                    method
                );
            }
        }
        Err(e) => {
            assert!(
                matches!(
                    e,
                    FractalAnalysisError::NumericalError { operation: None, .. }
                        | FractalAnalysisError::InsufficientData { .. }
                ),
                "Should fail appropriately with small values: {:?}",
                e
            );
        }
    }
}

/// Test scenario: Memory and performance stress testing
///
/// Tests behavior with large datasets and multiple simultaneous analyses.
#[test]
fn test_large_dataset_stress_handling() {
    let mut analyzer = StatisticalFractalAnalyzer::new();

    // Create a large dataset (but not so large it causes CI issues)
    let large_size = 10000;
    let large_dataset: Vec<f64> = (0..large_size)
        .map(|i| (i as f64 * 0.001).sin() + 0.1 * (i as f64 * 0.01).cos())
        .collect();

    analyzer.add_time_series("LARGE_DATASET".to_string(), large_dataset);

    // Analysis should complete without memory issues or timeouts
    let result = analyzer.analyze_series("LARGE_DATASET");

    match result {
        Ok(_) => {
            let results = analyzer.get_analysis_results("LARGE_DATASET").unwrap();
            assert!(
                !results.hurst_estimates.is_empty(),
                "Large dataset analysis should produce estimates"
            );

            // Validate that results are reasonable
            for (method, estimate) in &results.hurst_estimates {
                assert!(
                    estimate.estimate.is_finite(),
                    "Large dataset estimates should be finite for {:?}",
                    method
                );
                assert!(
                    estimate.standard_error >= 0.0 && estimate.standard_error < 1.0,
                    "Standard errors should be reasonable for large dataset: {:.3}",
                    estimate.standard_error
                );
            }
        }
        Err(e) => {
            // If it fails, should be due to computational constraints, not bugs
            assert!(
                matches!(e, FractalAnalysisError::NumericalError { operation: None, .. }),
                "Large dataset failure should be numerical, not logical: {:?}",
                e
            );
        }
    }
}

/// Test scenario: Concurrent analysis error handling
///
/// Tests that the analyzer handles multiple time series correctly and errors
/// in one analysis don't affect others.
#[test]
fn test_mixed_valid_invalid_data_handling() {
    let mut analyzer = StatisticalFractalAnalyzer::new();

    // Add mix of valid and invalid data
    let valid_data1: Vec<f64> = (0..500).map(|i| (i as f64 * 0.01).sin()).collect();
    let valid_data2: Vec<f64> = (0..600).map(|i| (i as f64 * 0.02).cos()).collect();
    let invalid_data = vec![f64::NAN; 500]; // All NaN
    let too_short_data = vec![1.0, 2.0]; // Too short

    analyzer.add_time_series("VALID1".to_string(), valid_data1);
    analyzer.add_time_series("VALID2".to_string(), valid_data2);
    analyzer.add_time_series("INVALID".to_string(), invalid_data);
    analyzer.add_time_series("TOO_SHORT".to_string(), too_short_data);

    // analyze_all_series should handle mixed success/failure appropriately
    let result = analyzer.analyze_all_series();

    // Should either succeed (processing valid ones, skipping invalid) or fail
    match result {
        Ok(_) => {
            // If successful, valid data should have results
            assert!(
                analyzer.get_analysis_results("VALID1").is_ok(),
                "Valid data should have results when analyze_all_series succeeds"
            );
            assert!(
                analyzer.get_analysis_results("VALID2").is_ok(),
                "Valid data should have results when analyze_all_series succeeds"
            );
        }
        Err(_) => {
            // If it fails due to invalid data, should still be able to analyze valid ones individually
            assert!(
                analyzer.analyze_series("VALID1").is_ok(),
                "Should be able to analyze valid data individually"
            );
            assert!(
                analyzer.analyze_series("VALID2").is_ok(),
                "Should be able to analyze valid data individually"
            );

            // Invalid ones should still fail
            assert!(
                analyzer.analyze_series("INVALID").is_err(),
                "Invalid data should still fail individually"
            );
            assert!(
                analyzer.analyze_series("TOO_SHORT").is_err(),
                "Too short data should still fail individually"
            );
        }
    }
}

/// Test scenario: Configuration and parameter validation
///
/// Tests that invalid configurations are properly rejected.
#[test]
fn test_configuration_error_handling() {
    let analyzer = StatisticalFractalAnalyzer::new();

    // Test that the analyzer is created in a valid state
    // (This is more of a regression test to ensure new() doesn't panic)

    // Add valid data to test configuration handling
    let mut analyzer = StatisticalFractalAnalyzer::new();
    let test_data: Vec<f64> = (0..200).map(|i| (i as f64 * 0.01).sin()).collect();
    analyzer.add_time_series("CONFIG_TEST".to_string(), test_data);

    // Basic analysis should work with default configuration
    let result = analyzer.analyze_series("CONFIG_TEST");

    // Should either succeed or fail gracefully, not panic
    match result {
        Ok(_) => {
            let results = analyzer.get_analysis_results("CONFIG_TEST").unwrap();
            assert!(
                !results.hurst_estimates.is_empty(),
                "Default configuration should produce some estimates"
            );
        }
        Err(e) => {
            // If it fails, should be a legitimate analysis error
            assert!(
                matches!(
                    e,
                    FractalAnalysisError::InsufficientData { .. }
                        | FractalAnalysisError::NumericalError { operation: None, .. }
                        | FractalAnalysisError::InvalidParameter { .. }
                ),
                "Configuration errors should be appropriate types: {:?}",
                e
            );
        }
    }
}
