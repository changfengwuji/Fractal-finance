//! Bootstrap validation and confidence interval methods.
//!
//! This module provides comprehensive bootstrap resampling methods for statistical
//! validation of fractal analysis estimators. It includes various bootstrap methods
//! and confidence interval construction techniques specifically designed for
//! time series with potential long-range dependence.

// Re-export configuration types
pub use crate::bootstrap_config::{
    BootstrapConfiguration, BootstrapMethod, BootstrapValidation, 
    ConfidenceInterval, ConfidenceIntervalMethod, EstimatorComplexity,
};

// Re-export block size functions
pub use crate::block_size::{
    politis_white_block_size, multivariate_politis_white_block_size,
};

// Re-export confidence interval functions
pub use crate::confidence_intervals::{
    calculate_bootstrap_confidence_interval, calculate_normal_confidence_interval,
    calculate_bca_confidence_interval,
};

// Re-export sampling functions
pub use crate::bootstrap_sampling::{
    generate_bootstrap_sample, generate_bootstrap_sample_inplace,
    generate_bootstrap_sample_inplace_with_rng, mix_seed,
};

use crate::{
    errors::{validate_all_finite, validate_data_length, FractalAnalysisError, FractalResult},
    memory_pool::{get_f64_buffer, return_f64_buffer},
    secure_rng::SecureRng,
    // Internal imports for functions not publicly re-exported
    bootstrap_config::{
        validate_bootstrap_config, MIN_DATA_POINTS, MIN_BOOTSTRAP_SAMPLES,
        MAX_BOOTSTRAP_SAMPLES_DEFAULT,
    },
    bootstrap_sampling::{
        standard_bootstrap_sample_inplace_with_rng, block_bootstrap_sample_inplace_with_rng,
        stationary_bootstrap_sample_inplace_with_rng, circular_bootstrap_sample_inplace_with_rng,
    },
    confidence_intervals::calculate_studentized_confidence_interval,
};
use std::collections::HashMap;

/// Multivariate block bootstrap for paired time series.
///
/// Performs block bootstrap on bivariate time series, preserving both
/// temporal dependence and cross-sectional relationships. Uses an optimal
/// block size that accounts for dependencies in both series.
///
/// # Arguments
/// * `data` - Paired time series data as (series1, series2) tuples
/// * `estimator` - Function that computes a statistic from paired data
/// * `config` - Bootstrap configuration
///
/// # Returns
/// Bootstrap validation results including confidence intervals
///
/// # Examples
/// ```rust,ignore
/// let paired_data = vec![(1.0, 2.0), (1.5, 2.5), (2.0, 3.0)];
/// let estimator = |data: &[(f64, f64)]| {
///     // Compute correlation coefficient or other paired statistic
///     let n = data.len() as f64;
///     let sum_xy: f64 = data.iter().map(|(x, y)| x * y).sum();
///     sum_xy / n
/// };
/// let config = BootstrapConfiguration::default();
/// let results = multivariate_block_bootstrap(&paired_data, estimator, &config)?;
/// ```
pub fn multivariate_block_bootstrap<F>(
    data: &[(f64, f64)],
    estimator: F,
    config: &BootstrapConfiguration,
) -> FractalResult<BootstrapValidation>
where
    F: Fn(&[(f64, f64)]) -> f64 + Send + Sync,
{
    // Validate data
    if data.len() < MIN_DATA_POINTS {
        return Err(FractalAnalysisError::InsufficientData {
            required: MIN_DATA_POINTS,
            actual: data.len(),
        });
    }

    // Validate all data points are finite
    for (i, (x, y)) in data.iter().enumerate() {
        if !x.is_finite() || !y.is_finite() {
            return Err(FractalAnalysisError::NumericalError {
                reason: format!("Non-finite value at index {}: {}", i, 
                    if !x.is_finite() { *x } else { *y }),
                operation: Some("multivariate_block_bootstrap".to_string()),
            });
        }
    }

    // Validate configuration
    validate_bootstrap_config(config)?;

    // Calculate optimal block size for multivariate data
    let block_size = config
        .block_size
        .unwrap_or_else(|| multivariate_politis_white_block_size(data));

    // Original estimate
    let original_estimate = estimator(data);

    // Generate bootstrap samples
    let mut bootstrap_estimates = Vec::with_capacity(config.num_bootstrap_samples);
    let n = data.len();

    // Use seeded RNG if provided for reproducibility
    let mut rng = if let Some(seed) = config.seed {
        SecureRng::with_seed(seed)
    } else {
        SecureRng::new()
    };

    for i in 0..config.num_bootstrap_samples {
        // Mix seed with iteration index for decorrelated streams
        if let Some(seed) = config.seed {
            rng = SecureRng::with_seed(mix_seed(seed, i));
        }

        // Generate bootstrap sample using block bootstrap
        let mut bootstrap_sample = Vec::with_capacity(n);

        while bootstrap_sample.len() < n {
            let start = rng.usize(0..n.saturating_sub(block_size).saturating_add(1));
            let end = (start + block_size).min(n);

            for j in start..end {
                if bootstrap_sample.len() < n {
                    bootstrap_sample.push(data[j]);
                }
            }
        }

        bootstrap_sample.truncate(n);

        // Calculate bootstrap estimate
        let estimate = estimator(&bootstrap_sample);
        if estimate.is_finite() {
            bootstrap_estimates.push(estimate);
        }
    }

    if bootstrap_estimates.is_empty() {
        return Err(FractalAnalysisError::BootstrapError {
            reason: "No valid bootstrap estimates generated".to_string(),
        });
    }

    // Calculate bootstrap statistics
    let mean = bootstrap_estimates.iter().sum::<f64>() / bootstrap_estimates.len() as f64;
    let bias = mean - original_estimate;

    let variance = bootstrap_estimates
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / (bootstrap_estimates.len() - 1) as f64;
    let standard_error = variance.sqrt();

    // Calculate confidence intervals
    let mut confidence_intervals = Vec::new();

    for &confidence_level in &config.confidence_levels {
        let ci = match config.confidence_interval_method {
            ConfidenceIntervalMethod::Normal => {
                calculate_normal_confidence_interval(original_estimate, standard_error, confidence_level)?
            }
            ConfidenceIntervalMethod::BootstrapPercentile => {
                calculate_bootstrap_confidence_interval(&bootstrap_estimates, original_estimate, confidence_level)?
            }
            ConfidenceIntervalMethod::BootstrapBca => {
                // For BCa, we need to convert back to univariate for jackknife
                // This is a limitation - BCa is not fully defined for multivariate estimators
                // Fall back to percentile method
                calculate_bootstrap_confidence_interval(&bootstrap_estimates, original_estimate, confidence_level)?
            }
            ConfidenceIntervalMethod::StudentizedBootstrap => {
                // Studentized bootstrap for multivariate is complex, fall back to percentile
                calculate_bootstrap_confidence_interval(&bootstrap_estimates, original_estimate, confidence_level)?
            }
        };
        confidence_intervals.push(ci);
    }

    Ok(BootstrapValidation {
        original_estimate,
        bootstrap_estimates,
        bias,
        standard_error,
        confidence_intervals,
    })
}

/// Bootstrap validation for paired data with correlation preservation.
///
/// Specialized bootstrap validation for analyzing relationships between
/// two time series, ensuring that both temporal and cross-sectional
/// dependencies are preserved during resampling.
///
/// # Arguments
/// * `x` - First time series
/// * `y` - Second time series  
/// * `estimator` - Function computing a statistic from paired data
/// * `config` - Bootstrap configuration
///
/// # Returns
/// Bootstrap validation results
///
/// # Examples
/// ```rust,ignore
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
/// let estimator = |data: &[(f64, f64)]| {
///     // Compute correlation or regression coefficient
///     correlation_coefficient(data)
/// };
/// let results = bootstrap_validate_pairs(&x, &y, estimator, &config)?;
/// ```
pub fn bootstrap_validate_pairs<F>(
    x: &[f64],
    y: &[f64],
    estimator: F,
    config: &BootstrapConfiguration,
) -> FractalResult<BootstrapValidation>
where
    F: Fn(&[(f64, f64)]) -> f64 + Send + Sync,
{
    // Validate inputs
    if x.len() != y.len() {
        return Err(FractalAnalysisError::InvalidParameter {
            parameter: "data dimensions".to_string(),
            value: y.len() as f64,
            constraint: format!("Must match first series length {}", x.len()),
        });
    }

    // Create paired data
    let paired_data: Vec<(f64, f64)> = x.iter().zip(y.iter()).map(|(&a, &b)| (a, b)).collect();

    // Use multivariate block bootstrap
    multivariate_block_bootstrap(&paired_data, estimator, config)
}

/// Core bootstrap validation function for univariate time series.
///
/// Performs comprehensive bootstrap validation of an estimator, including
/// bias estimation, standard error calculation, and confidence interval
/// construction using various methods.
///
/// # Arguments
/// * `data` - Time series data
/// * `estimator` - Function that computes a statistic from data
/// * `config` - Bootstrap configuration
///
/// # Returns
/// Bootstrap validation results including confidence intervals
///
/// # Mathematical Foundation
///
/// The bootstrap principle approximates the sampling distribution of an
/// estimator by resampling from the empirical distribution. For a statistic
/// T(X₁,...,Xₙ), the bootstrap distribution is:
///
/// T*(X₁*,...,Xₙ*) where X* ~ F̂ₙ (empirical CDF)
///
/// This provides estimates of:
/// - Bias: E[T*] - T
/// - Standard Error: SD[T*]
/// - Confidence Intervals: Quantiles of T*
///
/// # Examples
/// ```rust,ignore
/// use fractal_analysis::{bootstrap_validate, BootstrapConfiguration};
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let mean_estimator = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
/// let config = BootstrapConfiguration::default();
///
/// let results = bootstrap_validate(&data, mean_estimator, &config)?;
/// println!("Estimate: {} ± {}", results.original_estimate, results.standard_error);
/// ```
pub fn bootstrap_validate<F>(
    data: &[f64],
    estimator: F,
    config: &BootstrapConfiguration,
) -> FractalResult<BootstrapValidation>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    // Validate data
    validate_data_length(data, MIN_DATA_POINTS, "bootstrap_validate")?;
    validate_all_finite(data, "input data")?;
    validate_bootstrap_config(config)?;

    // Original estimate
    let original_estimate = estimator(data);
    if !original_estimate.is_finite() {
        return Err(FractalAnalysisError::NumericalError {
            reason: "Non-finite original estimate".to_string(),
            operation: Some("bootstrap_validate".to_string()),
        });
    }

    // Pre-compute block size if using block methods
    let block_size = if matches!(
        config.bootstrap_method,
        BootstrapMethod::Block | BootstrapMethod::Stationary
    ) {
        config
            .block_size
            .unwrap_or_else(|| politis_white_block_size(data))
    } else {
        0 // Not used for other methods
    };

    // Generate bootstrap estimates
    let mut bootstrap_estimates = Vec::with_capacity(config.num_bootstrap_samples);

    // Get a reusable buffer for in-place sampling
    let mut buffer = get_f64_buffer(data.len())?;
    // Ensure buffer has the correct length, not just capacity
    buffer.resize(data.len(), 0.0);

    // Initialize RNG
    let mut rng = if let Some(seed) = config.seed {
        SecureRng::with_seed(seed)
    } else {
        SecureRng::new()
    };

    for i in 0..config.num_bootstrap_samples {
        // Mix seed with iteration for decorrelated streams
        if let Some(seed) = config.seed {
            rng = SecureRng::with_seed(mix_seed(seed, i));
        }

        // Generate bootstrap sample in-place
        match config.bootstrap_method {
            BootstrapMethod::Standard => {
                standard_bootstrap_sample_inplace_with_rng(data, &mut buffer, &mut rng);
            }
            BootstrapMethod::Block => {
                block_bootstrap_sample_inplace_with_rng(data, &mut buffer, block_size, &mut rng);
            }
            BootstrapMethod::Stationary => {
                stationary_bootstrap_sample_inplace_with_rng(data, &mut buffer, block_size, &mut rng);
            }
            BootstrapMethod::Circular => {
                circular_bootstrap_sample_inplace_with_rng(data, &mut buffer, &mut rng);
            }
        }

        // Calculate bootstrap estimate
        let estimate = estimator(&buffer);
        if estimate.is_finite() {
            bootstrap_estimates.push(estimate);
        } else {
            // Debug: this shouldn't happen with valid data and estimator
            eprintln!("WARNING: Non-finite bootstrap estimate: {} from buffer len {}", estimate, buffer.len());
        }
    }

    // Return buffer to pool
    return_f64_buffer(buffer);

    if bootstrap_estimates.is_empty() {
        return Err(FractalAnalysisError::BootstrapError {
            reason: "No valid bootstrap estimates generated".to_string(),
        });
    }

    // Calculate bootstrap statistics
    let mean = bootstrap_estimates.iter().sum::<f64>() / bootstrap_estimates.len() as f64;
    let bias = mean - original_estimate;

    let variance = bootstrap_estimates
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / (bootstrap_estimates.len() - 1) as f64;
    let standard_error = variance.sqrt();

    // Calculate confidence intervals
    let mut confidence_intervals = Vec::new();

    for &confidence_level in &config.confidence_levels {
        let ci = match config.confidence_interval_method {
            ConfidenceIntervalMethod::Normal => {
                calculate_normal_confidence_interval(original_estimate, standard_error, confidence_level)?
            }
            ConfidenceIntervalMethod::BootstrapPercentile => {
                calculate_bootstrap_confidence_interval(&bootstrap_estimates, original_estimate, confidence_level)?
            }
            ConfidenceIntervalMethod::BootstrapBca => {
                calculate_bca_confidence_interval(
                    data,
                    &bootstrap_estimates,
                    original_estimate,
                    confidence_level,
                    &estimator,
                    config,
                )?
            }
            ConfidenceIntervalMethod::StudentizedBootstrap => {
                calculate_studentized_confidence_interval(
                    data,
                    &bootstrap_estimates,
                    original_estimate,
                    confidence_level,
                    &estimator,
                    config,
                )?
            }
        };
        confidence_intervals.push(ci);
    }

    Ok(BootstrapValidation {
        original_estimate,
        bootstrap_estimates,
        bias,
        standard_error,
        confidence_intervals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_configuration_defaults() {
        let config = BootstrapConfiguration::default();
        assert_eq!(config.num_bootstrap_samples, 1000);
        assert_eq!(config.confidence_levels, vec![0.90, 0.95, 0.99]);
        assert_eq!(
            config.confidence_interval_method,
            ConfidenceIntervalMethod::BootstrapBca
        );
    }

    #[test]
    fn test_bootstrap_sample_generation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = BootstrapConfiguration::default();

        let sample = generate_bootstrap_sample(&data, &config).unwrap();
        assert_eq!(sample.len(), data.len());

        // All values should come from original data
        for value in &sample {
            assert!(data.contains(value));
        }
    }

    #[test]
    fn test_bootstrap_validation() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mean_estimator = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;

        let config = BootstrapConfiguration {
            num_bootstrap_samples: 100,
            bootstrap_method: BootstrapMethod::Standard,
            block_size: None,
            confidence_levels: vec![0.95],
            confidence_interval_method: ConfidenceIntervalMethod::BootstrapPercentile,
            seed: Some(42),
            studentized_outer: None,
            studentized_inner: None,
            jackknife_block_size: None,
            force_block_jackknife: None,
        };

        let result = bootstrap_validate(&data, mean_estimator, &config).unwrap();

        // Check basic properties
        assert!((result.original_estimate - 49.5).abs() < 0.01);
        assert_eq!(result.bootstrap_estimates.len(), 100);
        assert_eq!(result.confidence_intervals.len(), 1);
    }

    #[test]
    fn test_bootstrap_reproducibility() {
        let data: Vec<f64> = (0..50).map(|i| (i as f64).sin()).collect();
        let estimator = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;

        let config = BootstrapConfiguration {
            num_bootstrap_samples: 100,
            bootstrap_method: BootstrapMethod::Standard,
            block_size: None,
            confidence_levels: vec![0.95],
            confidence_interval_method: ConfidenceIntervalMethod::BootstrapPercentile,
            seed: Some(12345),
            studentized_outer: None,
            studentized_inner: None,
            jackknife_block_size: None,
            force_block_jackknife: None,
        };

        let result1 = bootstrap_validate(&data, estimator, &config).unwrap();
        let result2 = bootstrap_validate(&data, estimator, &config).unwrap();

        // Results should be identical with same seed
        assert_eq!(result1.bootstrap_estimates, result2.bootstrap_estimates);
        assert_eq!(result1.bias, result2.bias);
        assert_eq!(result1.standard_error, result2.standard_error);
    }

    #[test]
    fn test_confidence_interval_calculation() {
        let bootstrap_estimates = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let original_estimate = 5.5;

        let ci = calculate_bootstrap_confidence_interval(&bootstrap_estimates, original_estimate, 0.80)
            .unwrap();

        assert_eq!(ci.confidence_level, 0.80);
        assert!(ci.lower_bound < ci.upper_bound);
        assert!(ci.lower_bound >= 1.0);
        assert!(ci.upper_bound <= 10.0);
    }

    #[test]
    fn test_block_bootstrap() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let estimator = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;

        let config = BootstrapConfiguration {
            num_bootstrap_samples: 100,
            bootstrap_method: BootstrapMethod::Block,
            block_size: Some(10),
            confidence_levels: vec![0.95],
            confidence_interval_method: ConfidenceIntervalMethod::BootstrapPercentile,
            seed: Some(42),
            studentized_outer: None,
            studentized_inner: None,
            jackknife_block_size: None,
            force_block_jackknife: None,
        };

        let result = bootstrap_validate(&data, estimator, &config).unwrap();
        assert_eq!(result.bootstrap_estimates.len(), 100);
    }

    #[test]
    fn test_different_bootstrap_methods() {
        let data: Vec<f64> = (0..50).map(|i| (i as f64).cos()).collect();
        let estimator = |x: &[f64]| x[0]; // First element

        let methods = vec![
            BootstrapMethod::Standard,
            BootstrapMethod::Block,
            BootstrapMethod::Stationary,
            BootstrapMethod::Circular,
        ];

        for method in methods {
            let config = BootstrapConfiguration {
                num_bootstrap_samples: 100,
                bootstrap_method: method.clone(),
                block_size: Some(5),
                confidence_levels: vec![0.95],
                confidence_interval_method: ConfidenceIntervalMethod::BootstrapPercentile,
                seed: Some(123),
                studentized_outer: None,
                studentized_inner: None,
                jackknife_block_size: None,
                force_block_jackknife: None,
            };

            let result = bootstrap_validate(&data, estimator, &config);
            if let Err(e) = &result {
                eprintln!("Bootstrap method {:?} failed with error: {:?}", method, e);
            }
            assert!(result.is_ok(), "Method {:?} failed", method);
        }
    }

    #[test]
    fn test_different_confidence_interval_methods() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let estimator = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;

        let ci_methods = vec![
            ConfidenceIntervalMethod::Normal,
            ConfidenceIntervalMethod::BootstrapPercentile,
            ConfidenceIntervalMethod::BootstrapBca,
        ];

        for ci_method in ci_methods {
            let config = BootstrapConfiguration {
                num_bootstrap_samples: 200,
                bootstrap_method: BootstrapMethod::Standard,
                block_size: None,
                confidence_levels: vec![0.95],
                confidence_interval_method: ci_method.clone(),
                seed: Some(42),
                studentized_outer: None,
                studentized_inner: None,
                jackknife_block_size: None,
                force_block_jackknife: None,
            };

            let result = bootstrap_validate(&data, estimator, &config);
            assert!(result.is_ok(), "CI method {:?} failed", ci_method);
        }
    }

    #[test]
    fn test_error_conditions() {
        // Test with too little data
        let small_data = vec![1.0, 2.0];
        let estimator = |x: &[f64]| x[0];
        let config = BootstrapConfiguration::default();

        let result = bootstrap_validate(&small_data, estimator, &config);
        assert!(result.is_err());

        // Test with non-finite values
        let bad_data = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
        let result = bootstrap_validate(&bad_data, estimator, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_bootstrap_mean_normal_data() {
        // Test bootstrap on normally distributed data
        let data: Vec<f64> = (0..200).map(|i| (i as f64 - 100.0) / 50.0).collect();
        let mean_estimator = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;

        let config = BootstrapConfiguration {
            num_bootstrap_samples: 500,
            bootstrap_method: BootstrapMethod::Standard,
            block_size: None,
            confidence_levels: vec![0.90, 0.95, 0.99],
            confidence_interval_method: ConfidenceIntervalMethod::BootstrapPercentile,
            seed: Some(42),
            studentized_outer: None,
            studentized_inner: None,
            jackknife_block_size: None,
            force_block_jackknife: None,
        };

        let result = bootstrap_validate(&data, mean_estimator, &config).unwrap();

        // Mean should be close to 0
        assert!((result.original_estimate - 0.0).abs() < 0.1);

        // Check confidence intervals are nested
        assert_eq!(result.confidence_intervals.len(), 3);
        let ci_90 = &result.confidence_intervals[0];
        let ci_95 = &result.confidence_intervals[1];
        let ci_99 = &result.confidence_intervals[2];

        assert!(ci_90.lower_bound >= ci_95.lower_bound);
        assert!(ci_90.upper_bound <= ci_95.upper_bound);
        assert!(ci_95.lower_bound >= ci_99.lower_bound);
        assert!(ci_95.upper_bound <= ci_99.upper_bound);
    }

    #[test]
    fn test_bootstrap_variance_estimation() {
        // Generate data with known variance
        let data: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 10.0).collect();
        let var_estimator = |x: &[f64]| {
            let mean = x.iter().sum::<f64>() / x.len() as f64;
            x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (x.len() - 1) as f64
        };

        let config = BootstrapConfiguration {
            num_bootstrap_samples: 200,
            bootstrap_method: BootstrapMethod::Standard,
            block_size: None,
            confidence_levels: vec![0.95],
            confidence_interval_method: ConfidenceIntervalMethod::BootstrapPercentile,
            seed: Some(999),
            studentized_outer: None,
            studentized_inner: None,
            jackknife_block_size: None,
            force_block_jackknife: None,
        };

        let result = bootstrap_validate(&data, var_estimator, &config).unwrap();

        // Theoretical variance should be around 8.5
        assert!(result.original_estimate > 7.0 && result.original_estimate < 10.0);
        assert!(result.standard_error > 0.0);
    }

    #[test]
    fn test_bootstrap_with_custom_estimator() {
        // Test with median estimator
        let data: Vec<f64> = (0..101).map(|i| i as f64).collect();
        let median_estimator = |x: &[f64]| {
            let mut sorted = x.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };

        let config = BootstrapConfiguration {
            num_bootstrap_samples: 100,
            bootstrap_method: BootstrapMethod::Standard,
            block_size: None,
            confidence_levels: vec![0.95],
            confidence_interval_method: ConfidenceIntervalMethod::BootstrapPercentile,
            seed: Some(42),
            studentized_outer: None,
            studentized_inner: None,
            jackknife_block_size: None,
            force_block_jackknife: None,
        };

        let result = bootstrap_validate(&data, median_estimator, &config).unwrap();

        // Median should be 50
        assert_eq!(result.original_estimate, 50.0);
        assert_eq!(result.confidence_intervals.len(), 1);
    }

    #[test]
    fn test_bootstrap_bias_correction() {
        // Create biased estimator (sample variance without Bessel's correction)
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let biased_var = |x: &[f64]| {
            let mean = x.iter().sum::<f64>() / x.len() as f64;
            x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64 // Biased
        };

        let config = BootstrapConfiguration {
            num_bootstrap_samples: 200,
            bootstrap_method: BootstrapMethod::Standard,
            block_size: None,
            confidence_levels: vec![0.95],
            confidence_interval_method: ConfidenceIntervalMethod::BootstrapPercentile,
            seed: Some(123),
            studentized_outer: None,
            studentized_inner: None,
            jackknife_block_size: None,
            force_block_jackknife: None,
        };

        let result = bootstrap_validate(&data, biased_var, &config).unwrap();

        // Bootstrap should detect the bias
        assert!(result.bias.abs() > 0.0);
    }

    #[test]
    fn test_normal_approximation_ci() {
        let estimate = 10.0;
        let standard_error = 2.0;

        let ci = calculate_normal_confidence_interval(estimate, standard_error, 0.95).unwrap();

        // For 95% CI with SE=2, the margin should be approximately 1.96 * 2 = 3.92
        let expected_margin = 1.96 * standard_error;
        assert!((ci.lower_bound - (estimate - expected_margin)).abs() < 0.1);
        assert!((ci.upper_bound - (estimate + expected_margin)).abs() < 0.1);
    }

    #[test]
    fn test_bca_confidence_intervals() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let estimator = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;

        let config = BootstrapConfiguration {
            num_bootstrap_samples: 200,
            bootstrap_method: BootstrapMethod::Standard,
            block_size: None,
            confidence_levels: vec![0.95],
            confidence_interval_method: ConfidenceIntervalMethod::BootstrapBca,
            seed: Some(42),
            studentized_outer: None,
            studentized_inner: None,
            jackknife_block_size: None,
            force_block_jackknife: None,
        };

        let result = bootstrap_validate(&data, estimator, &config).unwrap();
        assert_eq!(result.confidence_intervals.len(), 1);

        let ci = &result.confidence_intervals[0];
        assert!(ci.lower_bound < result.original_estimate);
        assert!(ci.upper_bound > result.original_estimate);
    }

    #[test]
    fn test_numerical_stability() {
        // Test with very small values
        let small_data: Vec<f64> = (0..50).map(|i| (i as f64) * 1e-10).collect();
        let estimator = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;

        let config = BootstrapConfiguration {
            num_bootstrap_samples: 100,
            bootstrap_method: BootstrapMethod::Standard,
            block_size: None,
            confidence_levels: vec![0.95],
            confidence_interval_method: ConfidenceIntervalMethod::BootstrapPercentile,
            seed: Some(42),
            studentized_outer: None,
            studentized_inner: None,
            jackknife_block_size: None,
            force_block_jackknife: None,
        };

        let result = bootstrap_validate(&small_data, estimator, &config);
        assert!(result.is_ok());

        // Test with very large values
        let large_data: Vec<f64> = (0..50).map(|i| (i as f64) * 1e10).collect();
        let result = bootstrap_validate(&large_data, estimator, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_coverage_probability_simulation() {
        // Simple test - not a full coverage study
        let true_mean = 5.0;
        let data: Vec<f64> = (0..100)
            .map(|i| true_mean + (i as f64 - 50.0) / 20.0)
            .collect();
        let mean_estimator = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;

        let config = BootstrapConfiguration {
            num_bootstrap_samples: 200,
            bootstrap_method: BootstrapMethod::Standard,
            block_size: None,
            confidence_levels: vec![0.95],
            confidence_interval_method: ConfidenceIntervalMethod::BootstrapPercentile,
            seed: Some(42),
            studentized_outer: None,
            studentized_inner: None,
            jackknife_block_size: None,
            force_block_jackknife: None,
        };

        let result = bootstrap_validate(&data, mean_estimator, &config).unwrap();
        let ci = &result.confidence_intervals[0];

        // The true mean should be within the confidence interval
        assert!(ci.lower_bound <= true_mean && ci.upper_bound >= true_mean);
    }

    #[test]
    fn test_end_to_end_reproducibility_with_seed() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).cos()).collect();
        let estimator = |x: &[f64]| {
            let mean = x.iter().sum::<f64>() / x.len() as f64;
            x.iter().map(|&v| (v - mean).abs()).sum::<f64>() / x.len() as f64
        };

        let config = BootstrapConfiguration {
            num_bootstrap_samples: 200,
            bootstrap_method: BootstrapMethod::Block,
            block_size: Some(10),
            confidence_levels: vec![0.90, 0.95],
            confidence_interval_method: ConfidenceIntervalMethod::BootstrapBca,
            seed: Some(54321),
            studentized_outer: None,
            studentized_inner: None,
            jackknife_block_size: None,
            force_block_jackknife: None,
        };

        let result1 = bootstrap_validate(&data, estimator, &config).unwrap();
        let result2 = bootstrap_validate(&data, estimator, &config).unwrap();

        // All results should be identical
        assert_eq!(result1.original_estimate, result2.original_estimate);
        assert_eq!(result1.bootstrap_estimates, result2.bootstrap_estimates);
        assert_eq!(result1.bias, result2.bias);
        assert_eq!(result1.standard_error, result2.standard_error);

        for (ci1, ci2) in result1
            .confidence_intervals
            .iter()
            .zip(result2.confidence_intervals.iter())
        {
            assert_eq!(ci1.lower_bound, ci2.lower_bound);
            assert_eq!(ci1.upper_bound, ci2.upper_bound);
        }
    }

    #[test]
    fn test_paired_bootstrap_disabled_for_safety() {
        // Paired bootstrap with correlation test - disabled for now
        // This would test bootstrap_validate_pairs when re-enabled
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        // Note: This test is disabled as paired bootstrap needs more validation
        // The infrastructure exists but needs statistical verification
        assert_eq!(x.len(), y.len());
    }

    #[test]
    fn test_bootstrap_correlation_coefficient_disabled() {
        // Test for correlation coefficient bootstrap - disabled
        // This requires special handling to preserve correlation structure
        let n = 100;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| 2.0 * v + 1.0).collect();

        // Correlation should be 1.0 for perfectly linear relationship
        // But bootstrap needs special care to preserve this
        assert_eq!(x.len(), y.len());
    }

    #[test]
    fn test_multivariate_block_size_selection() {
        // Test that multivariate_politis_white_block_size considers both series

        // Test 1: Identical series should give same block size as individual
        let identical_data: Vec<(f64, f64)> = (0..100)
            .map(|i| {
                let val = (i as f64 * 0.1).sin();
                (val, val)
            })
            .collect();

        let block_size_multi = multivariate_politis_white_block_size(&identical_data);
        let first_series: Vec<f64> = identical_data.iter().map(|(x, _)| *x).collect();
        let second_series: Vec<f64> = identical_data.iter().map(|(_, y)| *y).collect();
        let block_size_single = politis_white_block_size(&first_series);
        let block_size_second = politis_white_block_size(&second_series);

        // println!("First series block size: {}", block_size_single);
        // println!("Second series block size: {}", block_size_second);
        // println!("Multivariate block size: {}", block_size_multi);

        // For identical series, block sizes should be the same
        assert_eq!(block_size_single, block_size_second);
        // Multivariate should be similar (within 1 due to rounding)
        assert!((block_size_multi as i32 - block_size_single as i32).abs() <= 1);

        // Test 2: Different series with different dependencies
        let different_data: Vec<(f64, f64)> = (0..100)
            .map(|i| {
                let smooth = (i as f64 * 0.05).sin(); // Smooth series
                let noisy = if i % 2 == 0 { 1.0 } else { -1.0 }; // Noisy series
                (smooth, noisy)
            })
            .collect();

        let block_size_multi_diff = multivariate_politis_white_block_size(&different_data);
        let smooth_series: Vec<f64> = different_data.iter().map(|(x, _)| *x).collect();
        let noisy_series: Vec<f64> = different_data.iter().map(|(_, y)| *y).collect();
        let block_size_smooth = politis_white_block_size(&smooth_series);
        let block_size_noisy = politis_white_block_size(&noisy_series);

        // println!("\nSmooth series block size: {}", block_size_smooth);
        // println!("Noisy series block size: {}", block_size_noisy);
        // println!("Multivariate block size: {}", block_size_multi_diff);

        // Multivariate should be at least as large as the maximum
        assert!(block_size_multi_diff >= block_size_smooth.max(block_size_noisy));

        // Test 3: Test multivariate_block_bootstrap works
        let estimator = |data: &[(f64, f64)]| {
            // Simple sum of products
            data.iter().map(|(x, y)| x * y).sum::<f64>() / data.len() as f64
        };

        let config = BootstrapConfiguration {
            num_bootstrap_samples: 100,
            bootstrap_method: BootstrapMethod::Block,
            block_size: None, // Let it auto-select
            confidence_levels: vec![0.95],
            confidence_interval_method: ConfidenceIntervalMethod::BootstrapPercentile,
            seed: Some(42),
            studentized_outer: None,
            studentized_inner: None,
            jackknife_block_size: None,
            force_block_jackknife: None,
        };

        let result = multivariate_block_bootstrap(&identical_data, estimator, &config);
        assert!(result.is_ok(), "Multivariate block bootstrap should succeed");
    }
}