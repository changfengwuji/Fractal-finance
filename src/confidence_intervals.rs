//! Confidence interval calculation methods for bootstrap analysis.
//!
//! This module provides various methods for constructing confidence intervals
//! from bootstrap distributions, including normal approximation, percentile,
//! bias-corrected and accelerated (BCa), and studentized bootstrap methods.

use crate::{
    bootstrap_config::{
        BootstrapConfiguration, BootstrapMethod, ConfidenceInterval, 
        ConfidenceIntervalMethod, MAX_BOOTSTRAP_SAMPLES_DEFAULT,
    },
    bootstrap_sampling::{
        block_bootstrap_sample_with_rng, circular_bootstrap_sample_with_rng,
        mix_seed, standard_bootstrap_sample_with_rng, stationary_bootstrap_sample_with_rng,
    },
    block_size::politis_white_block_size,
    errors::{FractalAnalysisError, FractalResult},
    secure_rng::SecureRng,
};
use once_cell::sync::Lazy;
use statrs::distribution::{ContinuousCDF, Normal};

// Quantile interpolation epsilon for numerical stability
pub(crate) const QUANTILE_EPSILON: f64 = 1e-12;

// Cached standard normal distribution for performance
static STANDARD_NORMAL: Lazy<Normal> = Lazy::new(|| {
    Normal::new(0.0, 1.0).expect("Failed to create standard normal distribution")
});

/// Compute interpolated quantile using Hyndman-Fan Type 7 method.
/// 
/// This is the R default quantile method (type=7) which provides smooth,
/// unbiased quantile estimation that avoids stepwise jumps in confidence intervals.
/// Reference: Hyndman, R.J. and Fan, Y. (1996) Sample quantiles in statistical packages.
///
/// # Arguments
/// * `sorted` - Sorted array of finite values
/// * `p` - Quantile to compute (0 < p < 1)
///
/// # Returns
/// Interpolated quantile value using Type 7 method
pub(crate) fn quantile_type7(sorted: &[f64], p: f64) -> f64 {
    debug_assert!(!sorted.is_empty(), "quantile_type7 requires non-empty input");
    
    // Clamp p to valid range to prevent extrapolation, using same epsilon as BCa
    let p = p.clamp(QUANTILE_EPSILON, 1.0 - QUANTILE_EPSILON);
    
    let n = sorted.len() as f64;
    let h = p * (n - 1.0);
    let h_floor = h.floor() as usize;
    let h_frac = h - h_floor as f64;
    
    if h_floor + 1 < sorted.len() {
        sorted[h_floor] * (1.0 - h_frac) + sorted[h_floor + 1] * h_frac
    } else {
        sorted[h_floor.min(sorted.len() - 1)]
    }
}

/// Calculate percentile-based confidence interval from bootstrap distribution.
///
/// The percentile method uses quantiles of the bootstrap distribution directly
/// as confidence interval bounds. This is the simplest bootstrap CI method.
///
/// # Arguments
/// * `bootstrap_estimates` - Bootstrap sample estimates
/// * `original_estimate` - Original point estimate (used as fallback for degenerate cases)
/// * `confidence_level` - Confidence level (e.g., 0.95)
///
/// # Returns
/// Confidence interval using percentile method
pub fn calculate_bootstrap_confidence_interval(
    bootstrap_estimates: &[f64],
    original_estimate: f64,
    confidence_level: f64,
) -> FractalResult<ConfidenceInterval> {
    if bootstrap_estimates.is_empty() {
        return Err(FractalAnalysisError::BootstrapError {
            reason: "Empty bootstrap estimates".to_string(),
        });
    }

    let mut sorted_estimates = bootstrap_estimates.to_vec();
    // Filter out NaN values first
    sorted_estimates.retain(|x| x.is_finite());
    if sorted_estimates.is_empty() {
        return Err(FractalAnalysisError::BootstrapError {
            reason: "No finite bootstrap estimates available".to_string(),
        });
    }
    // After retain, all values are finite, so simple sort is sufficient
    sorted_estimates.sort_unstable_by(|a, b| 
        a.partial_cmp(b).unwrap() // Safe since we filtered NaN
    );

    let alpha = 1.0 - confidence_level;
    
    // Use interpolated quantiles (Type 7 / Hyndman-Fan) to prevent bias
    let p_lo = alpha / 2.0;
    let p_hi = 1.0 - alpha / 2.0;
    
    let lower_bound = quantile_type7(&sorted_estimates, p_lo);
    let upper_bound = quantile_type7(&sorted_estimates, p_hi);

    // Guard against non-finite values
    if !lower_bound.is_finite() || !upper_bound.is_finite() {
        // Degenerate CI when variance is effectively zero
        return Ok(ConfidenceInterval {
            confidence_level,
            lower_bound: original_estimate,
            upper_bound: original_estimate,
            method: ConfidenceIntervalMethod::BootstrapPercentile,
        });
    }
    
    Ok(ConfidenceInterval {
        confidence_level,
        lower_bound,
        upper_bound,
        method: ConfidenceIntervalMethod::BootstrapPercentile,
    })
}

/// Calculate normal approximation confidence interval.
///
/// Uses the normal approximation to construct confidence intervals based on
/// the point estimate and its standard error.
///
/// # Arguments
/// * `estimate` - Point estimate
/// * `standard_error` - Standard error of the estimate
/// * `confidence_level` - Confidence level (e.g., 0.95)
///
/// # Returns
/// Confidence interval using normal approximation
pub fn calculate_normal_confidence_interval(
    estimate: f64,
    standard_error: f64,
    confidence_level: f64,
) -> FractalResult<ConfidenceInterval> {
    let alpha = 1.0 - confidence_level;
    let z_critical = STANDARD_NORMAL.inverse_cdf(1.0 - alpha / 2.0);

    let margin = z_critical * standard_error;
    let lower_bound = estimate - margin;
    let upper_bound = estimate + margin;

    // Guard against non-finite values
    if !lower_bound.is_finite() || !upper_bound.is_finite() || !estimate.is_finite() {
        // Degenerate CI when variance is effectively zero or estimate is non-finite
        return Ok(ConfidenceInterval {
            confidence_level,
            lower_bound: estimate,
            upper_bound: estimate,
            method: ConfidenceIntervalMethod::Normal,
        });
    }

    Ok(ConfidenceInterval {
        confidence_level,
        lower_bound,
        upper_bound,
        method: ConfidenceIntervalMethod::Normal,
    })
}

/// Calculate bias-corrected and accelerated (BCa) confidence interval.
///
/// The BCa method provides more accurate confidence intervals by correcting
/// for bias and skewness in the bootstrap distribution. It uses jackknife
/// estimates to calculate the acceleration parameter.
///
/// **IMPORTANT**: BCa is theoretically derived for i.i.d. data. When using
/// block bootstrap (for time series), the code attempts to use block jackknife
/// for the acceleration parameter, but this is a heuristic approximation.
/// For rigorous time series inference with dependence, consider using
/// percentile or studentized bootstrap methods instead.
///
/// # Arguments
/// * `data` - Original data
/// * `bootstrap_estimates` - Bootstrap estimates
/// * `original_estimate` - Original point estimate
/// * `confidence_level` - Confidence level
/// * `estimator` - Estimator function
/// * `config` - Bootstrap configuration
///
/// # Returns
/// BCa confidence interval (falls back to percentile for extreme cases)
pub fn calculate_bca_confidence_interval(
    data: &[f64],
    bootstrap_estimates: &[f64],
    original_estimate: f64,
    confidence_level: f64,
    estimator: impl Fn(&[f64]) -> f64,
    config: &BootstrapConfiguration,
) -> FractalResult<ConfidenceInterval> {
    // Calculate bias correction with numerical safeguards
    let num_below = bootstrap_estimates
        .iter()
        .filter(|&&x| x < original_estimate)
        .count();

    // Prevent division by zero and extreme quantiles that cause overflow
    let proportion_below = num_below as f64 / bootstrap_estimates.len() as f64;
    let clamped_proportion = proportion_below.max(1e-6).min(1.0 - 1e-6);
    let bias_correction = STANDARD_NORMAL.inverse_cdf(clamped_proportion);

    // Calculate acceleration parameter
    // Note: BCa theory assumes i.i.d. data. For dependent data (block bootstrap),
    // we attempt block jackknife but this is a heuristic - the theoretical
    // properties of BCa do not hold. Consider using percentile intervals instead.
    let use_block_jackknife = config.force_block_jackknife.unwrap_or(false)
        || matches!(config.bootstrap_method, 
            BootstrapMethod::Block | BootstrapMethod::Stationary | BootstrapMethod::Circular) 
        || config.block_size.is_some()
        || detect_temporal_dependence(data);
    
    let acceleration = if use_block_jackknife {
        calculate_block_jackknife_acceleration(data, &estimator, config.jackknife_block_size)?
    } else {
        calculate_jackknife_acceleration(data, &estimator)?
    };

    // Calculate adjusted quantiles
    let alpha = 1.0 - confidence_level;
    let z_alpha_2 = STANDARD_NORMAL.inverse_cdf(alpha / 2.0);
    let z_1_alpha_2 = STANDARD_NORMAL.inverse_cdf(1.0 - alpha / 2.0);

    // Calculate adjusted quantiles with clamping to prevent out-of-bounds
    let eps = 1e-12;
    let adjusted_lower = STANDARD_NORMAL.cdf(
        bias_correction
            + (bias_correction + z_alpha_2) / (1.0 - acceleration * (bias_correction + z_alpha_2)),
    ).clamp(eps, 1.0 - eps);
    let adjusted_upper = STANDARD_NORMAL.cdf(
        bias_correction
            + (bias_correction + z_1_alpha_2)
                / (1.0 - acceleration * (bias_correction + z_1_alpha_2)),
    ).clamp(eps, 1.0 - eps);

    let mut sorted_estimates = bootstrap_estimates.to_vec();
    // CRITICAL FIX: Remove NaN values before sorting to avoid NaN propagation in interpolation
    sorted_estimates.retain(|x| x.is_finite());
    
    if sorted_estimates.is_empty() {
        return Err(FractalAnalysisError::BootstrapError {
            reason: "No finite bootstrap estimates for BCa calculation".to_string(),
        });
    }
    
    // Now sort the finite values
    sorted_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Use interpolated quantiles for BCa as well
    let lower_bound = quantile_type7(&sorted_estimates, adjusted_lower);
    let upper_bound = quantile_type7(&sorted_estimates, adjusted_upper);

    // Guard against non-finite values
    if !lower_bound.is_finite() || !upper_bound.is_finite() {
        // Degenerate CI when variance is effectively zero
        return Ok(ConfidenceInterval {
            confidence_level,
            lower_bound: original_estimate,
            upper_bound: original_estimate,
            method: ConfidenceIntervalMethod::BootstrapBca,
        });
    }

    // Ensure bounds are properly ordered
    // BCa can sometimes produce inverted bounds due to extreme skewness
    let (mut final_lower, mut final_upper) = if lower_bound > upper_bound {
        (upper_bound, lower_bound)
    } else {
        (lower_bound, upper_bound)
    };
    
    // Additional safety check: ensure the interval includes the estimate
    // BCa can sometimes produce intervals that don't include the original estimate
    // due to extreme bias or skewness. In such cases, expand the interval to include it.
    if final_lower > original_estimate {
        // Lower bound is above the estimate - adjust it
        final_lower = original_estimate - (original_estimate - final_lower).abs() * 0.1;
    }
    if final_upper < original_estimate {
        // Upper bound is below the estimate - adjust it
        final_upper = original_estimate + (final_upper - original_estimate).abs() * 0.1;
    }
    
    // Final check: if the interval is still invalid, fall back to percentile method
    if final_lower > final_upper || (final_lower > original_estimate && final_upper > original_estimate) ||
       (final_lower < original_estimate && final_upper < original_estimate) {
        let alpha = 1.0 - confidence_level;
        let p_lo = alpha / 2.0;
        let p_hi = 1.0 - alpha / 2.0;
        final_lower = quantile_type7(&sorted_estimates, p_lo);
        final_upper = quantile_type7(&sorted_estimates, p_hi);
        
        // Ensure ordering and inclusion of estimate
        if final_lower > final_upper {
            let temp = final_lower;
            final_lower = final_upper;
            final_upper = temp;
        }
        
        // Expand to include estimate if needed
        final_lower = final_lower.min(original_estimate);
        final_upper = final_upper.max(original_estimate);
    }
    
    Ok(ConfidenceInterval {
        confidence_level,
        lower_bound: final_lower,
        upper_bound: final_upper,
        method: ConfidenceIntervalMethod::BootstrapBca,
    })
}

/// Calculate studentized bootstrap confidence intervals.
///
/// The studentized bootstrap uses the t-statistic distribution instead of
/// the raw bootstrap distribution, providing better coverage for small samples
/// and heavy-tailed distributions common in financial data.
pub(crate) fn calculate_studentized_confidence_interval<F>(
    data: &[f64],
    bootstrap_estimates: &[f64],
    original_estimate: f64,
    confidence_level: f64,
    estimator: &F,
    config: &BootstrapConfiguration,
) -> FractalResult<ConfidenceInterval>
where
    F: Fn(&[f64]) -> f64,
{
    let num_bootstrap = config.num_bootstrap_samples;

    // Calculate standard error of the original sample using the same config
    let original_se = calculate_bootstrap_standard_error_with_config(
        data, 
        estimator, 
        config.studentized_inner.unwrap_or(50),  // Use consistent default of 50
        Some(config)
    )?;

    if original_se < 1e-10 {
        // Degenerate case - no variability
        return Ok(ConfidenceInterval {
            confidence_level,
            lower_bound: original_estimate,
            upper_bound: original_estimate,
            method: ConfidenceIntervalMethod::StudentizedBootstrap,
        });
    }

    // Use configurable bootstrap sizes for studentized CI
    // Default to num_bootstrap for outer, capped at reasonable limit for nested bootstrap
    // Note: While validation allows up to MAX_BOOTSTRAP_SAMPLES, nested bootstrap
    // is expensive so we cap at a lower practical limit
    let outer_samples = config.studentized_outer
        .unwrap_or(num_bootstrap)
        .min(num_bootstrap)
        .min(MAX_BOOTSTRAP_SAMPLES_DEFAULT);  // Cap at 1000 for practical nested bootstrap performance
    let inner_samples = config.studentized_inner.unwrap_or(50);
    
    // Generate second-level bootstrap samples to estimate standard errors
    let mut t_statistics = Vec::with_capacity(outer_samples);

    let mut rng = if let Some(seed) = config.seed {
        SecureRng::with_seed(seed)
    } else {
        SecureRng::new()
    };

    for i in 0..outer_samples {
        // Use proper seed mixing
        if let Some(seed) = config.seed {
            rng = SecureRng::with_seed(mix_seed(seed, i));
        }
        
        // Generate bootstrap sample using the configured method
        let bootstrap_sample = match config.bootstrap_method {
            BootstrapMethod::Block => block_bootstrap_sample_with_rng(
                data,
                config.block_size.unwrap_or_else(|| politis_white_block_size(data)),
                &mut rng
            ),
            BootstrapMethod::Stationary => stationary_bootstrap_sample_with_rng(
                data,
                config.block_size.unwrap_or_else(|| politis_white_block_size(data)),
                &mut rng
            ),
            BootstrapMethod::Circular => circular_bootstrap_sample_with_rng(data, &mut rng),
            _ => standard_bootstrap_sample_with_rng(data, &mut rng),
        };

        let bootstrap_estimate = estimator(&bootstrap_sample);

        // Calculate standard error for this bootstrap sample using nested bootstrap with same config
        let bootstrap_se = calculate_bootstrap_standard_error_with_config(
            &bootstrap_sample, 
            estimator, 
            inner_samples,
            Some(config)
        )?;

        if bootstrap_se > 1e-10 {
            // Calculate t-statistic
            let t_stat = (bootstrap_estimate - original_estimate) / bootstrap_se;
            if t_stat.is_finite() {
                t_statistics.push(t_stat);
            }
        }
    }

    if t_statistics.len() < 10 {
        // Fall back to percentile method if not enough valid t-statistics
        return calculate_bootstrap_confidence_interval(
            bootstrap_estimates,
            original_estimate,
            confidence_level,
        );
    }

    // Sort t-statistics
    t_statistics.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate quantiles
    let alpha = 1.0 - confidence_level;
    let lower_quantile = alpha / 2.0;
    let upper_quantile = 1.0 - alpha / 2.0;

    // Use consistent quantile function for t-statistics
    let t_lo = quantile_type7(&t_statistics, lower_quantile);
    let t_hi = quantile_type7(&t_statistics, upper_quantile);
    
    // Note: For studentized bootstrap, we use the REVERSED quantiles
    // Calculate confidence bounds
    let lower_bound = original_estimate - t_hi * original_se;
    let upper_bound = original_estimate - t_lo * original_se;

    // Guard against non-finite values
    if !lower_bound.is_finite() || !upper_bound.is_finite() || !original_estimate.is_finite() {
        // Degenerate CI when variance is effectively zero or estimate is non-finite
        return Ok(ConfidenceInterval {
            confidence_level,
            lower_bound: original_estimate,
            upper_bound: original_estimate,
            method: ConfidenceIntervalMethod::StudentizedBootstrap,
        });
    }

    Ok(ConfidenceInterval {
        confidence_level,
        lower_bound,
        upper_bound,
        method: ConfidenceIntervalMethod::StudentizedBootstrap,
    })
}

/// Detect temporal dependence in data using autocorrelation test.
///
/// Returns true if significant autocorrelation is detected at lag 1,
/// indicating the data has temporal structure that should be preserved.
pub(crate) fn detect_temporal_dependence(data: &[f64]) -> bool {
    let n = data.len();
    if n < 20 {
        // Too small to reliably detect autocorrelation
        return false;
    }

    // Calculate mean
    let mean = data.iter().sum::<f64>() / n as f64;

    // Calculate lag-1 autocorrelation
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..n {
        let deviation = data[i] - mean;
        denominator += deviation * deviation;

        if i > 0 {
            let prev_deviation = data[i - 1] - mean;
            numerator += deviation * prev_deviation;
        }
    }

    if denominator < 1e-10 {
        return false; // Constant data
    }

    let autocorr = numerator / denominator;

    // Test for significance using the approximate standard error
    // Under null hypothesis of no autocorrelation, SE ≈ 1/sqrt(n)
    let se = 1.0 / (n as f64).sqrt();
    let z_statistic = autocorr.abs() / se;

    // Use z > 2 as threshold (approximately 95% confidence)
    z_statistic > 2.0
}

/// Helper function to calculate bootstrap standard error with specific config
pub(crate) fn calculate_bootstrap_standard_error_with_config<F>(
    data: &[f64],
    estimator: &F,
    num_bootstrap: usize,
    config: Option<&BootstrapConfiguration>,
) -> FractalResult<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let mut estimates = Vec::with_capacity(num_bootstrap);
    let mut rng = SecureRng::new();
    
    // Use the same bootstrap method as the main config if provided
    let default_config = BootstrapConfiguration::default();
    let bootstrap_config = config.unwrap_or(&default_config);

    // Hoist block size computation to avoid O(B) calls to Politis-White
    let block_size = if matches!(bootstrap_config.bootstrap_method, BootstrapMethod::Block | BootstrapMethod::Stationary) {
        bootstrap_config.block_size.unwrap_or_else(|| politis_white_block_size(data))
    } else {
        0  // Not used for other methods
    };
    
    for i in 0..num_bootstrap {
        // Use proper seed mixing if seeded
        if let Some(seed) = bootstrap_config.seed {
            rng = SecureRng::with_seed(mix_seed(seed, i));
        }
        
        // Use the configured bootstrap method
        let bootstrap_sample = match bootstrap_config.bootstrap_method {
            BootstrapMethod::Block => block_bootstrap_sample_with_rng(
                data, 
                block_size,
                &mut rng
            ),
            BootstrapMethod::Stationary => stationary_bootstrap_sample_with_rng(
                data,
                block_size,
                &mut rng
            ),
            BootstrapMethod::Circular => circular_bootstrap_sample_with_rng(data, &mut rng),
            _ => standard_bootstrap_sample_with_rng(data, &mut rng),
        };

        let estimate = estimator(&bootstrap_sample);
        if estimate.is_finite() {
            estimates.push(estimate);
        }
    }

    if estimates.len() < 2 {
        return Ok(0.0);
    }

    let mean = estimates.iter().sum::<f64>() / estimates.len() as f64;
    let variance = estimates
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f64>()
        / (estimates.len() - 1) as f64;

    Ok(variance.sqrt())
}

/// Calculate block jackknife acceleration for temporal data.
///
/// For time series with temporal dependence, block jackknife preserves
/// the temporal structure better than standard leave-one-out jackknife.
fn calculate_block_jackknife_acceleration<F>(
    data: &[f64],
    estimator: &F,
    block_size: Option<usize>,
) -> FractalResult<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = data.len();
    if n < 20 {
        // Fall back to standard jackknife for very small samples
        return calculate_jackknife_acceleration(data, estimator);
    }

    // Determine block size if not provided
    let block_size = block_size.unwrap_or_else(|| {
        // Default block size: n^(1/3) rule for temporal data
        ((n as f64).powf(1.0 / 3.0) as usize).max(2).min(n / 10)
    });

    if block_size >= n / 2 {
        // Block too large, fall back to standard jackknife
        return calculate_jackknife_acceleration(data, estimator);
    }

    let num_blocks = (n + block_size - 1) / block_size;
    let mut jackknife_estimates = Vec::with_capacity(num_blocks);

    // Leave-block-out estimates
    for block_idx in 0..num_blocks {
        let block_start = block_idx * block_size;
        let block_end = (block_start + block_size).min(n);

        let mut jackknife_data = Vec::with_capacity(n - (block_end - block_start));
        jackknife_data.extend_from_slice(&data[..block_start]);
        jackknife_data.extend_from_slice(&data[block_end..]);

        if !jackknife_data.is_empty() {
            let estimate = estimator(&jackknife_data);
            if estimate.is_finite() {
                jackknife_estimates.push(estimate);
            }
        }
    }

    if jackknife_estimates.len() < 3 {
        return Ok(0.0);
    }

    let mean_jackknife = jackknife_estimates.iter().sum::<f64>() / jackknife_estimates.len() as f64;

    // Calculate skewness for acceleration parameter
    let mut sum_cubed = 0.0;
    let mut sum_squared = 0.0;

    for &estimate in &jackknife_estimates {
        let diff = estimate - mean_jackknife;
        sum_squared += diff * diff;
        sum_cubed += diff * diff * diff;
    }

    let variance = sum_squared / jackknife_estimates.len() as f64;
    if variance < 1e-10 {
        return Ok(0.0);
    }

    let skewness = sum_cubed / (jackknife_estimates.len() as f64 * variance.powf(1.5));

    // Acceleration parameter from skewness
    Ok(skewness / 6.0)
}

/// Calculate jackknife acceleration parameter for BCa intervals.
///
/// The acceleration parameter accounts for the rate of change of the
/// standard error with respect to the true parameter value.
fn calculate_jackknife_acceleration<F>(data: &[f64], estimator: &F) -> FractalResult<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = data.len();
    if n < 10 {
        return Ok(0.0); // No acceleration for small samples
    }

    let _original_estimate = estimator(data);
    let mut jackknife_estimates = Vec::with_capacity(n);

    // Leave-one-out estimates
    for i in 0..n {
        let mut jackknife_data = Vec::with_capacity(n - 1);
        jackknife_data.extend_from_slice(&data[..i]);
        jackknife_data.extend_from_slice(&data[i + 1..]);

        let estimate = estimator(&jackknife_data);
        if estimate.is_finite() {
            jackknife_estimates.push(estimate);
        }
    }

    if jackknife_estimates.is_empty() {
        return Ok(0.0);
    }

    let mean_jackknife = jackknife_estimates.iter().sum::<f64>() / jackknife_estimates.len() as f64;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for &estimate in &jackknife_estimates {
        let diff = mean_jackknife - estimate;
        numerator += diff.powi(3);
        denominator += diff.powi(2);
    }

    if denominator > 0.0 {
        Ok(numerator / (6.0 * denominator.powf(1.5)))
    } else {
        Ok(0.0)
    }
}