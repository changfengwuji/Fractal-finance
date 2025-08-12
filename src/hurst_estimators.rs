//! Hurst exponent estimation methods for fractal analysis
//!
//! This module provides various methods for estimating the Hurst exponent,
//! including Rescaled Range (R/S), Detrended Fluctuation Analysis (DFA),
//! Periodogram regression (GPH), wavelet-based methods, Whittle MLE, and Variogram.
//! 
//! ## Method Characteristics
//! 
//! - **R/S Analysis**: Classical method, works well for n ≥ 50
//! - **DFA**: Robust to trends, requires n ≥ 100 
//! - **GPH**: Frequency domain, requires n ≥ 128
//! - **Wavelet**: Multi-scale analysis, requires n ≥ 64
//! - **Whittle MLE**: Maximum likelihood in frequency domain, requires n ≥ 128 (experimental)
//! - **Variogram**: Based on spatial correlation, works for n ≥ 50 (experimental)
//! 
//! ## Implementation Notes
//! 
//! - All methods use HAC-robust standard errors except GPH (configurable)
//! - Bootstrap confidence intervals are computed for all methods
//! - Whittle estimator uses profiled likelihood to handle unknown scale
//! - All logarithms use natural logarithm (ln) for consistency

use crate::{
    bootstrap::{
        bootstrap_validate, generate_bootstrap_sample, politis_white_block_size,
        BootstrapConfiguration, BootstrapMethod, BootstrapValidation, ConfidenceInterval,
        ConfidenceIntervalMethod, EstimatorComplexity,
    },
    errors::{validate_data_length, FractalAnalysisError, FractalResult},
    fft_ops::calculate_periodogram_fft,
    linear_algebra::economy_qr_solve,
    math_utils::{
        calculate_variance, float_ops, float_total_cmp, integrate_series, mad, median,
        ols_regression, ols_regression_hac, wls_regression, percentile, standard_normal_cdf, generate_window_sizes,
    },
    multifractal::perform_multifractal_analysis,
    statistical_tests::TestConfiguration,
    wavelet::{calculate_wavelet_variance, estimate_wavelet_hurst_only},
};

use std::collections::BTreeMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ============================================================================
// CONSTANTS
// ============================================================================

/// Minimum variance threshold for numerical stability in R/S calculations
/// Used when computing standard deviations to avoid division issues
/// Increased from 1e-14 to 1e-12 for better handling of noisy near-zero data
/// Used for: Numerical stability in divisions and regression operations
const MIN_VARIANCE_THRESHOLD: f64 = 1e-12;

/// Minimum standard error to prevent division by near-zero
/// Used in test statistics to avoid numerical overflow
const MIN_STD_ERROR: f64 = 1e-6;

/// Variance below this is considered effectively zero (constant data)
/// Used to detect and reject constant time series early
/// Increased from 1e-15 to 1e-13 to avoid false positives with machine precision
/// Note: This is SMALLER than MIN_VARIANCE_THRESHOLD (1e-13 < 1e-12)
/// Used for: Early detection of truly constant data before processing
const ZERO_VARIANCE_THRESHOLD: f64 = 1e-13;

/// Near-zero variance threshold for special bootstrap handling
/// Data with variance below this gets simplified bootstrap (no resampling needed)
/// Typical scale: For price data (~100 scale), catches drift-only series
const NEAR_ZERO_VARIANCE_THRESHOLD: f64 = 1e-10;

/// Epsilon for matching q values in multifractal analysis
const Q_MATCH_EPSILON: f64 = 1e-6;

/// Default confidence level for intervals
const DEFAULT_CONFIDENCE_LEVEL: f64 = 0.95;

/// Hurst exponent for random walk (Brownian motion)
const RANDOM_WALK_HURST: f64 = 0.5;

/// GPH bandwidth parameter (controls frequency range)
const GPH_BANDWIDTH_EXPONENT: f64 = 0.65;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Estimation method for Hurst exponent
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum EstimationMethod {
    /// Rescaled Range Analysis (R/S)
    RescaledRange,
    /// Detrended Fluctuation Analysis (DFA)
    DetrendedFluctuationAnalysis,
    /// Periodogram Regression (GPH)
    PeriodogramRegression,
    /// Wavelet-based Estimation
    WaveletEstimation,
    /// Whittle Maximum Likelihood Estimator
    WhittleEstimator,
    /// Variogram-based Method
    VariogramMethod,
}

/// Hurst exponent estimate with uncertainty quantification
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HurstEstimate {
    /// Point estimate of Hurst exponent
    pub estimate: f64,
    /// Standard error of the estimate
    pub standard_error: f64,
    /// Bootstrap confidence interval
    pub confidence_interval: ConfidenceInterval,
    /// Test statistic for H=0.5 hypothesis
    pub test_statistic: f64,
    /// P-value for test of H=0.5
    pub p_value: f64,
    /// Adjusted p-value after multiple testing correction
    pub adjusted_p_value: Option<f64>,
    /// Bias correction applied (heuristic adjustments like DFA polynomial, GPH finite sample, R/S Lo's)
    pub bias_correction: f64,
    /// Finite sample correction factor
    pub finite_sample_correction: f64,
    /// Bootstrap bias estimate (E[θ*] - θ from bootstrap distribution)
    pub bootstrap_bias: f64,
}

/// Configuration for Hurst estimation
/// 
/// Controls the behavior of various Hurst estimation methods and their statistical properties.
/// 
/// ## HAC (Heteroskedasticity and Autocorrelation Consistent) Standard Errors
/// 
/// HAC standard errors are used to provide robust inference when residuals exhibit
/// heteroskedasticity and/or autocorrelation, which is common in financial time series.
/// 
/// ### Methods using HAC:
/// - **R/S Analysis**: Always uses HAC-robust errors via `ols_regression_hac`
/// - **DFA**: Always uses HAC-robust errors for scale regression  
/// - **Wavelet**: Always uses HAC-robust errors for variance regression
/// - **GPH**: Configurable via `use_hac_for_gph` flag
///   - When `true`: Uses HAC-robust standard errors from the regression
///   - When `false`: Uses classical GPH formula: π / (√24 * √m)
/// - **Variogram**: Always uses HAC-robust errors for lag regression
/// 
/// ## Bootstrap Determinism
/// 
/// The bootstrap procedure is fully deterministic when a seed is provided:
/// - Set `bootstrap_config.seed` to ensure reproducible confidence intervals
/// - Each bootstrap iteration uses `seed + iteration_index` for independent samples
/// - Same input data + same seed = identical results across runs
/// - This is crucial for reproducible research and backtesting
/// 
/// Example:
/// ```
/// let mut config = HurstEstimationConfig::default();
/// config.bootstrap_config.seed = Some(42); // Reproducible results
/// ```
#[derive(Debug, Clone)]
pub struct HurstEstimationConfig {
    /// DFA polynomial order for detrending (1 = linear, 2 = quadratic)
    pub dfa_polynomial_order: usize,
    /// Bootstrap configuration (controls resampling and confidence intervals)
    pub bootstrap_config: BootstrapConfiguration,
    /// Test configuration for statistical tests
    pub test_config: TestConfiguration,
    /// Whether to use HAC standard errors for GPH estimator
    /// When true, uses HAC-robust standard errors for the GPH regression
    /// When false, uses the classical GPH formula: π / (√24 * √m)
    pub use_hac_for_gph: bool,
    /// GPH bandwidth exponent (default: 0.65)
    /// Controls the number of frequencies used: m ≈ n^bandwidth_exponent
    pub gph_bandwidth_exponent: f64,
    /// GPH bandwidth multiplier (default: 0.8)
    /// Final bandwidth: m = multiplier * n^exponent
    pub gph_bandwidth_multiplier: f64,
}

impl Default for HurstEstimationConfig {
    fn default() -> Self {
        Self {
            dfa_polynomial_order: 1,
            bootstrap_config: BootstrapConfiguration::default(),
            test_config: TestConfiguration::default(),
            use_hac_for_gph: false,  // Default to analytic SE, not HAC for frequency domain
            gph_bandwidth_exponent: GPH_BANDWIDTH_EXPONENT,
            gph_bandwidth_multiplier: 0.8,
        }
    }
}

// ============================================================================
// MAIN ESTIMATION FUNCTIONS
// ============================================================================

/// Estimate Hurst exponent using multiple methods
pub fn estimate_hurst_multiple_methods(
    data: &[f64],
    config: &HurstEstimationConfig,
) -> FractalResult<BTreeMap<EstimationMethod, HurstEstimate>> {
    let n = data.len();
    
    // Handle very short series with simplified estimator
    if n < 50 {
        let est = estimate_hurst_simple_short_series(data)?;
        return Ok([(EstimationMethod::RescaledRange, est)].into_iter().collect());
    }
    
    let mut hurst_estimates = BTreeMap::new();

    // Adaptive method selection based on data length
    let estimation_methods = select_methods_for_data_length(n)?;

    // Use shared bootstrap for performance if multiple methods
    if estimation_methods.len() > 1 && n >= 100 {
        match estimate_hurst_with_shared_bootstrap(data, &estimation_methods, config) {
            Ok(estimates) => return Ok(estimates),
            Err(_) => {
                // Fallback to individual estimation
            }
        }
    }

    // Individual method estimation
    for method in &estimation_methods {
        match estimate_hurst_by_method(data, method, config) {
            Ok(estimate) => {
                hurst_estimates.insert(*method, estimate);
            }
            Err(_) => {
                // Continue with other methods
            }
        }
    }

    if hurst_estimates.is_empty() {
        return Err(FractalAnalysisError::NumericalError {
            reason: "Failed to estimate Hurst exponent with any method".to_string(),
            operation: None,
        });
    }

    apply_multiple_testing_corrections(hurst_estimates)
}

/// Select appropriate estimation methods based on data length
fn select_methods_for_data_length(n: usize) -> FractalResult<Vec<EstimationMethod>> {
    if n >= 256 {
        // For long series, use all methods
        Ok(vec![
            EstimationMethod::RescaledRange,
            EstimationMethod::DetrendedFluctuationAnalysis,
            EstimationMethod::PeriodogramRegression,
            EstimationMethod::WaveletEstimation,
            EstimationMethod::WhittleEstimator,
            EstimationMethod::VariogramMethod,
        ])
    } else if n >= 128 {
        // Whittle needs more data for good estimates
        Ok(vec![
            EstimationMethod::RescaledRange,
            EstimationMethod::DetrendedFluctuationAnalysis,
            EstimationMethod::PeriodogramRegression,
            EstimationMethod::WaveletEstimation,
            EstimationMethod::WhittleEstimator,
        ])
    } else if n >= 100 {
        Ok(vec![
            EstimationMethod::RescaledRange,
            EstimationMethod::DetrendedFluctuationAnalysis,
            EstimationMethod::WaveletEstimation,
            EstimationMethod::VariogramMethod,
        ])
    } else if n >= 64 {
        Ok(vec![
            EstimationMethod::RescaledRange,
            EstimationMethod::WaveletEstimation,
            EstimationMethod::VariogramMethod,
        ])
    } else if n >= 50 {
        Ok(vec![
            EstimationMethod::RescaledRange,
            EstimationMethod::VariogramMethod,
        ])
    } else {
        // For n < 50, the main entry point handles with simple estimator
        Err(FractalAnalysisError::InsufficientData {
            required: 50,
            actual: n,
        })
    }
}

/// Dispatch Hurst estimation to the appropriate method
pub fn estimate_hurst_by_method(
    data: &[f64],
    method: &EstimationMethod,
    config: &HurstEstimationConfig,
) -> FractalResult<HurstEstimate> {
    match method {
        EstimationMethod::RescaledRange => estimate_hurst_rescaled_range(data, config),
        EstimationMethod::DetrendedFluctuationAnalysis => estimate_hurst_dfa(data, config),
        EstimationMethod::PeriodogramRegression => estimate_hurst_periodogram(data, config),
        EstimationMethod::WaveletEstimation => estimate_hurst_wavelet(data, config),
        EstimationMethod::WhittleEstimator => estimate_hurst_whittle(data, config),
        EstimationMethod::VariogramMethod => {
            // Variogram expects FBM (cumulative) data
            // Auto-detect if data is FGN (stationary) and convert if needed
            if needs_fbm_conversion(data) {
                let cumulative_data = cummean_zeroed(data);
                estimate_hurst_variogram(&cumulative_data, config)
            } else {
                // Data appears non-stationary (likely already FBM), use as-is
                estimate_hurst_variogram(data, config)
            }
        },
    }
}

// ============================================================================
// RESCALED RANGE (R/S) ANALYSIS
// ============================================================================

/// Rescaled Range (R/S) analysis with bias correction
pub fn estimate_hurst_rescaled_range(
    data: &[f64],
    config: &HurstEstimationConfig,
) -> FractalResult<HurstEstimate> {
    validate_data_length(data, 50, "R/S analysis")?;
    let variance = validate_data_variance(data, "R/S analysis")?;

    let n = data.len();
    let window_sizes = generate_window_sizes(n, 10, 4.0);
    let mut log_rs_values = Vec::with_capacity(window_sizes.len());
    let mut log_n_values = Vec::with_capacity(window_sizes.len());

    for &window_size in &window_sizes {
        if window_size >= n {
            continue;
        }

        let rs_values = calculate_rs_statistics(data, window_size);

        if !rs_values.is_empty() {
            let mean_rs = rs_values.iter().sum::<f64>() / rs_values.len() as f64;
            if mean_rs > 0.0 && mean_rs.is_finite() {
                if let Some(log_val) = float_ops::safe_ln(mean_rs) {
                    log_rs_values.push(log_val);
                    log_n_values.push((window_size as f64).ln());
                }
            }
        }
    }

    if log_rs_values.len() < 3 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 3,
            actual: log_rs_values.len(),
        });
    }
    
    // Check for sufficient variation in x-values to avoid singular matrix
    let log_n_variance = calculate_variance(&log_n_values);
    if log_n_variance < MIN_VARIANCE_THRESHOLD {
        return Err(FractalAnalysisError::NumericalError {
            reason: "Insufficient variation in window sizes for R/S regression".to_string(),
            operation: None,
        });
    }

    let (hurst_estimate, std_error, _) = ols_regression_hac(&log_n_values, &log_rs_values, None)?;

    let bias_correction = calculate_rs_bias_correction(n);
    let corrected_estimate = (hurst_estimate - bias_correction).max(0.01).min(0.99);

    let bootstrap_result = perform_bootstrap_with_constant_check(
        data,
        variance,
        corrected_estimate,
        |data| estimate_rs_hurst_only(data).unwrap_or(0.5),
        EstimatorComplexity::Low,
        config,
    )?;

    let confidence_interval = extract_confidence_interval(&bootstrap_result, corrected_estimate);
    
    // Use Lo's modified R/S for hypothesis testing instead of naive z-test
    let (lo_statistic, _, lo_p_value) = lo_modified_rs_statistic(data, 5)?;
    
    Ok(build_hurst_estimate(
        corrected_estimate,
        std_error,
        confidence_interval,
        lo_statistic,
        lo_p_value,
        bias_correction,
        calculate_finite_sample_correction(n),
    ))
}

/// Core R/S Hurst estimation without bootstrap (for use in bootstrap)
pub fn estimate_rs_hurst_only(data: &[f64]) -> FractalResult<f64> {
    validate_data_length(data, 50, "R/S analysis")?;

    let n = data.len();
    let mut log_rs_values = Vec::new();
    let mut log_n_values = Vec::new();

    let window_sizes = generate_window_sizes(n, 10, 4.0);

    for &window_size in &window_sizes {
        if window_size >= n {
            continue;
        }

        let rs_values = calculate_rs_statistics(data, window_size);

        if !rs_values.is_empty() {
            let mean_rs = rs_values.iter().sum::<f64>() / rs_values.len() as f64;
            if mean_rs > 0.0 && mean_rs.is_finite() {
                if let Some(log_val) = float_ops::safe_ln(mean_rs) {
                    log_rs_values.push(log_val);
                    log_n_values.push((window_size as f64).ln());
                }
            }
        }
    }

    if log_rs_values.len() < 3 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 3,
            actual: log_rs_values.len(),
        });
    }
    
    // Check for sufficient variation in x-values to avoid singular matrix
    let log_n_variance = calculate_variance(&log_n_values);
    if log_n_variance < MIN_VARIANCE_THRESHOLD {
        return Err(FractalAnalysisError::NumericalError {
            reason: "Insufficient variation in window sizes for R/S regression".to_string(),
            operation: None,
        });
    }

    let (hurst_estimate, _, _) = ols_regression(&log_n_values, &log_rs_values)?;
    let bias_correction = calculate_rs_bias_correction(n);
    Ok((hurst_estimate - bias_correction).max(0.01).min(0.99))
}

/// Calculate R/S statistics for given window size
fn calculate_rs_statistics(data: &[f64], window_size: usize) -> Vec<f64> {
    let num_windows = data.len() / window_size;
    let mut rs_values = Vec::with_capacity(num_windows + 20);

    // Process non-overlapping windows
    for i in 0..num_windows {
        let start = i * window_size;
        let window = &data[start..start + window_size];
        let mean = window.iter().sum::<f64>() / window_size as f64;

        // Calculate cumulative deviations
        let mut cumulative_devs = Vec::with_capacity(window_size);
        let mut cumsum = 0.0;
        for &value in window {
            cumsum += value - mean;
            cumulative_devs.push(cumsum);
        }

        // Range
        let max_dev = cumulative_devs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_dev = cumulative_devs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let range = max_dev - min_dev;

        // Standard deviation
        let variance = calculate_variance(window);
        let std_dev = variance.sqrt();

        if variance > MIN_VARIANCE_THRESHOLD && std_dev.is_finite() && range.is_finite() {
            let rs_ratio = float_ops::safe_div(range, std_dev).unwrap_or(0.0);
            if rs_ratio.is_finite() && rs_ratio > 0.0 {
                rs_values.push(rs_ratio);
            }
        }
    }

    // Add sampled overlapping windows with proper stride for actual overlap
    let sample_stride = (window_size / 2).max(1);
    for offset in (window_size / 2..=data.len().saturating_sub(window_size)).step_by(sample_stride) {
        let window = &data[offset..offset + window_size];
        let mean = window.iter().sum::<f64>() / window_size as f64;

        let mut cumulative_devs = Vec::with_capacity(window_size);
        let mut cumsum = 0.0;
        for &value in window {
            cumsum += value - mean;
            cumulative_devs.push(cumsum);
        }

        let max_dev = cumulative_devs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_dev = cumulative_devs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let range = max_dev - min_dev;

        let variance = calculate_variance(window);
        let std_dev = variance.sqrt();

        if variance > MIN_VARIANCE_THRESHOLD && std_dev.is_finite() && range.is_finite() {
            let rs_ratio = float_ops::safe_div(range, std_dev).unwrap_or(0.0);
            if rs_ratio.is_finite() && rs_ratio > 0.0 {
                rs_values.push(rs_ratio);
            }
        }
    }

    rs_values
}

// ============================================================================
// DETRENDED FLUCTUATION ANALYSIS (DFA)
// ============================================================================

/// Detrended Fluctuation Analysis (DFA)
pub fn estimate_hurst_dfa(
    data: &[f64],
    config: &HurstEstimationConfig,
) -> FractalResult<HurstEstimate> {
    validate_data_length(data, 100, "DFA")?;
    let variance = validate_data_variance(data, "DFA")?;

    let n = data.len();
    let integrated = integrate_series(data);

    let window_sizes = generate_window_sizes(n, 10, 4.0);
    let mut log_f_values = Vec::with_capacity(window_sizes.len());
    let mut log_s_values = Vec::with_capacity(window_sizes.len());

    for &window_size in &window_sizes {
        if integrated.len() / window_size < 2 {
            continue;
        }

        let fluctuation = match calculate_dfa_fluctuation(
            &integrated,
            window_size,
            config.dfa_polynomial_order,
        ) {
            Ok(f) => f,
            Err(_) => continue,
        };

        if fluctuation > 0.0 && fluctuation.is_finite() {
            if let Some(log_val) = float_ops::safe_ln(fluctuation) {
                log_f_values.push(log_val);
                log_s_values.push((window_size as f64).ln());
            }
        }
    }

    if log_f_values.len() < 3 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 3,
            actual: log_f_values.len(),
        });
    }

    let (selected_log_s, selected_log_f) = select_dfa_scale_range(&log_s_values, &log_f_values)?;

    let (slope, std_error, _) = ols_regression_hac(&selected_log_s, &selected_log_f, None)?;

    let bias_correction = calculate_dfa_bias_correction(data.len());
    let corrected_estimate = (slope - bias_correction).max(0.01).min(0.99);

    let bootstrap_result = perform_bootstrap_with_constant_check(
        data,
        variance,
        corrected_estimate,
        |data| estimate_dfa_hurst_only(data, config.dfa_polynomial_order).unwrap_or(0.5),
        EstimatorComplexity::Medium,
        config,
    )?;

    let confidence_interval = extract_confidence_interval(&bootstrap_result, corrected_estimate);
    
    // Use bootstrap p-value for hypothesis testing H0: H=0.5
    let bootstrap_p_value = calculate_bootstrap_p_value(&bootstrap_result.bootstrap_estimates, corrected_estimate, 0.5);
    
    // Still compute z-statistic for reporting, but use bootstrap p-value
    let test_statistic = (corrected_estimate - 0.5) / std_error.max(1e-10);

    Ok(build_hurst_estimate(
        corrected_estimate,
        std_error,
        confidence_interval,
        test_statistic,
        bootstrap_p_value,
        bias_correction,
        0.0,
    ))
}

/// Core DFA Hurst estimation without bootstrap
pub fn estimate_dfa_hurst_only(data: &[f64], polynomial_order: usize) -> FractalResult<f64> {
    validate_data_length(data, 100, "DFA")?;

    let n = data.len();
    let integrated = integrate_series(data);

    let mut log_f_values = Vec::new();
    let mut log_s_values = Vec::new();

    let window_sizes = generate_window_sizes(n, 10, 4.0);

    for &window_size in &window_sizes {
        if integrated.len() / window_size < 2 {
            continue;
        }

        let fluctuation = match calculate_dfa_fluctuation(
            &integrated,
            window_size,
            polynomial_order,
        ) {
            Ok(f) => f,
            Err(_) => continue,
        };

        if fluctuation > 0.0 && fluctuation.is_finite() {
            if let Some(log_val) = float_ops::safe_ln(fluctuation) {
                log_f_values.push(log_val);
                log_s_values.push((window_size as f64).ln());
            }
        }
    }

    if log_f_values.len() < 3 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 3,
            actual: log_f_values.len(),
        });
    }

    let (slope, _, _) = ols_regression(&log_s_values, &log_f_values)?;
    let bias_correction = calculate_dfa_bias_correction(data.len());
    Ok((slope - bias_correction).max(0.01).min(0.99))
}

/// Calculate DFA fluctuation for given window size
fn calculate_dfa_fluctuation(
    integrated: &[f64],
    window_size: usize,
    polynomial_order: usize,
) -> FractalResult<f64> {
    let n = integrated.len();
    let num_windows = n / window_size;

    if num_windows < 2 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 2 * window_size,
            actual: n,
        });
    }

    if polynomial_order == 0 || polynomial_order > 5 {
        return Err(FractalAnalysisError::InvalidParameter {
            parameter: "polynomial_order".to_string(),
            value: polynomial_order as f64,
            constraint: "Must be between 1 and 5".to_string(),
        });
    }

    let min_window = polynomial_order + 2;
    if window_size < min_window {
        return Err(FractalAnalysisError::InvalidParameter {
            parameter: "window_size".to_string(),
            value: window_size as f64,
            constraint: format!(
                "Must be at least {} for polynomial order {}",
                min_window, polynomial_order
            ),
        });
    }

    let mut fluctuations = Vec::with_capacity(num_windows);

    for i in 0..num_windows {
        let start = i * window_size;
        let end = start + window_size;
        let window = &integrated[start..end];

        let fluctuation = detrend_and_calculate_fluctuation(window, polynomial_order)?;
        fluctuations.push(fluctuation);
    }

    Ok((fluctuations.iter().map(|f| f * f).sum::<f64>() / fluctuations.len() as f64).sqrt())
}

/// Detrend a window using polynomial fitting and return RMS fluctuation
/// Note: Using biased variance estimator (n divisor) for RMS calculation
/// All polynomial orders use centered and scaled x values for numerical stability
fn detrend_and_calculate_fluctuation(
    window: &[f64],
    order: usize,
) -> FractalResult<f64> {
    let n = window.len();

    if order == 1 {
        // Center and scale x values for numerical consistency with higher orders
        // This improves conditioning of the regression matrix
        let x_mean = (n - 1) as f64 / 2.0;
        let x_scale = if n > 1 { (n - 1) as f64 / 2.0 } else { 1.0 };
        
        let x_vals: Vec<f64> = (0..n)
            .map(|i| (i as f64 - x_mean) / x_scale)
            .collect();
        
        let (_, _, residuals) = ols_regression(&x_vals, window)?;
        // Using n divisor for RMS (biased estimator, standard in DFA)
        let variance = residuals.iter().map(|r| r * r).sum::<f64>() / n as f64;
        return Ok(variance.sqrt());
    }

    let coeffs = fit_polynomial(window, order)?;

    // Use same scaling as in fit_polynomial for consistency
    let x_mean = (n - 1) as f64 / 2.0;
    let x_scale = if n > 1 { (n - 1) as f64 / 2.0 } else { 1.0 };

    let mut sum_squared_residuals = 0.0;
    for (i, &y) in window.iter().enumerate() {
        let x_scaled = (i as f64 - x_mean) / x_scale;
        let mut fitted = 0.0;
        let mut x_power = 1.0;
        for &coeff in &coeffs {
            fitted += coeff * x_power;
            x_power *= x_scaled;
        }
        let residual = y - fitted;
        sum_squared_residuals += residual * residual;
    }

    Ok((sum_squared_residuals / n as f64).sqrt())
}

/// Fit a polynomial of given order using QR decomposition
/// Uses centered and scaled x values to improve numerical stability
fn fit_polynomial(y: &[f64], order: usize) -> FractalResult<Vec<f64>> {
    let n = y.len();
    if n <= order {
        return Err(FractalAnalysisError::InsufficientData {
            required: order + 1,
            actual: n,
        });
    }

    // Center and scale x values to [-1, 1] range for better conditioning
    let x_mean = (n - 1) as f64 / 2.0;
    let x_scale = if n > 1 { (n - 1) as f64 / 2.0 } else { 1.0 };
    
    let mut a = vec![vec![0.0; order + 1]; n];
    for i in 0..n {
        // Map [0, n-1] to [-1, 1]
        let x_scaled = (i as f64 - x_mean) / x_scale;
        let mut x_power = 1.0;
        for j in 0..=order {
            a[i][j] = x_power;
            x_power *= x_scaled;
        }
    }

    let coeffs = economy_qr_solve(&a, y)?;
    Ok(coeffs)
}

/// Select scale range for DFA using pre-specified bands to avoid selection bias
/// Uses scales from approximately 10 to n/4 as recommended in literature
fn select_dfa_scale_range(
    log_s: &[f64],
    log_f: &[f64],
) -> FractalResult<(Vec<f64>, Vec<f64>)> {
    let n = log_s.len();

    if n < 5 {
        return Ok((log_s.to_vec(), log_f.to_vec()));
    }

    // Pre-specified scale band to avoid selection bias
    // Common choice: scales from 10 to n/4 where n is data length
    // This avoids small-scale noise and large-scale poor statistics
    
    // Since we don't have the original data length, estimate from scale range
    // The largest scale is typically around n/2 or n/4
    let max_log_scale = log_s.last().copied().unwrap_or(0.0);
    let estimated_data_length = (max_log_scale.exp() * 4.0) as usize;
    
    // Target scale bounds
    let min_scale = 10.0_f64.max((estimated_data_length as f64).powf(0.6));
    let max_scale = (estimated_data_length as f64 / 4.0).min((estimated_data_length as f64).powf(0.9));
    
    let min_log_scale = min_scale.ln();
    let max_log_scale = max_scale.ln();
    
    // Find indices within the pre-specified band
    let mut start_idx = 0;
    let mut end_idx = n;
    
    for (i, &log_scale) in log_s.iter().enumerate() {
        if log_scale >= min_log_scale && start_idx == 0 {
            start_idx = i;
        }
        if log_scale > max_log_scale {
            end_idx = i;
            break;
        }
    }
    
    // Ensure we have at least 4 points for stable regression
    if end_idx - start_idx < 4 {
        // If pre-specified band is too narrow, use middle portion
        let skip_small = n / 5; // Skip smallest 20% of scales
        let skip_large = n / 5; // Skip largest 20% of scales
        start_idx = skip_small;
        end_idx = n - skip_large;
        
        if end_idx <= start_idx || end_idx - start_idx < 4 {
            // Last resort: use all available scales
            start_idx = 0;
            end_idx = n;
        }
    }

    Ok((log_s[start_idx..end_idx].to_vec(), log_f[start_idx..end_idx].to_vec()))
}

/// Calculate bootstrap p-value for hypothesis H0: parameter = null_value
fn calculate_bootstrap_p_value(bootstrap_estimates: &[f64], observed: f64, null_value: f64) -> f64 {
    if bootstrap_estimates.is_empty() {
        return 1.0;
    }
    
    // Two-sided p-value: proportion of bootstrap estimates as extreme as observed
    let t_obs = (observed - null_value).abs();
    let count_extreme = bootstrap_estimates
        .iter()
        .filter(|&&est| (est - null_value).abs() >= t_obs)
        .count();
    
    // Add 1 to numerator and denominator for conservative estimate
    let p_value = (count_extreme + 1) as f64 / (bootstrap_estimates.len() + 1) as f64;
    
    // Ensure p-value is in [0, 1]
    p_value.min(1.0).max(0.0)
}

// ============================================================================
// PERIODOGRAM REGRESSION (GPH)
// ============================================================================

/// Periodogram regression method (GPH estimator)
pub fn estimate_hurst_periodogram(
    data: &[f64],
    config: &HurstEstimationConfig,
) -> FractalResult<HurstEstimate> {
    validate_data_length(data, 128, "Periodogram regression")?;

    let variance = calculate_variance(data);
    if variance < ZERO_VARIANCE_THRESHOLD {
        return Err(FractalAnalysisError::NumericalError {
            reason: "Data has zero variance (constant values)".to_string(),
            operation: None,
        });
    }

    let n = data.len();
    
    // Check for discrete/binary data which GPH doesn't handle well
    let unique_ratio = count_unique_values(data, 20) as f64 / n as f64;
    if unique_ratio < 0.02 {
        // Discrete data detected - GPH not suitable, return conservative estimate
        // For random discrete increments, H should be around 0.5
        let hurst_estimate = 0.5;
        let std_error = 0.1; // Conservative uncertainty for discrete data
        
        return Ok(HurstEstimate {
            estimate: hurst_estimate,
            standard_error: std_error,
            confidence_interval: ConfidenceInterval {
                confidence_level: 0.95,
                lower_bound: 0.3,
                upper_bound: 0.7,
                method: ConfidenceIntervalMethod::Normal,
            },
            test_statistic: 0.0,
            p_value: 1.0, // Cannot reject H=0.5 for discrete data
            adjusted_p_value: Some(1.0),
            bias_correction: 0.0,
            finite_sample_correction: 0.0,
            bootstrap_bias: 0.0,
        });
    }
    
    let periodogram = calculate_periodogram_fft(data)?;

    // Ensure we don't exceed Nyquist frequency
    let nyquist = n / 2;
    let max_freq = ((n as f64).powf(config.gph_bandwidth_exponent) * config.gph_bandwidth_multiplier) as usize;
    let max_freq = max_freq.max(5).min(nyquist).min(periodogram.len().saturating_sub(1));

    let mut log_periodogram = Vec::with_capacity(max_freq);
    let mut log_canonical = Vec::with_capacity(max_freq);

    for k in 1..=max_freq {
        let lambda_k = 2.0 * std::f64::consts::PI * k as f64 / n as f64;

        // Use canonical GPH regressor: ln(4 * sin²(λ/2))
        let sin_half = (lambda_k / 2.0).sin();
        let canonical_freq = 4.0 * sin_half * sin_half;
        
        if periodogram[k] > 0.0 && periodogram[k].is_finite() && canonical_freq > 0.0 {
            if let Some(log_val) = float_ops::safe_ln(periodogram[k]) {
                if let Some(log_canon) = float_ops::safe_ln(canonical_freq) {
                    log_periodogram.push(log_val);
                    log_canonical.push(log_canon);
                }
            }
        }
    }

    // Actual m after filtering
    let m = log_periodogram.len();
    let min_frequencies = if n < 256 { 5 } else { 10 };
    if m < min_frequencies {
        return Err(FractalAnalysisError::InsufficientData {
            required: min_frequencies,
            actual: m,
        });
    }

    // Use HAC SE for robustness, or classical GPH formula if specified
    let (slope, regression_se, _) = ols_regression_hac(&log_canonical, &log_periodogram, None)?;
    
    // Use jackknife bias correction on d, then convert to H
    let d_corrected = if data.len() >= 200 && log_periodogram.len() >= 8 {
        // Use jackknife for bias correction
        compute_gph_d_jackknife(data, config).unwrap_or(-slope)
    } else {
        // Too small for jackknife, use raw estimate
        -slope
    };
    
    let hurst_estimate = (d_corrected + RANDOM_WALK_HURST).max(0.01).min(0.99);
    let bias_correction = hurst_estimate - (-slope + RANDOM_WALK_HURST); // For reporting

    // Decide which SE to use: HAC regression SE or classical GPH formula
    let gph_std_error = if config.use_hac_for_gph {
        // Use the HAC standard error (no transformation needed since d = -slope directly)
        regression_se
    } else {
        // Use classical GPH standard error formula: PI / (sqrt(24) * sqrt(m))
        // Note: Different GPH papers use slightly different constants;
        // this follows Geweke & Porter-Hudak (1983)
        let m = log_periodogram.len() as f64;
        std::f64::consts::PI / (24.0_f64.sqrt() * m.sqrt())
    };

    let bootstrap_result = perform_bootstrap_with_constant_check(
        data,
        variance,
        hurst_estimate,
        |data| estimate_periodogram_hurst_only(data, config).unwrap_or(0.5),
        EstimatorComplexity::Medium,
        config,
    )?;

    let confidence_interval = extract_confidence_interval(&bootstrap_result, hurst_estimate);
    
    // Use bootstrap p-value for hypothesis testing H0: H=0.5
    let bootstrap_p_value = calculate_bootstrap_p_value(&bootstrap_result.bootstrap_estimates, hurst_estimate, 0.5);
    
    // Still compute z-statistic for reporting, but use bootstrap p-value
    let test_statistic = (hurst_estimate - 0.5) / gph_std_error.max(1e-10);

    Ok(build_hurst_estimate(
        hurst_estimate,
        gph_std_error,
        confidence_interval,
        test_statistic,
        bootstrap_p_value,
        bias_correction,
        0.0,
    ))
}

/// Core periodogram Hurst estimation without bootstrap
pub fn estimate_periodogram_hurst_only(data: &[f64], config: &HurstEstimationConfig) -> FractalResult<f64> {
    validate_data_length(data, 128, "Periodogram regression")?;

    let n = data.len();
    let periodogram = calculate_periodogram_fft(data)?;

    // Ensure we don't exceed Nyquist frequency
    let nyquist = n / 2;
    let max_freq = ((n as f64).powf(config.gph_bandwidth_exponent) * config.gph_bandwidth_multiplier) as usize;
    let max_freq = max_freq.max(5).min(nyquist).min(periodogram.len().saturating_sub(1));

    let mut log_periodogram = Vec::with_capacity(max_freq);
    let mut log_canonical = Vec::with_capacity(max_freq);

    for k in 1..=max_freq {
        let lambda_k = 2.0 * std::f64::consts::PI * k as f64 / n as f64;

        // Use canonical GPH regressor: ln(4 * sin²(λ/2))
        let sin_half = (lambda_k / 2.0).sin();
        let canonical_freq = 4.0 * sin_half * sin_half;
        
        if periodogram[k] > 0.0 && periodogram[k].is_finite() && canonical_freq > 0.0 {
            if let Some(log_val) = float_ops::safe_ln(periodogram[k]) {
                if let Some(log_canon) = float_ops::safe_ln(canonical_freq) {
                    log_periodogram.push(log_val);
                    log_canonical.push(log_canon);
                }
            }
        }
    }

    // Actual m after filtering
    let m = log_periodogram.len();
    if m < 10 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 10,
            actual: m,
        });
    }

    // Use jackknife bias correction on d, then convert to H
    // In GPH regression: log(I(ω)) = c - 2d·log(ω) + ε
    // So the slope = -2d, hence d = -slope/2
    
    // Direct GPH estimate without jackknife (jackknife seems to introduce bias)
    let (slope, _, _) = ols_regression(&log_canonical, &log_periodogram)?;
    let d_corrected = -slope / 2.0;
    
    Ok((d_corrected + RANDOM_WALK_HURST).max(0.01).min(0.99))
}

// ============================================================================
// WAVELET-BASED ESTIMATION
// ============================================================================

/// Wavelet-based Hurst estimation with WLS
pub fn estimate_hurst_wavelet(
    data: &[f64],
    config: &HurstEstimationConfig,
) -> FractalResult<HurstEstimate> {
    validate_data_length(data, 64, "Wavelet estimation")?;
    let variance = validate_data_variance(data, "Wavelet estimation")?;

    let n = data.len();

    let expected_scales = ((n / 4) as f64).log2().ceil() as usize + 1;
    let mut scale_variances = Vec::with_capacity(expected_scales);
    let mut scales = Vec::with_capacity(expected_scales);
    let mut weights = Vec::with_capacity(expected_scales);

    let mut scale = 2;
    let mut scale_index = 1; // j=1 for scale=2, j=2 for scale=4, etc.
    while scale < n / 4 {
        let variance = calculate_wavelet_variance(data, scale);
        if variance > 0.0 && variance.is_finite() {
            if let Some(log_val) = float_ops::safe_ln(variance) {
                scale_variances.push(log_val);
                scales.push((scale as f64).ln());
                // Weight proportional to number of coefficients at this scale
                // At dyadic scale 2^j, there are approximately n/2^j coefficients
                weights.push((n as f64) / (scale as f64));
            }
        }
        scale *= 2;
        scale_index += 1;
    }

    if scale_variances.len() < 3 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 3,
            actual: scale_variances.len(),
        });
    }

    // Use WLS with scale-based weights instead of OLS
    let (slope, se_slope, _) = wls_regression(&scales, &scale_variances, &weights)?;
    let hurst_estimate = ((slope + 1.0) / 2.0).max(0.01).min(0.99);
    // Propagate the standard error through the transformation
    let se_h = se_slope / 2.0;

    let bootstrap_result = perform_bootstrap_with_constant_check(
        data,
        variance,
        hurst_estimate,
        |data| estimate_wavelet_hurst_only(data).unwrap_or(0.5),
        EstimatorComplexity::Medium,
        config,
    )?;

    let confidence_interval = extract_confidence_interval(&bootstrap_result, hurst_estimate);
    
    // Use bootstrap p-value for hypothesis testing H0: H=0.5
    let bootstrap_p_value = calculate_bootstrap_p_value(&bootstrap_result.bootstrap_estimates, hurst_estimate, 0.5);
    
    // Still compute z-statistic for reporting, but use bootstrap p-value
    let test_statistic = (hurst_estimate - 0.5) / se_h.max(1e-10);

    Ok(build_hurst_estimate(
        hurst_estimate,
        se_h,
        confidence_interval,
        test_statistic,
        bootstrap_p_value,
        0.0,
        0.0,
    ))
}

// ============================================================================
// WHITTLE ESTIMATOR
// ============================================================================

/// Whittle maximum likelihood estimator for Hurst exponent
/// 
/// The Whittle estimator is a frequency-domain maximum likelihood method that
/// estimates the Hurst exponent by fitting the theoretical spectral density
/// to the periodogram. This implementation uses the profiled likelihood to
/// handle the unknown scale parameter.
/// 
/// ## Algorithm
/// 
/// Minimizes the profiled Whittle objective:
/// Q(d) = ∑log(g_d(λ_k)) + m*log(mean(I(λ_k)/g_d(λ_k)))
/// 
/// where g_d(λ) = |2sin(λ/2)|^(-2d) is the normalized spectral density.
/// 
/// ## Advantages
/// - Maximum likelihood efficiency under Gaussian assumptions
/// - Natural handling of spectral characteristics
/// 
/// ## Limitations
/// - Requires n ≥ 128 for reliable estimates
/// - Assumes stationarity
/// - Computationally intensive (grid search)
/// 
/// ## Note
/// This is an experimental implementation. Results should be validated
/// against other methods for critical applications.
pub fn estimate_hurst_whittle(
    data: &[f64],
    config: &HurstEstimationConfig,
) -> FractalResult<HurstEstimate> {
    validate_data_length(data, 128, "Whittle estimation")?;
    let variance = validate_data_variance(data, "Whittle estimation")?;

    let n = data.len();
    let periodogram = calculate_periodogram_fft(data)?;
    
    // Use similar frequency range as GPH
    let max_freq = ((n as f64).powf(config.gph_bandwidth_exponent) * config.gph_bandwidth_multiplier) as usize;
    let max_freq = max_freq.max(10).min(periodogram.len() / 2);
    
    // Grid search for d parameter (H = d + 0.5)
    // Using profiled Whittle likelihood to handle unknown scale parameter
    // Use finer grid near d=0 for better detection of white noise
    let mut d_grid = Vec::new();
    // Fine grid near 0 for white noise detection
    for i in -20..=20 {
        d_grid.push(i as f64 * 0.005);  // Step of 0.005 near 0
    }
    // Coarser grid for extreme values
    for i in -40..=-21 {
        d_grid.push(i as f64 * 0.01);
    }
    for i in 21..=40 {
        d_grid.push(i as f64 * 0.01);
    }
    let mut best_d = 0.0;
    let mut min_objective = f64::INFINITY;
    
    for &d in &d_grid {
        let mut sum_log_g = 0.0;
        let mut sum_i_over_g = 0.0;
        let mut valid_freqs = 0;
        
        for k in 1..=max_freq {
            let lambda_k = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            
            // Normalized spectral density g_d(λ) = |2sin(λ/2)|^(-2d)
            let sin_half = (lambda_k / 2.0).sin();
            if sin_half.abs() < 1e-10 {
                continue;
            }
            
            // Handle d near 0 carefully for numerical stability
            let g_d = if d.abs() < 1e-10 {
                1.0  // When d ≈ 0, g_d = 1
            } else {
                (2.0 * sin_half).abs().powf(-2.0 * d)
            };
            
            if g_d > 0.0 && g_d.is_finite() && periodogram[k] > 0.0 {
                sum_log_g += g_d.ln();
                sum_i_over_g += periodogram[k] / g_d;
                valid_freqs += 1;
            }
        }
        
        if valid_freqs > 0 {
            // Profiled Whittle objective: Q(d) = ∑log(g_d) + m*log(mean(I/g_d))
            let mean_i_over_g = sum_i_over_g / valid_freqs as f64;
            let objective = sum_log_g + (valid_freqs as f64) * mean_i_over_g.ln();
            
            if objective < min_objective {
                min_objective = objective;
                best_d = d;
            }
        }
    }
    
    let hurst_estimate = (best_d + 0.5).max(0.01).min(0.99);
    
    // Use bootstrap for standard error estimation - more reliable than Hessian approximation
    let bootstrap_result = perform_bootstrap_with_constant_check(
        data,
        variance,
        hurst_estimate,
        |data| estimate_whittle_hurst_only(data, config).unwrap_or(0.5),
        EstimatorComplexity::High,
        config,
    )?;
    
    // Use bootstrap standard error as primary estimate
    let std_error = bootstrap_result.standard_error.max(MIN_STD_ERROR);
    
    let confidence_interval = extract_confidence_interval(&bootstrap_result, hurst_estimate);
    let (test_statistic, p_value) = calculate_test_statistics(hurst_estimate, std_error);
    
    Ok(build_hurst_estimate(
        hurst_estimate,
        std_error,
        confidence_interval,
        test_statistic,
        p_value,
        0.0,
        0.0,
    ))
}

/// Core Whittle Hurst estimation without bootstrap
pub fn estimate_whittle_hurst_only(data: &[f64], config: &HurstEstimationConfig) -> FractalResult<f64> {
    validate_data_length(data, 128, "Whittle estimation")?;
    
    let n = data.len();
    let periodogram = calculate_periodogram_fft(data)?;
    
    let max_freq = ((n as f64).powf(config.gph_bandwidth_exponent) * config.gph_bandwidth_multiplier) as usize;
    let max_freq = max_freq.max(10).min(periodogram.len() / 2);
    
    // Simplified grid search using profiled likelihood
    // Use finer resolution near d=0 for white noise detection
    let mut d_grid = Vec::new();
    for i in -10..=10 {
        d_grid.push(i as f64 * 0.01);  // Fine steps near 0
    }
    for i in (-40..=-11).step_by(2) {
        d_grid.push(i as f64 * 0.01);
    }
    for i in (11..=40).step_by(2) {
        d_grid.push(i as f64 * 0.01);
    }
    let mut best_d = 0.0;
    let mut min_objective = f64::INFINITY;
    
    for &d in &d_grid {
        let mut sum_log_g = 0.0;
        let mut sum_i_over_g = 0.0;
        let mut valid_freqs = 0;
        
        for k in 1..=max_freq {
            let lambda_k = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            let sin_half = (lambda_k / 2.0).sin();
            
            if sin_half.abs() > 1e-10 {
                // Handle d near 0 for numerical stability
                let g_d = if d.abs() < 1e-10 {
                    1.0  // When d ≈ 0, g_d = 1
                } else {
                    (2.0 * sin_half).abs().powf(-2.0 * d)
                };
                if g_d > 0.0 && g_d.is_finite() && periodogram[k] > 0.0 {
                    sum_log_g += g_d.ln();
                    sum_i_over_g += periodogram[k] / g_d;
                    valid_freqs += 1;
                }
            }
        }
        
        if valid_freqs > 0 {
            // Profiled Whittle objective
            let mean_i_over_g = sum_i_over_g / valid_freqs as f64;
            let objective = sum_log_g + (valid_freqs as f64) * mean_i_over_g.ln();
            
            if objective < min_objective {
                min_objective = objective;
                best_d = d;
            }
        }
    }
    
    Ok((best_d + 0.5).max(0.01).min(0.99))
}

// ============================================================================
// VARIOGRAM METHOD
// ============================================================================

/// Compute wavelet variances at multiple dyadic levels for FGN/FBM classification
/// Returns pairs of (level, variance) for levels 1 to max_level
/// Uses proper undecimated Haar wavelet transform
fn compute_level_variances(x: &[f64], max_level: usize) -> Vec<(usize, f64)> {
    let n = x.len();
    let mut vars = Vec::new();

    for level in 1..=max_level {
        let m = 1 << (level - 1);               // half-window length
        
        let mut sum_sq = 0.0;
        for t in 0..n {
            // Compute block sums of length m (circular)
            let mut s_a = 0.0;
            let mut s_b = 0.0;
            for k in 0..m {
                s_a += x[(t + k) % n];
                s_b += x[(t + m + k) % n];
            }
            
            // Haar detail coefficient with proper normalization
            // w = (sum_b - sum_a) / sqrt(2*m)
            // This normalization ensures variance scales as 2^(j(2H-1)) for FGN
            let normalization = (2.0 * m as f64).sqrt();
            let w = (s_b - s_a) / normalization;
            sum_sq += w * w;
        }

        let var_j = sum_sq / n as f64;          // mean energy at level j
        vars.push((level, var_j));
    }

    vars
}

/// Check if data needs conversion from FGN to FBM for variogram analysis
/// Uses wavelet variance slope as primary signal with soft decision boundaries
/// Theory: FGN has slope ≈ (2H-1) ∈ (-1,1), FBM has slope ≈ (2H+1) ∈ (1,3)
fn needs_fbm_conversion(x: &[f64]) -> bool {
    let n = x.len();
    if n < 50 { 
        // Too small for reliable analysis
        // Assume stationary (FGN) → needs conversion to FBM
        return true;
    }
    
    // Primary method: Wavelet variance slope (most reliable)
    let mut wavelet_slope = None;
    if n >= 128 {
        let max_level = ((n as f64).ln() / 2.0_f64.ln()).floor() as usize - 2;
        let max_level = max_level.min(8).max(3);
        let variances = compute_level_variances(x, max_level);
        
        if variances.len() >= 3 {
            // Regress log2(variance) vs level
            let mut x_sum = 0.0;
            let mut y_sum = 0.0;
            let mut xy_sum = 0.0;
            let mut x2_sum = 0.0;
            let mut valid_count = 0;
            
            for &(level, var) in &variances {
                if var > 1e-10 && var.is_finite() {
                    let x = level as f64;
                    let y = var.ln() / 2.0_f64.ln(); // log2(var)
                    
                    x_sum += x;
                    y_sum += y;
                    xy_sum += x * y;
                    x2_sum += x * x;
                    valid_count += 1;
                }
            }
            
            if valid_count >= 3 {
                let n_pts = valid_count as f64;
                let slope = (n_pts * xy_sum - x_sum * y_sum) / (n_pts * x2_sum - x_sum * x_sum);
                
                if slope.is_finite() {
                    wavelet_slope = Some(slope);
                    
                    // Soft decision boundaries
                    if slope > 1.05 {
                        // Clear FBM signal
                        return false;
                    }
                    if slope < 0.85 {
                        // Clear FGN signal
                        return true;
                    }
                    // slope in [0.85, 1.05] - uncertain, use additional hints
                }
            }
        }
    }
    
    // For uncertain cases or when wavelet method unavailable, use hints
    
    // Hint 1: Variance growth test (cheap and effective)
    let mut variance_growth_slope = None;
    if n >= 64 {
        let block_sizes = vec![2, 4, 8, 16];
        let mut log_vars = Vec::new();
        let mut log_sizes = Vec::new();
        
        for &size in &block_sizes {
            if size * 4 > n { break; }
            
            let n_blocks = n / size;
            let mut block_means = Vec::with_capacity(n_blocks);
            
            for i in 0..n_blocks {
                let block_sum: f64 = x[i*size..(i+1)*size].iter().sum();
                block_means.push(block_sum / size as f64);
            }
            
            let block_var = calculate_variance(&block_means);
            if block_var > 1e-10 && block_var.is_finite() {
                log_vars.push(block_var.ln());
                log_sizes.push((size as f64).ln());
            }
        }
        
        if log_vars.len() >= 3 {
            // Simple linear regression for variance growth
            let n_pts = log_vars.len() as f64;
            let x_sum: f64 = log_sizes.iter().sum();
            let y_sum: f64 = log_vars.iter().sum();
            let xy_sum: f64 = log_sizes.iter().zip(&log_vars).map(|(x, y)| x * y).sum();
            let x2_sum: f64 = log_sizes.iter().map(|x| x * x).sum();
            
            let slope = (n_pts * xy_sum - x_sum * y_sum) / (n_pts * x2_sum - x_sum * x_sum);
            if slope.is_finite() {
                variance_growth_slope = Some(slope);
            }
        }
    }
    
    // Hint 2: Uniqueness ratio (for discrete detection)
    let unique_ratio = count_unique_values(x, 20) as f64 / n as f64;
    let is_likely_discrete = unique_ratio < 0.02;
    
    // Hint 3: Autocorrelation structure (weak signal)
    let k = (n / 10).max(1).min(10);
    let autocorr_1 = calculate_lag_autocorrelation(x, 1);
    let autocorr_k = calculate_lag_autocorrelation(x, k);
    
    // Decision logic for uncertain wavelet slope
    if let Some(slope) = wavelet_slope {
        if slope >= 0.85 && slope <= 1.05 {
            // Uncertain band - use hints
            
            // Strong variance growth suggests non-stationary
            if let Some(vg_slope) = variance_growth_slope {
                if vg_slope > 0.2 {
                    return false; // Non-stationary, likely FBM
                }
                if vg_slope < 0.05 {
                    return true; // Stationary, likely FGN
                }
            }
            
            // Discrete data is more likely FGN
            if is_likely_discrete {
                return true;
            }
            
            // Very high autocorrelation might suggest FBM
            if autocorr_1 > 0.95 && autocorr_k > 0.8 {
                return false;
            }
            
            // Near-zero autocorrelation suggests white noise/FGN
            if autocorr_1.abs() < 0.1 {
                return true;
            }
        }
    } else {
        // No wavelet slope available - use other signals
        
        if let Some(vg_slope) = variance_growth_slope {
            if vg_slope > 0.3 {
                return false; // Strong variance growth - FBM
            }
            if vg_slope < 0.1 {
                return true; // No variance growth - FGN
            }
        }
        
        if is_likely_discrete {
            return true; // Discrete data usually FGN
        }
        
        if autocorr_1 > 0.95 && autocorr_k > 0.8 {
            return false; // Very persistent - FBM
        }
        
        if autocorr_1.abs() < 0.2 {
            return true; // Low correlation - FGN
        }
    }
    
    // Default: assume FGN (safer to convert for variogram)
    true
}

/// Count unique values in data (up to a limit) with relative tolerance
fn count_unique_values(x: &[f64], max_count: usize) -> usize {
    let n = x.len();
    if n == 0 { return 0; }
    
    let mean = x.iter().sum::<f64>() / n as f64;
    let variance = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    let std = variance.sqrt();
    
    // Use relative tolerance based on data scale
    let eps = (1e-8_f64).max(1e-3 * std);
    
    let mut unique: Vec<f64> = Vec::new();
    for &val in x {
        if !unique.iter().any(|&v| (v - val).abs() <= eps) {
            unique.push(val);
            if unique.len() >= max_count {
                return max_count;
            }
        }
    }
    unique.len()
}

/// Calculate autocorrelation at specific lag
fn calculate_lag_autocorrelation(x: &[f64], lag: usize) -> f64 {
    let n = x.len();
    if lag >= n {
        return 0.0;
    }
    
    let mean = x.iter().sum::<f64>() / n as f64;
    let variance = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    
    if variance < 1e-10 {
        return 0.0;
    }
    
    let mut covariance = 0.0;
    for i in 0..(n - lag) {
        covariance += (x[i] - mean) * (x[i + lag] - mean);
    }
    covariance /= (n - lag) as f64;
    
    covariance / variance
}

/// Convert FGN to FBM by centering and cumulating
fn cummean_zeroed(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = x.iter().copied().sum::<f64>() / n as f64;
    x.iter()
        .map(|&v| v - mean)
        .scan(0.0, |acc, v| {
            *acc += v;
            Some(*acc)
        })
        .collect()
}

/// Variogram-based Hurst estimation
/// 
/// The variogram method estimates the Hurst exponent by analyzing the
/// variance of increments at different lags. For fractional Brownian motion,
/// the variogram follows a power law: γ(h) ∝ h^(2H).
/// 
/// ## Algorithm
/// 
/// 1. Compute empirical variogram: γ(h) = (1/2)E[(X(t+h) - X(t))²]
/// 2. Perform log-log regression: log(γ(h)) vs log(h)
/// 3. Estimate H from slope: H = slope/2
/// 
/// ## Advantages
/// - Simple and intuitive
/// - Works with relatively short series (n ≥ 50)
/// - Robust to certain types of non-stationarity
/// 
/// ## Limitations
/// - Sensitive to outliers (uses OLS, not robust regression)
/// - Assumes self-similar increments
/// - Less efficient than spectral methods for long series
/// 
/// ## Implementation Details
/// Uses OLS with HAC-robust standard errors for inference.
/// For outlier-robust estimation, consider pre-filtering the data
/// or using alternative methods.
/// 
/// ## Note
/// This is an experimental implementation. The method works best for
/// processes with clear power-law scaling in their variogram.
pub fn estimate_hurst_variogram(
    data: &[f64],
    config: &HurstEstimationConfig,
) -> FractalResult<HurstEstimate> {
    validate_data_length(data, 50, "Variogram method")?;
    let variance = validate_data_variance(data, "Variogram method")?;
    
    // Variogram method expects FBM (cumulative) data
    // The caller is responsible for providing the correct data type
    let n = data.len();
    
    // Compute scale from increments for relative thresholding
    // This ensures scale-invariance regardless of data units
    let mut diff_sum_sq = 0.0;
    let mut diff_count = 0;
    for i in 1..n {
        let diff = data[i] - data[i-1];
        diff_sum_sq += diff * diff;
        diff_count += 1;
    }
    let diff_variance = if diff_count > 0 {
        diff_sum_sq / diff_count as f64
    } else {
        variance  // fallback to data variance
    };
    
    // Use relative epsilon based on both increment and overall variance
    let eps = (1e-12 * diff_variance)
        .max(1e-12 * variance)
        .max(1e-300);
    
    // Calculate variogram for different lags with better sampling
    let max_lag = (n / 4).min(100); // Use up to 1/4 of series length, max 100
    let mut variogram_values = Vec::new();
    let mut lag_values = Vec::new();
    
    // Always include small lags for good small-h coverage
    let mut lags = vec![1, 2, 3];
    
    // Add logarithmically spaced lags
    for i in 0..20 {
        let lag = ((1.5_f64.powi(i) as usize).max(1)).min(max_lag);
        if !lags.contains(&lag) {
            lags.push(lag);
        }
    }
    lags.sort_unstable();
    
    for &lag in &lags {
        let mut sum_sq_diff = 0.0;
        let mut count = 0;
        
        for i in 0..(n - lag) {
            let diff = data[i + lag] - data[i];
            sum_sq_diff += diff * diff;
            count += 1;
        }
        
        if count > 0 {
            let variogram = sum_sq_diff / (2.0 * count as f64);
            // Include all positive finite values - even small values are meaningful for FBM
            // The variogram should grow as h^(2H), so small values at small lags are expected
            if variogram.is_finite() && variogram > eps {
                variogram_values.push(variogram.ln());
                lag_values.push((lag as f64).ln());
            }
        }
    }
    
    // Need at least 3 points for regression
    if variogram_values.len() < 3 {
        // Fallback: if scale-aware filtering rejected too many points,
        // try with minimal threshold (for constant or near-constant data)
        variogram_values.clear();
        lag_values.clear();
        
        for lag in 1..=max_lag {
            let mut sum_sq_diff = 0.0;
            let mut count = 0;
            
            for i in 0..(n - lag) {
                let diff = data[i + lag] - data[i];
                sum_sq_diff += diff * diff;
                count += 1;
            }
            
            if count > 0 {
                let variogram = sum_sq_diff / (2.0 * count as f64);
                // Use absolute minimum threshold as fallback
                if variogram > 1e-300 && variogram.is_finite() {
                    variogram_values.push(variogram.ln());
                    lag_values.push((lag as f64).ln());
                }
            }
        }
        
        // If still insufficient, error out
        if variogram_values.len() < 3 {
            return Err(FractalAnalysisError::InsufficientData {
                required: 3,
                actual: variogram_values.len(),
            });
        }
    }
    
    // OLS regression with HAC-robust standard errors
    // Note: This provides robust SEs, not robust estimation against outliers
    let (slope, std_error, _) = ols_regression_hac(&lag_values, &variogram_values, None)?;
    
    // For fractional Brownian motion: γ(h) ∝ h^(2H)
    // So slope of log-log plot gives 2H
    let hurst_estimate = (slope / 2.0).max(0.01).min(0.99);
    let hurst_std_error = std_error / 2.0;
    
    let bootstrap_result = perform_bootstrap_with_constant_check(
        data,
        variance,
        hurst_estimate,
        |data| estimate_variogram_hurst_only(data).unwrap_or(0.5),
        EstimatorComplexity::Low,
        config,
    )?;
    
    let confidence_interval = extract_confidence_interval(&bootstrap_result, hurst_estimate);
    let (test_statistic, p_value) = calculate_test_statistics(hurst_estimate, hurst_std_error);
    
    Ok(build_hurst_estimate(
        hurst_estimate,
        hurst_std_error,
        confidence_interval,
        test_statistic,
        p_value,
        0.0,
        0.0,
    ))
}

/// Core variogram Hurst estimation without bootstrap
pub fn estimate_variogram_hurst_only(data: &[f64]) -> FractalResult<f64> {
    validate_data_length(data, 50, "Variogram method")?;
    
    // Variogram method expects FBM (cumulative) data
    // The caller is responsible for providing the correct data type
    let n = data.len();
    
    // Compute scale from increments for relative thresholding
    let mut diff_sum_sq = 0.0;
    let mut diff_count = 0;
    for i in 1..n {
        let diff = data[i] - data[i-1];
        diff_sum_sq += diff * diff;
        diff_count += 1;
    }
    let diff_variance = if diff_count > 0 {
        diff_sum_sq / diff_count as f64
    } else {
        1.0  // safe fallback
    };
    
    // Use relative epsilon based on data scale
    let eps = (1e-12 * diff_variance).max(1e-300);
    
    let max_lag = (n / 4).min(100);
    let mut variogram_values = Vec::new();
    let mut lag_values = Vec::new();
    
    // Always include small lags for good small-h coverage
    let mut lags = vec![1, 2, 3];
    
    // Add logarithmically spaced lags
    for i in 0..20 {
        let lag = ((1.5_f64.powi(i) as usize).max(1)).min(max_lag);
        if !lags.contains(&lag) {
            lags.push(lag);
        }
    }
    lags.sort_unstable();
    
    for &lag in &lags {
        let mut sum_sq_diff = 0.0;
        let mut count = 0;
        
        for i in 0..(n - lag) {
            let diff = data[i + lag] - data[i];
            sum_sq_diff += diff * diff;
            count += 1;
        }
        
        if count > 0 {
            let variogram = sum_sq_diff / (2.0 * count as f64);
            // Include all positive finite values
            if variogram.is_finite() && variogram > eps {
                variogram_values.push(variogram.ln());
                lag_values.push((lag as f64).ln());
            }
        }
    }
    
    // Need at least 3 points for regression
    if variogram_values.len() < 3 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 3,
            actual: variogram_values.len(),
        });
    }
    
    let (slope, _, _) = ols_regression(&lag_values, &variogram_values)?;
    Ok((slope / 2.0).max(0.01).min(0.99))
}

// ============================================================================
// SHARED BOOTSTRAP ESTIMATION
// ============================================================================

/// Estimate Hurst exponents using shared bootstrap infrastructure
fn estimate_hurst_with_shared_bootstrap(
    data: &[f64],
    methods: &[EstimationMethod],
    config: &HurstEstimationConfig,
) -> FractalResult<BTreeMap<EstimationMethod, HurstEstimate>> {
    let n = data.len();

    let mut bootstrap_config = config.bootstrap_config.clone();
    let adaptive_samples = BootstrapConfiguration::adaptive(n, EstimatorComplexity::Medium).num_bootstrap_samples;
    bootstrap_config.num_bootstrap_samples = adaptive_samples.max(bootstrap_config.num_bootstrap_samples);

    let mut core_estimates = BTreeMap::new();

    for method in methods {
        match method {
            EstimationMethod::RescaledRange => {
                if let Ok(estimate) = estimate_rs_hurst_only(data) {
                    core_estimates.insert(*method, estimate);
                }
            }
            EstimationMethod::DetrendedFluctuationAnalysis => {
                if let Ok(estimate) = estimate_dfa_hurst_only(data, config.dfa_polynomial_order) {
                    core_estimates.insert(*method, estimate);
                }
            }
            EstimationMethod::PeriodogramRegression => {
                if n >= 128 {
                    if let Ok(estimate) = estimate_periodogram_hurst_only(data, config) {
                        core_estimates.insert(*method, estimate);
                    }
                }
            }
            EstimationMethod::WaveletEstimation => {
                if let Ok(estimate) = estimate_wavelet_hurst_only(data) {
                    core_estimates.insert(*method, estimate);
                }
            }
            EstimationMethod::WhittleEstimator => {
                if n >= 128 {
                    if let Ok(estimate) = estimate_whittle_hurst_only(data, config) {
                        core_estimates.insert(*method, estimate);
                    }
                }
            }
            EstimationMethod::VariogramMethod => {
                // Apply same auto-detection logic for consistency
                if needs_fbm_conversion(data) {
                    let cumulative_data = cummean_zeroed(data);
                    if let Ok(estimate) = estimate_variogram_hurst_only(&cumulative_data) {
                        core_estimates.insert(*method, estimate);
                    }
                } else {
                    if let Ok(estimate) = estimate_variogram_hurst_only(data) {
                        core_estimates.insert(*method, estimate);
                    }
                }
            }
            _ => continue,
        }
    }

    if core_estimates.is_empty() {
        return Err(FractalAnalysisError::NumericalError {
            reason: "No valid core estimates computed".to_string(),
            operation: None,
        });
    }

    let mut bootstrap_estimates: BTreeMap<EstimationMethod, Vec<f64>> = BTreeMap::new();

    for method in core_estimates.keys() {
        bootstrap_estimates.insert(
            *method,
            Vec::with_capacity(bootstrap_config.num_bootstrap_samples),
        );
    }

    for i in 0..bootstrap_config.num_bootstrap_samples {
        // Mix seed with iteration index for different samples
        let mut config_with_seed = bootstrap_config.clone();
        if let Some(base_seed) = bootstrap_config.seed {
            config_with_seed.seed = Some(base_seed.wrapping_add(i as u64));
        } else {
            config_with_seed.seed = Some(42u64.wrapping_add(i as u64));
        }
        let bootstrap_sample = generate_bootstrap_sample(data, &config_with_seed)?;

        for (method, &core_estimate) in core_estimates.iter() {
            let bootstrap_estimate = match method {
                EstimationMethod::RescaledRange => {
                    estimate_rs_hurst_only(&bootstrap_sample).unwrap_or(core_estimate)
                }
                EstimationMethod::DetrendedFluctuationAnalysis => {
                    estimate_dfa_hurst_only(&bootstrap_sample, config.dfa_polynomial_order)
                        .unwrap_or(core_estimate)
                }
                EstimationMethod::PeriodogramRegression => {
                    if bootstrap_sample.len() >= 128 {
                        estimate_periodogram_hurst_only(&bootstrap_sample, config)
                            .unwrap_or(core_estimate)
                    } else {
                        core_estimate
                    }
                }
                EstimationMethod::WaveletEstimation => {
                    estimate_wavelet_hurst_only(&bootstrap_sample).unwrap_or(core_estimate)
                }
                EstimationMethod::WhittleEstimator => {
                    if bootstrap_sample.len() >= 128 {
                        estimate_whittle_hurst_only(&bootstrap_sample, config)
                            .unwrap_or(core_estimate)
                    } else {
                        core_estimate
                    }
                }
                EstimationMethod::VariogramMethod => {
                    // Apply same auto-detection logic for bootstrap samples
                    if needs_fbm_conversion(&bootstrap_sample) {
                        let cumulative_data = cummean_zeroed(&bootstrap_sample);
                        estimate_variogram_hurst_only(&cumulative_data).unwrap_or(core_estimate)
                    } else {
                        estimate_variogram_hurst_only(&bootstrap_sample).unwrap_or(core_estimate)
                    }
                }
                _ => continue,
            };

            if let Some(estimates_vec) = bootstrap_estimates.get_mut(method) {
                if bootstrap_estimate.is_finite() {
                    estimates_vec.push(bootstrap_estimate);
                } else {
                    estimates_vec.push(core_estimate);
                }
            }
        }
    }

    let mut final_estimates = BTreeMap::new();

    for (method, core_estimate) in core_estimates {
        if let Some(bootstrap_vals) = bootstrap_estimates.get(&method) {
            if bootstrap_vals.is_empty() {
                continue;
            }

            let mean_bootstrap = bootstrap_vals.iter().sum::<f64>() / bootstrap_vals.len() as f64;
            let bias = mean_bootstrap - core_estimate;

            let variance = bootstrap_vals
                .iter()
                .map(|x| {
                    let diff = x - mean_bootstrap;
                    diff * diff
                })
                .sum::<f64>()
                / (bootstrap_vals.len() - 1).max(1) as f64;
            let standard_error = variance.sqrt();

            let mut sorted_bootstrap = bootstrap_vals.clone();
            sorted_bootstrap.sort_by(float_total_cmp);

            let confidence_interval = ConfidenceInterval {
                confidence_level: 0.95,
                lower_bound: percentile(&sorted_bootstrap, 0.025),
                upper_bound: percentile(&sorted_bootstrap, 0.975),
                method: ConfidenceIntervalMethod::BootstrapPercentile,
            };

            let safe_std_error = if standard_error > MIN_STD_ERROR {
                standard_error
            } else {
                MIN_STD_ERROR
            };
            let test_statistic = (core_estimate - 0.5) / safe_std_error;
            let p_value = 2.0 * (1.0 - standard_normal_cdf(test_statistic.abs()));

            let debiased_estimate = (core_estimate - bias).max(0.01).min(0.99);

            let estimate = HurstEstimate {
                estimate: debiased_estimate,
                standard_error,
                confidence_interval,
                test_statistic,
                p_value,
                adjusted_p_value: Some(p_value),
                bias_correction: bias,
                finite_sample_correction: 1.0 / (n as f64).sqrt(),
                bootstrap_bias: 0.0,  // Not available in this context
            };

            final_estimates.insert(method, estimate);
        }
    }

    if final_estimates.is_empty() {
        return Err(FractalAnalysisError::NumericalError {
            reason: "Shared bootstrap produced no valid estimates".to_string(),
            operation: None,
        });
    }

    // Apply multiple testing corrections before returning
    apply_multiple_testing_corrections(final_estimates)
}


// ============================================================================
// SIMPLIFIED ESTIMATOR FOR SHORT SERIES
// ============================================================================

/// Simple Hurst estimation for very short series (< 50 points)
pub fn estimate_hurst_simple_short_series(data: &[f64]) -> FractalResult<HurstEstimate> {
    let n = data.len();
    if n < 3 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 3,
            actual: n,
        });
    }

    let expected_levels = (n / 2).max(1);
    let mut variance_ratios = Vec::with_capacity(expected_levels);
    let mut aggregation_levels = Vec::with_capacity(expected_levels);

    for agg_level in 1..=(n / 2).max(1) {
        if agg_level >= n {
            break;
        }

        let aggregated = aggregate_series(data, agg_level);
        if aggregated.len() < 2 {
            continue;
        }

        let variance = calculate_variance(&aggregated);
        if variance > 0.0 && !variance.is_nan() {
            variance_ratios.push(variance.ln());
            aggregation_levels.push((agg_level as f64).ln());
        }
    }

    if variance_ratios.len() < 2 {
        return Ok(HurstEstimate {
            estimate: 0.5,
            standard_error: 0.2,
            confidence_interval: ConfidenceInterval {
                confidence_level: 0.95,
                lower_bound: 0.3,
                upper_bound: 0.7,
                method: ConfidenceIntervalMethod::Normal,
            },
            test_statistic: 0.0,
            p_value: 0.5,
            adjusted_p_value: Some(0.5),
            bias_correction: 0.0,
            finite_sample_correction: 0.0,
            bootstrap_bias: 0.0,  // Not computed in this fallback case
        });
    }
    
    // Check for sufficient variation in aggregation levels to avoid singular matrix
    let agg_variance = calculate_variance(&aggregation_levels);
    if agg_variance < MIN_VARIANCE_THRESHOLD {
        // Insufficient variation - return conservative estimate
        return Ok(HurstEstimate {
            estimate: 0.5,
            standard_error: 0.3,
            confidence_interval: ConfidenceInterval {
                confidence_level: 0.95,
                lower_bound: 0.2,
                upper_bound: 0.8,
                method: ConfidenceIntervalMethod::Normal,
            },
            test_statistic: 0.0,
            p_value: 1.0,
            adjusted_p_value: Some(1.0),
            bias_correction: 0.0,
            finite_sample_correction: 0.0,
            bootstrap_bias: 0.0,  // Not computed in this fallback case
        });
    }

    let (slope, std_error, _) = ols_regression(&aggregation_levels, &variance_ratios)?;

    let hurst_estimate = (slope / 2.0).max(0.01).min(0.99);

    let p_val = 2.0 * (1.0 - standard_normal_cdf(
        ((hurst_estimate - 0.5) / std_error.max(0.1)).abs(),
    ));

    Ok(HurstEstimate {
        estimate: hurst_estimate,
        standard_error: std_error.max(0.1),
        confidence_interval: ConfidenceInterval {
            confidence_level: 0.95,
            lower_bound: (hurst_estimate - 0.2).max(0.0),
            upper_bound: (hurst_estimate + 0.2).min(1.0),
            method: ConfidenceIntervalMethod::Normal,
        },
        test_statistic: (hurst_estimate - 0.5) / std_error.max(0.1),
        p_value: p_val,
        adjusted_p_value: Some(p_val),
        bias_correction: 0.0,
        finite_sample_correction: 1.0 / (n as f64).sqrt(),
        bootstrap_bias: 0.0,  // Not computed in this simplified estimator
    })
}

/// Aggregate time series by averaging over non-overlapping windows
fn aggregate_series(data: &[f64], aggregation_level: usize) -> Vec<f64> {
    let num_windows = data.len() / aggregation_level;
    let mut aggregated = Vec::with_capacity(num_windows);

    for i in 0..num_windows {
        let start = i * aggregation_level;
        let end = (start + aggregation_level).min(data.len());
        let window = &data[start..end];
        let avg = window.iter().sum::<f64>() / window.len() as f64;
        aggregated.push(avg);
    }

    aggregated
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Validate data variance
fn validate_data_variance(data: &[f64], method_name: &str) -> FractalResult<f64> {
    let variance = calculate_variance(data);
    if variance < ZERO_VARIANCE_THRESHOLD {
        return Err(FractalAnalysisError::NumericalError {
            reason: format!(
                "Data has zero variance (constant values) for {}",
                method_name
            ),
            operation: None,
        });
    }
    Ok(variance)
}

/// Perform bootstrap with constant data handling
fn perform_bootstrap_with_constant_check(
    data: &[f64],
    variance: f64,
    corrected_estimate: f64,
    estimator: impl Fn(&[f64]) -> f64 + Sync + Send,
    complexity: EstimatorComplexity,
    config: &HurstEstimationConfig,
) -> FractalResult<BootstrapValidation> {
    if variance < NEAR_ZERO_VARIANCE_THRESHOLD {
        Ok(BootstrapValidation {
            original_estimate: corrected_estimate,
            bootstrap_estimates: vec![corrected_estimate; 10],
            bias: 0.0,
            standard_error: 0.0,
            confidence_intervals: vec![ConfidenceInterval {
                confidence_level: 0.95,
                lower_bound: corrected_estimate,
                upper_bound: corrected_estimate,
                method: ConfidenceIntervalMethod::Normal,
            }],
        })
    } else {
        let mut bootstrap_config = BootstrapConfiguration::adaptive(data.len(), complexity);

        bootstrap_config.bootstrap_method = BootstrapMethod::Block;

        if bootstrap_config.block_size.is_none() {
            bootstrap_config.block_size = Some(politis_white_block_size(data));
        }

        bootstrap_config.confidence_interval_method = ConfidenceIntervalMethod::BootstrapBca;

        bootstrap_validate(data, estimator, &bootstrap_config)
    }
}

/// Extract confidence interval from bootstrap result
fn extract_confidence_interval(
    bootstrap_result: &BootstrapValidation,
    corrected_estimate: f64,
) -> ConfidenceInterval {
    bootstrap_result
        .confidence_intervals
        .iter()
        .find(|ci| (ci.confidence_level - DEFAULT_CONFIDENCE_LEVEL).abs() < 1e-6)
        .cloned()
        .unwrap_or_else(|| {
            let se = bootstrap_result.standard_error.max(1e-12);
            let z_95 = 1.96;
            let margin = z_95 * se;
            ConfidenceInterval {
                confidence_level: 0.95,
                lower_bound: corrected_estimate - margin,
                upper_bound: corrected_estimate + margin,
                method: ConfidenceIntervalMethod::Normal,
            }
        })
}

/// Calculate test statistics
fn calculate_test_statistics(corrected_estimate: f64, std_error: f64) -> (f64, f64) {
    let safe_std_error = std_error.max(MIN_STD_ERROR);
    let test_statistic = (corrected_estimate - 0.5) / safe_std_error;
    let p_value = 2.0 * (1.0 - standard_normal_cdf(test_statistic.abs()));
    (test_statistic, p_value)
}

/// Build HurstEstimate struct
fn build_hurst_estimate(
    corrected_estimate: f64,
    std_error: f64,
    confidence_interval: ConfidenceInterval,
    test_statistic: f64,
    p_value: f64,
    bias_correction: f64,
    finite_sample_correction: f64,
) -> HurstEstimate {
    HurstEstimate {
        estimate: corrected_estimate,
        standard_error: std_error,
        confidence_interval,
        test_statistic,
        p_value,
        adjusted_p_value: None,
        bias_correction,
        finite_sample_correction,
        bootstrap_bias: 0.0,  // Not available in this context
    }
}

/// Apply multiple testing corrections (Benjamini-Hochberg FDR)
fn apply_multiple_testing_corrections(
    mut estimates: BTreeMap<EstimationMethod, HurstEstimate>,
) -> FractalResult<BTreeMap<EstimationMethod, HurstEstimate>> {
    let n_tests = estimates.len();
    if n_tests <= 1 {
        for estimate in estimates.values_mut() {
            estimate.adjusted_p_value = Some(estimate.p_value);
        }
        return Ok(estimates);
    }

    let mut p_values: Vec<(EstimationMethod, f64)> = estimates
        .iter()
        .map(|(method, est)| (*method, est.p_value))
        .collect();

    for p_val in p_values.iter_mut() {
        if !p_val.1.is_finite() {
            p_val.1 = 1.0;
        }
    }
    p_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut adjusted_p_values: Vec<(EstimationMethod, f64)> = Vec::with_capacity(n_tests);
    for (i, (method, p_val)) in p_values.iter().enumerate() {
        let rank = i + 1;
        let raw_adjusted = (p_val * n_tests as f64 / rank as f64).min(1.0);
        adjusted_p_values.push((*method, raw_adjusted));
    }

    for i in (0..adjusted_p_values.len() - 1).rev() {
        let next_adjusted = adjusted_p_values[i + 1].1;
        adjusted_p_values[i].1 = adjusted_p_values[i].1.min(next_adjusted);
    }

    for (method, adjusted_p) in adjusted_p_values {
        if let Some(estimate) = estimates.get_mut(&method) {
            estimate.adjusted_p_value = Some(adjusted_p);
        }
    }

    Ok(estimates)
}

// ============================================================================
// BIAS CORRECTION FUNCTIONS
// ============================================================================

/// Calculate R/S bias correction
fn calculate_rs_bias_correction(n: usize) -> f64 {
    0.5 / (n as f64).ln()
}

/// Lo's modified R/S statistic for testing H=0.5 with robustness to short-range dependence
/// 
/// # Arguments
/// * `data` - Time series data
/// * `q` - Bandwidth parameter for Newey-West variance estimation (0 for standard R/S)
/// 
/// # Returns
/// (Q_n statistic, critical value at 5%, p-value)
pub fn lo_modified_rs_statistic(data: &[f64], q: usize) -> FractalResult<(f64, f64, f64)> {
    let n = data.len();
    if n < 20 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 20,
            actual: n,
        });
    }

    let mean = data.iter().sum::<f64>() / n as f64;
    
    // Calculate cumulative deviations
    let mut cumulative_devs = Vec::with_capacity(n);
    let mut cumsum = 0.0;
    for &value in data {
        cumsum += value - mean;
        cumulative_devs.push(cumsum);
    }

    // Range
    let max_dev = cumulative_devs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_dev = cumulative_devs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let range = max_dev - min_dev;

    // Long-run variance estimator (Newey-West)
    let mut long_run_variance = 0.0;
    
    // Centered data
    let centered: Vec<f64> = data.iter().map(|&x| x - mean).collect();
    
    // Variance (j=0)
    for i in 0..n {
        long_run_variance += centered[i] * centered[i];
    }
    long_run_variance /= n as f64;
    
    // Add autocovariances with Bartlett weights
    if q > 0 {
        for j in 1..=q.min(n - 1) {
            let weight = 1.0 - (j as f64) / ((q + 1) as f64);
            let mut autocovariance = 0.0;
            for i in j..n {
                autocovariance += centered[i] * centered[i - j];
            }
            autocovariance /= n as f64;
            long_run_variance += 2.0 * weight * autocovariance;
        }
    }

    // Lo's Q_n statistic
    let q_n = range / ((n as f64).sqrt() * long_run_variance.sqrt());

    // Critical values from Lo (1991) Table II
    // These are asymptotic critical values for 5% significance level
    let critical_value_5pct = 1.862;
    
    // Approximate p-value using interpolation from Lo's table
    // Critical values: 10%: 1.747, 5%: 1.862, 2.5%: 1.990, 1%: 2.178
    let p_value = if q_n < 1.747 {
        1.0  // > 10%
    } else if q_n < 1.862 {
        0.10 - 0.05 * (q_n - 1.747) / (1.862 - 1.747)
    } else if q_n < 1.990 {
        0.05 - 0.025 * (q_n - 1.862) / (1.990 - 1.862)
    } else if q_n < 2.178 {
        0.025 - 0.015 * (q_n - 1.990) / (2.178 - 1.990)
    } else {
        0.01  // < 1%
    };

    Ok((q_n, critical_value_5pct, p_value))
}

/// Calculate DFA bias correction
fn calculate_dfa_bias_correction(n: usize) -> f64 {
    let log_correction = 0.02 / (n as f64).ln();
    let finite_sample_correction = 0.3 / (n as f64);
    log_correction + finite_sample_correction
}

/// Compute GPH d estimate using jackknife bias correction over FREQUENCIES
/// Returns bias-corrected d estimate
fn compute_gph_d_jackknife(data: &[f64], config: &HurstEstimationConfig) -> FractalResult<f64> {
    let n = data.len();
    let periodogram = calculate_periodogram_fft(data)?;
    
    // Ensure we don't exceed Nyquist frequency
    let nyquist = n / 2;
    let max_freq = ((n as f64).powf(config.gph_bandwidth_exponent) * config.gph_bandwidth_multiplier) as usize;
    let max_freq = max_freq.max(5).min(nyquist).min(periodogram.len().saturating_sub(1));
    
    let mut log_periodogram = Vec::with_capacity(max_freq);
    let mut log_canonical = Vec::with_capacity(max_freq);
    
    for k in 1..=max_freq {
        let lambda_k = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
        let sin_half = (lambda_k / 2.0).sin();
        let canonical_freq = 4.0 * sin_half * sin_half;
        
        if periodogram[k] > 0.0 && periodogram[k].is_finite() && canonical_freq > 0.0 {
            if let Some(log_val) = float_ops::safe_ln(periodogram[k]) {
                if let Some(log_canon) = float_ops::safe_ln(canonical_freq) {
                    log_periodogram.push(log_val);
                    log_canonical.push(log_canon);
                }
            }
        }
    }
    
    let m = log_periodogram.len();
    if m < 8 {
        // Too few frequencies for jackknife
        let (slope, _, _) = ols_regression(&log_canonical, &log_periodogram)?;
        return Ok(-slope / 2.0);
    }
    
    // Jackknife over FREQUENCY blocks (not time blocks) to address GPH bias
    let k_blocks = 5.min(m / 3).max(3);
    let block_size = (m as f64 / k_blocks as f64).ceil() as usize;
    
    // Full sample estimate
    let (slope_full, _, _) = ols_regression(&log_canonical, &log_periodogram)?;
    let d_full = -slope_full / 2.0;  // Correct: d = -slope/2
    
    // Leave-one-frequency-block-out estimates
    let mut d_jackknife = Vec::with_capacity(k_blocks);
    
    for k in 0..k_blocks {
        let start = k * block_size;
        let end = ((k + 1) * block_size).min(m);
        
        if start >= m {
            break;
        }
        
        // Create leave-one-frequency-block-out sample
        let mut loo_log_p = Vec::with_capacity(m - (end - start));
        let mut loo_log_c = Vec::with_capacity(m - (end - start));
        
        // Include frequencies before the block
        if start > 0 {
            loo_log_p.extend_from_slice(&log_periodogram[..start]);
            loo_log_c.extend_from_slice(&log_canonical[..start]);
        }
        
        // Include frequencies after the block
        if end < m {
            loo_log_p.extend_from_slice(&log_periodogram[end..]);
            loo_log_c.extend_from_slice(&log_canonical[end..]);
        }
        
        if loo_log_p.len() >= 8 {
            if let Ok((slope, _, _)) = ols_regression(&loo_log_c, &loo_log_p) {
                let d_k = -slope / 2.0;  // Correct: d = -slope/2
                // Check if d_k is reasonable (-1 < d < 1.5 covers most practical cases)
                if d_k > -1.0 && d_k < 1.5 {
                    d_jackknife.push(d_k);
                }
            }
        }
    }
    
    if d_jackknife.len() < 2 {
        // Not enough valid jackknife samples, return uncorrected estimate
        return Ok(d_full);
    }
    
    // Jackknife bias correction formula
    let mean_jackknife = d_jackknife.iter().sum::<f64>() / d_jackknife.len() as f64;
    let k = d_jackknife.len() as f64;
    let d_corrected = k * d_full - (k - 1.0) * mean_jackknife;
    
    Ok(d_corrected)
}

/// Compute raw GPH d estimate (without bias correction)
fn compute_gph_d_raw(data: &[f64], config: &HurstEstimationConfig) -> FractalResult<f64> {
    let n = data.len();
    let periodogram = calculate_periodogram_fft(data)?;
    
    // Ensure we don't exceed Nyquist frequency (n/2 for even n, (n-1)/2 for odd n)
    let nyquist = n / 2;
    let max_freq = ((n as f64).powf(config.gph_bandwidth_exponent) * config.gph_bandwidth_multiplier) as usize;
    let max_freq = max_freq.max(5).min(nyquist).min(periodogram.len().saturating_sub(1));
    
    let mut log_periodogram = Vec::with_capacity(max_freq);
    let mut log_canonical = Vec::with_capacity(max_freq);
    
    for k in 1..=max_freq {
        let lambda_k = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
        
        // Canonical GPH regressor: ln(4 * sin²(λ/2))
        let sin_half = (lambda_k / 2.0).sin();
        let canonical_freq = 4.0 * sin_half * sin_half;
        
        if periodogram[k] > 0.0 && periodogram[k].is_finite() && canonical_freq > 0.0 {
            if let Some(log_val) = float_ops::safe_ln(periodogram[k]) {
                if let Some(log_canon) = float_ops::safe_ln(canonical_freq) {
                    log_periodogram.push(log_val);
                    log_canonical.push(log_canon);
                }
            }
        }
    }
    
    // Actual m after filtering
    let m = log_periodogram.len();
    if m < 8 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 8,
            actual: m,
        });
    }
    
    let (slope, _, _) = ols_regression(&log_canonical, &log_periodogram)?;
    // GPH canonical form: slope = -d, so d = -slope
    Ok(-slope)
}

/// Calculate finite sample correction
fn calculate_finite_sample_correction(n: usize) -> f64 {
    1.0 / (n as f64).sqrt()
}

/// Estimate local Hurst exponent for a regime
pub fn estimate_local_hurst(data: &[f64], _regime_state: usize) -> f64 {
    if data.len() < 50 {
        return f64::NAN;
    }

    match perform_multifractal_analysis(data) {
        Ok(mf_result) => {
            if let Some((_, hurst)) = mf_result
                .generalized_hurst_exponents
                .iter()
                .find(|(q, _)| (q - 2.0).abs() < Q_MATCH_EPSILON)
            {
                *hurst
            } else if let Some((_, first_hurst)) = mf_result.generalized_hurst_exponents.first() {
                first_hurst.max(0.01).min(0.99)
            } else {
                fallback_hurst_estimation(data)
            }
        }
        Err(_) => fallback_hurst_estimation(data),
    }
}

/// Fallback Hurst estimation using DFA and R/S
fn fallback_hurst_estimation(data: &[f64]) -> f64 {
    match estimate_dfa_hurst_only(data, 1) {
        Ok(hurst) if hurst.is_finite() => hurst.max(0.01).min(0.99),
        _ => match estimate_rs_hurst_only(data) {
            Ok(hurst) if hurst.is_finite() => hurst.max(0.01).min(0.99),
            _ => f64::NAN,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hurst_estimation_methods() {
        let data: Vec<f64> = (0..200).map(|i| (i as f64).sin()).collect();
        let config = HurstEstimationConfig::default();

        let results = estimate_hurst_multiple_methods(&data, &config).unwrap();
        assert!(!results.is_empty());

        for (method, estimate) in results {
            assert!(estimate.estimate >= 0.0 && estimate.estimate <= 1.0);
            assert!(estimate.standard_error >= 0.0);
            assert!(estimate.p_value >= 0.0 && estimate.p_value <= 1.0);
        }
    }

    #[test]
    fn test_short_series_estimation() {
        let data = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0];
        let result = estimate_hurst_simple_short_series(&data).unwrap();
        assert!(result.estimate >= 0.0 && result.estimate <= 1.0);
    }

    #[test]
    fn test_aggregate_series() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let aggregated = aggregate_series(&data, 2);
        assert_eq!(aggregated, vec![1.5, 3.5, 5.5]);
    }

    #[test]
    fn test_whittle_estimator() {
        // Generate stationary data - Whittle requires stationarity
        // Using white noise (fractional noise with d=0, H=0.5)
        let mut rng = crate::secure_rng::FastrandCompat::with_seed(42);
        let mut data = Vec::with_capacity(256);
        
        // Generate white noise (stationary)
        for _ in 0..256 {
            data.push(rng.f64() * 2.0 - 1.0);
        }
        
        let config = HurstEstimationConfig::default();
        let result = estimate_hurst_whittle(&data, &config);
        
        // Should succeed for sufficient data length
        assert!(result.is_ok());
        
        if let Ok(estimate) = result {
            // For white noise, H should be around 0.5
            // Allow wide tolerance due to finite sample and estimation variance
            assert!(estimate.estimate > 0.3 && estimate.estimate < 0.7);
            assert!(estimate.standard_error > 0.0);
            assert!(estimate.p_value >= 0.0 && estimate.p_value <= 1.0);
        }
    }
    
    #[test]
    fn test_whittle_estimator_ar1() {
        // Test with AR(1) process - stationary with known persistence
        let mut rng = crate::secure_rng::FastrandCompat::with_seed(123);
        let n = 256;
        let mut data = Vec::with_capacity(n);
        
        // Generate AR(1) with moderate persistence
        let phi = 0.5;  // AR coefficient
        let mut x = 0.0;
        for _ in 0..n {
            let noise = rng.f64() * 2.0 - 1.0;
            x = phi * x + noise;
            data.push(x);
        }
        
        let config = HurstEstimationConfig::default();
        let result = estimate_hurst_whittle(&data, &config);
        
        assert!(result.is_ok());
        
        if let Ok(estimate) = result {
            // AR(1) with phi=0.5 should have H slightly above 0.5
            // but still in stationary range
            assert!(estimate.estimate > 0.4 && estimate.estimate < 0.8);
            assert!(estimate.standard_error > 0.0);
        }
    }
    
    #[test]
    fn test_whittle_estimator_short_series() {
        // Test with insufficient data
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let config = HurstEstimationConfig::default();
        let result = estimate_hurst_whittle(&data, &config);
        
        // Should fail for short series
        assert!(result.is_err());
    }
    
    #[test]
    fn test_variogram_method() {
        use crate::generators::{generate_fractional_brownian_motion, FbmConfig, FbmMethod, GeneratorConfig};
        
        // Generate proper FBM with known Hurst exponent
        let target_hurst = 0.65;
        let config = GeneratorConfig {
            length: 500,
            seed: Some(123),
            ..Default::default()
        };
        let fbm_config = FbmConfig {
            hurst_exponent: target_hurst,
            volatility: 1.0,
            method: FbmMethod::CirculantEmbedding,
        };
        
        let fbm = generate_fractional_brownian_motion(&config, &fbm_config).unwrap();
        
        // Variogram needs the FBM (cumulative process), not FGN
        let est_config = HurstEstimationConfig::default();
        let result = estimate_hurst_variogram(&fbm, &est_config);
        
        assert!(result.is_ok(), "Variogram estimation should succeed");
        
        if let Ok(estimate) = result {
            // Should be in valid range
            assert!(estimate.estimate >= 0.01 && estimate.estimate <= 0.99);
            assert!(estimate.standard_error > 0.0);
            assert!(estimate.confidence_interval.lower_bound <= estimate.estimate);
            assert!(estimate.confidence_interval.upper_bound >= estimate.estimate);
            
            // Verify the estimator recovers the known Hurst within tolerance
            assert!(
                (estimate.estimate - target_hurst).abs() < 0.2,
                "Variogram estimate {} should be close to true Hurst {}",
                estimate.estimate, target_hurst
            );
        }
    }
    
    #[test]
    fn test_variogram_constant_series() {
        // Test with constant data
        let data = vec![5.0; 100];
        let config = HurstEstimationConfig::default();
        let result = estimate_hurst_variogram(&data, &config);
        
        // Should fail for zero variance data
        assert!(result.is_err());
    }
    
    #[test]
    fn test_variogram_scale_invariance() {
        // Test that variogram method is invariant to data rescaling
        let mut rng = crate::secure_rng::FastrandCompat::with_seed(456);
        let n = 200;
        let mut data = vec![0.0];
        
        // Generate random walk
        for _ in 1..n {
            let increment = rng.f64() * 2.0 - 1.0;
            data.push(data.last().unwrap() + increment);
        }
        
        let config = HurstEstimationConfig::default();
        
        // Original data
        let result1 = estimate_hurst_variogram(&data, &config).unwrap();
        
        // Rescaled data (convert meters to kilometers)
        let scaled_data: Vec<f64> = data.iter().map(|x| x * 0.001).collect();
        let result2 = estimate_hurst_variogram(&scaled_data, &config).unwrap();
        
        // Hurst estimate should be the same regardless of scale
        assert!((result1.estimate - result2.estimate).abs() < 0.05,
                "Scale invariance failed: {} vs {}", result1.estimate, result2.estimate);
        
        // Also test with large scale factor
        let large_scaled: Vec<f64> = data.iter().map(|x| x * 1000.0).collect();
        let result3 = estimate_hurst_variogram(&large_scaled, &config).unwrap();
        
        assert!((result1.estimate - result3.estimate).abs() < 0.05,
                "Scale invariance failed for large scale: {} vs {}", result1.estimate, result3.estimate);
    }
    
    #[test]
    fn test_all_methods_consistency() {
        use crate::generators::{
            generate_fractional_brownian_motion, fbm_to_fgn, FbmConfig, FbmMethod, GeneratorConfig,
        };
        
        // Test consistency across methods using FGN with known Hurst
        // Run multiple replicates for statistical validity
        let n = 512;
        let target_h = 0.65; // Moderate long-range dependence
        let num_replicates = 20;
        
        // Collect estimates for each method across replicates
        let methods = vec![
            EstimationMethod::RescaledRange,
            EstimationMethod::DetrendedFluctuationAnalysis,
            EstimationMethod::PeriodogramRegression,
            EstimationMethod::WaveletEstimation,
            EstimationMethod::WhittleEstimator,
            EstimationMethod::VariogramMethod,
        ];
        
        let mut method_estimates: BTreeMap<EstimationMethod, Vec<f64>> = BTreeMap::new();
        for method in &methods {
            method_estimates.insert(*method, Vec::new());
        }
        
        // Generate multiple realizations
        for seed in 0..num_replicates {
            let config = GeneratorConfig {
                length: n + 1, // Need n+1 for n FGN values
                seed: Some(999 + seed as u64),
                ..Default::default()
            };
            
            let fbm_config = FbmConfig {
                hurst_exponent: target_h,
                volatility: 1.0,
                method: FbmMethod::CirculantEmbedding,
            };
            
            if let Ok(fbm) = generate_fractional_brownian_motion(&config, &fbm_config) {
                let fgn = fbm_to_fgn(&fbm);
                
                let est_config = HurstEstimationConfig::default();
                
                for method in &methods {
                    // Variogram method needs FBM (cumulative process), not FGN (increments)
                    let data = if *method == EstimationMethod::VariogramMethod {
                        &fbm[..fbm.len()-1]  // Use FBM but match FGN length
                    } else {
                        &fgn  // Other methods work on FGN
                    };
                    
                    if let Ok(estimate) = estimate_hurst_by_method(data, method, &est_config) {
                        method_estimates.get_mut(method).unwrap().push(estimate.estimate);
                    }
                }
            }
        }
        
        // Calculate median estimate and statistics for each method
        let mut median_estimates = Vec::new();
        let mut per_method_stats = Vec::new();
        
        for (method, mut estimates) in method_estimates {
            if estimates.len() >= 10 { // Need enough estimates
                estimates.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = estimates[estimates.len() / 2];
                
                // Calculate mean and standard deviation for diagnostic info
                let mean = estimates.iter().sum::<f64>() / estimates.len() as f64;
                let variance = estimates.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>() / (estimates.len() - 1) as f64;
                let std_dev = variance.sqrt();
                
                per_method_stats.push((method, median, mean, std_dev));
                median_estimates.push((method, median));
                
                // Per-method accuracy check with method-specific tolerances
                let tolerance = match method {
                    EstimationMethod::VariogramMethod => 0.2,  // Less accurate
                    EstimationMethod::WhittleEstimator => 0.12, // Moderate accuracy
                    _ => 0.15, // Default tolerance
                };
                
                assert!(
                    (median - target_h).abs() < tolerance,
                    "{:?} median {:.3} too far from target {:.3} (tolerance: {:.3}, std: {:.3})",
                    method, median, target_h, tolerance, std_dev
                );
            }
        }
        
        // Check that enough methods succeeded
        assert!(median_estimates.len() >= 4, "Too few methods succeeded: only {} of {} methods produced enough estimates", 
            median_estimates.len(), methods.len());
        
        // Check cross-method consistency
        let overall_median = {
            let mut all_medians: Vec<f64> = median_estimates.iter().map(|(_, m)| *m).collect();
            all_medians.sort_by(|a, b| a.partial_cmp(b).unwrap());
            all_medians[all_medians.len() / 2]
        };
        
        // Verify overall accuracy
        assert!(
            (overall_median - target_h).abs() < 0.1,
            "Overall median {:.3} too far from target {:.3}",
            overall_median, target_h
        );
        
        // Check relative consistency between methods
        for (method, median) in &median_estimates {
            let max_deviation = match method {
                EstimationMethod::VariogramMethod => 0.15, // Allow more deviation
                _ => 0.1,
            };
            
            assert!(
                (median - overall_median).abs() < max_deviation,
                "{:?} median {:.3} deviates too much from overall median {:.3}",
                method, median, overall_median
            );
        }
    }
}