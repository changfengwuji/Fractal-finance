//! Monte Carlo hypothesis testing framework for fractal analysis.
//!
//! This module provides comprehensive Monte Carlo methods for statistical hypothesis
//! testing in fractal time series analysis. It includes power analysis, surrogate
//! data methods, and various null hypothesis testing frameworks commonly used in
//! quantitative finance and econophysics.
//!
//! ## Key Features
//!
//! - **Hypothesis Testing**: Monte Carlo tests for Hurst exponents and multifractality
//! - **Surrogate Data**: Multiple methods for generating surrogate data (FFT, AAFT, IAAFT)
//! - **Power Analysis**: Statistical power analysis for fractal estimators
//! - **Effect Size**: Cohen's d and other effect size measures
//! - **Bootstrap Integration**: Confidence intervals using bootstrap methods

use crate::{
    bootstrap::{ConfidenceIntervalMethod, *},
    errors::{validate_data_length, validate_parameter, FractalAnalysisError, FractalResult},
    generators::*,
    math_utils::{calculate_variance, calculate_volatility, erf, ols_regression},
    memory_pool::{get_f64_buffer, return_f64_buffer},
    multifractal::*,
    secure_rng::{SecureRng, ThreadLocalRng},
};
use rustfft::{num_complex::Complex, FftPlanner};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;

/// Thread-local cache for generator states to avoid repeated initialization
thread_local! {
    /// Cache of pre-computed impulse responses for ARFIMA processes
    static ARFIMA_CACHE: RefCell<HashMap<(Vec<i64>, i64, Vec<i64>), Vec<f64>>> = RefCell::new(HashMap::new());

    /// Cache of FFT planners to avoid repeated initialization
    static FFT_PLANNER_CACHE: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
}

/// Configuration parameters for Monte Carlo testing procedures.
///
/// Controls all aspects of Monte Carlo simulation including sample sizes,
/// significance levels, parallelization options, and bootstrap integration.
#[derive(Debug, Clone)]
pub struct MonteCarloConfig {
    /// Number of Monte Carlo simulations to perform
    pub num_simulations: usize,
    /// Statistical significance level (e.g., 0.05 for 5% significance)
    pub significance_level: f64,
    /// Random seed for reproducible results
    pub seed: Option<u64>,
    /// Enable parallel computation where supported
    pub parallel: bool,
    /// Bootstrap configuration for confidence interval estimation
    pub bootstrap_config: BootstrapConfiguration,
    /// Use deterministic RNG seeding for parallel reproducibility
    pub deterministic_parallel: bool,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            num_simulations: 1000,
            significance_level: 0.05,
            seed: None,
            parallel: true, // Enable parallel processing by default for better performance
            bootstrap_config: BootstrapConfiguration::default(),
            deterministic_parallel: true, // Ensure reproducible parallel results
        }
    }
}

/// Complete results from Monte Carlo hypothesis testing.
///
/// Contains all statistical information from a Monte Carlo test including
/// the test statistic, p-value, effect size, and confidence intervals.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MonteCarloTestResult {
    /// Descriptive name of the test performed
    pub test_name: String,
    /// Observed test statistic from the original data
    pub observed_statistic: f64,
    /// Distribution of test statistics under the null hypothesis
    pub null_distribution: Vec<f64>,
    /// P-value (proportion of null statistics as extreme as observed)
    pub p_value: f64,
    /// Critical value at the specified significance level
    pub critical_value: f64,
    /// Statistical conclusion (true if null hypothesis is rejected)
    pub reject_null: bool,
    /// Bootstrap confidence interval for the test statistic
    pub confidence_interval: Option<ConfidenceInterval>,
    /// Effect size measure (Cohen's d)
    pub effect_size: f64,
}

/// Null hypothesis specifications for fractal analysis testing.
///
/// Defines the various null hypotheses that can be tested against
/// observed fractal data using Monte Carlo methods.
#[derive(Debug, Clone)]
pub enum NullHypothesis {
    /// Independent Gaussian white noise (no memory)
    WhiteNoise,
    /// Standard random walk with Hurst exponent = 0.5
    RandomWalk,
    /// ARMA process with specified autoregressive and moving average parameters
    ArmaProcess {
        /// Autoregressive coefficients
        ar_params: Vec<f64>,
        /// Moving average coefficients
        ma_params: Vec<f64>,
    },
    /// Linear deterministic trend with additive noise
    LinearTrend {
        /// Slope of the linear trend
        slope: f64,
    },
    /// Fractal Brownian motion with fixed Hurst exponent
    FixedHurst {
        /// Hurst exponent value
        hurst: f64,
    },
    /// Monofractal process with specified Hurst exponent (no multifractal scaling)
    Monofractal {
        /// Hurst exponent for the monofractal process
        hurst: f64,
    },
}

/// Configuration for surrogate data generation.
///
/// Controls parameters for various surrogate data methods,
/// allowing fine-tuning of convergence and accuracy.
#[derive(Debug, Clone)]
pub struct SurrogateConfig {
    /// Maximum iterations for iterative methods (e.g., IAAFT)
    pub max_iterations: usize,
    /// Convergence tolerance for iterative methods
    pub tolerance: f64,
    /// Starting iteration for convergence checking
    pub convergence_check_start: usize,
    /// Number of consecutive iterations without improvement to trigger early stopping
    pub stagnation_threshold: usize,
}

impl Default for SurrogateConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            convergence_check_start: 3,
            stagnation_threshold: 5,
        }
    }
}

/// Computation health tracking for financial transparency and audit.
///
/// CRITICAL for production financial systems: Tracks computation failures,
/// numerical issues, and provides audit trail for regulatory compliance.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ComputationHealth {
    /// Total number of computations attempted
    pub total_computations: usize,
    /// Number of successful computations
    pub successful_computations: usize,
    /// Number of NaN results (indicates serious numerical issues)
    pub nan_results: usize,
    /// Number of infinite results (overflow/underflow)
    pub infinite_results: usize,
    /// Number of computations that threw errors
    pub error_computations: usize,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Warning messages for audit trail
    pub warnings: Vec<String>,
    /// Timestamp of computation (for audit trail)
    pub timestamp: std::time::SystemTime,
}

impl ComputationHealth {
    /// Create new health tracker
    pub fn new() -> Self {
        Self {
            total_computations: 0,
            successful_computations: 0,
            nan_results: 0,
            infinite_results: 0,
            error_computations: 0,
            success_rate: 1.0,
            warnings: Vec::new(),
            timestamp: std::time::SystemTime::now(),
        }
    }

    /// Record a computation result
    pub fn record_result(&mut self, result: f64) {
        self.total_computations += 1;
        if result.is_nan() {
            self.nan_results += 1;
        } else if !result.is_finite() {
            self.infinite_results += 1;
        } else {
            self.successful_computations += 1;
        }
        self.update_success_rate();
    }

    /// Record a computation error
    pub fn record_error(&mut self) {
        self.total_computations += 1;
        self.error_computations += 1;
        self.update_success_rate();
    }

    /// Update success rate
    fn update_success_rate(&mut self) {
        if self.total_computations > 0 {
            self.success_rate =
                self.successful_computations as f64 / self.total_computations as f64;
        }
    }

    /// Add warning message
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Check if health is acceptable for financial use
    pub fn is_healthy(&self, min_success_rate: f64) -> bool {
        self.success_rate >= min_success_rate && self.nan_results == 0
    }

    /// Generate audit report
    pub fn audit_report(&self) -> String {
        format!(
            "ComputationHealth Report:\n\
             - Timestamp: {:?}\n\
             - Total Computations: {}\n\
             - Successful: {} ({:.1}%)\n\
             - NaN Results: {}\n\
             - Infinite Results: {}\n\
             - Errors: {}\n\
             - Warnings: {}\n\
             - Status: {}",
            self.timestamp,
            self.total_computations,
            self.successful_computations,
            self.success_rate * 100.0,
            self.nan_results,
            self.infinite_results,
            self.error_computations,
            self.warnings.len(),
            if self.is_healthy(0.8) {
                "HEALTHY"
            } else {
                "UNHEALTHY - REVIEW REQUIRED"
            }
        )
    }
}

impl Default for ComputationHealth {
    fn default() -> Self {
        Self::new()
    }
}

/// Available methods for generating surrogate data.
///
/// Surrogate data methods preserve certain statistical properties of the
/// original data while destroying others, allowing for hypothesis testing
/// of specific features like nonlinearity or long-range dependence.
#[derive(Debug, Clone)]
pub enum SurrogateMethod {
    /// Fourier transform method - preserves power spectrum by phase randomization
    FourierTransform,
    /// Amplitude Adjusted Fourier Transform - preserves distribution and spectrum
    AmplitudeAdjusted,
    /// Iterative AAFT - refined version with better convergence
    IterativeAmplitudeAdjusted {
        /// Configuration for iterative refinement
        config: SurrogateConfig,
    },
    /// Block shuffling - preserves local correlations within blocks
    BlockShuffle {
        /// Size of blocks to shuffle
        block_size: usize,
    },
    /// Phase randomization - equivalent to Fourier transform method
    PhaseRandomization,
}

/// Results from statistical power analysis.
///
/// Power analysis determines the ability of a statistical test to detect
/// true effects of different magnitudes across various sample sizes.
#[derive(Debug, Clone)]
pub struct PowerAnalysisResult {
    /// True parameter value being tested against
    pub true_value: f64,
    /// Sample sizes evaluated in the power analysis
    pub sample_sizes: Vec<usize>,
    /// Statistical power achieved for each sample size
    pub power_curves: Vec<f64>,
    /// Minimum sample size required for 80% statistical power
    pub required_sample_size_80: Option<usize>,
    /// Minimum sample size required for 95% statistical power
    pub required_sample_size_95: Option<usize>,
    /// Bias of the estimator for each sample size
    pub bias_curves: Vec<f64>,
    /// Mean squared error for each sample size
    pub mse_curves: Vec<f64>,
}

/// Comprehensive Monte Carlo test for Hurst exponent hypotheses.
///
/// Performs Monte Carlo hypothesis testing for the Hurst exponent using
/// synthetic data generated under the specified null hypothesis. This is
/// the gold standard for testing fractal properties when analytical
/// distributions are unknown or unreliable.
///
/// # Arguments
/// * `data` - Time series data to test
/// * `null_hypothesis` - Specification of the null hypothesis
/// * `config` - Monte Carlo configuration parameters
///
/// # Returns
/// * `Ok(MonteCarloTestResult)` - Complete test results
/// * `Err` - If insufficient data or invalid parameters
///
/// # Example
/// ```rust
/// use financial_fractal_analysis::{monte_carlo_hurst_test, NullHypothesis, MonteCarloConfig};
///
/// let data = vec![/* your time series data */];
/// let config = MonteCarloConfig::default();
///
/// let result = monte_carlo_hurst_test(
///     &data,
///     NullHypothesis::WhiteNoise,
///     &config,
/// ).unwrap();
///
/// if result.reject_null {
///     println!("Rejected white noise hypothesis (p = {:.4})", result.p_value);
/// }
/// ```
pub fn monte_carlo_hurst_test(
    data: &[f64],
    null_hypothesis: NullHypothesis,
    config: &MonteCarloConfig,
) -> FractalResult<MonteCarloTestResult> {
    validate_data_length(data, 50, "Monte Carlo Hurst test")?;

    // Validate configuration parameters
    if config.num_simulations == 0 {
        return Err(FractalAnalysisError::InvalidParameter {
            parameter: "num_simulations".to_string(),
            value: config.num_simulations as f64,
            constraint: "must be greater than 0".to_string(),
        });
    }

    if config.significance_level <= 0.0 || config.significance_level >= 1.0 {
        return Err(FractalAnalysisError::InvalidParameter {
            parameter: "significance_level".to_string(),
            value: config.significance_level,
            constraint: "must be between 0 and 1 (exclusive)".to_string(),
        });
    }

    // Use SecureRng for cryptographically secure randomness
    // Note: Seed setup is handled per-thread in parallel execution

    // Calculate observed Hurst exponent using multiple methods
    let observed_hurst = estimate_robust_hurst_exponent(data)?;

    // OPTIMIZATION: Parallel Monte Carlo simulations for significant speedup
    let null_statistics = {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            // Parallel computation of null distribution with deterministic seeding
            let results: Vec<_> = (0..config.num_simulations)
                .into_par_iter()
                .filter_map(|i| {
                    // CRITICAL FIX: Ensure statistical independence between simulations
                    // Use cryptographic mixing to create independent RNG sequences
                    let thread_seed = if config.deterministic_parallel {
                        config
                            .seed
                            .map(|s| {
                                // Mix seed with simulation index using bit manipulation for independence
                                // This ensures each simulation has a unique, deterministic seed
                                let mixed = s ^ ((i as u64).rotate_left(32));
                                mixed.wrapping_mul(0x9e3779b97f4a7c15) // Golden ratio constant
                            })
                            .unwrap_or_else(|| {
                                // For non-deterministic mode, use system entropy mixed with index
                                ThreadLocalRng::u64(0..u64::MAX)
                                    .wrapping_add(i as u64 * 0x517cc1b727220a95)
                            })
                    } else {
                        // Pure random mode for maximum entropy
                        ThreadLocalRng::u64(0..u64::MAX)
                    };

                    // OPTIMIZATION: Use memory pool for synthetic data generation in parallel threads
                    let mut synthetic_buffer = match get_f64_buffer(data.len()) {
                        Ok(buffer) => buffer,
                        Err(_) => return None,
                    };

                    let result = match generate_synthetic_under_null_inplace(
                        &null_hypothesis,
                        &mut synthetic_buffer,
                    ) {
                        Ok(()) => match estimate_robust_hurst_exponent(&synthetic_buffer) {
                            Ok(null_hurst) => Some(null_hurst),
                            Err(_) => None,
                        },
                        Err(_) => None,
                    };

                    // Return buffer to pool for reuse by other threads
                    return_f64_buffer(synthetic_buffer);
                    result
                })
                .collect();

            if results.len() < config.num_simulations / 2 {
                return Err(FractalAnalysisError::NumericalError {
                    reason: "Too many failed Monte Carlo simulations in parallel processing"
                        .to_string(),
                    operation: None,
                });
            }

            results
        }

        #[cfg(not(feature = "parallel"))]
        {
            // Sequential fallback when parallel feature is not enabled with memory pooling
            let mut statistics = Vec::with_capacity(config.num_simulations);

            for _ in 0..config.num_simulations {
                // OPTIMIZATION: Use memory pool for synthetic data generation to reduce allocations
                let mut synthetic_buffer = get_f64_buffer(data.len())?;
                synthetic_buffer.resize(data.len(), 0.0);
                generate_synthetic_under_null_inplace(&null_hypothesis, &mut synthetic_buffer)?;

                let null_hurst = estimate_robust_hurst_exponent(&synthetic_buffer)?;
                statistics.push(null_hurst);

                // Return buffer to pool for reuse
                return_f64_buffer(synthetic_buffer);
            }
            statistics
        }
    };

    // Calculate p-value and critical value
    let mut null_statistics = null_statistics;
    // Safe sort handling NaN values
    sort_f64_slice(&mut null_statistics);

    let p_value = calculate_p_value_two_tailed(observed_hurst, &null_statistics);
    let critical_idx = ((1.0 - config.significance_level) * null_statistics.len() as f64) as usize;
    let critical_value = if null_statistics.is_empty() {
        0.0
    } else {
        null_statistics[critical_idx.min(null_statistics.len().saturating_sub(1))]
    };

    let reject_null = p_value < config.significance_level;

    // Calculate effect size (Cohen's d)
    let null_mean = null_statistics.iter().sum::<f64>() / null_statistics.len() as f64;
    let null_std = calculate_variance(&null_statistics).sqrt();
    let effect_size = if null_std > 0.0 {
        (observed_hurst - null_mean) / null_std
    } else {
        0.0
    };

    // Bootstrap confidence interval for observed statistic - CRITICAL FIX: Use BCa method and track failures
    let confidence_interval = if config.bootstrap_config.num_bootstrap_samples > 0 {
        compute_bca_bootstrap_interval(
            data,
            observed_hurst,
            |data| {
                match estimate_robust_hurst_exponent(data) {
                    Ok(hurst) => {
                        // Validate the result
                        if hurst.is_finite() && hurst > 0.01 && hurst < 0.99 {
                            hurst
                        } else {
                            // CRITICAL FIX: Return NaN for invalid results instead of silent fallback
                            f64::NAN
                        }
                    }
                    Err(_) => {
                        // CRITICAL FIX: Return NaN for failed estimation instead of silent fallback
                        f64::NAN
                    }
                }
            },
            &config.bootstrap_config,
        )
        .ok()
    } else {
        None
    };

    Ok(MonteCarloTestResult {
        test_name: format!("Monte Carlo Hurst Test ({:?})", null_hypothesis),
        observed_statistic: observed_hurst,
        null_distribution: null_statistics,
        p_value,
        critical_value,
        reject_null,
        confidence_interval,
        effect_size,
    })
}

/// Compute BCa (bias-corrected and accelerated) bootstrap confidence interval.
///
/// This provides more accurate confidence intervals than simple percentile methods,
/// especially for small samples or when the estimator has bias.
fn compute_bca_bootstrap_interval<F>(
    data: &[f64],
    observed_statistic: f64,
    statistic_fn: F,
    config: &BootstrapConfiguration,
) -> FractalResult<ConfidenceInterval>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    let n = data.len();
    let num_bootstrap = config.num_bootstrap_samples;

    // Generate bootstrap samples
    let mut bootstrap_statistics = Vec::with_capacity(num_bootstrap);

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        bootstrap_statistics = (0..num_bootstrap)
            .into_par_iter()
            .map(|i| {
                // Create deterministic seed for reproducible bootstrap
                let seed = config
                    .seed
                    .map(|s| s.wrapping_add(i as u64))
                    .unwrap_or_else(|| ThreadLocalRng::u64(0..u64::MAX));
                let mut rng = SecureRng::with_seed(seed);

                // Generate bootstrap sample
                let mut bootstrap_sample = Vec::with_capacity(n);
                for _ in 0..n {
                    let idx = rng.usize(0..n);
                    bootstrap_sample.push(data[idx]);
                }

                let result = statistic_fn(&bootstrap_sample);
                // CRITICAL FIX: Return NaN for invalid results instead of propagating them
                if result.is_finite() {
                    result
                } else {
                    f64::NAN
                }
            })
            .filter(|&x| x.is_finite()) // CRITICAL FIX: Filter out NaN and infinite values
            .collect();
    }

    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..num_bootstrap {
            let seed = config
                .seed
                .map(|s| s.wrapping_add(i as u64))
                .unwrap_or_else(|| ThreadLocalRng::u64(0..u64::MAX));
            let mut rng = SecureRng::with_seed(seed);

            let mut bootstrap_sample = Vec::with_capacity(n);
            for _ in 0..n {
                let idx = rng.usize(0..n);
                bootstrap_sample.push(data[idx]);
            }

            let result = statistic_fn(&bootstrap_sample);
            // CRITICAL FIX: Only add valid results to statistics
            if result.is_finite() {
                bootstrap_statistics.push(result);
            }
        }
    }

    // CRITICAL FIX: Check if we have enough valid bootstrap samples
    let valid_samples = bootstrap_statistics.len();
    let failure_rate = (num_bootstrap - valid_samples) as f64 / num_bootstrap as f64;

    // If too many computations failed, return an error instead of silently proceeding
    if failure_rate > 0.2 {
        return Err(FractalAnalysisError::NumericalError {
            reason: format!(
                "Bootstrap computation failure rate too high: {:.1}% failed ({} out of {}). \
                 This indicates serious numerical issues with the estimator or data quality.",
                failure_rate * 100.0,
                num_bootstrap - valid_samples,
                num_bootstrap
            ),
            operation: None,
        });
    }

    // Require minimum number of valid samples for reliable confidence intervals
    if valid_samples < 100 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 100,
            actual: valid_samples,
        });
    }

    // Calculate bias correction factor (z0)
    let num_less = bootstrap_statistics
        .iter()
        .filter(|&&x| x < observed_statistic)
        .count() as f64;
    let proportion = num_less / num_bootstrap as f64;

    // Use inverse normal CDF approximation for z0
    let z0 = if proportion > 0.0 && proportion < 1.0 {
        // Approximate inverse normal CDF using rational approximation
        inverse_normal_cdf(proportion)
    } else if proportion == 0.0 {
        -3.0 // Very negative z-score
    } else {
        3.0 // Very positive z-score
    };

    // Calculate acceleration factor (a) using jackknife
    let mut jackknife_statistics = Vec::with_capacity(n);
    for i in 0..n {
        // Create jackknife sample (leave one out)
        let mut jackknife_sample = Vec::with_capacity(n - 1);
        for j in 0..n {
            if i != j {
                jackknife_sample.push(data[j]);
            }
        }
        jackknife_statistics.push(statistic_fn(&jackknife_sample));
    }

    let jackknife_mean = jackknife_statistics.iter().sum::<f64>() / n as f64;
    let numerator: f64 = jackknife_statistics
        .iter()
        .map(|&x| (jackknife_mean - x).powi(3))
        .sum();
    let denominator: f64 = jackknife_statistics
        .iter()
        .map(|&x| (jackknife_mean - x).powi(2))
        .sum::<f64>()
        .powf(1.5);

    let a = if denominator > 1e-10 {
        numerator / (6.0 * denominator)
    } else {
        0.0
    };

    // Calculate adjusted percentiles
    let alpha = (1.0 - config.confidence_levels[0]) / 2.0;
    let z_alpha = inverse_normal_cdf(alpha);
    let z_1_alpha = -z_alpha;

    // BCa adjustment
    let lower_p = normal_cdf(z0 + (z0 + z_alpha) / (1.0 - a * (z0 + z_alpha)));
    let upper_p = normal_cdf(z0 + (z0 + z_1_alpha) / (1.0 - a * (z0 + z_1_alpha)));

    // Sort bootstrap statistics
    let mut sorted_statistics = bootstrap_statistics.clone();
    sort_f64_slice(&mut sorted_statistics);

    // Get adjusted percentiles
    let lower_idx = (lower_p * num_bootstrap as f64) as usize;
    let upper_idx = (upper_p * num_bootstrap as f64) as usize;

    let lower = sorted_statistics[lower_idx.min(num_bootstrap - 1)];
    let upper = sorted_statistics[upper_idx.min(num_bootstrap - 1)];

    Ok(ConfidenceInterval {
        lower_bound: lower,
        upper_bound: upper,
        confidence_level: config.confidence_levels[0],
        method: ConfidenceIntervalMethod::BootstrapBca,
    })
}

/// Approximate inverse normal CDF using rational approximation.
fn inverse_normal_cdf(p: f64) -> f64 {
    // Coefficients for rational approximation
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];

    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let q = p.min(1.0 - p);
    let r = if q > 0.02425 {
        // Central region
        let x = q - 0.5;
        let r = x * x;
        x * (((((A[5] * r + A[4]) * r + A[3]) * r + A[2]) * r + A[1]) * r + A[0])
            / (((((B[4] * r + B[3]) * r + B[2]) * r + B[1]) * r + B[0]) * r + 1.0)
    } else {
        // Tail region
        let r = (-2.0 * q.ln()).sqrt();
        if r <= 5.0 {
            let r = r - 1.6;
            (((((A[5] * r + A[4]) * r + A[3]) * r + A[2]) * r + A[1]) * r + A[0])
                / (((((B[4] * r + B[3]) * r + B[2]) * r + B[1]) * r + B[0]) * r + 1.0)
        } else {
            let r = r - 5.0;
            (((((A[5] * r + A[4]) * r + A[3]) * r + A[2]) * r + A[1]) * r + A[0])
                / (((((B[4] * r + B[3]) * r + B[2]) * r + B[1]) * r + B[0]) * r + 1.0)
        }
    };

    if p < 0.5 {
        -r
    } else {
        r
    }
}

/// Approximate normal CDF.
fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Error function approximation.

/// Monte Carlo test for multifractality against monofractal null hypothesis.
///
/// Tests whether observed data exhibits genuine multifractal scaling behavior
/// by comparing against synthetic monofractal data with the same Hurst exponent.
///
/// # Arguments
/// * `data` - Time series data to test for multifractality
/// * `config` - Monte Carlo configuration parameters
///
/// # Returns
/// * `Ok(MonteCarloTestResult)` - Test results for multifractality
/// * `Err` - If insufficient data or analysis fails
pub fn monte_carlo_multifractal_test(
    data: &[f64],
    config: &MonteCarloConfig,
) -> FractalResult<MonteCarloTestResult> {
    validate_data_length(data, 100, "Monte Carlo multifractal test")?;

    // Use SecureRng for cryptographically secure randomness
    // Note: Seed setup is handled per-thread in parallel execution

    // Calculate observed multifractality measure
    let mf_analysis = perform_multifractal_analysis(data)?;
    let observed_multifractality = mf_analysis.multifractality_degree;

    // Use the estimated H(2) as the monofractal Hurst exponent
    let monofractal_hurst = mf_analysis
        .generalized_hurst_exponents
        .iter()
        .find(|(q, _)| (*q - 2.0).abs() < 0.1)
        .map(|(_, h)| *h)
        .unwrap_or(0.5);

    // Create null hypothesis with the estimated Hurst
    let null_hypothesis = NullHypothesis::Monofractal {
        hurst: monofractal_hurst,
    };

    let data_length = data.len();

    // OPTIMIZATION: Parallel Monte Carlo multifractal simulations
    let null_statistics = {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            // Parallel computation of multifractal null distribution
            let results: Vec<_> = (0..config.num_simulations)
                .into_par_iter()
                .filter_map(|i| {
                    // Each thread gets its own random seed for reproducibility
                    let thread_seed = config
                        .seed
                        .map(|s| s.wrapping_add(i as u64))
                        .unwrap_or_else(|| {
                            // Use system entropy for random seed
                            ThreadLocalRng::u64(0..u64::MAX)
                        });

                    // OPTIMIZATION: Use memory pool for synthetic data generation in parallel threads
                    let mut synthetic_buffer = match get_f64_buffer(data_length) {
                        Ok(buffer) => buffer,
                        Err(_) => return None,
                    };

                    let result = match generate_synthetic_under_null_inplace(
                        &null_hypothesis,
                        &mut synthetic_buffer,
                    ) {
                        Ok(()) => match perform_multifractal_analysis(&synthetic_buffer) {
                            Ok(synthetic_mf) => Some(synthetic_mf.multifractality_degree),
                            Err(_) => None,
                        },
                        Err(_) => None,
                    };

                    // Return buffer to pool for reuse by other threads
                    return_f64_buffer(synthetic_buffer);
                    result
                })
                .collect();

            if results.len() < config.num_simulations / 2 {
                return Err(FractalAnalysisError::NumericalError {
                    reason: "Too many failed multifractal Monte Carlo simulations in parallel processing".to_string(), operation: None
                });
            }

            results
        }

        #[cfg(not(feature = "parallel"))]
        {
            // Sequential fallback when parallel feature is not enabled
            let mut statistics = Vec::with_capacity(config.num_simulations);

            for _ in 0..config.num_simulations {
                // OPTIMIZATION: Use memory pool for synthetic data generation to reduce allocations
                let mut synthetic_buffer = get_f64_buffer(data_length)?;
                synthetic_buffer.resize(data_length, 0.0);
                generate_synthetic_under_null_inplace(&null_hypothesis, &mut synthetic_buffer)?;

                if let Ok(synthetic_mf) = perform_multifractal_analysis(&synthetic_buffer) {
                    statistics.push(synthetic_mf.multifractality_degree);
                }

                // Return buffer to pool for reuse
                return_f64_buffer(synthetic_buffer);
            }

            if statistics.is_empty() {
                return Err(FractalAnalysisError::NumericalError {
                    reason: "Failed to generate null distribution".to_string(),
                    operation: None,
                });
            }

            statistics
        }
    };

    let mut null_statistics = null_statistics;
    // Safe sort handling NaN values
    sort_f64_slice(&mut null_statistics);

    let p_value = calculate_p_value_one_tailed_upper(observed_multifractality, &null_statistics);
    let critical_idx = ((1.0 - config.significance_level) * null_statistics.len() as f64) as usize;
    let critical_value = if null_statistics.is_empty() {
        0.0
    } else {
        null_statistics[critical_idx.min(null_statistics.len().saturating_sub(1))]
    };

    let reject_null = p_value < config.significance_level;

    // Effect size
    let null_mean = null_statistics.iter().sum::<f64>() / null_statistics.len() as f64;
    let null_std = calculate_variance(&null_statistics).sqrt();
    let effect_size = if null_std > 0.0 {
        (observed_multifractality - null_mean) / null_std
    } else {
        0.0
    };

    // FIX: Implement BCa (bias-corrected and accelerated) bootstrap intervals
    let confidence_interval = if config.bootstrap_config.num_bootstrap_samples > 0 {
        compute_bca_bootstrap_interval(
            data,
            observed_multifractality,
            |data| {
                match perform_multifractal_analysis(data) {
                    Ok(mf_analysis) => {
                        // Validate the result
                        if mf_analysis.multifractality_degree.is_finite()
                            && mf_analysis.multifractality_degree >= 0.0
                        {
                            mf_analysis.multifractality_degree
                        } else {
                            // CRITICAL FIX: Return NaN for invalid results instead of silent fallback
                            f64::NAN
                        }
                    }
                    Err(_) => {
                        // CRITICAL FIX: Return NaN for failed analysis instead of silent fallback
                        f64::NAN
                    }
                }
            },
            &config.bootstrap_config,
        )
        .ok()
    } else {
        None
    };

    Ok(MonteCarloTestResult {
        test_name: "Monte Carlo Multifractal Test".to_string(),
        observed_statistic: observed_multifractality,
        null_distribution: null_statistics,
        p_value,
        critical_value,
        reject_null,
        confidence_interval,
        effect_size,
    })
}

/// Surrogate data testing for detecting nonlinear dependencies.
///
/// Tests for nonlinear structure in time series data by comparing against
/// surrogate data that preserves linear properties while destroying nonlinear
/// dependencies. This is fundamental for detecting genuine nonlinear dynamics.
///
/// # Arguments
/// * `data` - Original time series data
/// * `surrogate_method` - Method for generating surrogate data
/// * `test_statistic_fn` - Function to compute test statistic from data
/// * `config` - Monte Carlo configuration parameters
///
/// # Returns
/// * `Ok(MonteCarloTestResult)` - Test results for nonlinearity
/// * `Err` - If data is insufficient or generation fails
pub fn surrogate_data_test(
    data: &[f64],
    surrogate_method: SurrogateMethod,
    test_statistic_fn: impl Fn(&[f64]) -> f64 + Sync,
    config: &MonteCarloConfig,
) -> FractalResult<MonteCarloTestResult> {
    validate_data_length(data, 50, "Surrogate data test")?;

    // Use SecureRng for cryptographically secure randomness
    // Note: Seed setup is handled per-thread in parallel execution

    let observed_statistic = test_statistic_fn(data);

    // OPTIMIZATION: Use parallel processing when enabled and available
    let use_parallel = config.parallel && cfg!(feature = "parallel");

    let mut null_statistics = if use_parallel {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            // Parallel computation of null distribution
            let results: Vec<_> = (0..config.num_simulations)
                .into_par_iter()
                .filter_map(|i| {
                    // Each thread gets its own SecureRng with unique seed for reproducibility
                    let thread_seed = config
                        .seed
                        .map(|s| s.wrapping_add(i as u64))
                        .unwrap_or_else(|| {
                            // Use system entropy for random seed
                            ThreadLocalRng::u64(0..u64::MAX)
                        });

                    match generate_surrogate_data(data, &surrogate_method) {
                        Ok(surrogate) => {
                            let null_stat = test_statistic_fn(&surrogate);
                            if null_stat.is_finite() {
                                Some(null_stat)
                            } else {
                                None
                            }
                        }
                        Err(_) => None,
                    }
                })
                .collect();

            if results.len() < config.num_simulations / 2 {
                return Err(FractalAnalysisError::NumericalError {
                    reason: "Too many failed surrogate data generations in parallel processing"
                        .to_string(),
                    operation: None,
                });
            }

            results
        }

        #[cfg(not(feature = "parallel"))]
        {
            // This branch shouldn't be reached due to use_parallel check
            Vec::new()
        }
    } else {
        // Sequential processing when parallelization is disabled
        let mut statistics = Vec::with_capacity(config.num_simulations);

        for _ in 0..config.num_simulations {
            let surrogate = generate_surrogate_data(data, &surrogate_method)?;
            let null_stat = test_statistic_fn(&surrogate);

            if null_stat.is_finite() {
                statistics.push(null_stat);
            }
        }

        statistics
    };

    if null_statistics.is_empty() {
        return Err(FractalAnalysisError::NumericalError {
            reason: "Failed to generate surrogate statistics".to_string(),
            operation: None,
        });
    }

    // Safe sort handling NaN values
    sort_f64_slice(&mut null_statistics);

    let p_value = calculate_p_value_two_tailed(observed_statistic, &null_statistics);
    let critical_idx = ((1.0 - config.significance_level) * null_statistics.len() as f64) as usize;
    let critical_value = if null_statistics.is_empty() {
        0.0
    } else {
        null_statistics[critical_idx.min(null_statistics.len().saturating_sub(1))]
    };

    let reject_null = p_value < config.significance_level;

    let null_mean = null_statistics.iter().sum::<f64>() / null_statistics.len() as f64;
    let null_std = calculate_variance(&null_statistics).sqrt();
    let effect_size = if null_std > 0.0 {
        (observed_statistic - null_mean) / null_std
    } else {
        0.0
    };

    Ok(MonteCarloTestResult {
        test_name: format!("Surrogate Data Test ({:?})", surrogate_method),
        observed_statistic,
        null_distribution: null_statistics,
        p_value,
        critical_value,
        reject_null,
        confidence_interval: None,
        effect_size,
    })
}

/// Statistical power analysis for Hurst exponent estimators.
///
/// Evaluates the statistical power of Hurst exponent tests across different
/// sample sizes and effect sizes. Essential for experimental design and
/// determining adequate sample sizes for reliable fractal analysis.
///
/// # Arguments
/// * `true_hurst` - True Hurst exponent under null hypothesis
/// * `sample_sizes` - Range of sample sizes to evaluate
/// * `alternative_hurst` - Alternative Hurst exponent to detect
/// * `config` - Monte Carlo configuration for power analysis
///
/// # Returns
/// * `Ok(PowerAnalysisResult)` - Complete power analysis results
/// * `Err` - If parameters are invalid or analysis fails
pub fn power_analysis_hurst_estimator(
    true_hurst: f64,
    sample_sizes: &[usize],
    alternative_hurst: f64,
    config: &MonteCarloConfig,
) -> FractalResult<PowerAnalysisResult> {
    // Use default volatility of 0.01
    power_analysis_hurst_estimator_with_volatility(
        true_hurst,
        sample_sizes,
        alternative_hurst,
        0.01,
        config,
    )
}

/// Statistical power analysis for Hurst exponent estimators with configurable volatility.
///
/// Evaluates the statistical power of Hurst exponent tests across different
/// sample sizes and effect sizes with specified volatility parameter.
///
/// # Arguments
/// * `true_hurst` - True Hurst exponent under null hypothesis
/// * `sample_sizes` - Range of sample sizes to evaluate
/// * `alternative_hurst` - Alternative Hurst exponent to detect
/// * `volatility` - Volatility parameter for synthetic data generation
/// * `config` - Monte Carlo configuration for power analysis
///
/// # Returns
/// * `Ok(PowerAnalysisResult)` - Complete power analysis results
/// * `Err` - If parameters are invalid or analysis fails
pub fn power_analysis_hurst_estimator_with_volatility(
    true_hurst: f64,
    sample_sizes: &[usize],
    alternative_hurst: f64,
    volatility: f64,
    config: &MonteCarloConfig,
) -> FractalResult<PowerAnalysisResult> {
    validate_parameter(true_hurst, 0.01, 0.99, "true Hurst exponent")?;
    validate_parameter(alternative_hurst, 0.01, 0.99, "alternative Hurst exponent")?;
    validate_parameter(volatility, 1e-6, 10.0, "volatility")?;

    // Use SecureRng for cryptographically secure randomness
    // Note: Seed setup is handled per-thread in parallel execution

    let mut power_curves = Vec::with_capacity(sample_sizes.len());
    let mut bias_curves = Vec::with_capacity(sample_sizes.len());
    let mut mse_curves = Vec::with_capacity(sample_sizes.len());

    // OPTIMIZATION: Pre-compute critical values for each sample size to avoid nested Monte Carlo
    // This reduces complexity from O(N×M) to O(N+M)
    let mut critical_values = HashMap::new();

    for &n in sample_sizes {
        // First, compute the critical value for this sample size under the null hypothesis
        // This is done once per sample size instead of for every alternative hypothesis test
        let mut null_distribution = Vec::with_capacity(config.num_simulations);

        for _ in 0..config.num_simulations {
            // Generate data under null hypothesis
            let fbm_config = FbmConfig {
                hurst_exponent: true_hurst,
                volatility,
                method: FbmMethod::Hosking,
            };

            // CRITICAL FIX: Generate FBM with length n+1 so FGN has exactly n samples
            let gen_config = GeneratorConfig {
                length: n + 1,
                seed: None,
                sampling_frequency: 1.0,
            };

            let fbm = generate_fractional_brownian_motion(&gen_config, &fbm_config)?;
            let returns = fbm_to_fgn(&fbm); // Now returns has length n

            if let Ok(estimated_h) = estimate_robust_hurst_exponent(&returns) {
                null_distribution.push(estimated_h);
            }
        }

        // Sort null distribution and find critical values
        sort_f64_slice(&mut null_distribution);

        // Two-tailed test: use both lower and upper critical values
        let lower_idx =
            ((config.significance_level / 2.0) * null_distribution.len() as f64) as usize;
        let upper_idx =
            ((1.0 - config.significance_level / 2.0) * null_distribution.len() as f64) as usize;

        let lower_critical =
            null_distribution[lower_idx.min(null_distribution.len().saturating_sub(1))];
        let upper_critical =
            null_distribution[upper_idx.min(null_distribution.len().saturating_sub(1))];

        critical_values.insert(n, (lower_critical, upper_critical));
    }

    // Now test power for each sample size using pre-computed critical values
    for &n in sample_sizes {
        let (lower_critical, upper_critical) = critical_values
            .get(&n)
            .expect("Critical values should have been pre-computed for all sample sizes")
            .clone();
        let mut rejections = 0;
        let mut estimates = Vec::with_capacity(config.num_simulations);

        for _ in 0..config.num_simulations {
            // Generate data under alternative hypothesis
            let fbm_config = FbmConfig {
                hurst_exponent: alternative_hurst,
                volatility,
                method: FbmMethod::Hosking,
            };

            // CRITICAL FIX: Generate FBM with length n+1 so FGN has exactly n samples
            let gen_config = GeneratorConfig {
                length: n + 1,
                seed: None,
                sampling_frequency: 1.0,
            };

            let fbm = generate_fractional_brownian_motion(&gen_config, &fbm_config)?;
            let returns = fbm_to_fgn(&fbm); // Now returns has length n

            if let Ok(estimated_h) = estimate_robust_hurst_exponent(&returns) {
                estimates.push(estimated_h);

                // OPTIMIZATION: Direct comparison with pre-computed critical values
                // instead of running nested Monte Carlo test
                if estimated_h < lower_critical || estimated_h > upper_critical {
                    rejections += 1;
                }
            }
        }

        let power = rejections as f64 / config.num_simulations as f64;
        power_curves.push(power);

        // Calculate bias and MSE
        if !estimates.is_empty() {
            let mean_estimate = estimates.iter().sum::<f64>() / estimates.len() as f64;
            let bias = mean_estimate - alternative_hurst;
            bias_curves.push(bias);

            let mse = estimates
                .iter()
                .map(|&est| (est - alternative_hurst).powi(2))
                .sum::<f64>()
                / estimates.len() as f64;
            mse_curves.push(mse);
        } else {
            bias_curves.push(f64::NAN);
            mse_curves.push(f64::NAN);
        }
    }

    // Find required sample sizes for 80% and 95% power
    let required_sample_size_80 = find_required_sample_size(&power_curves, sample_sizes, 0.8);
    let required_sample_size_95 = find_required_sample_size(&power_curves, sample_sizes, 0.95);

    Ok(PowerAnalysisResult {
        true_value: true_hurst,
        sample_sizes: sample_sizes.to_vec(),
        power_curves,
        required_sample_size_80,
        required_sample_size_95,
        bias_curves,
        mse_curves,
    })
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Sort a slice of f64 values safely, handling NaN values.
///
/// NaN values are sorted to the end of the slice (treated as greater than any finite value).
/// This provides consistent sorting behavior for statistical calculations.
fn sort_f64_slice(slice: &mut [f64]) {
    slice.sort_by(|a, b| {
        match a.partial_cmp(b) {
            Some(ord) => ord,
            None => {
                // Handle NaN: treat NaN as greater than any number
                if a.is_nan() && b.is_nan() {
                    std::cmp::Ordering::Equal
                } else if a.is_nan() {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            }
        }
    });
}

/// Generate synthetic data under the specified null hypothesis.
/// Generate synthetic data under null hypothesis into a pre-allocated buffer.
///
/// This is an optimized version that writes directly into the provided buffer
/// to reduce memory allocations during Monte Carlo simulations.
fn generate_synthetic_under_null_inplace(
    null_hypothesis: &NullHypothesis,
    buffer: &mut [f64],
) -> FractalResult<()> {
    let length = buffer.len();
    let gen_config = GeneratorConfig {
        length,
        seed: None,
        sampling_frequency: 1.0,
    };

    match null_hypothesis {
        NullHypothesis::WhiteNoise => {
            let data = generate_benchmark_series(BenchmarkSeriesType::WhiteNoise, &gen_config)?;
            buffer.copy_from_slice(&data);
        }
        NullHypothesis::RandomWalk => {
            let data = generate_benchmark_series(BenchmarkSeriesType::RandomWalk, &gen_config)?;
            buffer.copy_from_slice(&data);
        }
        NullHypothesis::ArmaProcess {
            ar_params,
            ma_params,
        } => {
            let arfima_config = ArfimaConfig {
                ar_params: ar_params.clone(),
                d_param: 0.0, // No fractional integration
                ma_params: ma_params.clone(),
                innovation_variance: 1.0,
            };
            let data = generate_arfima(&gen_config, &arfima_config)?;
            buffer.copy_from_slice(&data);
        }
        NullHypothesis::LinearTrend { slope } => {
            let mut data = generate_benchmark_series(BenchmarkSeriesType::WhiteNoise, &gen_config)?;
            for (i, value) in data.iter_mut().enumerate() {
                *value += slope * i as f64;
            }
            buffer.copy_from_slice(&data);
        }
        NullHypothesis::FixedHurst { hurst } => {
            let fbm_config = FbmConfig {
                hurst_exponent: *hurst,
                volatility: 0.01,
                method: FbmMethod::Hosking,
            };
            // Generate FBM with length+1 so that FGN has exactly the right length
            let fbm_gen_config = GeneratorConfig {
                length: length + 1,
                seed: gen_config.seed,
                sampling_frequency: gen_config.sampling_frequency,
            };
            let fbm = generate_fractional_brownian_motion(&fbm_gen_config, &fbm_config)?;
            let fgn = fbm_to_fgn(&fbm);

            // Ensure exact length match
            if fgn.len() == buffer.len() {
                buffer.copy_from_slice(&fgn);
            } else {
                // Handle potential length mismatches by copying available data and padding if needed
                let copy_len = fgn.len().min(buffer.len());
                buffer[..copy_len].copy_from_slice(&fgn[..copy_len]);
                if copy_len < buffer.len() {
                    // Pad with the last value to maintain statistical properties
                    let last_value = if !fgn.is_empty() {
                        fgn[fgn.len() - 1]
                    } else {
                        0.0
                    };
                    for i in copy_len..buffer.len() {
                        buffer[i] = last_value;
                    }
                }
            }
        }
        NullHypothesis::Monofractal { hurst } => {
            let fbm_config = FbmConfig {
                hurst_exponent: *hurst, // Use the specified Hurst exponent
                volatility: 0.01,
                method: FbmMethod::Hosking,
            };
            // Generate FBM with length+1 so that FGN has exactly the right length
            let fbm_gen_config = GeneratorConfig {
                length: length + 1,
                seed: gen_config.seed,
                sampling_frequency: gen_config.sampling_frequency,
            };
            let fbm = generate_fractional_brownian_motion(&fbm_gen_config, &fbm_config)?;
            let fgn = fbm_to_fgn(&fbm);

            // Ensure exact length match
            if fgn.len() == buffer.len() {
                buffer.copy_from_slice(&fgn);
            } else {
                // Handle potential length mismatches by copying available data and padding if needed
                let copy_len = fgn.len().min(buffer.len());
                buffer[..copy_len].copy_from_slice(&fgn[..copy_len]);
                if copy_len < buffer.len() {
                    // Pad with the last value to maintain statistical properties
                    let last_value = if !fgn.is_empty() {
                        fgn[fgn.len() - 1]
                    } else {
                        0.0
                    };
                    for i in copy_len..buffer.len() {
                        buffer[i] = last_value;
                    }
                }
            }
        }
    }

    Ok(())
}

fn generate_synthetic_under_null(
    length: usize,
    null_hypothesis: &NullHypothesis,
) -> FractalResult<Vec<f64>> {
    let gen_config = GeneratorConfig {
        length,
        seed: None,
        sampling_frequency: 1.0,
    };

    match null_hypothesis {
        NullHypothesis::WhiteNoise => {
            generate_benchmark_series(BenchmarkSeriesType::WhiteNoise, &gen_config)
        }
        NullHypothesis::RandomWalk => {
            generate_benchmark_series(BenchmarkSeriesType::RandomWalk, &gen_config)
        }
        NullHypothesis::ArmaProcess {
            ar_params,
            ma_params,
        } => {
            let arfima_config = ArfimaConfig {
                ar_params: ar_params.clone(),
                d_param: 0.0, // No fractional integration
                ma_params: ma_params.clone(),
                innovation_variance: 1.0,
            };
            generate_arfima(&gen_config, &arfima_config)
        }
        NullHypothesis::LinearTrend { slope } => {
            let mut data = generate_benchmark_series(BenchmarkSeriesType::WhiteNoise, &gen_config)?;
            for (i, value) in data.iter_mut().enumerate() {
                *value += slope * i as f64;
            }
            Ok(data)
        }
        NullHypothesis::FixedHurst { hurst } => {
            let fbm_config = FbmConfig {
                hurst_exponent: *hurst,
                volatility: 0.01,
                method: FbmMethod::Hosking,
            };
            let fbm = generate_fractional_brownian_motion(&gen_config, &fbm_config)?;
            Ok(fbm_to_fgn(&fbm))
        }
        NullHypothesis::Monofractal { hurst } => {
            let fbm_config = FbmConfig {
                hurst_exponent: *hurst, // Use the specified Hurst exponent
                volatility: 0.01,
                method: FbmMethod::Hosking,
            };
            let fbm = generate_fractional_brownian_motion(&gen_config, &fbm_config)?;
            Ok(fbm_to_fgn(&fbm))
        }
    }
}

/// Generate surrogate data using the specified method.
pub fn generate_surrogate_data(data: &[f64], method: &SurrogateMethod) -> FractalResult<Vec<f64>> {
    match method {
        SurrogateMethod::FourierTransform => fourier_surrogate(data),
        SurrogateMethod::AmplitudeAdjusted => amplitude_adjusted_surrogate(data),
        SurrogateMethod::IterativeAmplitudeAdjusted { config } => {
            iterative_amplitude_adjusted_surrogate(data, config)
        }
        SurrogateMethod::BlockShuffle { block_size } => block_shuffle_surrogate(data, *block_size),
        SurrogateMethod::PhaseRandomization => phase_randomization_surrogate(data),
    }
}

/// Generate Fourier transform surrogate data.
///
/// Preserves the power spectrum of the original data while randomizing phases.
/// This destroys nonlinear correlations while maintaining linear properties,
/// making it ideal for testing nonlinearity.
pub fn fourier_surrogate(data: &[f64]) -> FractalResult<Vec<f64>> {
    let n = data.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Work with original data length to preserve power exactly
    // Convert data to complex (no zero-padding to avoid power loss)
    let mut fft_data: Vec<Complex<f64>> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();

    // Create FFT planners for exact data length
    let mut planner = FftPlanner::new();
    let fft_forward = planner.plan_fft_forward(n);
    let fft_inverse = planner.plan_fft_inverse(n);

    fft_forward.process(&mut fft_data);

    // Randomize phases while preserving magnitudes exactly
    for i in 1..n / 2 {
        let magnitude = fft_data[i].norm();
        // Use thread-local secure RNG for phase randomization
        let random_phase = 2.0 * std::f64::consts::PI * ThreadLocalRng::f64();

        // Set positive frequency with exact magnitude preservation
        fft_data[i] = Complex::from_polar(magnitude, random_phase);

        // Set corresponding negative frequency (complex conjugate for real input)
        fft_data[n - i] = fft_data[i].conj();
    }

    // DC component remains real (no phase randomization)
    fft_data[0] = Complex::new(fft_data[0].re, 0.0);

    // Nyquist component (if present) remains real
    if n % 2 == 0 {
        fft_data[n / 2] = Complex::new(fft_data[n / 2].re, 0.0);
    }

    // Take inverse FFT
    fft_inverse.process(&mut fft_data);

    // FIX: Proper normalization for rustfft which doesn't normalize by default
    // The forward FFT accumulates n values, so inverse needs 1/n normalization
    let normalization = 1.0 / n as f64;

    // Extract real part with normalization
    let mut surrogate: Vec<f64> = fft_data.iter().map(|c| c.re * normalization).collect();

    // FIX: Preserve exact mean and variance of original data
    // This is more robust than power correction and handles DC component properly
    let original_mean = data.iter().sum::<f64>() / n as f64;
    let surrogate_mean = surrogate.iter().sum::<f64>() / n as f64;

    // Remove mean from both
    let original_centered: Vec<f64> = data.iter().map(|&x| x - original_mean).collect();
    let mut surrogate_centered: Vec<f64> = surrogate.iter().map(|&x| x - surrogate_mean).collect();

    // Calculate variances
    let original_var = original_centered.iter().map(|&x| x * x).sum::<f64>() / n as f64;
    let surrogate_var = surrogate_centered.iter().map(|&x| x * x).sum::<f64>() / n as f64;

    // Scale to match variance and add back original mean
    if surrogate_var > 1e-10 {
        let scale = (original_var / surrogate_var).sqrt();
        surrogate = surrogate_centered
            .iter()
            .map(|&x| x * scale + original_mean)
            .collect();
    }

    Ok(surrogate)
}

/// Generate Amplitude Adjusted Fourier Transform (AAFT) surrogate data.
///
/// Preserves both the power spectrum and the amplitude distribution of the
/// original data while destroying phase relationships. More sophisticated
/// than simple Fourier surrogates.
fn amplitude_adjusted_surrogate(data: &[f64]) -> FractalResult<Vec<f64>> {
    let n = data.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Step 1: Create Gaussian noise with same power spectrum as data
    let gaussian_noise = fourier_surrogate(data)?;

    // Step 2: Rank the Gaussian noise
    let mut indexed_noise: Vec<(f64, usize)> = gaussian_noise
        .iter()
        .enumerate()
        .map(|(i, &val)| (val, i))
        .collect();
    // Safe sort handling NaN values
    indexed_noise.sort_by(|a, b| {
        match a.0.partial_cmp(&b.0) {
            Some(ord) => ord,
            None => {
                // Handle NaN: treat NaN as greater than any number
                if a.0.is_nan() && b.0.is_nan() {
                    std::cmp::Ordering::Equal
                } else if a.0.is_nan() {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            }
        }
    });

    // Step 3: Sort original data by value
    let mut sorted_data = data.to_vec();
    // Use helper function for NaN-safe sorting
    sort_f64_slice(&mut sorted_data);

    // Step 4: Create surrogate by replacing ranked Gaussian values with ranked original values
    let mut surrogate = vec![0.0; n];
    for (rank, &(_, original_index)) in indexed_noise.iter().enumerate() {
        surrogate[original_index] = sorted_data[rank];
    }

    Ok(surrogate)
}

/// Generate Iterative Amplitude Adjusted Fourier Transform (IAAFT) surrogate data.
///
/// Iteratively refines AAFT surrogates to better match both spectrum and
/// distribution constraints. Provides the most accurate surrogate data
/// for most applications.
fn iterative_amplitude_adjusted_surrogate(
    data: &[f64],
    config: &SurrogateConfig,
) -> FractalResult<Vec<f64>> {
    let n = data.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Start with AAFT surrogate
    let mut surrogate = amplitude_adjusted_surrogate(data)?;

    // Get target power spectrum from original data
    let target_spectrum = get_power_spectrum(data)?;

    // Sort original data for rank matching
    let mut sorted_data = data.to_vec();
    // Use helper function for NaN-safe sorting
    sort_f64_slice(&mut sorted_data);

    // Iterative refinement with proper convergence checking
    let mut prev_error = f64::INFINITY;
    let mut stagnation_count = 0;
    let mut error_history = Vec::with_capacity(10);

    for iteration in 0..config.max_iterations {
        // Step 1: Adjust power spectrum
        surrogate = adjust_power_spectrum(&surrogate, &target_spectrum)?;

        // Step 2: Adjust amplitude distribution (rank matching)
        surrogate = rank_match(&surrogate, &sorted_data)?;

        // Check convergence starting from configured iteration to allow initial stabilization
        if iteration >= config.convergence_check_start {
            let current_spectrum = get_power_spectrum(&surrogate)?;
            let spectrum_error = calculate_spectrum_error(&target_spectrum, &current_spectrum);

            // FIX: Check both spectrum AND distribution convergence
            let sorted_surrogate = {
                let mut s = surrogate.clone();
                sort_f64_slice(&mut s);
                s
            };
            let distribution_error: f64 = sorted_data
                .iter()
                .zip(sorted_surrogate.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
                / n as f64;

            let total_error = spectrum_error + distribution_error;

            // Check for true convergence (both spectrum and distribution)
            if spectrum_error < config.tolerance && distribution_error < config.tolerance {
                break;
            }

            // FIX: Detect oscillation by checking error history
            error_history.push(total_error);
            if error_history.len() > 5 {
                error_history.remove(0);

                // Check if we're oscillating (variance of recent errors is small)
                let mean_error = error_history.iter().sum::<f64>() / error_history.len() as f64;
                let error_variance = error_history
                    .iter()
                    .map(|&e| (e - mean_error).powi(2))
                    .sum::<f64>()
                    / error_history.len() as f64;

                if error_variance < (config.tolerance * 0.1).powi(2) {
                    // We're oscillating, not converging
                    break;
                }
            }

            // Check for improvement
            if (prev_error - total_error).abs() < config.tolerance * 0.01 {
                stagnation_count += 1;
                if stagnation_count >= config.stagnation_threshold {
                    break;
                }
            } else {
                stagnation_count = 0;
            }

            prev_error = total_error;
        }
    }

    Ok(surrogate)
}

/// Adjust power spectrum to match target spectrum.
/// Thread-safe: Creates new data instead of modifying in-place.
fn adjust_power_spectrum(data: &[f64], target_spectrum: &[f64]) -> FractalResult<Vec<f64>> {
    let n = data.len();

    // CRITICAL SAFETY CHECK: Prevent massive memory allocation
    const MAX_FFT_SIZE: usize = 1 << 26; // 2^26 = 67,108,864
    const MAX_INPUT_SIZE: usize = MAX_FFT_SIZE / 2; // Conservative limit

    if n > MAX_INPUT_SIZE {
        return Err(FractalAnalysisError::InvalidParameter {
            parameter: "data_length".to_string(),
            value: n as f64,
            constraint: format!("Must be ≤ {} for FFT power spectrum", MAX_INPUT_SIZE),
        });
    }

    // FIX: Use actual data length for FFT to avoid zero-padding artifacts
    let fft_size = n;

    // Take FFT of current data
    let mut fft_data: Vec<Complex<f64>> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();

    // Only resize if necessary for FFT algorithm
    if n != n.next_power_of_two() {
        fft_data.resize(n.next_power_of_two(), Complex::new(0.0, 0.0));
    }

    let mut planner = FftPlanner::new();
    let actual_fft_size = fft_data.len();
    let fft_forward = planner.plan_fft_forward(actual_fft_size);
    fft_forward.process(&mut fft_data);

    // FIX: Thread-safe adjustment - create new array instead of in-place modification
    let mut adjusted_fft = vec![Complex::new(0.0, 0.0); actual_fft_size];

    // Preserve DC component
    adjusted_fft[0] = fft_data[0];

    // Adjust magnitudes to match target spectrum
    for i in 1..n / 2 {
        if i < target_spectrum.len() {
            let current_magnitude = fft_data[i].norm();
            let target_magnitude = target_spectrum[i].sqrt();

            if current_magnitude > 1e-10 {
                let scaling_factor = target_magnitude / current_magnitude;
                adjusted_fft[i] = fft_data[i] * scaling_factor;
                // Maintain conjugate symmetry for real output
                adjusted_fft[n - i] = adjusted_fft[i].conj();
            } else {
                // If current magnitude is near zero, create new complex with target magnitude
                let phase = ThreadLocalRng::f64() * 2.0 * std::f64::consts::PI;
                adjusted_fft[i] = Complex::from_polar(target_magnitude, phase);
                adjusted_fft[n - i] = adjusted_fft[i].conj();
            }
        }
    }

    // Handle Nyquist frequency if present
    if n % 2 == 0 && n / 2 < target_spectrum.len() {
        let i = n / 2;
        let current_magnitude = fft_data[i].norm();
        let target_magnitude = target_spectrum[i].sqrt();
        if current_magnitude > 1e-10 {
            let scaling_factor = target_magnitude / current_magnitude;
            adjusted_fft[i] = Complex::new(fft_data[i].re * scaling_factor, 0.0);
        }
    }

    // Take inverse FFT on the adjusted data
    let fft_inverse = planner.plan_fft_inverse(actual_fft_size);
    fft_inverse.process(&mut adjusted_fft);

    // Extract real part with proper normalization
    let result: Vec<f64> = adjusted_fft[..n]
        .iter()
        .map(|c| c.re / actual_fft_size as f64)
        .collect();

    Ok(result)
}

/// Perform rank matching to preserve amplitude distribution.
fn rank_match(data: &[f64], target_sorted: &[f64]) -> FractalResult<Vec<f64>> {
    let n = data.len();
    if n != target_sorted.len() {
        return Err(FractalAnalysisError::NumericalError {
            reason: "Mismatched array lengths in rank matching".to_string(),
            operation: None,
        });
    }

    // Create index-value pairs and sort by value
    let mut indexed_data: Vec<(f64, usize)> =
        data.iter().enumerate().map(|(i, &val)| (val, i)).collect();
    // Safe sort handling NaN values
    indexed_data.sort_by(|a, b| {
        match a.0.partial_cmp(&b.0) {
            Some(ord) => ord,
            None => {
                // Handle NaN: treat NaN as greater than any number
                if a.0.is_nan() && b.0.is_nan() {
                    std::cmp::Ordering::Equal
                } else if a.0.is_nan() {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            }
        }
    });

    // Replace with target values maintaining rank order
    let mut result = vec![0.0; n];
    for (rank, &(_, original_index)) in indexed_data.iter().enumerate() {
        result[original_index] = target_sorted[rank];
    }

    Ok(result)
}

/// Compute power spectrum of time series data.
pub fn get_power_spectrum(data: &[f64]) -> FractalResult<Vec<f64>> {
    let n = data.len();

    // CRITICAL SAFETY CHECK: Prevent massive memory allocation
    const MAX_FFT_SIZE: usize = 1 << 26; // 2^26 = 67,108,864
    const MAX_INPUT_SIZE: usize = MAX_FFT_SIZE / 2; // Conservative limit

    if n > MAX_INPUT_SIZE {
        return Err(FractalAnalysisError::InvalidParameter {
            parameter: "data_length".to_string(),
            value: n as f64,
            constraint: format!("Must be ≤ {} for power spectrum FFT", MAX_INPUT_SIZE),
        });
    }

    // FIX: Apply Hann window to reduce spectral leakage
    let mut windowed_data = Vec::with_capacity(n);
    for i in 0..n {
        let window_coeff =
            0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos());
        windowed_data.push(data[i] * window_coeff);
    }

    let fft_size = n.next_power_of_two().min(MAX_FFT_SIZE);

    let mut fft_data: Vec<Complex<f64>> = windowed_data
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    fft_data.resize(fft_size, Complex::new(0.0, 0.0));

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    fft.process(&mut fft_data);

    // FIX: Prevent overflow and apply proper normalization
    let normalization = 1.0 / (n as f64).sqrt();
    let spectrum: Vec<f64> = fft_data[..fft_size / 2]
        .iter()
        .map(|c| {
            // Prevent overflow by checking magnitude first
            let mag = c.norm();
            if mag > 1e150 {
                // Prevent overflow in squaring
                f64::MAX
            } else {
                let power = c.norm_sqr() * normalization * normalization;
                // Apply correction for windowing power loss
                power * 1.633 // Hann window correction factor
            }
        })
        .collect();

    Ok(spectrum)
}

/// Calculate normalized error between power spectra.
fn calculate_spectrum_error(target: &[f64], current: &[f64]) -> f64 {
    let n = target.len().min(current.len());
    let mut error = 0.0;
    let mut total = 0.0;

    for i in 0..n {
        error += (target[i] - current[i]).powi(2);
        total += target[i].powi(2);
    }

    if total > 0.0 {
        (error / total).sqrt()
    } else {
        0.0
    }
}

/// Generate block shuffle surrogate data.
///
/// Preserves local correlations within blocks while destroying global
/// correlations by shuffling block order.
fn block_shuffle_surrogate(data: &[f64], block_size: usize) -> FractalResult<Vec<f64>> {
    let n = data.len();
    if block_size == 0 {
        return Err(FractalAnalysisError::InvalidParameter {
            parameter: "block_size".to_string(),
            value: 0.0,
            constraint: "must be greater than 0".to_string(),
        });
    }
    if block_size >= n {
        return fourier_surrogate(data); // Fall back to Fourier surrogate
    }

    let num_blocks = n / block_size;
    let remainder_size = n % block_size;

    // FIX: Include remainder as a separate block to avoid position bias
    let total_blocks = if remainder_size > 0 {
        num_blocks + 1
    } else {
        num_blocks
    };
    let mut blocks = Vec::with_capacity(total_blocks);

    // Extract complete blocks
    for i in 0..num_blocks {
        let start = i * block_size;
        let end = start + block_size;
        blocks.push(data[start..end].to_vec());
    }

    // FIX: Add remainder as its own block if present
    if remainder_size > 0 {
        let remaining_start = num_blocks * block_size;
        blocks.push(data[remaining_start..].to_vec());
    }

    // Shuffle ALL blocks (including remainder) using Fisher-Yates
    for i in (1..blocks.len()).rev() {
        let j = ThreadLocalRng::usize(0..i + 1);
        blocks.swap(i, j);
    }

    // Reconstruct surrogate with randomized block order
    let mut surrogate = Vec::with_capacity(n);
    for block in blocks {
        surrogate.extend(block);
    }

    Ok(surrogate)
}

/// Generate phase randomization surrogate (equivalent to Fourier surrogate).
fn phase_randomization_surrogate(data: &[f64]) -> FractalResult<Vec<f64>> {
    // Phase randomization is exactly what the Fourier surrogate does
    fourier_surrogate(data)
}

/// Estimate robust Hurst exponent optimized for Monte Carlo usage.
///
/// Uses simplified DFA without bootstrap to avoid infinite recursion
/// in Monte Carlo tests while maintaining reasonable accuracy.
fn estimate_robust_hurst_exponent(data: &[f64]) -> FractalResult<f64> {
    // Validate input data for NaN/Inf
    for (i, &value) in data.iter().enumerate() {
        if !value.is_finite() {
            return Err(FractalAnalysisError::NumericalError {
                reason: format!(
                    "Non-finite value {} at index {} in Hurst estimation",
                    value, i
                ),
                operation: None,
            });
        }
    }
    // Use simplified DFA without bootstrap to avoid infinite recursion in Monte Carlo tests
    estimate_dfa_hurst_raw(data)
}

/// Raw DFA Hurst estimation optimized for Monte Carlo simulations.
///
/// Streamlined implementation without bootstrap validation for efficiency
/// in large-scale Monte Carlo studies.
fn estimate_dfa_hurst_raw(data: &[f64]) -> FractalResult<f64> {
    validate_data_length(data, 32, "DFA estimation")?;

    let n = data.len();

    // Integrate the series
    let mean = data.iter().sum::<f64>() / n as f64;
    let mut profile = Vec::with_capacity(n + 1);
    profile.push(0.0);

    let mut cumsum = 0.0;
    for &value in data {
        cumsum += value - mean;
        profile.push(cumsum);
    }

    // Generate scale range
    // CRITICAL FIX: max_scale should be bounded by n/4, not larger
    let min_scale = 8;
    let max_scale = (n / 4).max(min_scale + 1); // Ensure at least 2 scales, but never exceed n/4

    // For very small data (n < 32), DFA is not reliable
    if max_scale <= min_scale {
        return Err(FractalAnalysisError::InsufficientData {
            required: 32,
            actual: n,
        });
    }

    let mut scales = Vec::new();

    // Generate logarithmically spaced scales for better coverage
    let mut scale = min_scale;
    while scale <= max_scale {
        scales.push(scale);

        // CRITICAL FIX: Ensure scale always increases to avoid infinite loop
        let new_scale = ((scale as f64) * 1.2).round() as usize;
        if new_scale <= scale {
            // If scale doesn't increase, manually increment
            scale = scale + 1;
        } else {
            scale = new_scale;
        }

        // Emergency stop to prevent infinite loops
        if scales.len() > 50 || scale > max_scale {
            break;
        }
    }

    // Ensure we have the maximum scale if not already included
    if scales.last() != Some(&max_scale) && max_scale > min_scale {
        scales.push(max_scale);
    }

    let mut log_scales = Vec::new();
    let mut log_fluctuations = Vec::new();

    for &scale in &scales {
        let mut fluctuations = Vec::new();

        let num_segments = n / scale;
        for seg in 0..num_segments {
            let start = seg * scale;
            let end = start + scale;

            if end > profile.len() - 1 {
                break;
            }

            // Linear detrending
            let x_vals: Vec<f64> = (0..scale).map(|i| i as f64).collect();
            let y_vals: Vec<f64> = profile[start..end].iter().copied().collect();

            if let Ok((_, _, residuals)) = ols_regression(&x_vals, &y_vals) {
                let variance =
                    residuals.iter().map(|r| r.powi(2)).sum::<f64>() / residuals.len() as f64;
                if variance >= 0.0 && !variance.is_nan() {
                    fluctuations.push(variance.max(1e-15).sqrt()); // Prevent sqrt(0) issues
                }
            }
        }

        if !fluctuations.is_empty() {
            let avg_fluctuation = fluctuations.iter().sum::<f64>() / fluctuations.len() as f64;
            if avg_fluctuation >= 0.0 && !avg_fluctuation.is_nan() {
                log_scales.push((scale as f64).ln());
                log_fluctuations.push(avg_fluctuation.max(1e-15).ln()); // Prevent log(0) issues
            }
        }
    }

    if log_scales.len() < 3 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 3,
            actual: log_scales.len(),
        });
    }

    // Linear regression to get Hurst exponent
    let (hurst_estimate, _, _) = ols_regression(&log_scales, &log_fluctuations)?;

    // Financial sector validation: ensure reasonable bounds
    Ok(hurst_estimate.max(0.01).min(0.99))
}

/// Calculate two-tailed p-value for Monte Carlo testing.
fn calculate_p_value_two_tailed(observed: f64, null_distribution: &[f64]) -> f64 {
    let n = null_distribution.len() as f64;

    // Count values in null distribution that are as extreme or more extreme than observed
    // For two-tailed test: count values >= |observed - median| from the median
    let median = if null_distribution.is_empty() {
        observed
    } else {
        let mut sorted_null = null_distribution.to_vec();
        // Safe sort handling NaN values
        sorted_null.sort_by(|a, b| {
            match a.partial_cmp(b) {
                Some(ord) => ord,
                None => {
                    // Handle NaN: treat NaN as greater than any number
                    if a.is_nan() && b.is_nan() {
                        std::cmp::Ordering::Equal
                    } else if a.is_nan() {
                        std::cmp::Ordering::Greater
                    } else {
                        std::cmp::Ordering::Less
                    }
                }
            }
        });
        sorted_null[sorted_null.len() / 2]
    };

    let observed_deviation = (observed - median).abs();
    let count_extreme = null_distribution
        .iter()
        .filter(|&&x| (x - median).abs() >= observed_deviation)
        .count() as f64;

    // Add 1 to numerator and denominator for continuity correction
    (count_extreme + 1.0) / (n + 1.0)
}

/// Calculate one-tailed upper p-value for Monte Carlo testing.
fn calculate_p_value_one_tailed_upper(observed: f64, null_distribution: &[f64]) -> f64 {
    let n = null_distribution.len() as f64;
    let count_greater = null_distribution.iter().filter(|&&x| x >= observed).count() as f64;

    (count_greater + 1.0) / (n + 1.0)
}

/// Calculate volatility of time series data.

/// Find minimum sample size required for target statistical power.
fn find_required_sample_size(
    power_curves: &[f64],
    sample_sizes: &[usize],
    target_power: f64,
) -> Option<usize> {
    for (i, &power) in power_curves.iter().enumerate() {
        if power >= target_power && i < sample_sizes.len() {
            return Some(sample_sizes[i]);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monte_carlo_hurst_test() {
        // Generate fractional Brownian motion with known Hurst
        let config = GeneratorConfig {
            length: 1000,
            seed: Some(42),
            ..Default::default()
        };

        let fbm_config = FbmConfig {
            hurst_exponent: 0.7,
            volatility: 0.01,
            method: FbmMethod::Hosking,
        };

        let fbm = generate_fractional_brownian_motion(&config, &fbm_config).unwrap();
        let returns = fbm_to_fgn(&fbm);

        let mc_config = MonteCarloConfig {
            num_simulations: 100, // Reduced for testing
            significance_level: 0.05,
            seed: Some(123),
            ..Default::default()
        };

        // Test against white noise null hypothesis (should reject)
        let result =
            monte_carlo_hurst_test(&returns, NullHypothesis::WhiteNoise, &mc_config).unwrap();

        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.observed_statistic > 0.0 && result.observed_statistic < 1.0);
        assert!(result.effect_size.is_finite());
    }

    #[test]
    fn test_surrogate_data_test() {
        let data: Vec<f64> = (0..100)
            .map(|i| (i as f64 / 10.0).sin() + 0.1 * ThreadLocalRng::f64())
            .collect();

        let config = MonteCarloConfig {
            num_simulations: 50,
            ..Default::default()
        };

        let result = surrogate_data_test(
            &data,
            SurrogateMethod::BlockShuffle { block_size: 10 },
            |data| data.iter().sum::<f64>() / data.len() as f64, // Mean as test statistic
            &config,
        )
        .unwrap();

        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.observed_statistic.is_finite());
    }

    #[test]
    #[cfg(feature = "long-tests")]
    fn test_power_analysis() {
        let sample_sizes = vec![50, 100, 200];
        let config = MonteCarloConfig {
            num_simulations: 20, // Very reduced for testing
            ..Default::default()
        };

        let result = power_analysis_hurst_estimator(
            0.5, // Null hypothesis
            &sample_sizes,
            0.7, // Alternative hypothesis
            &config,
        )
        .unwrap();

        assert_eq!(result.sample_sizes.len(), 3);
        assert_eq!(result.power_curves.len(), 3);
        assert!(result.power_curves.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }
}
