//! Bootstrap configuration structures and validation.
//!
//! This module contains all configuration types for bootstrap resampling,
//! including method selection, confidence interval configuration, and
//! adaptive parameter sizing based on statistical theory.

use crate::errors::{FractalAnalysisError, FractalResult};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// Safety constants
pub(crate) const MIN_DATA_POINTS: usize = 20;
pub(crate) const MIN_BOOTSTRAP_SAMPLES: usize = 100;  // Minimum for reliable estimates
pub(crate) const MIN_BOOTSTRAP_SAMPLES_BCA: usize = 200;  // BCa needs more samples for stability
pub(crate) const MAX_BOOTSTRAP_SAMPLES: usize = 10000;  // Higher cap for high-precision work
pub(crate) const MAX_BOOTSTRAP_SAMPLES_DEFAULT: usize = 1000;  // Default cap for standard use
pub(crate) const MIN_CONFIDENCE_LEVEL: f64 = 0.5;
pub(crate) const MAX_CONFIDENCE_LEVEL: f64 = 0.999;
pub(crate) const MIN_BLOCK_SIZE: usize = 2;
pub(crate) const MAX_BLOCK_SIZE_RATIO: f64 = 0.5; // Block size shouldn't exceed half the data

/// Bootstrap configuration for time series resampling.
///
/// Configures all aspects of bootstrap resampling including the method,
/// sample size, block parameters, and confidence interval construction.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BootstrapConfiguration {
    /// Number of bootstrap samples
    pub num_bootstrap_samples: usize,
    /// Bootstrap method
    pub bootstrap_method: BootstrapMethod,
    /// Block size for block bootstrap
    pub block_size: Option<usize>,
    /// Confidence levels for intervals
    pub confidence_levels: Vec<f64>,
    /// Method for constructing confidence intervals
    pub confidence_interval_method: ConfidenceIntervalMethod,
    /// Random seed for reproducible results
    pub seed: Option<u64>,
    /// Outer bootstrap samples for studentized CI (default: 100)
    pub studentized_outer: Option<usize>,
    /// Inner bootstrap samples for studentized CI (default: 50)
    pub studentized_inner: Option<usize>,
    /// Block size for jackknife in BCa acceleration calculation
    pub jackknife_block_size: Option<usize>,
    /// Force block jackknife for BCa even without detected dependence.
    /// 
    /// When set to `true`, block jackknife will be used for BCa acceleration
    /// calculation even if no significant temporal dependence is detected in
    /// the data. This can be useful for series where dependence may be weak
    /// but still present.
    pub force_block_jackknife: Option<bool>,
}

/// Available bootstrap methods for different data dependencies.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BootstrapMethod {
    /// Standard bootstrap (iid resampling)
    Standard,
    /// Block bootstrap for dependent data
    Block,
    /// Stationary bootstrap
    Stationary,
    /// Circular bootstrap
    Circular,
}

/// Confidence interval with method information.
///
/// Represents a confidence interval computed using a specific method,
/// with metadata about the confidence level and construction technique.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConfidenceInterval {
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Lower bound
    pub lower_bound: f64,
    /// Upper bound
    pub upper_bound: f64,
    /// Method used for CI construction
    pub method: ConfidenceIntervalMethod,
}

/// Methods for constructing confidence intervals.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ConfidenceIntervalMethod {
    /// Normal approximation
    Normal,
    /// Bootstrap percentile
    BootstrapPercentile,
    /// Bootstrap bias-corrected and accelerated (BCa)
    BootstrapBca,
    /// Studentized bootstrap.
    /// 
    /// Note: Nested bootstrap sizing is controlled by the `studentized_outer` and 
    /// `studentized_inner` configuration parameters. The outer samples are capped
    /// at 1000 for practical performance reasons, even though validation allows
    /// higher values. Inner samples default to 50 if not specified.
    StudentizedBootstrap,
}

/// Complete results from bootstrap validation.
///
/// Contains all bootstrap statistics including the original estimate,
/// bootstrap distribution, bias estimates, and confidence intervals.
#[derive(Debug, Clone)]
pub struct BootstrapValidation {
    /// Original estimate
    pub original_estimate: f64,
    /// Bootstrap estimates
    pub bootstrap_estimates: Vec<f64>,
    /// Bootstrap bias
    pub bias: f64,
    /// Bootstrap standard error
    pub standard_error: f64,
    /// Confidence intervals
    pub confidence_intervals: Vec<ConfidenceInterval>,
}

impl Default for BootstrapConfiguration {
    fn default() -> Self {
        Self {
            num_bootstrap_samples: 1000, // Default value - use adaptive() for data-driven sizing
            bootstrap_method: BootstrapMethod::Block,
            block_size: None, // Auto-select via Politis-White
            confidence_levels: vec![0.90, 0.95, 0.99],
            confidence_interval_method: ConfidenceIntervalMethod::BootstrapBca,
            seed: None,
            studentized_outer: None,
            studentized_inner: None,
            jackknife_block_size: None,
            force_block_jackknife: None,
        }
    }
}

impl BootstrapConfiguration {
    /// Calculate optimal bootstrap sample size based on data characteristics and statistical theory.
    ///
    /// This method balances computational efficiency with statistical accuracy using established
    /// theoretical results from bootstrap literature:
    ///
    /// - Efron & Tibshirani (1993): Bootstrap accuracy ∝ 1/√B for most statistics
    /// - Hall (1995): For bias-corrected estimators, B ≈ n^0.4 provides optimal MSE
    /// - Davison & Hinkley (1997): B ≥ 200 sufficient for 95% confidence intervals
    /// - Shao & Tu (1995): Percentile methods converge slower, need B ≥ 400
    ///
    /// # Arguments
    /// * `data_length` - Length of the original time series
    /// * `confidence_level` - Highest confidence level requested (affects precision requirements)
    /// * `estimator_complexity` - Computational complexity of the estimator (Low/Medium/High)
    ///
    /// # Returns
    /// Optimal number of bootstrap samples that maintains statistical rigor while optimizing performance
    ///
    /// # Mathematical Justification
    /// For Hurst exponent estimation, the bootstrap MSE decomposes as:
    /// MSE(B) = Bias² + Variance + Monte Carlo Error
    ///
    /// Where Monte Carlo Error ≈ σ²/B, leading to optimal B ∝ n^α with α ∈ [0.3, 0.5]
    /// depending on the specific estimator properties.
    pub fn calculate_adaptive_bootstrap_size(
        data_length: usize,
        confidence_level: f64,
        estimator_complexity: EstimatorComplexity,
    ) -> usize {
        // Base formula: B = c * n^α where α ∈ [0.3, 0.5] based on statistical theory
        let n = data_length as f64;
        let alpha = match estimator_complexity {
            EstimatorComplexity::Low => 0.3,    // R/S, simple statistics
            EstimatorComplexity::Medium => 0.4, // DFA, GPH regression
            EstimatorComplexity::High => 0.5,   // Multifractal, regime detection
        };

        // Base sample calculation using established theory
        let base_samples = (n.powf(alpha) * 25.0) as usize; // Scaling factor from empirical studies

        // Adjustment for confidence level precision requirements
        let confidence_multiplier = if confidence_level >= 0.99 {
            1.5 // Higher precision needed for 99% intervals
        } else if confidence_level >= 0.95 {
            1.2 // Standard precision for 95% intervals
        } else {
            1.0 // Lower precision acceptable for 90% intervals
        };

        let adjusted_samples = (base_samples as f64 * confidence_multiplier) as usize;

        // Enforce statistical bounds based on bootstrap literature:
        // - Minimum 100: Below this, CLT approximations become unreliable
        // - Minimum 200 for BCa: Hall & Wilson (1991) minimum for bias-correction
        // - Maximum 10000: Clamped to MAX_BOOTSTRAP_SAMPLES for practical limits
        let min_samples = match estimator_complexity {
            EstimatorComplexity::Low => 100,
            EstimatorComplexity::Medium => 150,
            EstimatorComplexity::High => 200,
        };

        adjusted_samples.max(min_samples).min(MAX_BOOTSTRAP_SAMPLES)
    }

    /// Create adaptive configuration that automatically optimizes bootstrap sample size
    /// based on data characteristics while maintaining mathematical rigor.
    pub fn adaptive(data_length: usize, estimator_complexity: EstimatorComplexity) -> Self {
        let highest_confidence = 0.95; // Standard requirement for most applications
        let optimal_samples = Self::calculate_adaptive_bootstrap_size(
            data_length,
            highest_confidence,
            estimator_complexity,
        );

        // BCa requires minimum 200 samples regardless of complexity
        // Ensure we meet this requirement when using BCa
        let final_samples = optimal_samples.max(MIN_BOOTSTRAP_SAMPLES_BCA);

        Self {
            num_bootstrap_samples: final_samples,
            bootstrap_method: BootstrapMethod::Block, // Preserves temporal dependence
            block_size: Some((data_length / 20).max(10).min(100)), // Adaptive block sizing
            confidence_levels: vec![0.90, 0.95, 0.99],
            confidence_interval_method: ConfidenceIntervalMethod::BootstrapBca,
            seed: None,
            studentized_outer: None,
            studentized_inner: None,
            jackknife_block_size: None,
            force_block_jackknife: None,
        }
    }
}

/// Estimator computational complexity levels for adaptive bootstrap sizing.
///
/// This classification helps determine optimal bootstrap sample sizes based on
/// the computational cost and statistical properties of different estimators.
#[derive(Debug, Clone, Copy)]
pub enum EstimatorComplexity {
    /// Simple estimators like R/S, basic sample statistics
    /// Fast to compute, stable convergence properties
    Low,
    /// Moderate complexity like DFA, regression-based methods
    /// Moderate computational cost, good convergence  
    Medium,
    /// Complex estimators like multifractal analysis, regime detection
    /// High computational cost, slower convergence
    High,
}

/// Validate bootstrap configuration parameters for financial-grade safety
pub(crate) fn validate_bootstrap_config(config: &BootstrapConfiguration) -> FractalResult<()> {
    // Validate number of bootstrap samples
    let min_samples = if matches!(config.confidence_interval_method, ConfidenceIntervalMethod::BootstrapBca) {
        MIN_BOOTSTRAP_SAMPLES_BCA  // BCa needs more samples
    } else {
        MIN_BOOTSTRAP_SAMPLES
    };
    
    if config.num_bootstrap_samples < min_samples {
        return Err(FractalAnalysisError::InvalidParameter {
            parameter: "num_bootstrap_samples".to_string(),
            value: config.num_bootstrap_samples as f64,
            constraint: format!("Must be at least {} for reliable estimates", min_samples),
        });
    }
    
    if config.num_bootstrap_samples > MAX_BOOTSTRAP_SAMPLES {
        return Err(FractalAnalysisError::InvalidParameter {
            parameter: "num_bootstrap_samples".to_string(),
            value: config.num_bootstrap_samples as f64,
            constraint: format!("Must be at most {} for practical computation", MAX_BOOTSTRAP_SAMPLES),
        });
    }
    
    // Validate confidence levels
    for &level in &config.confidence_levels {
        if level < MIN_CONFIDENCE_LEVEL || level > MAX_CONFIDENCE_LEVEL {
            return Err(FractalAnalysisError::InvalidParameter {
                parameter: "confidence_level".to_string(),
                value: level,
                constraint: format!("Must be between {} and {}", MIN_CONFIDENCE_LEVEL, MAX_CONFIDENCE_LEVEL),
            });
        }
    }
    
    // Validate block size if specified
    if let Some(block_size) = config.block_size {
        if block_size < MIN_BLOCK_SIZE {
            return Err(FractalAnalysisError::InvalidParameter {
                parameter: "block_size".to_string(),
                value: block_size as f64,
                constraint: format!("Must be at least {} for block bootstrap", MIN_BLOCK_SIZE),
            });
        }
    }
    
    Ok(())
}