//! Time-series cross-validation for fractal model selection and validation.
//!
//! This module provides comprehensive cross-validation frameworks specifically
//! designed for time series fractal analysis. Unlike standard machine learning
//! cross-validation, these methods respect the temporal structure and dependencies
//! inherent in financial time series data.
//!
//! ## Key Features
//!
//! - **Time-Aware Cross-Validation**: Walk-forward, rolling window, expanding window methods
//! - **Model Selection**: Automated selection between multiple fractal estimators
//! - **Performance Metrics**: Comprehensive evaluation including stability and directional accuracy
//! - **Statistical Testing**: Significance tests for comparing model performance
//! - **Ensemble Methods**: Weighted combinations of multiple estimators

use crate::{
    bootstrap::*,
    errors::{validate_data_length, FractalAnalysisError, FractalResult},
    fft_ops::calculate_periodogram_fft,
    generators::{
        fbm_to_fgn, generate_fractional_brownian_motion, FbmConfig, FbmMethod, GeneratorConfig,
    },
    math_utils::{calculate_variance, ols_regression},
    multifractal::*,
};
use log::warn;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// CONFIGURATION STRUCTURES - Flexible Parameter Settings
// ============================================================================

/// Configuration for Detrended Fluctuation Analysis (DFA)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DfaConfig {
    /// Minimum scale size (default: 8, Peng et al., 1994)
    pub min_scale: usize,
    /// Maximum scale as fraction of data length (default: 0.25)
    pub max_scale_ratio: f64,
}

impl Default for DfaConfig {
    fn default() -> Self {
        Self {
            min_scale: 8,
            max_scale_ratio: 0.25,
        }
    }
}

/// Configuration for GPH Periodogram Regression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GphConfig {
    /// Frequency selection power (default: 0.8, Geweke & Porter-Hudak, 1983)
    pub frequency_power: f64,
    /// Minimum frequency for stable estimation (default: 1)
    pub min_frequency: usize,
}

impl Default for GphConfig {
    fn default() -> Self {
        Self {
            frequency_power: 0.8,
            min_frequency: 1,
        }
    }
}

/// Configuration for WTMM (Wavelet Transform Modulus Maxima)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WtmmEstimatorConfig {
    /// Empirical scaling factor for standard error estimation (default: 0.1)
    pub std_error_scale: f64,
    /// Minimum scale for WTMM analysis (default: 2.0)
    pub min_scale: f64,
    /// Maximum scale as fraction of data length (default: 0.25)
    pub max_scale_ratio: f64,
}

impl Default for WtmmEstimatorConfig {
    fn default() -> Self {
        Self {
            std_error_scale: 0.1,
            min_scale: 2.0,
            max_scale_ratio: 0.25,
        }
    }
}

/// Configuration for Hurst-based trading strategies
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TradingConfig {
    /// Minimum Hurst estimate value to prevent numerical issues (default: 0.01, Lo, 1991)
    pub hurst_min_bound: f64,
    /// Maximum Hurst estimate value to prevent numerical issues (default: 0.99, Lo, 1991)
    pub hurst_max_bound: f64,
    /// Trading signal threshold from neutral (0.5) for Hurst-based strategy (default: 0.05)
    pub hurst_trading_threshold: f64,
    /// Transaction cost as fraction of position size (default: 0.001 = 10 basis points)
    pub transaction_cost_rate: f64,
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            hurst_min_bound: 0.01,
            hurst_max_bound: 0.99,
            hurst_trading_threshold: 0.05,
            transaction_cost_rate: 0.001,
        }
    }
}

/// Configuration for financial metrics calculation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FinancialMetricsConfig {
    /// Data type of the input series
    pub data_type: DataType,
    /// Annual trading periods for annualization (default: 252 for daily data)
    pub annual_periods: f64,
    /// Risk-free rate for Sharpe ratio calculation (default: 0.0)
    pub risk_free_rate: f64,
}

/// Input data type for financial calculations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DataType {
    /// Raw price data
    Prices,
    /// Simple returns: (P_t - P_{t-1}) / P_{t-1}
    SimpleReturns,
    /// Log returns: ln(P_t / P_{t-1})
    LogReturns,
}

impl Default for FinancialMetricsConfig {
    fn default() -> Self {
        Self {
            data_type: DataType::LogReturns,
            annual_periods: 252.0,
            risk_free_rate: 0.0,
        }
    }
}

/// Default values as constants for backward compatibility
const Z_95_CONFIDENCE: f64 = 1.96;

/// Cross-validation methods for time series analysis.
///
/// These methods respect the temporal ordering of data, which is crucial
/// for financial time series where future information cannot be used to
/// predict the past (no look-ahead bias).
#[derive(Debug, Clone)]
pub enum CrossValidationMethod {
    /// Walk-forward validation with fixed window size
    /// Training window slides forward, maintaining constant size
    WalkForward {
        /// Size of the training window
        window_size: usize,
        /// Step size for moving the window forward
        step_size: usize,
    },
    /// Expanding window validation with growing training set
    /// Training set grows over time, more data = better estimates
    ExpandingWindow {
        /// Initial training set size
        initial_size: usize,
        /// Step size for expanding the window
        step_size: usize,
    },
    /// Time-based splits for irregular time series
    TimeBased {
        /// Fraction of each split to use for training
        train_fraction: f64,
        /// Number of time-based splits
        num_splits: usize,
    },
    /// Purged cross-validation for overlapping predictions
    /// Includes gaps to prevent information leakage
    Purged {
        /// Training window size
        window_size: usize,
        /// Purge gap to prevent leakage
        purge_size: usize,
        /// Embargo period after purge
        embargo_size: usize,
    },
}

/// Fractal estimator types available for model selection.
///
/// Each estimator has different computational complexity, accuracy,
/// and suitability for different types of fractal behavior.
#[derive(Debug, Clone)]
pub enum FractalEstimator {
    /// Rescaled Range (R/S) analysis - classic method
    RescaledRange,
    /// Detrended Fluctuation Analysis - robust and popular with config
    DetrendedFluctuation(DfaConfig),
    /// Multifractal DFA for specific q-moment
    MultifractalDFA {
        /// q-value for moment calculation
        q_value: f64,
        /// DFA configuration
        config: DfaConfig,
    },
    /// Periodogram regression (GPH - Geweke and Porter-Hudak method) with config
    PeriodogramRegression(GphConfig),
    /// Wavelet-based estimation
    WaveletBased,
    /// Wavelet Transform Modulus Maxima method with config
    WaveletModulusMaxima(WtmmEstimatorConfig),
    /// Ensemble method combining multiple estimators
    Ensemble {
        /// List of methods to combine (sorted for canonical representation)
        methods: Vec<FractalEstimator>,
    },
}

impl FractalEstimator {
    /// Create an Ensemble estimator with canonical ordering
    pub fn new_ensemble(mut methods: Vec<FractalEstimator>) -> Self {
        // Sort methods to ensure canonical representation
        methods.sort_by_key(|m| m.discriminant_value());
        FractalEstimator::Ensemble { methods }
    }

    /// Get a discriminant value for ordering (used for canonical representation)
    fn discriminant_value(&self) -> u8 {
        match self {
            FractalEstimator::RescaledRange => 0,
            FractalEstimator::DetrendedFluctuation(_) => 1,
            FractalEstimator::MultifractalDFA { .. } => 2,
            FractalEstimator::PeriodogramRegression(_) => 3,
            FractalEstimator::WaveletBased => 4,
            FractalEstimator::WaveletModulusMaxima(_) => 5,
            FractalEstimator::Ensemble { .. } => 6,
        }
    }
}

// Implement PartialEq and Eq for proper comparison
impl PartialEq for FractalEstimator {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (FractalEstimator::RescaledRange, FractalEstimator::RescaledRange) => true,
            (
                FractalEstimator::DetrendedFluctuation(c1),
                FractalEstimator::DetrendedFluctuation(c2),
            ) => {
                c1.min_scale == c2.min_scale
                    && (c1.max_scale_ratio - c2.max_scale_ratio).abs() < 1e-10
            }
            (
                FractalEstimator::MultifractalDFA {
                    q_value: q1,
                    config: c1,
                },
                FractalEstimator::MultifractalDFA {
                    q_value: q2,
                    config: c2,
                },
            ) => {
                (q1 - q2).abs() < 1e-10
                    && c1.min_scale == c2.min_scale
                    && (c1.max_scale_ratio - c2.max_scale_ratio).abs() < 1e-10
            }
            (
                FractalEstimator::PeriodogramRegression(c1),
                FractalEstimator::PeriodogramRegression(c2),
            ) => {
                (c1.frequency_power - c2.frequency_power).abs() < 1e-10
                    && c1.min_frequency == c2.min_frequency
            }
            (FractalEstimator::WaveletBased, FractalEstimator::WaveletBased) => true,
            (
                FractalEstimator::WaveletModulusMaxima(c1),
                FractalEstimator::WaveletModulusMaxima(c2),
            ) => {
                (c1.std_error_scale - c2.std_error_scale).abs() < 1e-10
                    && (c1.min_scale - c2.min_scale).abs() < 1e-10
                    && (c1.max_scale_ratio - c2.max_scale_ratio).abs() < 1e-10
            }
            (
                FractalEstimator::Ensemble { methods: m1 },
                FractalEstimator::Ensemble { methods: m2 },
            ) => m1 == m2,
            _ => false,
        }
    }
}

impl Eq for FractalEstimator {}

// Implement Hash for HashMap usage
impl std::hash::Hash for FractalEstimator {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.discriminant_value().hash(state);
        match self {
            FractalEstimator::RescaledRange => {}
            FractalEstimator::DetrendedFluctuation(c) => {
                c.min_scale.hash(state);
                ((c.max_scale_ratio * 1000.0) as i64).hash(state);
            }
            FractalEstimator::MultifractalDFA { q_value, config } => {
                ((q_value * 1000.0) as i64).hash(state);
                config.min_scale.hash(state);
                ((config.max_scale_ratio * 1000.0) as i64).hash(state);
            }
            FractalEstimator::PeriodogramRegression(c) => {
                ((c.frequency_power * 1000.0) as i64).hash(state);
                c.min_frequency.hash(state);
            }
            FractalEstimator::WaveletBased => {}
            FractalEstimator::WaveletModulusMaxima(c) => {
                ((c.std_error_scale * 1000.0) as i64).hash(state);
                ((c.min_scale * 1000.0) as i64).hash(state);
                ((c.max_scale_ratio * 1000.0) as i64).hash(state);
            }
            FractalEstimator::Ensemble { methods } => {
                methods.hash(state);
            }
        }
    }
}

/// Comprehensive performance metrics for fractal model evaluation.
///
/// Includes both statistical accuracy measures and financial performance
/// indicators relevant to trading and risk management applications.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PerformanceMetrics {
    /// Mean Absolute Stability Drift - average temporal inconsistency
    pub mae: f64,
    /// Mean Squared Stability Drift - penalizes large temporal changes
    pub mse: f64,
    /// Root Mean Squared Stability Drift
    pub rmse: f64,
    /// Mean Absolute Percentage Drift relative to train estimates
    pub mape: f64,
    /// Estimate stability concordance - consistency of estimate changes between folds
    /// NOT a measure of financial prediction accuracy
    pub estimate_stability_concordance: f64,
    /// Hit rate - percentage of predictions within confidence intervals
    pub hit_rate: f64,
    /// Sharpe ratio for trading strategies based on fractal signals
    pub sharpe_ratio: f64,
    /// Maximum drawdown from peak performance
    pub max_drawdown: f64,
    /// Stability measure - variance of estimates across folds
    pub stability: f64,
}

/// Complete cross-validation results for a single estimator.
///
/// Contains detailed information about performance across all folds,
/// confidence intervals, and stability measures.
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    /// The fractal estimator that was evaluated
    pub estimator: FractalEstimator,
    /// Aggregated performance metrics across all folds
    pub metrics: PerformanceMetrics,
    /// Detailed results for each individual fold
    pub fold_results: Vec<FoldResult>,
    /// Average performance across all folds
    pub average_performance: f64,
    /// Standard deviation of performance across folds
    pub performance_std: f64,
    /// Bootstrap confidence interval for performance
    pub performance_ci: ConfidenceInterval,
    /// Feature importance scores for ensemble methods
    pub feature_importance: Option<HashMap<String, f64>>,
}

/// Results from a single cross-validation fold.
///
/// Contains training and testing information for one temporal split
/// of the data, including estimates and stability measurements.
#[derive(Debug, Clone)]
pub struct FoldResult {
    /// Index of this fold in the cross-validation sequence
    pub fold_index: usize,
    /// Training period (start_index, end_index)
    pub train_period: (usize, usize),
    /// Testing period (start_index, end_index)
    pub test_period: (usize, usize),
    /// Parameter estimate from training data
    pub train_estimate: f64,
    /// Parameter estimate from test data
    pub test_estimate: f64,
    /// True value if known (for synthetic data validation)
    pub true_value: Option<f64>,
    /// Stability drift: |test_estimate - train_estimate|
    /// Measures temporal consistency of fractal characteristics
    pub error: f64,
    /// Confidence interval calculated from TRAINING data
    /// Used to check if test estimates fall within expected bounds
    pub train_confidence_interval: Option<ConfidenceInterval>,
    /// Confidence interval calculated from TEST data (for reference)
    pub test_confidence_interval: Option<ConfidenceInterval>,
}

/// Complete model selection results.
///
/// Contains the best performing model, performance comparisons,
/// and statistical significance tests between competing models.
#[derive(Debug, Clone)]
pub struct ModelSelectionResult {
    /// Best performing estimator according to selection criterion
    pub best_estimator: FractalEstimator,
    /// Complete results for all evaluated estimators
    pub all_results: HashMap<FractalEstimator, CrossValidationResult>,
    /// Ranking of estimators by performance (best to worst)
    pub ranking: Vec<(FractalEstimator, f64)>,
    /// Criterion used for model selection
    pub selection_criterion: SelectionCriterion,
    /// P-values from pairwise significance tests
    pub significance_tests: HashMap<(FractalEstimator, FractalEstimator), f64>,
}

/// Model selection criteria for choosing the best estimator.
///
/// Different criteria emphasize different aspects of model performance,
/// allowing for application-specific optimization.
#[derive(Debug, Clone)]
pub enum SelectionCriterion {
    /// Minimize average cross-validation error
    MinimizeError,
    /// Maximize directional accuracy for trend prediction
    MaximizeDirectionalAccuracy,
    /// Maximize stability (minimize variance across folds)
    MaximizeStability,
    /// Information criteria with complexity penalty
    InformationCriteria {
        /// Penalty parameter for model complexity
        penalty: f64,
    },
    /// Custom weighted combination of multiple criteria
    WeightedCombination {
        /// Weights for different performance metrics
        weights: HashMap<String, f64>,
    },
}

/// Configuration parameters for cross-validation procedures.
///
/// Controls all aspects of the cross-validation process including
/// the validation method, estimators to test, and statistical parameters.
#[derive(Debug, Clone)]
pub struct CrossValidationConfig {
    /// Cross-validation method to use
    pub method: CrossValidationMethod,
    /// List of fractal estimators to evaluate
    pub estimators: Vec<FractalEstimator>,
    /// Criterion for selecting the best model
    pub selection_criterion: SelectionCriterion,
    /// Bootstrap configuration for confidence intervals
    pub bootstrap_config: BootstrapConfiguration,
    /// Number of Monte Carlo runs for stability assessment
    pub stability_runs: usize,
    /// Significance level for statistical tests
    pub significance_level: f64,
    /// Random seed for reproducible results
    pub seed: Option<u64>,
    /// Trading strategy configuration
    pub trading_config: TradingConfig,
    /// Financial metrics configuration
    pub financial_config: FinancialMetricsConfig,
}

impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self {
            method: CrossValidationMethod::WalkForward {
                window_size: 200,
                step_size: 50,
            },
            estimators: vec![
                FractalEstimator::DetrendedFluctuation(DfaConfig::default()),
                FractalEstimator::MultifractalDFA {
                    q_value: 2.0,
                    config: DfaConfig::default(),
                },
                FractalEstimator::PeriodogramRegression(GphConfig::default()),
            ],
            selection_criterion: SelectionCriterion::MinimizeError,
            bootstrap_config: BootstrapConfiguration::default(),
            stability_runs: 100,
            significance_level: 0.05,
            seed: None,
            trading_config: TradingConfig::default(),
            financial_config: FinancialMetricsConfig::default(),
        }
    }
}

/// Perform comprehensive cross-validation for fractal model selection.
///
/// This is the main function for evaluating multiple fractal estimators
/// using time-series appropriate cross-validation methods. It returns
/// detailed performance comparisons and statistical significance tests.
///
/// # Arguments
/// * `data` - Time series data for analysis
/// * `config` - Cross-validation configuration parameters
///
/// # Returns
/// * `Ok(ModelSelectionResult)` - Complete model selection results
/// * `Err` - If insufficient data or configuration issues
///
/// # Example
/// ```rust
/// use financial_fractal_analysis::{cross_validate_fractal_models, CrossValidationConfig};
///
/// let data = vec![/* your time series data */];
/// let config = CrossValidationConfig::default();
///
/// let result = cross_validate_fractal_models(&data, &config).unwrap();
/// println!("Best estimator: {:?}", result.best_estimator);
/// ```
pub fn cross_validate_fractal_models(
    data: &[f64],
    config: &CrossValidationConfig,
) -> FractalResult<ModelSelectionResult> {
    validate_data_length(data, 100, "Cross-validation")?;

    // Note: The seed from config can be used in the future if random sampling
    // is needed (e.g., for bootstrap confidence intervals). Currently, this
    // implementation uses analytical methods to avoid nested bootstrap complexity.

    let mut all_results = HashMap::new();

    // Evaluate each estimator
    for estimator in &config.estimators {
        let cv_result = perform_cross_validation(data, estimator, config)?;
        all_results.insert(estimator.clone(), cv_result);
    }

    // Select best model based on criterion
    let best_estimator = select_best_model(&all_results, &config.selection_criterion)?;

    // Rank all models
    let ranking = rank_models(&all_results, &config.selection_criterion);

    // Perform significance tests between models
    let significance_tests =
        perform_pairwise_significance_tests(&all_results, config.significance_level);

    Ok(ModelSelectionResult {
        best_estimator,
        all_results,
        ranking,
        selection_criterion: config.selection_criterion.clone(),
        significance_tests,
    })
}

/// Perform cross-validation for a single fractal estimator.
fn perform_cross_validation(
    data: &[f64],
    estimator: &FractalEstimator,
    config: &CrossValidationConfig,
) -> FractalResult<CrossValidationResult> {
    let folds = generate_cv_folds_for_estimator(data.len(), &config.method, estimator)?;
    let mut fold_results = Vec::with_capacity(folds.len());
    let mut errors = Vec::with_capacity(folds.len());

    for (fold_index, (train_indices, test_indices)) in folds.iter().enumerate() {
        let train_data = extract_data_by_indices(data, train_indices);
        let test_data = extract_data_by_indices(data, test_indices);

        // Estimate on training data - skip fold if estimation fails
        let train_estimate = match estimate_with_method(&train_data, estimator) {
            Ok(est) => est,
            Err(_) => continue, // Skip this fold if training fails
        };

        // Predict/estimate on test data - skip fold if estimation fails
        let test_estimate = match estimate_with_method(&test_data, estimator) {
            Ok(est) => est,
            Err(_) => continue, // Skip this fold if testing fails
        };

        // Calculate stability drift - temporal consistency of Hurst estimates
        // This measures how stable the fractal characteristics are over time
        let stability_drift = (test_estimate - train_estimate).abs();
        errors.push(stability_drift);

        // Calculate confidence intervals from both train and test data
        let (train_ci, test_ci) = if config.bootstrap_config.num_bootstrap_samples > 0 {
            let train_ci = calculate_estimate_confidence_interval(
                &train_data,
                estimator,
                &config.bootstrap_config,
            )
            .ok();
            let test_ci = calculate_estimate_confidence_interval(
                &test_data,
                estimator,
                &config.bootstrap_config,
            )
            .ok();
            (train_ci, test_ci)
        } else {
            (None, None)
        };

        fold_results.push(FoldResult {
            fold_index,
            train_period: (train_indices[0], train_indices[train_indices.len() - 1]),
            test_period: (test_indices[0], test_indices[test_indices.len() - 1]),
            train_estimate,
            test_estimate,
            true_value: None, // Would be known for synthetic data
            error: stability_drift,
            train_confidence_interval: train_ci,
            test_confidence_interval: test_ci,
        });
    }

    // Check if we have any successful folds
    if fold_results.is_empty() {
        return Err(FractalAnalysisError::ValidationError {
            validation_type: format!("Cross-validation: all folds failed for estimator {:?}", estimator),
        });
    }

    // Calculate performance metrics
    let metrics = calculate_performance_metrics(
        &fold_results,
        data,
        &config.trading_config,
        &config.financial_config,
    )?;

    // Calculate performance statistics
    let average_performance = errors.iter().sum::<f64>() / errors.len() as f64;
    let performance_variance = if errors.len() > 1 {
        errors
            .iter()
            .map(|e| {
                let diff = e - average_performance;
                diff * diff
            })
            .sum::<f64>()
            / (errors.len() - 1) as f64
    } else {
        0.0
    };
    let performance_std = performance_variance.sqrt();

    // Bootstrap confidence interval for performance
    let performance_ci =
        calculate_bootstrap_confidence_interval(&errors, average_performance, 0.95)?;

    Ok(CrossValidationResult {
        estimator: estimator.clone(),
        metrics,
        fold_results,
        average_performance,
        performance_std,
        performance_ci,
        feature_importance: None, // Could be implemented for ensemble methods
    })
}

/// Get minimum data size required for a specific fractal estimator.
fn get_estimator_min_size(estimator: &FractalEstimator) -> usize {
    match estimator {
        FractalEstimator::DetrendedFluctuation(_) => 64,
        FractalEstimator::MultifractalDFA { .. } => 64,
        FractalEstimator::PeriodogramRegression(_) => 128,
        FractalEstimator::WaveletBased => 64,
        // WTMM requires 128 for rigorous analysis
        FractalEstimator::WaveletModulusMaxima(_) => 128,
        FractalEstimator::RescaledRange => 32,
        FractalEstimator::Ensemble { methods } => {
            // Use the maximum requirement of all methods in the ensemble
            methods
                .iter()
                .map(|m| get_estimator_min_size(m))
                .max()
                .unwrap_or(64)
        }
    }
}

/// Generate cross-validation folds with estimator-specific minimum sizes.
fn generate_cv_folds_for_estimator(
    data_length: usize,
    method: &CrossValidationMethod,
    estimator: &FractalEstimator,
) -> FractalResult<Vec<(Vec<usize>, Vec<usize>)>> {
    let mut folds = Vec::new();
    let min_test_size = get_estimator_min_size(estimator);

    match method {
        CrossValidationMethod::WalkForward {
            window_size,
            step_size,
        } => {
            // Validate that step_size meets minimum requirements upfront
            if *step_size < min_test_size {
                return Err(FractalAnalysisError::InvalidParameter {
                    parameter: "step_size".to_string(),
                    value: *step_size as f64,
                    constraint: format!(">= {} (minimum required for estimator)", min_test_size),
                });
            }

            let mut start = 0;

            // Test window size is exactly the step_size (no overlap)
            let test_window_size = *step_size;

            while start + window_size + test_window_size <= data_length {
                let train_end = start + window_size;
                let test_start = train_end;
                let test_end = test_start + test_window_size;

                // Create fold with non-overlapping test set
                let train_indices = (start..train_end).collect();
                let test_indices = (test_start..test_end).collect();
                folds.push((train_indices, test_indices));

                start += step_size;
            }
        }

        CrossValidationMethod::ExpandingWindow {
            initial_size,
            step_size,
        } => {
            // Validate that step_size meets minimum requirements upfront
            if *step_size < min_test_size {
                return Err(FractalAnalysisError::InvalidParameter {
                    parameter: "step_size".to_string(),
                    value: *step_size as f64,
                    constraint: format!(">= {} (minimum required for estimator)", min_test_size),
                });
            }

            let mut train_end = *initial_size;

            // Test window size is exactly the step_size (no overlap)
            let test_window_size = *step_size;

            while train_end + test_window_size <= data_length {
                let test_start = train_end;
                let test_end = test_start + test_window_size;

                // Create fold with non-overlapping test set
                let train_indices = (0..train_end).collect();
                let test_indices = (test_start..test_end).collect();
                folds.push((train_indices, test_indices));

                train_end += step_size;
            }
        }

        CrossValidationMethod::TimeBased {
            train_fraction,
            num_splits,
        } => {
            let split_size = data_length / num_splits;

            for i in 0..*num_splits {
                let split_start = i * split_size;
                let split_end = ((i + 1) * split_size).min(data_length);
                let split_length = split_end - split_start;

                let train_length = (split_length as f64 * train_fraction) as usize;
                let test_length = split_length - train_length;

                // Only create fold if test set meets minimum size requirement
                if test_length >= min_test_size {
                    let train_indices = (split_start..split_start + train_length).collect();
                    let test_indices: Vec<usize> =
                        (split_start + train_length..split_end).collect();
                    folds.push((train_indices, test_indices));
                }
            }
        }

        CrossValidationMethod::Purged {
            window_size,
            purge_size,
            embargo_size,
        } => {
            let total_gap = purge_size + embargo_size;
            let mut start = 0;

            while start + window_size + total_gap < data_length {
                let train_end = start + window_size;
                let test_start = train_end + total_gap;
                let test_end = (test_start + window_size).min(data_length);

                // Only create fold if test set meets minimum size requirement
                if test_end > test_start && (test_end - test_start) >= min_test_size {
                    let train_indices = (start..train_end).collect();
                    let test_indices = (test_start..test_end).collect();
                    folds.push((train_indices, test_indices));
                }

                start += window_size / 2; // Overlap by half
            }
        }
    }

    if folds.is_empty() {
        // Calculate minimum required based on actual test window size needed
        let test_window_size = min_test_size;
        let min_required = match method {
            CrossValidationMethod::WalkForward {
                window_size,
                step_size,
            } => window_size + (*step_size).max(test_window_size),
            CrossValidationMethod::ExpandingWindow {
                initial_size,
                step_size,
            } => initial_size + (*step_size).max(test_window_size),
            CrossValidationMethod::TimeBased {
                num_splits,
                train_fraction,
            } => {
                // Need enough data so that the smallest test set (1 - train_fraction)
                // has at least min_test_size samples
                let min_split_size =
                    (min_test_size as f64 / (1.0 - train_fraction)).ceil() as usize;
                // And we need num_splits such splits
                min_split_size * num_splits
            }
            CrossValidationMethod::Purged {
                window_size,
                purge_size,
                embargo_size,
            } => window_size * 2 + purge_size + embargo_size,
        };

        return Err(FractalAnalysisError::InsufficientData {
            required: min_required,
            actual: data_length,
        });
    }

    Ok(folds)
}

/// Extract data by indices with optimized memory allocation.
fn extract_data_by_indices(data: &[f64], indices: &[usize]) -> Vec<f64> {
    // Pre-allocate vector with known size
    let mut result = Vec::with_capacity(indices.len());

    // Use bounds checking to avoid panics
    for &index in indices {
        if index < data.len() {
            result.push(data[index]);
        }
    }

    result
}

/// Estimate fractal parameter using specified method.
///
/// Raw implementations without bootstrap to prevent recursion during
/// cross-validation. Each method is optimized for computational efficiency.
fn estimate_with_method(data: &[f64], estimator: &FractalEstimator) -> FractalResult<f64> {
    // VALIDATION: Check if data meets minimum requirements for the estimator
    let min_size = get_estimator_min_size(estimator);
    if data.len() < min_size {
        return Err(FractalAnalysisError::InsufficientData {
            required: min_size,
            actual: data.len(),
        });
    }

    match estimator {
        FractalEstimator::DetrendedFluctuation(config) => {
            // Raw DFA implementation without bootstrap confidence intervals
            estimate_dfa_raw_with_config(data, config)
        }

        FractalEstimator::MultifractalDFA { q_value, config } => {
            // Raw multifractal DFA implementation
            estimate_multifractal_dfa_raw_with_config(data, *q_value, config)
        }

        FractalEstimator::PeriodogramRegression(config) => {
            // Raw periodogram regression without bootstrap
            estimate_periodogram_raw_with_config(data, config)
        }

        FractalEstimator::WaveletBased => {
            // Raw wavelet estimation without bootstrap
            estimate_wavelet_raw(data)
        }

        FractalEstimator::WaveletModulusMaxima(config) => {
            // Raw WTMM estimation without bootstrap
            estimate_wtmm_raw_with_config(data, config)
        }

        FractalEstimator::RescaledRange => {
            // Raw R/S analysis without bootstrap
            estimate_rs_raw(data)
        }

        FractalEstimator::Ensemble { methods } => {
            // Prevent infinite recursion by limiting ensemble depth
            if methods.len() > 10 {
                return Err(FractalAnalysisError::NumericalError {
                    reason: "Ensemble method too complex - maximum 10 methods allowed".to_string(),
                    operation: None,
                });
            }

            // Check for nested ensembles to prevent exponential complexity
            for method in methods {
                if matches!(method, FractalEstimator::Ensemble { .. }) {
                    return Err(FractalAnalysisError::NumericalError {
                        reason: "Nested ensemble methods not allowed".to_string(),
                        operation: None,
                    });
                }
            }

            // Collect estimates with standard errors for optimal weighting
            let mut estimates_with_errors = Vec::with_capacity(methods.len());

            for method in methods {
                // Try to get estimate with standard error if available
                let result = match method {
                    FractalEstimator::PeriodogramRegression(config) => {
                        estimate_periodogram_raw_with_error_config(data, config)
                            .map(|r| (r.estimate, r.std_error, method))
                            .ok()
                    }
                    FractalEstimator::WaveletBased => estimate_wavelet_raw_with_error(data)
                        .map(|r| (r.estimate, r.std_error, method))
                        .ok(),
                    FractalEstimator::WaveletModulusMaxima(config) => {
                        estimate_wtmm_raw_with_error_config(data, config)
                            .map(|r| (r.estimate, r.std_error, method))
                            .ok()
                    }
                    _ => {
                        // For methods without analytical standard errors, use bootstrap
                        // to estimate variance for inverse-variance weighting
                        if let Ok(estimate) = estimate_with_method(data, method) {
                            // Quick bootstrap with fewer samples for performance
                            let bootstrap_std_error =
                                estimate_bootstrap_std_error(data, method, 100).ok();
                            Some((estimate, bootstrap_std_error, method))
                        } else {
                            None
                        }
                    }
                };

                if let Some((estimate, std_error, method)) = result {
                    // Validate estimate is reasonable (Hurst must be in [0,1])
                    if estimate >= 0.0 && estimate <= 1.0 {
                        estimates_with_errors.push((estimate, std_error));
                    } else {
                        warn!(
                            "Ensemble sub-estimator {:?} produced out-of-range Hurst estimate: {}",
                            method, estimate
                        );
                    }
                } else {
                    warn!(
                        "Ensemble sub-estimator {:?} failed on data segment of length {}",
                        method,
                        data.len()
                    );
                }
            }

            if estimates_with_errors.is_empty() {
                warn!(
                    "All {} ensemble methods failed for data of length {}",
                    methods.len(),
                    data.len()
                );
                return Err(FractalAnalysisError::NumericalError {
                    reason: "No valid estimates from ensemble methods".to_string(),
                    operation: None,
                });
            }

            // Use inverse-variance weighting for all methods
            // Methods with analytical SE use those, others use bootstrap SE
            let weighted_estimate = if estimates_with_errors.iter().any(|(_, se)| se.is_some()) {
                // Inverse-variance weighting
                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;

                // Calculate median standard error for methods without SE
                let ses_with_values: Vec<f64> = estimates_with_errors
                    .iter()
                    .filter_map(|(_, se)| *se)
                    .collect();
                let median_se = if !ses_with_values.is_empty() {
                    let mut sorted_ses = ses_with_values.clone();
                    sorted_ses.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    sorted_ses[sorted_ses.len() / 2]
                } else {
                    0.05 // Default standard error if none available
                };

                for (estimate, std_error) in &estimates_with_errors {
                    let se = std_error.unwrap_or(median_se);
                    // Inverse variance weight (1/se^2), with minimum SE to avoid extreme weights
                    let weight = 1.0 / (se.max(0.001).powi(2));

                    weighted_sum += estimate * weight;
                    weight_sum += weight;
                }

                if weight_sum > 0.0 {
                    weighted_sum / weight_sum
                } else {
                    // Fallback to simple mean
                    estimates_with_errors.iter().map(|(e, _)| *e).sum::<f64>()
                        / estimates_with_errors.len() as f64
                }
            } else {
                // If no standard errors at all, use trimmed mean for robustness
                let mut estimates: Vec<f64> =
                    estimates_with_errors.iter().map(|(e, _)| *e).collect();

                estimates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let trimmed_estimates = if estimates.len() > 4 {
                    // Remove top and bottom 20% for robustness
                    let trim_count = estimates.len() / 5;
                    &estimates[trim_count..estimates.len() - trim_count]
                } else {
                    &estimates[..]
                };

                trimmed_estimates.iter().sum::<f64>() / trimmed_estimates.len() as f64
            };

            Ok(weighted_estimate)
        }
    }
}

/// Estimate standard error using bootstrap for methods without analytical SE.
///
/// This function performs a quick bootstrap estimation with fewer samples
/// for performance in ensemble methods.
fn estimate_bootstrap_std_error(
    data: &[f64],
    estimator: &FractalEstimator,
    num_samples: usize,
) -> FractalResult<f64> {
    use crate::secure_rng::SecureRng;

    if data.len() < 10 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 10,
            actual: data.len(),
        });
    }

    let mut rng = SecureRng::new();
    let mut bootstrap_estimates = Vec::with_capacity(num_samples);
    let n = data.len();

    // Perform bootstrap resampling
    for _ in 0..num_samples {
        let mut resampled_data = Vec::with_capacity(n);

        // Resample with replacement
        for _ in 0..n {
            let idx = rng.usize(0..n);
            resampled_data.push(data[idx]);
        }

        // Calculate estimate on resampled data
        if let Ok(estimate) = estimate_with_method(&resampled_data, estimator) {
            bootstrap_estimates.push(estimate);
        }
    }

    if bootstrap_estimates.len() < num_samples / 2 {
        // Too many failures, can't reliably estimate standard error
        return Err(FractalAnalysisError::BootstrapError {
            reason: "Too many bootstrap samples failed".to_string(),
        });
    }

    // Calculate standard error from bootstrap distribution
    let mean = bootstrap_estimates.iter().sum::<f64>() / bootstrap_estimates.len() as f64;
    let variance = bootstrap_estimates
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / (bootstrap_estimates.len() - 1) as f64;

    Ok(variance.sqrt())
}

/// Raw DFA Hurst exponent estimation optimized for cross-validation.
fn estimate_dfa_raw(data: &[f64]) -> FractalResult<f64> {
    estimate_dfa_raw_with_config(data, &DfaConfig::default())
}

/// Raw DFA Hurst exponent estimation with custom configuration.
fn estimate_dfa_raw_with_config(data: &[f64], config: &DfaConfig) -> FractalResult<f64> {
    validate_data_length(data, 64, "DFA estimation")?;

    let n = data.len();

    // Integrate the series (Profile calculation)
    let mean = data.iter().sum::<f64>() / n as f64;
    let mut profile = Vec::with_capacity(n + 1);
    profile.push(0.0);

    let mut cumsum = 0.0;
    for &value in data {
        cumsum += value - mean;
        profile.push(cumsum);
    }

    // Generate scale range using configuration parameters
    let min_scale = config.min_scale;
    let max_scale = ((n as f64 * config.max_scale_ratio) as usize).max(min_scale * 4);

    // OPTIMIZATION: Pre-calculate scales capacity for better memory allocation
    let estimated_scales_count =
        ((max_scale as f64 / min_scale as f64).ln() / 1.2_f64.ln()).ceil() as usize;
    let scales_capacity = estimated_scales_count.min(50);
    let mut scales = Vec::with_capacity(scales_capacity);

    // Logarithmically spaced scales for better statistical properties
    let mut scale = min_scale;
    while scale <= max_scale {
        scales.push(scale);
        scale = ((scale as f64) * 1.2).round() as usize;
        if scales.len() > 50 {
            break;
        }
    }

    let mut log_scales = Vec::with_capacity(scales.len());
    let mut log_fluctuations = Vec::with_capacity(scales.len());

    // OPTIMIZATION: Enhanced parallel processing for scale calculations
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        // Parallel computation across scales for maximum performance
        let scale_results: Vec<_> = scales
            .par_iter()
            .map(|&scale| {
                let num_segments = n / scale;
                let fluctuations: Vec<f64> = (0..num_segments)
                    .into_par_iter()
                    .filter_map(|seg| {
                        let start = seg * scale;
                        let end = start + scale;

                        if end > profile.len() - 1 {
                            return None;
                        }

                        // Fit polynomial trend (linear for robust estimation)
                        let x_vals: Vec<f64> = (0..scale).map(|i| i as f64).collect();
                        let y_vals: Vec<f64> = profile[start..end].iter().copied().collect();

                        if let Ok((_, _, residuals)) = ols_regression(&x_vals, &y_vals) {
                            let variance = residuals.iter().map(|r| r * r).sum::<f64>()
                                / residuals.len() as f64;
                            if variance >= 0.0 && !variance.is_nan() {
                                Some(variance.max(1e-15).sqrt())
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .collect();

                if !fluctuations.is_empty() {
                    let avg_fluctuation =
                        fluctuations.iter().sum::<f64>() / fluctuations.len() as f64;
                    if avg_fluctuation > 0.0 && avg_fluctuation.is_finite() {
                        Some((scale as f64, avg_fluctuation))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .filter_map(|x| x)
            .collect();

        for (scale, fluctuation) in scale_results {
            log_scales.push(scale.ln());
            log_fluctuations.push(fluctuation.ln());
        }
    }

    #[cfg(not(feature = "parallel"))]
    for &scale in &scales {
        let num_segments = n / scale;
        let mut fluctuations = Vec::with_capacity(num_segments);

        // Calculate fluctuations for each segment
        for seg in 0..num_segments {
            let start = seg * scale;
            let end = start + scale;

            if end > profile.len() - 1 {
                break;
            }

            // Fit polynomial trend (linear for robust estimation)
            let x_vals: Vec<f64> = (0..scale).map(|i| i as f64).collect();
            let y_vals: Vec<f64> = profile[start..end].iter().copied().collect();

            if let Ok((_, _, residuals)) = ols_regression(&x_vals, &y_vals) {
                let variance =
                    residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64;
                if variance >= 0.0 && !variance.is_nan() {
                    fluctuations.push(variance.max(1e-15).sqrt());
                }
            }
        }

        if !fluctuations.is_empty() {
            let avg_fluctuation = fluctuations.iter().sum::<f64>() / fluctuations.len() as f64;
            if avg_fluctuation >= 0.0 && !avg_fluctuation.is_nan() {
                log_scales.push((scale as f64).ln());
                log_fluctuations.push(avg_fluctuation.max(1e-15).ln());
            }
        }
    }

    if log_scales.len() < 3 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 3,
            actual: log_scales.len(),
        });
    }

    // Linear regression: log(F) = log(a) + α * log(n), where α = H
    let (hurst_estimate, _, _) = ols_regression(&log_scales, &log_fluctuations)?;

    // Apply bounds clamping with warning
    let clamped_estimate = if hurst_estimate < 0.01 || hurst_estimate > 0.99 {
        warn!(
            "DFA estimator produced out-of-range Hurst estimate: {:.4}, clamping to [0.01, 0.99]",
            hurst_estimate
        );
        hurst_estimate.max(0.01).min(0.99)
    } else {
        hurst_estimate
    };
    Ok(clamped_estimate)
}

/// Raw multifractal DFA estimation for q-order moments.
fn estimate_multifractal_dfa_raw(data: &[f64], q: f64) -> FractalResult<f64> {
    estimate_multifractal_dfa_raw_with_config(data, q, &DfaConfig::default())
}

/// Raw multifractal DFA estimation with custom configuration.
fn estimate_multifractal_dfa_raw_with_config(
    data: &[f64],
    q: f64,
    config: &DfaConfig,
) -> FractalResult<f64> {
    validate_data_length(data, 64, "Multifractal DFA estimation")?;

    if q.abs() < 1e-10 {
        return estimate_dfa_raw_with_config(data, config); // q=0 case defaults to standard DFA
    }

    let n = data.len();

    // Profile calculation
    let mean = data.iter().sum::<f64>() / n as f64;
    let mut profile = Vec::with_capacity(n + 1);
    profile.push(0.0);

    let mut cumsum = 0.0;
    for &value in data {
        cumsum += value - mean;
        profile.push(cumsum);
    }

    // Scale range using configuration parameters
    let min_scale = config.min_scale;
    let max_scale = ((n as f64 * config.max_scale_ratio) as usize).max(min_scale * 4);
    let mut scales = Vec::new();

    let mut scale = min_scale;
    while scale <= max_scale {
        scales.push(scale);
        scale = ((scale as f64) * 1.2).round() as usize;
        if scales.len() > 50 {
            break;
        }
    }

    let mut log_scales = Vec::new();
    let mut log_q_fluctuations = Vec::new();

    for &scale in &scales {
        let mut fluctuations = Vec::new();

        let num_segments = n / scale;
        for seg in 0..num_segments {
            let start = seg * scale;
            let end = start + scale;

            if end > profile.len() - 1 {
                break;
            }

            let x_vals: Vec<f64> = (0..scale).map(|i| i as f64).collect();
            let y_vals: Vec<f64> = profile[start..end].iter().copied().collect();

            if let Ok((_, _, residuals)) = ols_regression(&x_vals, &y_vals) {
                let variance =
                    residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64;
                if variance >= 0.0 && !variance.is_nan() {
                    fluctuations.push(variance.max(1e-15).sqrt());
                }
            }
        }

        if !fluctuations.is_empty() {
            // Calculate q-order moment
            let q_moment = if q.abs() < 1e-10 {
                // q=0 case: geometric mean
                let log_sum = fluctuations.iter().map(|f| f.ln()).sum::<f64>();
                (log_sum / fluctuations.len() as f64).exp()
            } else {
                // q≠0 case: generalized mean
                let sum_q = fluctuations.iter().map(|f| f.powf(q)).sum::<f64>();
                (sum_q / fluctuations.len() as f64).powf(1.0 / q)
            };

            if q_moment >= 0.0 && q_moment.is_finite() {
                log_scales.push((scale as f64).ln());
                log_q_fluctuations.push(q_moment.max(1e-15).ln());
            }
        }
    }

    if log_scales.len() < 3 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 3,
            actual: log_scales.len(),
        });
    }

    let (hurst_q, _, _) = ols_regression(&log_scales, &log_q_fluctuations)?;

    // Apply bounds clamping with warning
    let clamped_estimate = if hurst_q < 0.01 || hurst_q > 0.99 {
        warn!(
            "Multifractal DFA (q={}) produced out-of-range Hurst estimate: {:.4}, clamping to [0.01, 0.99]",
            q, hurst_q
        );
        hurst_q.max(0.01).min(0.99)
    } else {
        hurst_q
    };
    Ok(clamped_estimate)
}

/// Result from a raw estimation method including standard error
struct EstimateWithError {
    estimate: f64,
    std_error: Option<f64>,
}

/// Raw periodogram regression (GPH) Hurst estimation.
fn estimate_periodogram_raw(data: &[f64]) -> FractalResult<f64> {
    estimate_periodogram_raw_with_config(data, &GphConfig::default())
}

/// Raw periodogram regression with custom configuration.
fn estimate_periodogram_raw_with_config(data: &[f64], config: &GphConfig) -> FractalResult<f64> {
    estimate_periodogram_raw_with_error_config(data, config).map(|r| r.estimate)
}

/// Raw periodogram regression (GPH) with standard error.
fn estimate_periodogram_raw_with_error(data: &[f64]) -> FractalResult<EstimateWithError> {
    estimate_periodogram_raw_with_error_config(data, &GphConfig::default())
}

/// Raw periodogram regression with standard error and custom configuration.
fn estimate_periodogram_raw_with_error_config(
    data: &[f64],
    config: &GphConfig,
) -> FractalResult<EstimateWithError> {
    validate_data_length(data, 128, "Periodogram regression")?;

    let n = data.len();

    // Calculate FFT periodogram
    let periodogram = calculate_periodogram_fft(data)?;

    // Use frequencies as per configuration
    let max_freq_idx = ((n as f64).powf(config.frequency_power) as usize)
        .max(config.min_frequency)
        .min(periodogram.len() - 1);

    let mut log_frequencies = Vec::new();
    let mut log_periodogram = Vec::new();

    for k in 1..=max_freq_idx {
        if k < periodogram.len() && periodogram[k] >= 0.0 && !periodogram[k].is_nan() {
            let frequency = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            log_frequencies.push(frequency.ln());
            log_periodogram.push(periodogram[k].max(1e-15).ln());
        }
    }

    if log_frequencies.len() < 10 {
        return Err(FractalAnalysisError::StatisticalTestError {
            test_name: "Insufficient frequencies for GPH estimation".to_string(),
        });
    }

    // GPH regression: log(I(λ)) = const - 2d*log(λ) + error
    // where d = H - 0.5 (fractional integration parameter)
    let (slope, std_error, _) = ols_regression(&log_frequencies, &log_periodogram)?;
    let d_estimate = -slope;
    let hurst_estimate = d_estimate + 0.5;

    // The standard error of d equals the standard error of the slope
    let hurst_std_error = std_error;

    // Apply bounds clamping with warning
    let clamped_estimate = if hurst_estimate < 0.01 || hurst_estimate > 0.99 {
        warn!(
            "GPH estimator produced out-of-range Hurst estimate: {:.4}, clamping to [0.01, 0.99]",
            hurst_estimate
        );
        hurst_estimate.max(0.01).min(0.99)
    } else {
        hurst_estimate
    };

    Ok(EstimateWithError {
        estimate: clamped_estimate,
        std_error: Some(hurst_std_error),
    })
}

/// Raw Rescaled Range (R/S) analysis.
fn estimate_rs_raw(data: &[f64]) -> FractalResult<f64> {
    validate_data_length(data, 32, "R/S analysis")?;

    let n = data.len();

    // Calculate cumulative deviations
    let mean = data.iter().sum::<f64>() / n as f64;
    let mut cumulative_deviations = Vec::with_capacity(n);
    let mut cumsum = 0.0;

    for &value in data {
        cumsum += value - mean;
        cumulative_deviations.push(cumsum);
    }

    // Window sizes for R/S analysis: logarithmically spaced
    let min_window = 8; // Same minimum as DFA default
    let max_window = ((n as f64 * 0.25) as usize).max(min_window * 4); // 25% of data length
    let mut window_sizes = Vec::new();

    let mut window = min_window;
    while window <= max_window {
        window_sizes.push(window);
        window = ((window as f64) * 1.3).round() as usize;
        if window_sizes.len() > 50 {
            break;
        }
    }

    let mut log_windows = Vec::new();
    let mut log_rs_values = Vec::new();

    for &window_size in &window_sizes {
        let num_windows = n / window_size;
        if num_windows < 2 {
            continue;
        }

        let mut rs_values = Vec::new();

        for w in 0..num_windows {
            let start = w * window_size;
            let end = start + window_size;

            if end > n {
                break;
            }

            // Calculate range in this window
            let window_cumdev = &cumulative_deviations[start..end];
            let min_cumdev = window_cumdev.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_cumdev = window_cumdev
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let range = max_cumdev - min_cumdev;

            // Calculate standard deviation for this window
            let window_data = &data[start..end];
            let window_variance = calculate_variance(window_data);
            let window_std = window_variance.sqrt();

            // Financial sector practice: avoid division by very small numbers
            if window_std > 1e-12 && range >= 0.0 {
                let rs_ratio = range / window_std;
                if rs_ratio >= 0.0 && rs_ratio.is_finite() {
                    rs_values.push(rs_ratio);
                }
            }
        }

        if !rs_values.is_empty() {
            let avg_rs = rs_values.iter().sum::<f64>() / rs_values.len() as f64;
            if avg_rs >= 0.0 && !avg_rs.is_nan() {
                log_windows.push((window_size as f64).ln());
                log_rs_values.push(avg_rs.max(1e-15).ln());
            }
        }
    }

    if log_windows.len() < 3 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 3,
            actual: log_windows.len(),
        });
    }

    // Linear regression: log(R/S) = log(c) + H * log(n)
    let (hurst_estimate, _std_error, _) = ols_regression(&log_windows, &log_rs_values)?;

    // Apply Lo's bias correction for financial data
    let bias_correction = 0.5 / (n as f64).ln();
    let corrected_hurst = hurst_estimate - bias_correction;

    // Apply bounds clamping with warning
    let clamped_estimate = if corrected_hurst < 0.01 || corrected_hurst > 0.99 {
        warn!(
            "R/S analysis produced out-of-range Hurst estimate: {:.4}, clamping to [0.01, 0.99]",
            corrected_hurst
        );
        corrected_hurst.max(0.01).min(0.99)
    } else {
        corrected_hurst
    };
    Ok(clamped_estimate)
}

/// Raw wavelet-based Hurst estimation.
fn estimate_wavelet_raw(data: &[f64]) -> FractalResult<f64> {
    estimate_wavelet_raw_with_error(data).map(|r| r.estimate)
}

/// Raw wavelet-based Hurst estimation with standard error.
fn estimate_wavelet_raw_with_error(data: &[f64]) -> FractalResult<EstimateWithError> {
    validate_data_length(data, 64, "Wavelet estimation")?;

    let n = data.len();
    let mut scale_variances = Vec::new();
    let mut scales = Vec::new();

    // Use dyadic scales appropriate for financial time series
    let mut scale = 2;
    let max_scale = (n / 8).max(32);
    while scale < max_scale {
        // Calculate wavelet variance at this scale using differences
        let mut variance_sum = 0.0;
        let mut count = 0;

        for i in 0..(n - scale) {
            let diff = data[i + scale] - data[i];
            variance_sum += diff * diff;
            count += 1;
        }

        if count > 10 {
            // Ensure sufficient data points
            let variance = variance_sum / count as f64;
            if variance >= 0.0 && variance.is_finite() {
                scale_variances.push(variance.max(1e-15).ln());
                scales.push((scale as f64).ln());
            }
        }
        scale *= 2;
        if scales.len() > 20 {
            break;
        }
    }

    if scales.len() < 3 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 3,
            actual: scales.len(),
        });
    }

    // Linear regression: log(Var) = log(c) + (2H-1) * log(scale)
    let (slope, std_error, _) = ols_regression(&scales, &scale_variances)?;
    let hurst_estimate = (slope + 1.0) / 2.0;
    let hurst_std_error = std_error / 2.0;

    // Apply bounds clamping with warning
    let clamped_estimate = if hurst_estimate < 0.01 || hurst_estimate > 0.99 {
        warn!(
            "Wavelet estimator produced out-of-range Hurst estimate: {:.4}, clamping to [0.01, 0.99]",
            hurst_estimate
        );
        hurst_estimate.max(0.01).min(0.99)
    } else {
        hurst_estimate
    };

    Ok(EstimateWithError {
        estimate: clamped_estimate,
        std_error: Some(hurst_std_error),
    })
}

/// Raw WTMM Hurst estimation.
fn estimate_wtmm_raw(data: &[f64]) -> FractalResult<f64> {
    estimate_wtmm_raw_with_config(data, &WtmmEstimatorConfig::default())
}

/// Raw WTMM Hurst estimation with custom configuration.
fn estimate_wtmm_raw_with_config(data: &[f64], config: &WtmmEstimatorConfig) -> FractalResult<f64> {
    estimate_wtmm_raw_with_error_config(data, config).map(|r| r.estimate)
}

/// Raw WTMM Hurst estimation with standard error.
///
/// # Standard Error Estimation
/// This function uses a heuristic approach to estimate standard error from the
/// variance of generalized dimensions D(q). This is an approximation method where:
/// - Higher variance in D(q) indicates greater uncertainty in the Hurst estimate
/// - The empirical scaling factor (0.1) was determined through validation studies
/// - This provides a reasonable but approximate confidence interval
///
/// For more precise standard errors, consider using bootstrap methods or
/// Monte Carlo simulations with the full WTMM analysis.
fn estimate_wtmm_raw_with_error(data: &[f64]) -> FractalResult<EstimateWithError> {
    estimate_wtmm_raw_with_error_config(data, &WtmmEstimatorConfig::default())
}

/// Raw WTMM Hurst estimation with standard error and custom configuration.
fn estimate_wtmm_raw_with_error_config(
    data: &[f64],
    estimator_config: &WtmmEstimatorConfig,
) -> FractalResult<EstimateWithError> {
    validate_data_length(data, 128, "WTMM estimation")?;

    // Configure WTMM for Hurst estimation
    // Use q=2 for standard Hurst exponent (corresponds to variance scaling)
    let config = WtmmConfig {
        q_range: (1.0, 3.0), // Focus around q=2 for Hurst
        num_q_values: 5,     // Just need a few q values centered on 2
        min_scale: estimator_config.min_scale,
        max_scale: (data.len() as f64 * estimator_config.max_scale_ratio).min(128.0),
        num_scales: 30,
        min_maxima_lines: 5, // Relax requirement for cross-validation
        embedding_dim: 1.0,  // 1D time series
    };

    // Perform WTMM analysis
    let wtmm_result = perform_wtmm_analysis_with_config(data, &config)?;

    // Extract Hurst exponent from τ(q=2)
    // We're analyzing FGN (increments), not FBM
    // For FGN: τ(2) = 2*H - 2, so H = (τ(2) + 2) / 2
    let tau_2 = wtmm_result
        .scaling_exponents
        .iter()
        .min_by_key(|(q, _)| ((q - 2.0).abs() * 1000.0) as i64)
        .map(|(_, tau)| *tau)
        .ok_or_else(|| FractalAnalysisError::NumericalError {
            reason: "Could not find τ(2) in WTMM results".to_string(),
            operation: None,
        })?;

    // Use FGN formula since we're analyzing increments
    let hurst_estimate = (tau_2 + 2.0) / 2.0;

    // Estimate standard error from the spread of generalized dimensions
    // More variation in D(q) indicates higher uncertainty
    let d_values: Vec<f64> = wtmm_result
        .generalized_dimensions
        .iter()
        .map(|(_, d)| *d)
        .collect();

    let std_error = if d_values.len() >= 3 {
        // Use variance of D(q) values as proxy for uncertainty
        let mean_d = d_values.iter().sum::<f64>() / d_values.len() as f64;
        let variance_d = d_values.iter().map(|d| (d - mean_d).powi(2)).sum::<f64>()
            / (d_values.len() - 1) as f64;

        // Scale by typical Hurst standard error relationship
        // std_error_scale is an empirical factor validated through studies
        let base_se = variance_d.sqrt() * estimator_config.std_error_scale;
        let n_factor = 1.0 / (data.len() as f64).sqrt();

        Some((base_se + n_factor).max(0.01).min(0.1))
    } else {
        // Fallback to theoretical standard error
        Some((1.0 / (data.len() as f64).sqrt()).max(0.01).min(0.1))
    };

    // Apply bounds clamping with warning
    let clamped_hurst = if hurst_estimate < 0.01 || hurst_estimate > 0.99 {
        warn!(
            "WTMM estimator produced out-of-range Hurst estimate: {:.4}, clamping to [0.01, 0.99]",
            hurst_estimate
        );
        hurst_estimate.max(0.01).min(0.99)
    } else {
        hurst_estimate
    };

    Ok(EstimateWithError {
        estimate: clamped_hurst,
        std_error,
    })
}

/// Calculate confidence interval for estimate during cross-validation.
///
/// Uses jackknife or analytical standard errors instead of bootstrap.
fn calculate_estimate_confidence_interval(
    data: &[f64],
    estimator: &FractalEstimator,
    _bootstrap_config: &BootstrapConfiguration,
) -> FractalResult<ConfidenceInterval> {
    let n = data.len();
    if n < 10 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 10,
            actual: n,
        });
    }

    // For methods with analytical standard errors, use them directly
    match estimator {
        FractalEstimator::PeriodogramRegression(config) => {
            // GPH has analytical standard error from regression
            let result = estimate_periodogram_raw_with_error_config(data, config)?;
            if let Some(std_error) = result.std_error {
                let z_95 = 1.96;
                let margin = z_95 * std_error;
                return Ok(ConfidenceInterval {
                    confidence_level: 0.95,
                    lower_bound: (result.estimate - margin).max(0.01),
                    upper_bound: (result.estimate + margin).min(0.99),
                    method: crate::bootstrap::ConfidenceIntervalMethod::Normal,
                });
            }
        }
        FractalEstimator::WaveletBased => {
            // Wavelet method has analytical standard error from regression
            let result = estimate_wavelet_raw_with_error(data)?;
            if let Some(std_error) = result.std_error {
                let z_95 = 1.96;
                let margin = z_95 * std_error;
                return Ok(ConfidenceInterval {
                    confidence_level: 0.95,
                    lower_bound: (result.estimate - margin).max(0.01),
                    upper_bound: (result.estimate + margin).min(0.99),
                    method: crate::bootstrap::ConfidenceIntervalMethod::Normal,
                });
            }
        }
        FractalEstimator::WaveletModulusMaxima(config) => {
            // WTMM has analytical standard error from generalized dimensions
            let result = estimate_wtmm_raw_with_error_config(data, config)?;
            if let Some(std_error) = result.std_error {
                let z_95 = 1.96;
                let margin = z_95 * std_error;
                return Ok(ConfidenceInterval {
                    confidence_level: 0.95,
                    lower_bound: (result.estimate - margin).max(0.01),
                    upper_bound: (result.estimate + margin).min(0.99),
                    method: crate::bootstrap::ConfidenceIntervalMethod::Normal,
                });
            }
        }
        _ => {
            // For other methods, use jackknife
        }
    }

    let full_estimate = estimate_with_method(data, estimator)?;

    // Jackknife: Leave-one-out estimates
    let mut jackknife_estimates = Vec::with_capacity(n);
    for i in 0..n.min(100) {
        // Limit to 100 for performance
        let mut jack_data = Vec::with_capacity(n - 1);
        jack_data.extend_from_slice(&data[..i]);
        jack_data.extend_from_slice(&data[i + 1..]);

        if let Ok(estimate) = estimate_with_method(&jack_data, estimator) {
            jackknife_estimates.push(estimate);
        }
    }

    if jackknife_estimates.len() < 5 {
        // Fallback to theoretical standard error for Hurst estimates
        // Based on theoretical analysis: SE(H) ≈ 1/sqrt(n) for large n
        let theoretical_se = (1.0 / (n as f64).sqrt()).max(0.01).min(0.1);
        let z_95 = 1.96;
        let margin = z_95 * theoretical_se;

        return Ok(ConfidenceInterval {
            confidence_level: 0.95,
            lower_bound: (full_estimate - margin).max(0.01),
            upper_bound: (full_estimate + margin).min(0.99),
            method: crate::bootstrap::ConfidenceIntervalMethod::Normal,
        });
    }

    // Calculate jackknife standard error
    let jack_mean = jackknife_estimates.iter().sum::<f64>() / jackknife_estimates.len() as f64;
    let jack_variance = jackknife_estimates
        .iter()
        .map(|&x| (x - jack_mean).powi(2))
        .sum::<f64>()
        / (jackknife_estimates.len() - 1) as f64;

    let jackknife_se = ((n as f64 - 1.0) / n as f64 * jack_variance).sqrt();

    // Use t-distribution for small samples
    let df = (n - 1) as f64;
    let t_critical = if df < 30.0 { 2.0 } else { 1.96 }; // Approximation
    let margin = t_critical * jackknife_se;

    // Apply jackknife bias correction
    let bias = (n as f64 - 1.0) * (jack_mean - full_estimate);
    let bias_corrected_estimate = full_estimate - bias;
    
    // Clamp the bias-corrected estimate to valid range BEFORE computing intervals
    let clamped_estimate = bias_corrected_estimate.max(0.01).min(0.99);
    
    // Compute bounds and ensure they're properly ordered
    let raw_lower = clamped_estimate - margin;
    let raw_upper = clamped_estimate + margin;
    
    // Clamp bounds to valid range and ensure lower < upper
    let lower_bound = raw_lower.max(0.01).min(0.98);
    let upper_bound = raw_upper.max(lower_bound + 0.01).min(0.99);

    Ok(ConfidenceInterval {
        confidence_level: 0.95,
        lower_bound,
        upper_bound,
        method: crate::bootstrap::ConfidenceIntervalMethod::Normal,
    })
}

/// Calculate comprehensive performance metrics from fold results.
///
/// # Arguments
/// * `fold_results` - Results from each cross-validation fold
/// * `data` - Time series data (prices or returns based on config)
/// * `trading_config` - Trading strategy configuration
/// * `financial_config` - Financial metrics configuration
///
/// # Returns
/// Performance metrics including MAE, MSE, Sharpe ratio, and maximum drawdown.
///
/// # Note
/// The P&L calculation performs event-driven backtesting using Hurst-based trading signals.
/// Trading decisions are based on the configured thresholds.
fn calculate_performance_metrics(
    fold_results: &[FoldResult],
    data: &[f64],
    trading_config: &TradingConfig,
    financial_config: &FinancialMetricsConfig,
) -> FractalResult<PerformanceMetrics> {
    if fold_results.is_empty() {
        return Err(FractalAnalysisError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    let errors: Vec<f64> = fold_results.iter().map(|f| f.error.abs()).collect();
    let squared_errors: Vec<f64> = errors.iter().map(|e| e.powi(2)).collect();

    let mae = errors.iter().sum::<f64>() / errors.len() as f64;
    let mse = squared_errors.iter().sum::<f64>() / squared_errors.len() as f64;
    let rmse = mse.sqrt();

    // MAPE calculation (simplified)
    let mape = errors
        .iter()
        .zip(fold_results.iter())
        .map(|(error, fold)| {
            if fold.train_estimate.abs() > 1e-8 {
                error / fold.train_estimate.abs()
            } else {
                0.0
            }
        })
        .sum::<f64>()
        / errors.len() as f64;

    // FINANCIAL-GRADE METRICS: Proper implementation for production use

    // Estimate stability concordance: How consistently the estimator behaves across time
    // This measures whether changes in Hurst estimates are consistent between train and test
    // NOT a measure of financial prediction accuracy - just estimator stability
    let estimate_stability_concordance = if fold_results.len() > 1 {
        let mut concordant_changes = 0;
        for i in 1..fold_results.len() {
            let test_change = fold_results[i].test_estimate - fold_results[i - 1].test_estimate;
            let train_change = fold_results[i].train_estimate - fold_results[i - 1].train_estimate;
            // Check if changes have same direction (both increase or both decrease)
            if test_change * train_change > 0.0 {
                concordant_changes += 1;
            }
        }
        concordant_changes as f64 / (fold_results.len() - 1) as f64
    } else {
        0.5 // No direction information with single fold
    };

    // Hit rate: Percentage of test estimates within training confidence bounds
    // This measures predictive accuracy - can the model trained on past data
    // provide valid confidence intervals for future estimates?
    let hit_rate = if !fold_results.is_empty() {
        let within_bounds = fold_results
            .iter()
            .filter(|f| {
                if let Some(train_ci) = &f.train_confidence_interval {
                    // Check if test estimate falls within training CI
                    // This validates the model's predictive capability
                    f.test_estimate >= train_ci.lower_bound
                        && f.test_estimate <= train_ci.upper_bound
                } else {
                    false
                }
            })
            .count();
        within_bounds as f64 / fold_results.len() as f64
    } else {
        0.0
    };

    // EVENT-DRIVEN BACKTESTING: Calculate real P&L from trading signals based on Hurst estimates
    let mut pnl_series = Vec::new();

    for (fold_idx, fold) in fold_results.iter().enumerate() {
        // Generate trading signal based on training set Hurst estimate
        let signal = if fold.train_estimate > 0.5 + trading_config.hurst_trading_threshold {
            1.0 // Long position (expect persistence)
        } else if fold.train_estimate < 0.5 - trading_config.hurst_trading_threshold {
            -1.0 // Short position (expect mean reversion)
        } else {
            0.0 // No position (neutral zone)
        };

        // Calculate actual returns from test period data
        if fold.test_period.1 < data.len() && fold.test_period.0 < fold.test_period.1 {
            let test_data = &data[fold.test_period.0..=fold.test_period.1];

            // Calculate period return based on data type
            if test_data.len() >= 2 {
                let period_return = match financial_config.data_type {
                    DataType::LogReturns => {
                        // For log returns: sum them to get cumulative return
                        test_data.iter().sum::<f64>()
                    }
                    DataType::SimpleReturns => {
                        // For simple returns: compound them
                        let mut cumulative = 1.0;
                        for &ret in test_data {
                            cumulative *= 1.0 + ret;
                        }
                        cumulative - 1.0
                    }
                    DataType::Prices => {
                        // For prices: calculate return from first to last
                        let first = test_data[0];
                        let last = test_data[test_data.len() - 1];
                        if first.abs() > 1e-10 {
                            (last - first) / first
                        } else {
                            0.0
                        }
                    }
                };

                // Apply position to get P&L
                let raw_pnl = signal * period_return;

                // Apply transaction costs for position changes
                let transaction_cost_applied = if fold_idx > 0 {
                    let prev_signal = if fold_results[fold_idx - 1].train_estimate
                        > 0.5 + trading_config.hurst_trading_threshold
                    {
                        1.0
                    } else if fold_results[fold_idx - 1].train_estimate
                        < 0.5 - trading_config.hurst_trading_threshold
                    {
                        -1.0
                    } else {
                        0.0
                    };

                    // Cost is proportional to position change
                    let position_change = (signal - prev_signal).abs();
                    position_change * trading_config.transaction_cost_rate
                } else {
                    // Initial position cost
                    signal.abs() * trading_config.transaction_cost_rate
                };

                pnl_series.push(raw_pnl - transaction_cost_applied);
            }
        }
    }

    // Calculate Sharpe ratio from P&L series
    let sharpe_ratio = if pnl_series.len() > 1 {
        let mean_return = pnl_series.iter().sum::<f64>() / pnl_series.len() as f64;
        let return_variance = pnl_series
            .iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>()
            / (pnl_series.len() - 1) as f64;
        let return_std = return_variance.sqrt();

        if return_std > 1e-10 {
            // Annualized Sharpe ratio using configured annual periods
            let annualization_factor = financial_config.annual_periods.sqrt();
            let excess_return =
                mean_return - financial_config.risk_free_rate / financial_config.annual_periods;
            excess_return / return_std * annualization_factor
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Maximum drawdown from P&L series
    let max_drawdown = if !pnl_series.is_empty() {
        let mut cumulative_pnl = Vec::with_capacity(pnl_series.len());
        let mut cum_sum = 0.0;
        for &ret in &pnl_series {
            cum_sum += ret;
            cumulative_pnl.push(cum_sum);
        }

        let mut max_dd = 0.0;
        let mut peak = cumulative_pnl[0];

        for &value in &cumulative_pnl {
            if value > peak {
                peak = value;
            }
            let drawdown = (peak - value) / peak.max(0.01); // Percentage drawdown
            if drawdown > max_dd {
                max_dd = drawdown;
            }
        }

        max_dd // Maximum drawdown is already a percentage, no normalization needed
    } else {
        0.0
    };

    // Stability measure (variance of estimates)
    let estimates: Vec<f64> = fold_results.iter().map(|f| f.test_estimate).collect();
    let mean_estimate = estimates.iter().sum::<f64>() / estimates.len() as f64;
    let stability = if estimates.len() > 1 {
        estimates
            .iter()
            .map(|e| (e - mean_estimate).powi(2))
            .sum::<f64>()
            / (estimates.len() - 1) as f64
    } else {
        // Single fold: no variance, perfect stability (0.0)
        0.0
    };

    Ok(PerformanceMetrics {
        mae,
        mse,
        rmse,
        mape,
        estimate_stability_concordance,
        hit_rate,
        sharpe_ratio,
        max_drawdown,
        stability,
    })
}

/// Select the best model based on the selection criterion.
fn select_best_model(
    results: &HashMap<FractalEstimator, CrossValidationResult>,
    criterion: &SelectionCriterion,
) -> FractalResult<FractalEstimator> {
    if results.is_empty() {
        return Err(FractalAnalysisError::NumericalError {
            reason: "No models to select from".to_string(),
            operation: None,
        });
    }

    let best = match criterion {
        SelectionCriterion::MinimizeError => results
            .iter()
            .min_by(|a, b| {
                a.1.average_performance
                    .partial_cmp(&b.1.average_performance)
                    .unwrap()
            })
            .map(|(estimator, _)| estimator.clone()),

        SelectionCriterion::MaximizeDirectionalAccuracy => results
            .iter()
            .max_by(|a, b| {
                a.1.metrics
                    .estimate_stability_concordance
                    .partial_cmp(&b.1.metrics.estimate_stability_concordance)
                    .unwrap()
            })
            .map(|(estimator, _)| estimator.clone()),

        SelectionCriterion::MaximizeStability => results
            .iter()
            .min_by(|a, b| {
                a.1.metrics
                    .stability
                    .partial_cmp(&b.1.metrics.stability)
                    .unwrap()
            })
            .map(|(estimator, _)| estimator.clone()),

        SelectionCriterion::InformationCriteria { penalty } => results
            .iter()
            .min_by(|a, b| {
                let score_a = a.1.average_performance + penalty * calculate_model_complexity(&a.0);
                let score_b = b.1.average_performance + penalty * calculate_model_complexity(&b.0);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .map(|(estimator, _)| estimator.clone()),

        SelectionCriterion::WeightedCombination { weights } => results
            .iter()
            .min_by(|a, b| {
                let score_a = calculate_weighted_score(&a.1, weights);
                let score_b = calculate_weighted_score(&b.1, weights);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .map(|(estimator, _)| estimator.clone()),
    };

    best.ok_or(FractalAnalysisError::NumericalError {
        reason: "Could not select best model".to_string(),
        operation: None,
    })
}

/// Rank models by performance according to the selection criterion.
fn rank_models(
    results: &HashMap<FractalEstimator, CrossValidationResult>,
    criterion: &SelectionCriterion,
) -> Vec<(FractalEstimator, f64)> {
    let mut ranking: Vec<(FractalEstimator, f64)> = results
        .iter()
        .map(|(estimator, result)| {
            let score = match criterion {
                SelectionCriterion::MinimizeError => result.average_performance,
                SelectionCriterion::MaximizeDirectionalAccuracy => {
                    -result.metrics.estimate_stability_concordance
                }
                SelectionCriterion::MaximizeStability => result.metrics.stability,
                SelectionCriterion::InformationCriteria { penalty } => {
                    result.average_performance + penalty * calculate_model_complexity(estimator)
                }
                SelectionCriterion::WeightedCombination { weights } => {
                    calculate_weighted_score(result, weights)
                }
            };
            (estimator.clone(), score)
        })
        .collect();

    ranking.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    ranking
}

/// Perform pairwise significance tests between all model pairs.
fn perform_pairwise_significance_tests(
    results: &HashMap<FractalEstimator, CrossValidationResult>,
    _significance_level: f64,
) -> HashMap<(FractalEstimator, FractalEstimator), f64> {
    let mut tests = HashMap::new();
    let estimators: Vec<_> = results.keys().cloned().collect();

    for i in 0..estimators.len() {
        for j in (i + 1)..estimators.len() {
            let est_a = &estimators[i];
            let est_b = &estimators[j];

            if let (Some(result_a), Some(result_b)) = (results.get(est_a), results.get(est_b)) {
                let p_value = paired_t_test_p_value(
                    &result_a
                        .fold_results
                        .iter()
                        .map(|f| f.error)
                        .collect::<Vec<_>>(),
                    &result_b
                        .fold_results
                        .iter()
                        .map(|f| f.error)
                        .collect::<Vec<_>>(),
                );

                tests.insert((est_a.clone(), est_b.clone()), p_value);
            }
        }
    }

    tests
}

/// Calculate model complexity penalty for information criteria.
fn calculate_model_complexity(estimator: &FractalEstimator) -> f64 {
    match estimator {
        FractalEstimator::DetrendedFluctuation(_) => 2.0,
        FractalEstimator::MultifractalDFA { .. } => 3.0,
        FractalEstimator::PeriodogramRegression(_) => 2.0,
        FractalEstimator::WaveletBased => 3.0,
        FractalEstimator::WaveletModulusMaxima(_) => 4.0,
        FractalEstimator::RescaledRange => 1.0,
        FractalEstimator::Ensemble { methods } => {
            methods.iter().map(calculate_model_complexity).sum::<f64>()
        }
    }
}

/// Calculate weighted score for custom model selection criteria.
fn calculate_weighted_score(result: &CrossValidationResult, weights: &HashMap<String, f64>) -> f64 {
    let mut score = 0.0;

    if let Some(&w) = weights.get("error") {
        score += w * result.average_performance;
    }

    if let Some(&w) = weights.get("stability") {
        score += w * result.metrics.stability;
    }

    if let Some(&w) = weights.get("directional_accuracy") {
        score -= w * result.metrics.estimate_stability_concordance; // Negative because higher is better
    }

    score
}

/// Paired t-test p-value calculation using proper t-distribution.
fn paired_t_test_p_value(sample_a: &[f64], sample_b: &[f64]) -> f64 {
    use statrs::distribution::{ContinuousCDF, StudentsT};

    if sample_a.len() != sample_b.len() || sample_a.len() < 2 {
        return 1.0; // No significant difference
    }

    let differences: Vec<f64> = sample_a
        .iter()
        .zip(sample_b.iter())
        .map(|(a, b)| a - b)
        .collect();

    let n = differences.len() as f64;
    let mean_diff = differences.iter().sum::<f64>() / n;
    let var_diff = differences
        .iter()
        .map(|d| (d - mean_diff).powi(2))
        .sum::<f64>()
        / (n - 1.0);

    if var_diff <= 1e-10 {
        // If variance is essentially zero, samples are identical
        return if mean_diff.abs() < 1e-10 { 1.0 } else { 0.0 };
    }

    let se_diff = (var_diff / n).sqrt();
    let t_stat = mean_diff / se_diff;
    let df = n - 1.0;

    // Use Student's t-distribution for proper p-value calculation
    match StudentsT::new(0.0, 1.0, df) {
        Ok(t_dist) => {
            // Two-tailed test
            let p_value = 2.0 * (1.0 - t_dist.cdf(t_stat.abs()));
            p_value.min(1.0).max(0.0) // Ensure valid probability
        }
        Err(_) => {
            // Fallback to normal approximation for df issues
            use statrs::distribution::Normal;
            match Normal::new(0.0, 1.0) {
                Ok(normal) => 2.0 * (1.0 - normal.cdf(t_stat.abs())),
                Err(_) => 1.0,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_validation_folds() {
        let data_length = 1000;
        let method = CrossValidationMethod::WalkForward {
            window_size: 200,
            step_size: 64,
        };

        // Test with DetrendedFluctuation estimator (requires 64 points minimum)
        let estimator = FractalEstimator::DetrendedFluctuation(DfaConfig::default());
        let folds = generate_cv_folds_for_estimator(data_length, &method, &estimator).unwrap();

        assert!(!folds.is_empty());

        for (train_indices, test_indices) in &folds {
            assert!(!train_indices.is_empty());
            assert!(!test_indices.is_empty());
            assert!(train_indices.last().unwrap() < test_indices.first().unwrap());
        }
    }

    #[test]
    fn test_model_selection() {
        // Generate synthetic fractional Brownian motion
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

        let cv_config = CrossValidationConfig {
            method: CrossValidationMethod::WalkForward {
                window_size: 512,
                step_size: 256,
            },
            estimators: vec![
                FractalEstimator::DetrendedFluctuation(DfaConfig::default()),
                FractalEstimator::MultifractalDFA {
                    q_value: 2.0,
                    config: DfaConfig::default(),
                },
            ],
            bootstrap_config: BootstrapConfiguration {
                num_bootstrap_samples: 10, // Reduced for testing
                ..Default::default()
            },
            stability_runs: 10,
            ..Default::default()
        };

        let result = cross_validate_fractal_models(&returns, &cv_config);
        assert!(
            result.is_ok(),
            "Cross-validation should succeed with adequate data"
        );

        let result = result.unwrap();
        assert!(!result.all_results.is_empty());
        assert!(!result.ranking.is_empty());
        assert_eq!(result.ranking.len(), cv_config.estimators.len());
    }

    #[test]
    fn test_wtmm_cross_validation() {
        // Generate synthetic fractional Brownian motion with known Hurst
        let target_hurst = 0.7;
        let config = GeneratorConfig {
            length: 512, // Sufficient for WTMM (requires 128 minimum)
            seed: Some(42),
            sampling_frequency: 1.0,
        };

        let fbm_config = FbmConfig {
            hurst_exponent: target_hurst,
            volatility: 0.01,
            method: FbmMethod::CirculantEmbedding,
        };

        let fbm = generate_fractional_brownian_motion(&config, &fbm_config).unwrap();
        // Convert to FGN for consistency with other methods
        let data_for_analysis = fbm_to_fgn(&fbm);

        // Test WTMM estimator - now handles FGN input
        let cv_config = CrossValidationConfig {
            method: CrossValidationMethod::ExpandingWindow {
                initial_size: 256,
                step_size: 128,  // Ensure all folds are >= 128
            },
            estimators: vec![
                FractalEstimator::WaveletModulusMaxima(WtmmEstimatorConfig::default()),
            ],
            bootstrap_config: BootstrapConfiguration {
                num_bootstrap_samples: 100, // Minimum required for bootstrap
                ..Default::default()
            },
            stability_runs: 5,
            ..Default::default()
        };

        let result = cross_validate_fractal_models(&data_for_analysis, &cv_config);
        assert!(result.is_ok(), "WTMM cross-validation should succeed");

        let result = result.unwrap();

        // Check that WTMM results are present
        let wtmm_key = FractalEstimator::WaveletModulusMaxima(WtmmEstimatorConfig::default());
        assert!(result.all_results.contains_key(&wtmm_key));

        // Get WTMM results
        let wtmm_result = &result.all_results[&wtmm_key];

        // Check that estimates are reasonable
        assert!(!wtmm_result.fold_results.is_empty());
        
        // Collect all estimates to verify accuracy against known Hurst
        let mut all_estimates = Vec::new();
        for fold in &wtmm_result.fold_results {
            // Hurst estimates should be in valid range
            assert!(fold.train_estimate >= 0.0 && fold.train_estimate <= 1.0);
            assert!(fold.test_estimate >= 0.0 && fold.test_estimate <= 1.0);
            
            all_estimates.push(fold.train_estimate);
            all_estimates.push(fold.test_estimate);

            // Should have confidence intervals with analytical errors
            assert!(fold.train_confidence_interval.is_some());
            let ci = fold.train_confidence_interval.as_ref().unwrap();
            assert!(ci.lower_bound < ci.upper_bound);
        }
        
        // Calculate mean estimate across all folds
        let mean_estimate = all_estimates.iter().sum::<f64>() / all_estimates.len() as f64;
        
        // WTMM should recover the Hurst exponent within reasonable tolerance
        assert!(
            (mean_estimate - target_hurst).abs() < 0.15,
            "WTMM mean estimate {} should be close to true Hurst {}",
            mean_estimate, target_hurst
        );

        // Performance metrics should be computed
        assert!(wtmm_result.metrics.stability >= 0.0);
        assert!(wtmm_result.metrics.mae >= 0.0);
        
        // Bootstrap intervals should be reasonable
        if let Some(ci) = &wtmm_result.fold_results[0].train_confidence_interval {
            let width = ci.upper_bound - ci.lower_bound;
            assert!(
                width > 0.01 && width < 0.5,
                "Bootstrap CI width {} should be reasonable",
                width
            );
        }
    }

    #[test]
    fn test_wtmm_ensemble() {
        // Test WTMM in ensemble with other methods
        let target_hurst = 0.6;
        let config = GeneratorConfig {
            length: 400,
            seed: Some(123),
            sampling_frequency: 1.0,
        };

        let fbm_config = FbmConfig {
            hurst_exponent: target_hurst,
            volatility: 0.01,
            method: FbmMethod::CirculantEmbedding,
        };

        let fbm = generate_fractional_brownian_motion(&config, &fbm_config).unwrap();
        // Convert to FGN for all methods in ensemble (WTMM now handles FGN)
        let data = fbm_to_fgn(&fbm);

        // Create ensemble including WTMM
        let ensemble = FractalEstimator::Ensemble {
            methods: vec![
                FractalEstimator::WaveletModulusMaxima(WtmmEstimatorConfig::default()),
                FractalEstimator::DetrendedFluctuation(DfaConfig::default()),
                FractalEstimator::PeriodogramRegression(GphConfig::default()),
            ],
        };

        // Test that WTMM works in ensemble - create a simple walk-forward validation
        let cv_config = CrossValidationConfig {
            method: CrossValidationMethod::WalkForward {
                window_size: 256,  // Ensure >= 128 for WTMM
                step_size: 128,    // Ensure all windows are large enough
            },
            estimators: vec![ensemble.clone()],
            bootstrap_config: BootstrapConfiguration {
                num_bootstrap_samples: 100, // Minimum required for bootstrap
                ..Default::default()
            },
            stability_runs: 3,
            ..Default::default()
        };

        let result = cross_validate_fractal_models(&data, &cv_config);
        assert!(
            result.is_ok(),
            "Ensemble with WTMM should work in cross-validation"
        );

        let result = result.unwrap();
        let ensemble_result = &result.all_results[&ensemble];
        
        // Verify that metrics are sensible
        assert!(ensemble_result.average_performance >= 0.0);
        
        // Collect estimates and verify ensemble recovers the Hurst exponent
        let mut estimates = Vec::new();
        for fold in &ensemble_result.fold_results {
            estimates.push(fold.train_estimate);
            estimates.push(fold.test_estimate);
        }
        
        let mean_ensemble_estimate = estimates.iter().sum::<f64>() / estimates.len() as f64;
        
        // Ensemble should provide reasonable estimate of true Hurst
        assert!(
            (mean_ensemble_estimate - target_hurst).abs() < 0.12,
            "Ensemble mean estimate {} should be close to true Hurst {}",
            mean_ensemble_estimate, target_hurst
        );
        
        // Check bootstrap confidence intervals are present and reasonable
        if let Some(ci) = &ensemble_result.fold_results[0].train_confidence_interval {
            let width = ci.upper_bound - ci.lower_bound;
            assert!(
                width > 0.01 && width < 0.4,
                "Bootstrap CI width {} should be reasonable for ensemble",
                width
            );
        }
    }
}
