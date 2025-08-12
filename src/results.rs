//! # Analysis Results Structures
//!
//! This module contains all the result structures used for storing and organizing
//! the outputs of fractal analysis, including estimation results, validation metrics,
//! regime analysis, and model selection criteria.

use crate::{
    hurst_estimators::{EstimationMethod, HurstEstimate},
    multifractal::MultifractalAnalysis,
    preprocessing::PreprocessingInfo,
    statistical_tests::{
        GoodnessOfFitTests, LongRangeDependenceTest, ShortRangeDependenceTest, StructuralBreakTest,
    },
};
use std::collections::{BTreeMap, HashMap};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Complete results from comprehensive fractal analysis
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FractalEstimationResults {
    /// Hurst exponent estimates from different methods (deterministic ordering)
    pub hurst_estimates: BTreeMap<EstimationMethod, HurstEstimate>,
    /// Multifractal spectrum analysis
    pub multifractal_analysis: MultifractalAnalysis,
    /// Regime change detection results
    pub regime_analysis: RegimeAnalysis,
    /// Statistical tests results
    pub statistical_tests: StatisticalTestResults,
    /// Model selection criteria
    pub model_selection: ModelSelectionCriteria,
    /// Assumption validation results
    pub assumption_checks: AssumptionValidation,
    /// Data preprocessing information
    pub preprocessing_info: PreprocessingInfo,
}

/// Assumption validation results for financial time series analysis
///
/// **IMPORTANT**: P-values for stationarity tests (ADF/KPSS) are approximate
/// based on interpolation or asymptotic methods. See `p_value_method` field
/// in test results for details. For compliance-critical applications, use
/// test statistics with appropriate critical value tables.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AssumptionValidation {
    /// Stationarity test results
    pub stationarity: StationarityTests,
    /// Normality test results
    pub normality: NormalityTests,
    /// Serial correlation test results
    pub serial_correlation: SerialCorrelationTests,
    /// Heteroskedasticity test results
    pub heteroskedasticity: HeteroskedasticityTests,
    /// Data quality metrics
    pub data_quality: DataQualityMetrics,
    /// Method-specific validity checks
    pub method_validity: BTreeMap<EstimationMethod, MethodValidity>,
    /// Overall validity assessment
    pub overall_valid: bool,
    /// Warning messages
    pub warnings: Vec<String>,
}

/// Stationarity test results
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StationarityTests {
    pub adf_statistic: f64,
    pub adf_p_value: f64,
    pub adf_is_stationary: bool,
    pub kpss_statistic: f64,
    pub kpss_p_value: f64,
    pub kpss_is_stationary: bool,
    pub conclusion: String,
}

/// Normality test results
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NormalityTests {
    pub jarque_bera_statistic: f64,
    pub jarque_bera_p_value: f64,
    pub is_normal: bool,
    pub skewness: f64,
    pub kurtosis: f64,
}

/// Serial correlation test results
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SerialCorrelationTests {
    pub ljung_box_statistic: f64,
    pub ljung_box_p_value: f64,
    pub has_serial_correlation: bool,
    pub first_order_autocorr: f64,
}

/// Heteroskedasticity test results
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HeteroskedasticityTests {
    pub arch_lm_statistic: f64,
    pub arch_lm_p_value: f64,
    pub has_arch_effects: bool,
    pub recommendation: String,
}

/// Data quality metrics
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DataQualityMetrics {
    pub sample_size: usize,
    pub missing_values: usize,
    pub outliers_detected: usize,
    pub outlier_percentage: f64,
    /// Near-zero variance flag (true if variance < 1e-10)
    /// Note: This indicates practically constant data, not mathematically zero variance
    pub zero_variance: bool,
    pub sufficient_variation: bool,
}

/// Method-specific validity assessment
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MethodValidity {
    pub method_name: String,
    pub sample_size_adequate: bool,
    pub assumptions_met: bool,
    pub reliability_score: f64, // 0-1 scale
    pub specific_warnings: Vec<String>,
}

/// Regime analysis results from HMM-based detection
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RegimeAnalysis {
    /// Detected regime changes with confidence intervals
    pub regime_changes: Vec<RegimeChange>,
    /// Regime-specific Hurst exponents
    pub regime_hurst_exponents: Vec<(usize, usize, f64)>, // (regime_id, start_time_index, hurst)
    /// Regime duration statistics
    pub regime_duration_stats: RegimeDurationStatistics,
    /// Transition probabilities
    pub transition_probabilities: Vec<Vec<f64>>,
}

/// Individual regime change detection result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RegimeChange {
    /// Change point location
    pub change_point: usize,
    /// Confidence interval for change point
    pub change_point_ci: (usize, usize),
    /// Pre-change Hurst exponent
    pub pre_change_hurst: f64,
    /// Post-change Hurst exponent
    pub post_change_hurst: f64,
    /// Statistical significance
    pub test_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Change magnitude
    pub change_magnitude: f64,
}

/// Statistics about regime durations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RegimeDurationStatistics {
    /// Mean regime duration
    pub mean_duration: f64,
    /// Standard deviation of duration
    pub std_duration: f64,
    /// Distribution of durations
    pub duration_distribution: Vec<(f64, f64)>, // (duration, frequency)
}

/// Results from comprehensive statistical testing
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StatisticalTestResults {
    /// Test for long-range dependence
    pub long_range_dependence_test: LongRangeDependenceTest,
    /// Test for short-range dependence
    pub short_range_dependence_test: ShortRangeDependenceTest,
    /// Test for structural breaks
    pub structural_break_tests: Vec<StructuralBreakTest>,
    /// Goodness-of-fit tests
    pub goodness_of_fit_tests: GoodnessOfFitTests,
}

/// Model selection criteria for comparing estimators
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ModelSelectionCriteria {
    /// Uncertainty score derived from confidence interval width (lower is better)
    pub uncertainty_score: f64,
    /// Number of parameters (always 1 for Hurst exponent)
    pub num_parameters: usize,
    /// Best model
    pub best_model: EstimationMethod,
}

/// Validation statistics for comprehensive model assessment
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ValidationStatistics {
    /// Out-of-sample prediction accuracy
    pub prediction_accuracy: PredictionAccuracy,
    /// Model robustness tests
    pub robustness_tests: RobustnessTests,
    /// Sensitivity analysis
    pub sensitivity_analysis: SensitivityAnalysis,
}

/// Prediction accuracy metrics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PredictionAccuracy {
    /// Mean squared prediction error
    pub mspe: f64,
    /// Mean absolute prediction error
    pub mape: f64,
    /// Estimate stability concordance
    pub estimate_stability_concordance: f64,
    /// Prediction intervals coverage
    pub coverage_probability: f64,
}

/// Robustness test results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RobustnessTests {
    /// Robustness to outliers
    pub outlier_robustness: f64,
    /// Robustness to sample size
    pub sample_size_robustness: f64,
    /// Robustness to detrending method
    pub detrending_robustness: f64,
}

/// Sensitivity analysis results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SensitivityAnalysis {
    /// Sensitivity to window size
    pub window_size_sensitivity: HashMap<usize, f64>,
    /// Sensitivity to polynomial order (for DFA)
    pub polynomial_order_sensitivity: HashMap<usize, f64>,
    /// Sensitivity to aggregation scale
    pub aggregation_sensitivity: HashMap<usize, f64>,
}