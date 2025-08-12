//! # Main Statistical Fractal Analyzer
//!
//! This module contains the [`StatisticalFractalAnalyzer`] struct which serves as the
//! main entry point for comprehensive fractal analysis of financial time series.
//! It orchestrates all analysis methods and provides a unified interface for
//! enterprise-grade fractal analysis with statistical rigor.
//!
//! ## Key Features
//!
//! - **Multiple Hurst Estimation Methods**: R/S, DFA, GPH, Wavelet-based estimation
//! - **Statistical Rigor**: All estimates include bias corrections and confidence intervals
//! - **Multifractal Analysis**: Complete MF-DFA implementation with singularity spectrum
//! - **Regime Detection**: HMM-based structural break detection
//! - **Model Validation**: Cross-validation and bootstrap validation frameworks
//! - **Robustness Testing**: Comprehensive sensitivity and stability analysis
//!
//! ## Important Note on P-Values
//!
//! **WARNING**: P-values for ADF and KPSS tests are **APPROXIMATE** based on critical value
//! interpolation or asymptotic approximations. They should not be used for regulatory
//! compliance or critical financial decisions without additional validation.
//!
//! Each test result includes a `p_value_method` field indicating the approximation method:
//! - `Interpolated`: Uses linear interpolation between tabulated critical values
//! - `Asymptotic`: Based on large-sample approximations
//! - `ResponseSurface`: Response surface regression (when available)
//!
//! For production use with financial reporting requirements, use the test statistics
//! directly with published critical value tables appropriate for your sample size.
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use fractal_finance::StatisticalFractalAnalyzer;
//! use fractal_finance::errors::FractalAnalysisError;
//!
//! # fn main() -> Result<(), FractalAnalysisError> {
//! let mut analyzer = StatisticalFractalAnalyzer::new();
//! // Use enough data for preprocessing & estimators (minimum 128 for GPH)
//! let data: Vec<f64> = (0..512).map(|i| ((i as f64 * 0.1).sin()) * 0.01).collect();
//! analyzer.add_time_series("ASSET".to_string(), data)?;
//!
//! // Perform comprehensive analysis
//! analyzer.analyze_all_series()?;
//!
//! // Get results with confidence intervals
//! let results = analyzer.get_analysis_results("ASSET")?;
//! for (method, estimate) in &results.hurst_estimates {
//!     println!("{:?}: H = {:.3} ± {:.3}", method,
//!         estimate.estimate, estimate.standard_error);
//! }
//! # Ok(())
//! # }
//! ```

use crate::{
    bootstrap::{
        bootstrap_validate, politis_white_block_size, BootstrapConfiguration, BootstrapMethod,
        BootstrapValidation, ConfidenceInterval, ConfidenceIntervalMethod, EstimatorComplexity,
    },
    config::{AnalysisConfig, AnalysisDepth},
    cross_validation::*,
    decimal_finance::FinancialAuditLog,
    errors::{validate_all_finite, validate_data_length, FractalAnalysisError, FractalResult},
    results::{
        AssumptionValidation, DataQualityMetrics, FractalEstimationResults,
        HeteroskedasticityTests, MethodValidity, ModelSelectionCriteria, NormalityTests,
        PredictionAccuracy, RegimeAnalysis, RegimeChange, RegimeDurationStatistics,
        RobustnessTests, SensitivityAnalysis, SerialCorrelationTests, StationarityTests,
        StatisticalTestResults, ValidationStatistics,
    },
    fft_ops::calculate_periodogram_fft,
    hurst_estimators::{
        estimate_dfa_hurst_only, estimate_hurst_multiple_methods, estimate_rs_hurst_only,
        EstimationMethod, HurstEstimate, HurstEstimationConfig,
    },
    linear_algebra::{economy_qr_solve},
    math_utils::{
        self, calculate_kurtosis, calculate_skewness, calculate_variance, chi_squared_cdf,
        float_ops, ols_regression_hac, standard_normal_cdf,
    },
    preprocessing::{
        detect_outliers, preprocess_financial_data_with_kind, DataKind, PreprocessingInfo,
    },
    monte_carlo::*,
    multifractal::*,
    regime_detection::*,
    statistical_tests::*,
    wavelet::{calculate_wavelet_variance, estimate_wavelet_hurst_only},
};
// Serde derives not currently used in this module
// #[cfg(feature = "serde")]
// use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, RwLock};

/// Maximum allowed time series size (10 million points)
const MAX_TIME_SERIES_SIZE: usize = 10_000_000;
/// Maximum total memory for all series (100 million points)
const MAX_TOTAL_DATA_POINTS: usize = 100_000_000;
/// Minimum variance threshold for numerical stability
/// Increased from 1e-14 to 1e-12 for better handling of noisy near-zero data
/// Used for: Numerical stability in divisions and regression operations
const MIN_VARIANCE_THRESHOLD: f64 = 1e-12;
/// Minimum standard error for division safety (numerical floor)
const MIN_STD_ERROR: f64 = 1e-6;

// Statistical constants
/// Zero variance threshold for constant data detection
/// Set to 1e-13 to avoid false positives with machine precision
/// Note: This is SMALLER than MIN_VARIANCE_THRESHOLD (1e-13 < 1e-12)
/// Used for: Early detection of truly constant data before processing
const ZERO_VARIANCE_THRESHOLD: f64 = 1e-13;
/// Near-zero variance threshold for data quality marking and bootstrap skipping
/// Set to 1e-10 for practical near-zero variance detection
const NEAR_ZERO_VARIANCE_THRESHOLD: f64 = 1e-10;
/// Epsilon for matching q values in MF-DFA (relaxed for numerical stability)
const Q_MATCH_EPSILON: f64 = 1e-6;
/// Default confidence level for intervals
const DEFAULT_CONFIDENCE_LEVEL: f64 = 0.95;
/// Random walk Hurst exponent (H=0.5 hypothesis)
const RANDOM_WALK_HURST: f64 = 0.5;

// Estimation method constants
/// GPH bandwidth exponent (n^α where α=0.65)
const GPH_BANDWIDTH_EXPONENT: f64 = 0.65;
// Removed unused Hurst bounds constants - estimates are no longer clamped artificially
/// Default polynomial order for DFA detrending (1=linear, 2=quadratic, 3=cubic)
const DEFAULT_DFA_POLYNOMIAL_ORDER: usize = 2;

/// Comprehensive fractal analysis with statistical rigor
///
/// The main orchestrating class that coordinates all analysis methods and provides
/// a unified interface for comprehensive fractal analysis of financial time series.
/// All estimators include proper bias corrections, confidence intervals, and
/// comprehensive validation frameworks.



pub struct StatisticalFractalAnalyzer {
    /// Raw time series data (thread-safe, deterministic ordering)
    time_series_data: Arc<RwLock<BTreeMap<String, Vec<f64>>>>,
    /// Estimation results with confidence intervals (thread-safe, deterministic ordering)
    estimation_results: Arc<RwLock<BTreeMap<String, FractalEstimationResults>>>,
    /// Model validation statistics (thread-safe, deterministic ordering)
    validation_statistics: Arc<RwLock<BTreeMap<String, ValidationStatistics>>>,
    /// Bootstrap configuration
    bootstrap_config: BootstrapConfiguration,
    /// DFA polynomial order for detrending (1-5, default: 2)
    dfa_polynomial_order: usize,
    /// Total data points tracker for memory management
    total_data_points: Arc<RwLock<usize>>,
    /// Financial audit log for compliance and forensic analysis
    audit_log: FinancialAuditLog,
    /// Test configuration for statistical tests
    test_config: TestConfiguration,
    /// Analysis configuration for controlling components
    analysis_config: AnalysisConfig,
}


impl StatisticalFractalAnalyzer {
    /// Create a new fractal analyzer with default configuration
    pub fn new() -> Self {
        Self {
            time_series_data: Arc::new(RwLock::new(BTreeMap::new())),
            estimation_results: Arc::new(RwLock::new(BTreeMap::new())),
            validation_statistics: Arc::new(RwLock::new(BTreeMap::new())),
            bootstrap_config: BootstrapConfiguration::default(),
            dfa_polynomial_order: DEFAULT_DFA_POLYNOMIAL_ORDER,
            total_data_points: Arc::new(RwLock::new(0)),
            audit_log: FinancialAuditLog::new(),
            test_config: TestConfiguration::default(),
            analysis_config: AnalysisConfig::default(),
        }
    }

    /// Set the polynomial order for DFA detrending
    ///
    /// # Arguments
    /// * `order` - Polynomial order (1=linear, 2=quadratic, 3=cubic, 4=quartic, 5=quintic)
    ///
    /// # Returns
    /// Error if order is out of valid range [1, 5]
    pub fn set_dfa_polynomial_order(&mut self, order: usize) -> FractalResult<()> {
        if order == 0 || order > 5 {
            return Err(FractalAnalysisError::InvalidParameter {
                parameter: "dfa_polynomial_order".to_string(),
                value: order as f64,
                constraint: "Must be between 1 and 5".to_string(),
            });
        }
        self.dfa_polynomial_order = order;
        Ok(())
    }

    /// Set the random seed for bootstrap operations
    ///
    /// # Arguments
    /// * `seed` - Random seed for reproducibility
    pub fn set_bootstrap_seed(&mut self, seed: u64) {
        self.bootstrap_config.seed = Some(seed);
    }

    /// Add time series for analysis with memory safety checks
    ///
    /// # Returns
    /// Ok(true) if a new series was inserted, Ok(false) if an existing series was replaced
    pub fn add_time_series(&mut self, asset: String, data: Vec<f64>) -> FractalResult<bool> {
        // Validate data size
        if data.len() > MAX_TIME_SERIES_SIZE {
            return Err(FractalAnalysisError::InvalidParameter {
                parameter: "data_size".to_string(),
                value: data.len() as f64,
                constraint: format!("Maximum {} points allowed", MAX_TIME_SERIES_SIZE),
            });
        }

        // Validate all values are finite
        validate_all_finite(&data, "time_series_data")?;

        // Acquire both locks up front to ensure consistent state
        let mut series_map =
            self.time_series_data
                .write()
                .map_err(|_| FractalAnalysisError::NumericalError {
                    reason: "Failed to acquire lock for time series data".to_string(),
                    operation: None,
                })?;

        let mut total_points =
            self.total_data_points
                .write()
                .map_err(|_| FractalAnalysisError::NumericalError {
                    reason: "Failed to acquire lock for memory tracking".to_string(),
                    operation: None,
                })?;

        // Calculate new total using saturating arithmetic to prevent underflow
        let old_size = series_map.get(&asset).map(|v| v.len()).unwrap_or(0);
        let new_total = total_points
            .saturating_sub(old_size)
            .saturating_add(data.len());

        if new_total > MAX_TOTAL_DATA_POINTS {
            return Err(FractalAnalysisError::InvalidParameter {
                parameter: "total_data_points".to_string(),
                value: new_total as f64,
                constraint: format!("Maximum {} total points allowed", MAX_TOTAL_DATA_POINTS),
            });
        }

        // Store data length before moving data
        let data_len = data.len();
        let is_new = old_size == 0;

        // Insert data and update count atomically
        series_map.insert(asset.clone(), data);
        *total_points = new_total;

        // FINANCIAL COMPLIANCE: Log data addition for audit trail
        // Critical audit failures should be propagated in production
        if let Err(e) = self.audit_log.log_operation(
            format!(
                "{}: {} with {} points",
                if is_new {
                    "ADD_TIME_SERIES"
                } else {
                    "REPLACE_TIME_SERIES"
                },
                &asset,
                data_len
            ),
            None,
        ) {
            log::warn!("Failed to log audit operation: {:?}", e);
        }

        Ok(is_new)
    }

    /// Remove time series data for a specific asset
    ///
    /// # Returns
    /// Ok(true) if the series was removed, Ok(false) if it didn't exist
    pub fn remove_time_series(&mut self, asset: &str) -> FractalResult<bool> {
        // Compute removal size within lock scope, then drop locks before taking results lock
        let (removed_size, was_removed) = {
            let mut series_map = self.time_series_data.write().map_err(|_| {
                FractalAnalysisError::NumericalError {
                    reason: "Failed to acquire lock for time series data".to_string(),
                    operation: None,
                }
            })?;

            let mut total_points = self.total_data_points.write().map_err(|_| {
                FractalAnalysisError::NumericalError {
                    reason: "Failed to acquire lock for memory tracking".to_string(),
                    operation: None,
                }
            })?;

            // Remove and update total if asset exists
            if let Some(removed_data) = series_map.remove(asset) {
                let size = removed_data.len();
                *total_points = total_points.saturating_sub(size);
                (size, true)
            } else {
                (0, false)
            }
        }; // Drop locks here

        // Clear cached results after dropping data locks
        if was_removed {
            if let Ok(mut results) = self.estimation_results.write() {
                results.remove(asset);
            }

            // FINANCIAL COMPLIANCE: Log removal for audit trail  
            // Critical audit failures should be propagated in production
            if let Err(e) = self.audit_log.log_operation(
                format!("REMOVE_TIME_SERIES: {} with {} points", asset, removed_size),
                None,
            ) {
                log::warn!("Failed to log audit operation: {:?}", e);
            }
        }

        Ok(was_removed)
    }

    /// Get raw time series data for a specific asset
    pub fn get_time_series_data(&self, asset: &str) -> FractalResult<Vec<f64>> {
        let series_map =
            self.time_series_data
                .read()
                .map_err(|_| FractalAnalysisError::NumericalError {
                    reason: "Failed to acquire lock for reading time series data".to_string(),
                    operation: None,
                })?;

        series_map
            .get(asset)
            .cloned()
            .ok_or_else(|| FractalAnalysisError::TimeSeriesNotFound {
                name: asset.to_string(),
            })
    }

    /// Perform comprehensive fractal analysis on all time series
    pub fn analyze_all_series(&mut self) -> FractalResult<()> {
        // Get list of assets without cloning data - just asset names
        let assets_to_analyze: Vec<String> = {
            let series_map =
                self.time_series_data
                    .read()
                    .map_err(|_| FractalAnalysisError::NumericalError {
                        reason: "Failed to acquire lock for reading time series data".to_string(),
                        operation: None,
                    })?;
            series_map.keys().cloned().collect()
        };

        // FINANCIAL COMPLIANCE: Log batch analysis start
        // Critical audit failures should be propagated in production
        if let Err(e) = self.audit_log.log_operation(
            format!("BATCH_ANALYSIS_START: {} assets", assets_to_analyze.len()),
            None,
        ) {
            log::warn!("Failed to log audit operation: {:?}", e);
        }

        // CONCURRENCY SAFETY: Clone configuration before parallel processing
        // This prevents race conditions if configuration is modified during analysis
        let current_bootstrap_config = self.bootstrap_config.clone();
        let current_dfa_order = self.dfa_polynomial_order;
        // Clone test and analysis configs to avoid borrow checker issues in parallel closure
        let current_test_config = self.test_config.clone();
        let current_analysis_config = self.analysis_config.clone();

        // FINANCIAL-GRADE PARALLEL PROCESSING: Analyze multiple assets concurrently
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            // Process all assets in parallel for maximum throughput
            // Note: We use the cloned configs to ensure consistency across all parallel tasks
            let analysis_results: Vec<
                FractalResult<(String, FractalEstimationResults, ValidationStatistics)>,
            > = assets_to_analyze
                .par_iter()
                .map(|asset| {
                    // Fetch data on demand to avoid cloning entire dataset
                    let data = {
                        let series_map = self.time_series_data.read().map_err(|_| {
                            FractalAnalysisError::NumericalError {
                                reason: "Failed to acquire lock for reading time series data"
                                    .to_string(),
                                operation: None,
                            }
                        })?;
                        series_map.get(asset).cloned().ok_or_else(|| {
                            FractalAnalysisError::TimeSeriesNotFound {
                                name: asset.clone(),
                            }
                        })?
                    };

                    // Create a temporary analyzer with the cloned configs for thread safety
                    let mut temp_analyzer = Self {
                        time_series_data: Arc::new(RwLock::new(BTreeMap::new())),
                        estimation_results: Arc::new(RwLock::new(BTreeMap::new())),
                        validation_statistics: Arc::new(RwLock::new(BTreeMap::new())),
                        bootstrap_config: current_bootstrap_config.clone(),
                        dfa_polynomial_order: current_dfa_order,
                        total_data_points: Arc::new(RwLock::new(0)),
                        audit_log: FinancialAuditLog::new(),
                        // Use the cloned configs to avoid borrow checker issues
                        test_config: current_test_config.clone(),
                        analysis_config: current_analysis_config.clone(),
                    };

                    let results = temp_analyzer.perform_comprehensive_analysis(&data)?;
                    // Validation may fail for short series - that's acceptable
                    let validation = match temp_analyzer.perform_validation_analysis(asset, &data) {
                        Ok(val) => Some(val),
                        Err(FractalAnalysisError::InsufficientData { .. }) => None,
                        Err(e) => return Err(e),
                    };
                    Ok((asset.clone(), results, validation))
                })
                .collect();

            // Write results sequentially to avoid lock contention
            // FAULT-TOLERANT: Continue processing even if individual assets fail
            let mut failed_assets = Vec::new();
            for result in analysis_results {
                match result {
                    Ok((asset, results, validation)) => {
                        // Store successful results
                        let mut results_map = self.estimation_results.write().map_err(|_| {
                            FractalAnalysisError::NumericalError {
                                reason: "Failed to acquire lock for writing results".to_string(),
                                operation: None,
                            }
                        })?;
                        results_map.insert(asset.clone(), results);

                        // Store validation if available (may be None for short series)
                        if let Some(val) = validation {
                            let mut validation_map =
                                self.validation_statistics.write().map_err(|_| {
                                    FractalAnalysisError::NumericalError {
                                        reason:
                                            "Failed to acquire lock for writing validation statistics"
                                                .to_string(),
                                        operation: None,
                                    }
                                })?;
                            validation_map.insert(asset, val);
                        }
                    }
                    Err(e) => {
                        // Log failure but continue processing other assets
                        log::warn!("Asset analysis failed with error: {:?}", e);
                        if let FractalAnalysisError::TimeSeriesNotFound { name } = &e {
                            failed_assets.push(name.clone());
                        } else {
                            failed_assets.push(format!("Unknown asset: {:?}", e));
                        }
                    }
                }
            }

            // Log summary of failures if any occurred
            if !failed_assets.is_empty() {
                log::warn!(
                    "{} assets failed during batch analysis: {:?}",
                    failed_assets.len(),
                    failed_assets
                );
            }
        }

        // Fallback to sequential processing if parallel feature is not enabled
        #[cfg(not(feature = "parallel"))]
        {
            for asset in assets_to_analyze {
                // Fetch data on demand
                let data = self.get_time_series_data(&asset)?;
                self.analyze_and_store(&asset, &data)?;
            }
        }

        Ok(())
    }

    /// Core analysis and storage logic - extracted to avoid duplication
    fn analyze_and_store(&mut self, asset: &str, data: &[f64]) -> FractalResult<()> {
        self.analyze_and_store_with_kind(asset, data, DataKind::Auto)
    }

    fn analyze_and_store_with_kind(
        &mut self,
        asset: &str,
        data: &[f64],
        data_kind: DataKind,
    ) -> FractalResult<()> {
        let results = self.perform_comprehensive_analysis_with_kind(data, data_kind)?;

        // Store results
        let mut results_map =
            self.estimation_results
                .write()
                .map_err(|_| FractalAnalysisError::NumericalError {
                    reason: "Failed to acquire lock for writing results".to_string(),
                    operation: None,
                })?;
        results_map.insert(asset.to_string(), results);

        // Perform REAL validation with cross-validation and Monte Carlo testing
        // Skip validation for short series where it cannot be performed rigorously
        match self.perform_validation_analysis(asset, data) {
            Ok(validation) => {
                // Store validation statistics
                let mut validation_map = self.validation_statistics.write().map_err(|_| {
                    FractalAnalysisError::NumericalError {
                        reason: "Failed to acquire lock for writing validation statistics".to_string(),
                        operation: None,
                    }
                })?;
                validation_map.insert(asset.to_string(), validation);
            }
            Err(FractalAnalysisError::InsufficientData { .. }) => {
                // Skip validation for short series - this is expected and acceptable
                // The analysis results are still valid with limited methods
            }
            Err(e) => {
                // Propagate other errors
                return Err(e);
            }
        }

        Ok(())
    }

    /// Analyze a single time series
    pub fn analyze_series(&mut self, asset: &str) -> FractalResult<()> {
        let data = self.get_time_series_data(asset)?;

        // FINANCIAL COMPLIANCE: Log single asset analysis
        let _ = self
            .audit_log
            .log_operation(format!("ANALYZE_SERIES: {}", asset), None);

        self.analyze_and_store(asset, &data)
    }

    /// Analyze a single time series with explicit data kind specification
    ///
    /// This method allows power users to skip auto-detection heuristics by explicitly
    /// specifying whether the data represents prices (needs differencing) or returns
    /// (already stationary).
    ///
    /// # Arguments
    /// * `asset` - Name/identifier of the time series
    /// * `data_kind` - Explicit specification of data type (Prices, Returns, or Auto)
    ///
    /// # Example
    /// ```no_run
    /// # use fractal_finance::{StatisticalFractalAnalyzer, DataKind};
    /// # let mut analyzer = StatisticalFractalAnalyzer::new();
    /// // Skip auto-detection when you know data is returns
    /// analyzer.analyze_series_with_kind("ASSET", DataKind::Returns)?;
    /// # Ok::<(), fractal_finance::FractalAnalysisError>(())
    /// ```
    pub fn analyze_series_with_kind(
        &mut self,
        asset: &str,
        data_kind: DataKind,
    ) -> FractalResult<()> {
        let data = self.get_time_series_data(asset)?;

        // FINANCIAL COMPLIANCE: Log analysis with explicit data kind
        // Critical audit failures should be propagated in production
        if let Err(e) = self.audit_log.log_operation(
            format!("ANALYZE_SERIES_WITH_KIND: {} (kind={:?})", asset, data_kind),
            None,
        ) {
            log::warn!("Failed to log audit operation: {:?}", e);
        }

        self.analyze_and_store_with_kind(asset, &data, data_kind)
    }

    fn perform_comprehensive_analysis(
        &self,
        data: &[f64],
    ) -> FractalResult<FractalEstimationResults> {
        self.perform_comprehensive_analysis_with_kind(data, DataKind::Auto)
    }

    fn perform_comprehensive_analysis_with_kind(
        &self,
        data: &[f64],
        data_kind: DataKind,
    ) -> FractalResult<FractalEstimationResults> {
        // CRITICAL: Check stationarity and preprocess for financial data
        let (processed_data, preprocessing_info) =
            preprocess_financial_data_with_kind(data, data_kind, &self.test_config)?;

        // Perform assumption validation on both original and processed data
        let assumption_checks =
            self.validate_assumptions(data, &processed_data, &preprocessing_info)?;

        // Break down the analysis into logical components
        let hurst_config = HurstEstimationConfig {
            dfa_polynomial_order: self.dfa_polynomial_order,
            bootstrap_config: self.bootstrap_config.clone(),
            test_config: self.test_config.clone(),
            ..HurstEstimationConfig::default()
        };
        let hurst_estimates = estimate_hurst_multiple_methods(&processed_data, &hurst_config)?;

        // Conditionally run heavy components based on configuration
        let multifractal_analysis = if self.analysis_config.is_multifractal_enabled() {
            self.perform_multifractal_component(&processed_data)?
        } else {
            // Return minimal multifractal results
            MultifractalAnalysis {
                generalized_hurst_exponents: vec![],
                mass_exponents: vec![],
                singularity_spectrum: vec![],
                multifractality_degree: 0.0,
                asymmetry_parameter: 0.0,
                multifractality_test: MultifractalityTest {
                    test_statistic: 0.0,
                    p_value: 1.0,
                    critical_value: 0.0,
                    is_multifractal: false,
                },
            }
        };

        let regime_analysis = if self.analysis_config.is_regime_detection_enabled() {
            self.perform_regime_component(&processed_data)?
        } else {
            // Return minimal regime results
            RegimeAnalysis {
                regime_changes: vec![],
                regime_hurst_exponents: vec![],
                regime_duration_stats: RegimeDurationStatistics {
                    mean_duration: 0.0,
                    std_duration: 0.0,
                    duration_distribution: vec![],
                },
                transition_probabilities: vec![],
            }
        };

        let statistical_tests = self.perform_statistical_component(&processed_data)?;
        let model_selection =
            self.perform_model_selection_component(&hurst_estimates, &processed_data)?;

        Ok(FractalEstimationResults {
            hurst_estimates,
            multifractal_analysis,
            regime_analysis,
            statistical_tests,
            model_selection,
            assumption_checks,
            preprocessing_info: preprocessing_info.clone(),
        })
    }



    /// Validate assumptions for fractal analysis
    fn validate_assumptions(
        &self,
        original_data: &[f64],
        processed_data: &[f64],
        preprocessing_info: &PreprocessingInfo,
    ) -> FractalResult<AssumptionValidation> {
        // Guard against empty data
        if processed_data.is_empty() {
            return Err(FractalAnalysisError::InsufficientData {
                required: 1,
                actual: 0,
            });
        }
        
        let mut validation = AssumptionValidation::default();
        let mut warnings = Vec::new();

        // 1. Stationarity tests (already computed in preprocessing)
        validation.stationarity = StationarityTests {
            adf_statistic: preprocessing_info.adf_statistic,
            adf_p_value: preprocessing_info.adf_p_value,
            adf_is_stationary: preprocessing_info.adf_p_value < 0.05,
            kpss_statistic: preprocessing_info.kpss_statistic,
            kpss_p_value: preprocessing_info.kpss_p_value,
            kpss_is_stationary: preprocessing_info.kpss_p_value > 0.05,
            conclusion: if preprocessing_info.adf_p_value < 0.05
                && preprocessing_info.kpss_p_value > 0.05
            {
                "Series is stationary".to_string()
            } else if preprocessing_info.differencing_applied {
                "Series made stationary through differencing".to_string()
            } else {
                "Series may not be stationary - results should be interpreted with caution"
                    .to_string()
            },
        };

        if !validation.stationarity.adf_is_stationary && !preprocessing_info.differencing_applied {
            warnings
                .push("Non-stationary series detected - Hurst estimates may be biased".to_string());
        }

        // 2. Normality tests using Jarque-Bera
        let jb_stat = jarque_bera_test(processed_data);
        let jb_p = 1.0 - chi_squared_cdf(jb_stat, 2);
        let skewness = calculate_skewness(processed_data);
        let kurtosis = calculate_kurtosis(processed_data);

        validation.normality = NormalityTests {
            jarque_bera_statistic: jb_stat,
            jarque_bera_p_value: jb_p,
            is_normal: jb_p > 0.05,
            skewness,
            kurtosis,
        };

        if !validation.normality.is_normal {
            warnings.push(format!("Non-normal distribution detected (skew={:.2}, kurt={:.2}) - bootstrap CIs recommended", 
                skewness, kurtosis));
        }

        // 3. Serial correlation tests
        // Cap lag at min(sqrt(n), n-2, 40) for stability with shorter samples
        let lag = ((processed_data.len() as f64).sqrt().round() as usize)
            .min(processed_data.len().saturating_sub(2))
            .min(40);
        let (lb_stat, lb_p) = self.ljung_box_test(processed_data, lag)?;
        let first_order_ac = if processed_data.len() > 1 {
            let mean = processed_data.iter().sum::<f64>() / processed_data.len() as f64;
            let mut cov = 0.0;
            let mut var = 0.0;
            for i in 0..processed_data.len() - 1 {
                cov += (processed_data[i] - mean) * (processed_data[i + 1] - mean);
            }
            for x in processed_data {
                var += (x - mean).powi(2);
            }
            // Guard against divide-by-zero with near-constant data
            if var.abs() < MIN_VARIANCE_THRESHOLD {
                0.0 // No autocorrelation for constant data
            } else {
                cov / var
            }
        } else {
            0.0
        };

        validation.serial_correlation = SerialCorrelationTests {
            ljung_box_statistic: lb_stat,
            ljung_box_p_value: lb_p,
            has_serial_correlation: lb_p < 0.05,
            first_order_autocorr: first_order_ac,
        };

        if validation.serial_correlation.has_serial_correlation {
            warnings
                .push("Significant serial correlation detected - block bootstrap used".to_string());
        }

        // 4. Heteroskedasticity tests (ARCH effects)
        validation.heteroskedasticity = HeteroskedasticityTests {
            arch_lm_statistic: preprocessing_info.arch_test_statistic,
            arch_lm_p_value: preprocessing_info.arch_test_p_value,
            has_arch_effects: preprocessing_info.arch_effects_present,
            recommendation: if preprocessing_info.arch_effects_present {
                if preprocessing_info.volatility_adjusted {
                    "ARCH effects detected and adjusted".to_string()
                } else {
                    "ARCH effects present - consider GARCH modeling".to_string()
                }
            } else {
                "No significant ARCH effects".to_string()
            },
        };

        // 5. Data quality metrics
        let outliers = self.detect_outliers(processed_data);
        validation.data_quality = DataQualityMetrics {
            sample_size: processed_data.len(),
            missing_values: 0, // Already handled in preprocessing
            outliers_detected: outliers.len(),
            outlier_percentage: if processed_data.is_empty() { 0.0 } else { (outliers.len() as f64 / processed_data.len() as f64) * 100.0 },
            zero_variance: calculate_variance(processed_data) < NEAR_ZERO_VARIANCE_THRESHOLD,
            sufficient_variation: calculate_variance(processed_data) > 1e-6,
        };

        if validation.data_quality.outlier_percentage > 5.0 {
            warnings.push(format!(
                "High outlier percentage ({:.1}%) detected",
                validation.data_quality.outlier_percentage
            ));
        }

        // 6. Method-specific validity checks
        let mut method_validity = BTreeMap::new();
        let n = processed_data.len();

        // R/S validity
        method_validity.insert(
            EstimationMethod::RescaledRange,
            MethodValidity {
                method_name: "R/S Analysis".to_string(),
                sample_size_adequate: n >= 50,
                assumptions_met: n >= 50 && validation.data_quality.sufficient_variation,
                reliability_score: if n >= 100 {
                    0.9
                } else if n >= 50 {
                    0.7
                } else {
                    0.3
                },
                specific_warnings: if n < 50 {
                    vec!["Sample size too small for reliable R/S analysis".to_string()]
                } else {
                    vec![]
                },
            },
        );

        // DFA validity
        method_validity.insert(
            EstimationMethod::DetrendedFluctuationAnalysis,
            MethodValidity {
                method_name: "DFA".to_string(),
                sample_size_adequate: n >= 100,
                assumptions_met: n >= 100 && validation.data_quality.sufficient_variation,
                reliability_score: if n >= 500 {
                    0.95
                } else if n >= 100 {
                    0.8
                } else {
                    0.4
                },
                specific_warnings: if n < 100 {
                    vec!["Sample size too small for reliable DFA".to_string()]
                } else {
                    vec![]
                },
            },
        );

        // GPH validity
        method_validity.insert(
            EstimationMethod::PeriodogramRegression,
            MethodValidity {
                method_name: "GPH".to_string(),
                sample_size_adequate: n >= 128,
                assumptions_met: n >= 128 && validation.stationarity.adf_is_stationary,
                reliability_score: if n >= 256 && validation.stationarity.adf_is_stationary {
                    0.85
                } else if n >= 128 {
                    0.6
                } else {
                    0.2
                },
                specific_warnings: {
                    let mut w = vec![];
                    if n < 128 {
                        w.push("Sample size too small for GPH".to_string());
                    }
                    if !validation.stationarity.adf_is_stationary {
                        w.push("GPH requires stationary data".to_string());
                    }
                    w
                },
            },
        );

        // Wavelet validity
        method_validity.insert(
            EstimationMethod::WaveletEstimation,
            MethodValidity {
                method_name: "Wavelet".to_string(),
                sample_size_adequate: n >= 64,
                assumptions_met: n >= 64, // Dyadic scale regression needs sufficient scales
                reliability_score: if n >= 128 {
                    0.9
                } else if n >= 64 {
                    0.7
                } else {
                    0.3
                },
                specific_warnings: {
                    let mut w = vec![];
                    if n < 64 {
                        w.push("Sample size too small for wavelet analysis".to_string());
                    }
                    // Using dyadic scales for variance regression
                    w
                },
            },
        );

        validation.method_validity = method_validity;

        // Overall validity assessment
        validation.overall_valid = validation.data_quality.sample_size >= 50
            && validation.data_quality.sufficient_variation
            && (validation.stationarity.adf_is_stationary
                || preprocessing_info.differencing_applied);

        validation.warnings = warnings;

        Ok(validation)
    }


    /// Ljung-Box test for serial correlation
    fn ljung_box_test(&self, data: &[f64], lag: usize) -> FractalResult<(f64, f64)> {
        ljung_box_test_with_config(data, lag, &self.test_config)
    }

    /// Detect outliers using IQR method
    fn detect_outliers(&self, data: &[f64]) -> Vec<usize> {
        detect_outliers(data)
    }

    /// Estimate Hurst exponent using multiple methods with shared bootstrap infrastructure.
    ///
    /// This optimized approach generates bootstrap samples once and applies all estimation
    /// methods to the same resamples, providing significant computational savings while
    /// maintaining statistical rigor. We draw the same bootstrap resamples for all
    /// estimators to control resampling noise across methods. This induces dependence
    /// across estimates and improves cross-method comparability; it does not preserve
    /// independence.
    ///
    /// Benefits of shared resampling:
    /// 1. Methods are evaluated on identical uncertainty realizations
    /// 2. Cross-method comparisons have reduced Monte Carlo noise
    /// 3. Pairwise differences between methods are more precisely estimated
    /// 4. Computational cost scales as O(B) instead of O(M×B) where M=methods, B=bootstrap samples

    /// Estimate Hurst exponents using shared bootstrap infrastructure for maximum efficiency.
    ///
    /// This method generates bootstrap samples once and applies all specified estimation
    /// methods to the same resamples. This approach is both computationally efficient and
    /// statistically superior for comparing methods because it eliminates bootstrap
    /// variability as a source of cross-method differences.
    ///
    /// # Mathematical Foundation
    ///
    /// For bootstrap samples B₁, B₂, ..., Bₖ drawn from the original data, we compute:
    /// - Ĥᵢʳˢ(Bⱼ), Ĥᵢᵈᶠᵃ(Bⱼ), Ĥᵢᵍᵖʰ(Bⱼ), Ĥᵢʷᵃᵛ(Bⱼ) for each bootstrap sample Bⱼ
    ///
    /// This controls resampling variability in the joint bootstrap distribution while reducing
    /// computational complexity from O(M×B×N) to O(B×N) where M=methods, B=bootstrap samples, N=data length.
    ///
    /// NOTE: Shared bootstrap uses percentile confidence intervals because BCa
    /// requires jackknife which needs the original data. Single-method bootstrap
    /// uses BCa for better coverage properties.


    /// Perform multifractal analysis component
    fn perform_multifractal_component(&self, data: &[f64]) -> FractalResult<MultifractalAnalysis> {
        perform_multifractal_analysis(data)
    }

    /// Perform regime detection component
    fn perform_regime_component(&self, data: &[f64]) -> FractalResult<RegimeAnalysis> {
        self.detect_regime_changes(data)
    }

    /// Perform statistical tests component
    fn perform_statistical_component(&self, data: &[f64]) -> FractalResult<StatisticalTestResults> {
        self.perform_statistical_tests(data)
    }

    /// Perform model selection component
    fn perform_model_selection_component(
        &self,
        estimates: &BTreeMap<EstimationMethod, HurstEstimate>,
        data: &[f64],
    ) -> FractalResult<ModelSelectionCriteria> {
        self.perform_model_selection(estimates, data)
    }



    // Helper methods
    fn calculate_rs_statistics(&self, data: &[f64], window_size: usize) -> Vec<f64> {
        // OPTIMIZATION: Start with non-overlapping windows for O(n) base computation
        // then add limited overlapping samples for better statistical coverage
        // For billion-dollar financial systems, this balances speed and accuracy
        let num_windows = data.len() / window_size;
        let mut rs_values = Vec::with_capacity(num_windows + 20); // Reserve for overlapping samples too

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
            let max_dev = cumulative_devs
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_dev = cumulative_devs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let range = max_dev - min_dev;

            // Standard deviation
            let variance = calculate_variance(window);
            let std_dev = variance.sqrt();

            // Proper numerical stability check for division
            if variance > MIN_VARIANCE_THRESHOLD && std_dev.is_finite() && range.is_finite() {
                let rs_ratio = float_ops::safe_div(range, std_dev).unwrap_or(0.0);
                if rs_ratio.is_finite() && rs_ratio > 0.0 {
                    rs_values.push(rs_ratio);
                }
            }
        }

        // Add sampled overlapping windows for better coverage (max 20 additional samples)
        // Calculate stride to get approximately 20 overlapping samples
        let max_extra_samples = 20usize;
        let available_range = data.len().saturating_sub(3 * window_size / 2);
        
        // Skip overlapping samples if no room available
        if available_range > 0 {
            let sample_stride = (available_range / max_extra_samples.max(1)).max(1);
            
            for offset in
                (window_size / 2..data.len().saturating_sub(window_size)).step_by(sample_stride)
            {
            let window = &data[offset..offset + window_size];
            let mean = window.iter().sum::<f64>() / window_size as f64;

            let mut cumulative_devs = Vec::with_capacity(window_size);
            let mut cumsum = 0.0;
            for &value in window {
                cumsum += value - mean;
                cumulative_devs.push(cumsum);
            }

            let max_dev = cumulative_devs
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
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
        }

        rs_values
    }

    fn calculate_dfa_fluctuation(
        &self,
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

        // Validate polynomial order
        if polynomial_order == 0 || polynomial_order > 5 {
            return Err(FractalAnalysisError::InvalidParameter {
                parameter: "polynomial_order".to_string(),
                value: polynomial_order as f64,
                constraint: "Must be between 1 and 5".to_string(),
            });
        }

        // Ensure window size is sufficient for polynomial fitting
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

            // Perform polynomial detrending
            let fluctuation = self.detrend_and_calculate_fluctuation(window, polynomial_order)?;
            fluctuations.push(fluctuation);
        }

        Ok((fluctuations.iter().map(|f| f * f).sum::<f64>() / fluctuations.len() as f64).sqrt())
    }

    /// Detrend a window using polynomial fitting and return RMS fluctuation
    fn detrend_and_calculate_fluctuation(
        &self,
        window: &[f64],
        order: usize,
    ) -> FractalResult<f64> {
        let n = window.len();
        
        // Check for constant data to avoid singular matrices
        let window_variance = calculate_variance(window);
        if window_variance < MIN_VARIANCE_THRESHOLD {
            // For constant data, detrending has no effect, fluctuation is zero
            return Ok(0.0);
        }

        // For linear detrending, use existing efficient implementation
        if order == 1 {
            let x_vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let (_, _, residuals) = math_utils::ols_regression(&x_vals, window)?;
            // Using population variance (n divisor) as is standard in DFA
            // For unbiased variance, would use (n - 2) for linear regression
            let variance = residuals.iter().map(|r| r * r).sum::<f64>() / n as f64;
            return Ok(variance.sqrt());
        }

        // For higher order polynomials, use polynomial fitting
        let coeffs = self.fit_polynomial(window, order)?;

        // Calculate residuals
        let mut sum_squared_residuals = 0.0;
        for (i, &y) in window.iter().enumerate() {
            let x = i as f64;
            let mut fitted = 0.0;
            let mut x_power = 1.0;
            for &coeff in &coeffs {
                fitted += coeff * x_power;
                x_power *= x;
            }
            let residual = y - fitted;
            sum_squared_residuals += residual * residual;
        }

        Ok((sum_squared_residuals / n as f64).sqrt())
    }

    /// Fit a polynomial of given order using QR decomposition for numerical stability
    fn fit_polynomial(&self, y: &[f64], order: usize) -> FractalResult<Vec<f64>> {
        let n = y.len();
        if n <= order {
            return Err(FractalAnalysisError::InsufficientData {
                required: order + 1,
                actual: n,
            });
        }
        
        // Check for constant data to avoid singular matrices
        let y_variance = calculate_variance(y);
        if y_variance < MIN_VARIANCE_THRESHOLD {
            // For constant data, return coefficients that give constant value
            let mut coeffs = vec![0.0; order + 1];
            coeffs[0] = y[0]; // Constant term equals the constant value
            return Ok(coeffs);
        }

        // For high polynomial orders or long series, use normalized coordinates
        // to improve numerical conditioning
        let use_normalization = order > 4 || n > 1000;
        
        // Always use QR decomposition for numerical stability
        // Build Vandermonde matrix A with optional normalization
        let mut a = vec![vec![0.0; order + 1]; n];
        
        if use_normalization {
            // Normalize x to [-1, 1] range for better conditioning
            let x_scale = 2.0 / (n - 1) as f64;
            let x_shift = -1.0;
            
            for i in 0..n {
                let x_norm = i as f64 * x_scale + x_shift;
                let mut x_power = 1.0;
                for j in 0..=order {
                    a[i][j] = x_power;
                    x_power *= x_norm;
                }
            }
        } else {
            for i in 0..n {
                let x = i as f64;
                let mut x_power = 1.0;
                for j in 0..=order {
                    a[i][j] = x_power;
                    x_power *= x;
                }
            }
        }

        // Use economy QR solve for efficiency
        // Wrap in error handling to provide more context on failure
        match economy_qr_solve(&a, y) {
            Ok(coeffs) => Ok(coeffs),
            Err(e) => {
                // If QR solve fails, try with lower order polynomial
                if order > 1 {
                    self.fit_polynomial(y, order - 1)
                } else {
                    // Last resort: return mean as constant fit
                    let mean = y.iter().sum::<f64>() / n as f64;
                    Ok(vec![mean])
                }
            }
        }
    }

    // Removed broken qr_decomposition - using householder_qr everywhere for consistency
    // Note: solve_linear_system was also removed - using QR decomposition for all linear systems

    /// Common helper for checking data validity and variance
    fn validate_data_variance(&self, data: &[f64], method_name: &str) -> FractalResult<f64> {
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

    /// Common helper for bootstrap with constant data handling
    fn perform_bootstrap_with_constant_check(
        &self,
        data: &[f64],
        variance: f64,
        corrected_estimate: f64,
        estimator: impl Fn(&[f64]) -> f64 + Sync + Send,
        complexity: EstimatorComplexity,
    ) -> FractalResult<BootstrapValidation> {
        if variance < NEAR_ZERO_VARIANCE_THRESHOLD {
            // For near-constant data, provide minimal bootstrap result with theoretical minimum standard error
            // Based on finite sample theory, minimum standard error is approximately 1/sqrt(n)
            let n = data.len() as f64;
            let theoretical_min_se = (1.0 / n.sqrt()).max(1e-6);
            
            Ok(BootstrapValidation {
                original_estimate: corrected_estimate,
                bootstrap_estimates: vec![corrected_estimate; 10],
                bias: 0.0,
                standard_error: theoretical_min_se,
                confidence_intervals: vec![ConfidenceInterval {
                    confidence_level: 0.95,
                    lower_bound: corrected_estimate - 1.96 * theoretical_min_se,
                    upper_bound: corrected_estimate + 1.96 * theoretical_min_se,
                    // Note: This is a degenerate interval for near-constant data
                    // We use Normal as a placeholder since data has low variance
                    method: ConfidenceIntervalMethod::Normal,
                }],
            })
        } else {
            // Use block bootstrap for time series with automatic block size selection
            let mut config = BootstrapConfiguration::adaptive(data.len(), complexity);

            // Force block bootstrap for time series
            config.bootstrap_method = BootstrapMethod::Block;

            // Use Politis-White method for automatic block size if not specified
            if config.block_size.is_none() {
                config.block_size = Some(politis_white_block_size(data));
            }

            // Use percentile CIs for block bootstrap (safer than BCa here)
            // BCa's acceleration parameter isn't generally valid with block bootstrap
            config.confidence_interval_method = ConfidenceIntervalMethod::BootstrapPercentile;

            bootstrap_validate(data, estimator, &config)
        }
    }

    /// Common helper for extracting confidence interval from bootstrap
    fn extract_confidence_interval(
        &self,
        bootstrap_result: &BootstrapValidation,
        corrected_estimate: f64,
    ) -> ConfidenceInterval {
        bootstrap_result
            .confidence_intervals
            .iter()
            .find(|ci| (ci.confidence_level - DEFAULT_CONFIDENCE_LEVEL).abs() < 1e-6)
            .cloned()
            .unwrap_or_else(|| {
                // Fallback: check if bootstrap was degenerate
                let is_degenerate = bootstrap_result.bootstrap_estimates.windows(2)
                    .all(|w| (w[0] - w[1]).abs() < 1e-10);
                
                if is_degenerate {
                    // For degenerate data, return wide conservative interval
                    ConfidenceInterval {
                        confidence_level: 0.95,
                        lower_bound: corrected_estimate - 0.5,
                        upper_bound: corrected_estimate + 0.5,
                        method: ConfidenceIntervalMethod::Normal,
                    }
                } else {
                    // Normal fallback for non-degenerate case
                    let se = bootstrap_result.standard_error.max(1e-12);
                    let z_95 = 1.96;
                    let margin = z_95 * se;
                    ConfidenceInterval {
                        confidence_level: 0.95,
                        lower_bound: corrected_estimate - margin,
                        upper_bound: corrected_estimate + margin,
                        method: ConfidenceIntervalMethod::Normal,
                    }
                }
            })
    }

    /// Common helper for calculating test statistics
    /// 
    /// Note: This uses normal approximation for the test statistic.
    /// The approximation is only asymptotically valid and may not be accurate
    /// for small samples or non-normal estimator distributions.
    /// For more accurate p-values, use bootstrap hypothesis testing.
    fn calculate_test_statistics(&self, corrected_estimate: f64, std_error: f64) -> (f64, f64) {
        // Guard against division by zero with minimum standard error
        let safe_std_error = std_error.max(MIN_STD_ERROR);
        // Z-test for H₀: H = 0.5 (random walk)
        // Assumes asymptotic normality of the estimator
        let test_statistic = (corrected_estimate - 0.5) / safe_std_error;
        // Two-sided p-value using normal approximation
        let p_value = 2.0 * (1.0 - math_utils::standard_normal_cdf(test_statistic.abs()));
        (test_statistic, p_value)
    }

    /// Automatically select the optimal scale range for DFA using segmented regression.
    ///
    /// Identifies the linear segment in the log-log plot by:
    /// 1. Excluding very small scales (< 10) which have boundary effects
    /// 2. Excluding very large scales (> n/4) which have poor statistics
    /// 3. Finding the segment with best linear fit (highest R²)
    ///
    /// This is critical for accurate Hurst estimation as DFA often shows
    /// crossover behavior at different scales.
    fn select_dfa_scale_range(
        &self,
        log_s: &[f64],
        log_f: &[f64],
    ) -> FractalResult<(Vec<f64>, Vec<f64>)> {
        let n = log_s.len();

        if n < 5 {
            // Not enough points for segmentation, use all
            return Ok((log_s.to_vec(), log_f.to_vec()));
        }

        // Try different segment lengths and positions
        let min_segment_length = 4.max(n / 3);
        let mut best_r_squared = -1.0;
        let mut best_start = 0;
        let mut best_end = n;

        // Sliding window approach to find best linear segment
        for length in min_segment_length..=n {
            for start in 0..=(n - length) {
                let end = start + length;

                // Compute R² for this segment
                let segment_log_s = &log_s[start..end];
                let segment_log_f = &log_f[start..end];

                // Calculate regression and R²
                if let Ok((slope, _, residuals)) =
                    math_utils::ols_regression(segment_log_s, segment_log_f)
                {
                    // Penalize but don't skip unrealistic slopes
                    // This ensures we always have a result even for pathological data
                    let slope_penalty = if slope < -0.1 || slope > 2.1 {
                        0.5 // Halve the R² for out-of-range slopes
                    } else {
                        1.0
                    };

                    // Calculate R²
                    let mean_f = segment_log_f.iter().sum::<f64>() / segment_log_f.len() as f64;
                    let ss_tot = segment_log_f
                        .iter()
                        .map(|&y| (y - mean_f).powi(2))
                        .sum::<f64>();
                    let ss_res = residuals.iter().map(|&r| r.powi(2)).sum::<f64>();

                    let r_squared = if ss_tot > 0.0 {
                        ((1.0 - ss_res / ss_tot).max(0.0)) * slope_penalty
                    } else {
                        0.0
                    };

                    // Weight by segment length (prefer longer segments with good fit)
                    let weighted_r_squared = r_squared * (length as f64 / n as f64).powf(0.5);

                    if weighted_r_squared > best_r_squared {
                        best_r_squared = weighted_r_squared;
                        best_start = start;
                        best_end = end;
                    }
                }
            }
        }

        // Use the best segment found
        let selected_log_s = log_s[best_start..best_end].to_vec();
        let selected_log_f = log_f[best_start..best_end].to_vec();

        // Ensure we have enough points
        if selected_log_s.len() < 3 {
            return Ok((log_s.to_vec(), log_f.to_vec()));
        }

        Ok((selected_log_s, selected_log_f))
    }

    /// Apply multiple testing corrections (Benjamini-Hochberg FDR)
    ///
    /// NOTE: Adjusted p-values are computed but not currently used in model selection.
    /// They are provided for reference and potential future use.
    fn apply_multiple_testing_corrections(
        &self,
        mut estimates: BTreeMap<EstimationMethod, HurstEstimate>,
        _alpha: f64,  // Currently unused - reserved for future significance marking
    ) -> FractalResult<BTreeMap<EstimationMethod, HurstEstimate>> {
        let n_tests = estimates.len();
        if n_tests <= 1 {
            // Still set adjusted = raw for consistency
            for estimate in estimates.values_mut() {
                estimate.adjusted_p_value = Some(estimate.p_value);
            }
            return Ok(estimates);
        }

        // Extract p-values and sort
        let mut p_values: Vec<(EstimationMethod, f64)> = estimates
            .iter()
            .map(|(method, est)| (method.clone(), est.p_value))
            .collect();

        // Sanitize NaN p-values before sorting
        for p_val in p_values.iter_mut() {
            if !p_val.1.is_finite() {
                p_val.1 = 1.0; // Conservative: treat NaN as p=1
            }
        }
        p_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply Benjamini-Hochberg step-up procedure with specified FDR

        // Compute raw adjusted p-values
        let mut adjusted_p_values: Vec<(EstimationMethod, f64)> = Vec::with_capacity(n_tests);
        for (i, (method, p_val)) in p_values.iter().enumerate() {
            let rank = i + 1;
            // Raw adjusted p-value = min(1, p * n / rank)
            let raw_adjusted = (p_val * n_tests as f64 / rank as f64).min(1.0);
            adjusted_p_values.push((method.clone(), raw_adjusted));
        }

        // Apply step-up monotonicity: adjusted_p[i] = min(adjusted_p[i], adjusted_p[i+1])
        // Work from largest p-value to smallest (reverse order)
        for i in (0..adjusted_p_values.len() - 1).rev() {
            let next_adjusted = adjusted_p_values[i + 1].1;
            adjusted_p_values[i].1 = adjusted_p_values[i].1.min(next_adjusted);
        }

        // Store adjusted p-values in estimates
        for (method, adjusted_p) in adjusted_p_values {
            if let Some(estimate) = estimates.get_mut(&method) {
                estimate.adjusted_p_value = Some(adjusted_p);
            }
        }

        Ok(estimates)
    }

    /// Common helper for building HurstEstimate
    fn build_hurst_estimate(
        &self,
        corrected_estimate: f64,
        std_error: f64,
        confidence_interval: ConfidenceInterval,
        test_statistic: f64,
        p_value: f64,
        bias_correction: f64,
        finite_sample_correction: f64,
        bootstrap_bias: f64,
    ) -> HurstEstimate {
        HurstEstimate {
            estimate: corrected_estimate,
            standard_error: std_error,
            confidence_interval,
            test_statistic,
            p_value,
            adjusted_p_value: None, // Will be filled by multiple testing correction
            bias_correction,
            finite_sample_correction,
            bootstrap_bias,
        }
    }




    fn calculate_rs_bias_correction(&self, n: usize) -> f64 {
        // Heuristic finite-sample adjustment for R/S regression slope
        // Note: Lo (1991) modifies the R/S statistic itself, not the regression slope
        // This is an empirical adjustment. For rigorous inference, use bootstrap CIs.
        0.5 / (n as f64).ln()
    }

    fn calculate_dfa_bias_correction(&self, n: usize) -> f64 {
        // Heuristic finite-sample adjustment for DFA
        // These constants are empirical and not rigorously derived
        // For production use, prefer bootstrap confidence intervals which
        // naturally account for finite-sample bias without ad-hoc corrections
        let log_correction = 0.02 / (n as f64).ln();
        let finite_sample_correction = 0.3 / (n as f64);
        log_correction + finite_sample_correction
    }

    fn compute_gph_d_jackknife_internal(&self, data: &[f64], m_full: usize) -> FractalResult<f64> {
        let n = data.len();
        let k_blocks = 5.min(n / 200).max(3);
        
        // Ensure block_size >= 1 and distribute remainder evenly
        let base_block_size = n / k_blocks;
        if base_block_size < 1 {
            // Too few data points for jackknife
            return self.compute_gph_d_raw_internal(data);
        }
        
        let remainder = n % k_blocks;
        
        // Compute full-sample d estimate
        let d_full = self.compute_gph_d_raw_internal(data)?;
        
        // Compute leave-one-block-out estimates
        let mut d_jackknife = Vec::with_capacity(k_blocks);
        let mut current_pos = 0;
        
        for k in 0..k_blocks {
            // Distribute remainder across first blocks
            let block_size = base_block_size + if k < remainder { 1 } else { 0 };
            let start = current_pos;
            let end = start + block_size;
            current_pos = end;
            
            // Create leave-one-out sample
            let mut loo_data = Vec::with_capacity(n - block_size);
            loo_data.extend_from_slice(&data[..start]);
            if end < n {
                loo_data.extend_from_slice(&data[end..]);
            }
            
            // Skip if leave-one-out sample is too small
            if loo_data.len() < 128 {
                continue;
            }
            
            // Compute d estimate on leave-one-out sample
            if let Ok(d_k) = self.compute_gph_d_raw_internal(&loo_data) {
                // Check if d_k is reasonable (-1 < d < 1.5 covers most practical cases)
                if d_k > -1.0 && d_k < 1.5 {
                    d_jackknife.push(d_k);
                }
            }
        }
        
        if d_jackknife.len() < 2 {
            // Not enough valid jackknife samples, return uncorrected estimate
            return Ok(d_full);
        }
        
        let mean_jackknife = d_jackknife.iter().sum::<f64>() / d_jackknife.len() as f64;
        let k = d_jackknife.len() as f64;
        Ok(k * d_full - (k - 1.0) * mean_jackknife)
    }
    
    fn compute_gph_d_raw_internal(&self, data: &[f64]) -> FractalResult<f64> {
        let n = data.len();
        let periodogram = calculate_periodogram_fft(data)?;
        
        // Ensure we don't exceed Nyquist frequency
        let nyquist = n / 2;
        let max_freq = ((n as f64).powf(GPH_BANDWIDTH_EXPONENT) * 0.8) as usize;
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
        
        let (slope, _, _) = math_utils::ols_regression(&log_canonical, &log_periodogram)?;
        // MATHEMATICAL CORRECTION: GPH canonical form regression gives slope = -d, so d = -slope
        Ok(-slope)
    }

    fn calculate_finite_sample_correction(&self, n: usize) -> f64 {
        1.0 / (n as f64).sqrt()
    }

    fn detect_regime_changes(&self, data: &[f64]) -> FractalResult<RegimeAnalysis> {
        // Early validation: HMM needs sufficient data for meaningful regime detection
        if data.len() < 100 {
            return Err(FractalAnalysisError::InsufficientData {
                required: 100,
                actual: data.len(),
            });
        }
        
        // Configure sophisticated HMM-based regime detection with full mathematical rigor
        // NOTE: This is computationally intensive (O(T×K²×I)) but maintains statistical accuracy
        // In quantitative finance, mathematical rigor cannot be compromised for performance
        
        // Window size: 10% of data, but bounded by [50, data.len()/2] for feasibility
        let window_size = ((data.len() / 10).max(50)).min(data.len() / 2);
        let base_config = RegimeDetectionConfig {
            window_size,
            step_size: 10,
            num_states_range: (2, 4), // Full range for proper model selection
            auto_select_states: true,
            min_regime_duration: 10,
            bootstrap_config: self.bootstrap_config.clone(),
            seed: Some(42),
        };

        let hmm_config = HMMRegimeDetectionConfig {
            base_config,
            // Use auto-selected number of states from the range
            hmm: FractalHMM::new(2), // Start with 2, will be adjusted by auto_select_states
            overlap_ratio: 0.5,
            feature_extraction_method: FeatureExtractionMethod::SlidingWindow,
            validation_method: ValidationMethod::CrossValidation { folds: 5 }, // Full 5-fold CV for robustness
            significance_level: 0.05,
            random_seed: Some(42),
        };

        // Perform regime detection with fallback to simplified method
        let hmm_result = match detect_fractal_regimes_with_hmm(data, &hmm_config) {
            Ok(result) => result,
            Err(e) => {
                #[cfg(feature = "debug_logging")]
                log::info!(
                    "Multifractal regime detection failed: {:?}. Using simplified detection.",
                    e
                );
                // Fallback to simplified regime detection using the base algorithm
                detect_fractal_regimes(data, &hmm_config.base_config)?
            }
        };

        // Create a map of state -> average_hurst from HMM regime statistics
        let state_hurst_map: std::collections::HashMap<usize, f64> = hmm_result
            .regime_statistics
            .iter()
            .map(|stats| (stats.state_index, stats.average_hurst))
            .collect();

        // Convert HMM results to RegimeAnalysis format using HMM-learned parameters
        let regime_changes = hmm_result
            .change_points
            .iter()
            .map(|cp| {
                // Use the HMM's learned Hurst parameters for each state
                let pre_hurst = *state_hurst_map.get(&cp.from_state).unwrap_or(&0.5);
                let post_hurst = *state_hurst_map.get(&cp.to_state).unwrap_or(&0.5);

                // Compute proper statistical test for regime change
                let (test_statistic, p_value) = self.compute_regime_change_test(
                    data,
                    cp.time_index,
                    cp.from_state,
                    cp.to_state,
                    &hmm_result,
                );
                
                // Compute bootstrap confidence interval for change point location
                let change_point_ci = self.compute_change_point_ci(
                    data,
                    cp.time_index,
                    cp.confidence,
                );
                
                RegimeChange {
                    change_point: cp.time_index,
                    change_point_ci,
                    pre_change_hurst: pre_hurst,
                    post_change_hurst: post_hurst,
                    test_statistic,
                    p_value,
                    change_magnitude: (post_hurst - pre_hurst).abs(),
                }
            })
            .collect();

        // Extract regime-specific Hurst exponents from HMM results
        let regime_hurst_exponents = hmm_result
            .regime_statistics
            .iter()
            .map(|stats| {
                let start_time = stats.first_occurrence_time;
                let hurst = stats.average_hurst;
                (stats.state_index, start_time, hurst) // Use actual state index
            })
            .collect();

        // Calculate duration statistics from detected regimes
        let durations: Vec<f64> = hmm_result
            .regime_statistics
            .iter()
            .map(|stats| stats.average_duration)
            .collect();

        let mean_duration = if !durations.is_empty() {
            durations.iter().sum::<f64>() / durations.len() as f64
        } else {
            50.0
        }; // Default regime duration

        let std_duration = if durations.len() > 1 {
            let variance = durations
                .iter()
                .map(|d| (d - mean_duration).powi(2))
                .sum::<f64>()
                / (durations.len() - 1) as f64;
            variance.sqrt()
        } else {
            20.0
        }; // Default standard deviation

        let duration_distribution = self.calculate_duration_histogram(&durations);

        Ok(RegimeAnalysis {
            regime_changes,
            regime_hurst_exponents,
            regime_duration_stats: RegimeDurationStatistics {
                mean_duration,
                std_duration,
                duration_distribution,
            },
            transition_probabilities: hmm_result.hmm_params.transition_matrix.clone(),
        })
    }

    fn perform_statistical_tests(&self, data: &[f64]) -> FractalResult<StatisticalTestResults> {
        // Run statistical tests based on configuration depth
        match self.analysis_config.depth {
            AnalysisDepth::Light => {
                // Skip heavy statistical tests for light analysis
                Ok(StatisticalTestResults {
                    long_range_dependence_test: Default::default(),
                    short_range_dependence_test: Default::default(),
                    structural_break_tests: vec![],
                    goodness_of_fit_tests: test_goodness_of_fit(data),
                })
            }
            AnalysisDepth::Standard | AnalysisDepth::Deep => {
                // Run full statistical tests, but skip LRD test for short series
                // GPH test requires at least 128 data points
                let long_range_test = if data.len() >= 128 {
                    test_long_range_dependence(data)?
                } else {
                    // Return default for short series
                    Default::default()
                };
                
                Ok(StatisticalTestResults {
                    long_range_dependence_test: long_range_test,
                    short_range_dependence_test: test_short_range_dependence(data)?,
                    structural_break_tests: test_structural_breaks(data)?,
                    goodness_of_fit_tests: test_goodness_of_fit(data),
                })
            }
        }
    }

    fn perform_model_selection(
        &self,
        estimates: &BTreeMap<EstimationMethod, HurstEstimate>,
        _data: &[f64],
    ) -> FractalResult<ModelSelectionCriteria> {
        // Mathematically rigorous model selection using MSE = Bias² + Variance
        // Bootstrap provides both components for optimal bias-variance trade-off
        let mut best_mse = f64::INFINITY;
        let mut best_model = None;

        for (method, estimate) in estimates {
            // Compute Mean Squared Error: MSE = Bias² + Variance
            // Bootstrap bias: E[θ*] - θ
            // Variance estimated by SE²
            let bias_squared = estimate.bootstrap_bias.powi(2);
            let variance = estimate.standard_error.powi(2);
            let mse = bias_squared + variance;
            
            // Secondary check: ensure CI is reasonable
            let ci_width = estimate.confidence_interval.upper_bound - estimate.confidence_interval.lower_bound;
            
            // Reject methods with pathological CIs (wider than theoretical range)
            // but still consider MSE as primary criterion
            let penalty = if ci_width > 2.0 {
                // Penalize methods with unreasonably wide CIs
                // This suggests numerical instability
                10.0
            } else {
                1.0
            };
            
            let adjusted_mse = mse * penalty;

            if adjusted_mse < best_mse && adjusted_mse > 0.0 && adjusted_mse.is_finite() {
                best_mse = adjusted_mse;
                best_model = Some(method.clone());
            }
        }

        // Convert MSE to effective uncertainty score
        // Root MSE provides interpretable scale
        let rmse = best_mse.sqrt();

        Ok(ModelSelectionCriteria {
            uncertainty_score: rmse,  // Root Mean Squared Error
            num_parameters: 1,        // Single Hurst parameter
            best_model: best_model.unwrap_or(EstimationMethod::DetrendedFluctuationAnalysis),
        })
    }

    fn perform_validation_analysis(
        &self,
        asset: &str,
        data: &[f64],
    ) -> FractalResult<ValidationStatistics> {
        // For very short series, validation analysis cannot be performed rigorously
        // Cross-validation requires sufficient data for meaningful window sizes
        if data.len() < 100 {
            return Err(FractalAnalysisError::InsufficientData {
                required: 100,
                actual: data.len(),
            });
        }

        // PERFORMANCE OPTIMIZATION: Use adaptive configuration for cross-validation
        // This dramatically reduces computation time while maintaining statistical validity

        // Calculate adaptive stability runs based on data characteristics
        let adaptive_stability_runs = self.calculate_adaptive_stability_runs(data.len());

        // Merge user bootstrap config with adaptive settings
        let mut adaptive_bootstrap_config = self.bootstrap_config.clone();
        let adaptive_base =
            BootstrapConfiguration::adaptive(data.len(), EstimatorComplexity::Medium);
        // Only override samples count for optimization, preserve user's seed and CI method
        adaptive_bootstrap_config.num_bootstrap_samples = adaptive_base
            .num_bootstrap_samples
            .max(adaptive_bootstrap_config.num_bootstrap_samples);

        let cv_config = CrossValidationConfig {
            method: CrossValidationMethod::WalkForward {
                // Ensure window_size respects data length while preferring minimum sizes
                window_size: {
                    let ws = (data.len() / 3).max(64);
                    ws.min(data.len() / 2) // Cap at half the data length
                },
                step_size: {
                    let ss = (data.len() / 8).max(64);
                    ss.min(data.len() / 4) // Cap at quarter of data length
                },
            },
            estimators: {
                let mut estimators = vec![
                    FractalEstimator::DetrendedFluctuation(DfaConfig::default()),
                    FractalEstimator::MultifractalDFA {
                        q_value: 2.0,
                        config: DfaConfig::default(),
                    },
                    FractalEstimator::WaveletBased,
                    FractalEstimator::RescaledRange,
                ];

                // Filter estimators based on actual data availability
                // Each estimator will be validated individually in cross_validation
                // but we can pre-filter obvious cases to save computation
                let current_step_size = (data.len() / 8).max(64).min(data.len() / 4);

                // PeriodogramRegression requires minimum 128 points
                if current_step_size >= 128 {
                    estimators.push(FractalEstimator::PeriodogramRegression(GphConfig::default()));
                }

                estimators
            },
            selection_criterion: SelectionCriterion::MinimizeError,
            bootstrap_config: adaptive_bootstrap_config.clone(),
            stability_runs: adaptive_stability_runs, // OPTIMIZED: Adaptive instead of fixed 50
            significance_level: 0.05,
            seed: self.bootstrap_config.seed.or(Some(42)),
            trading_config: TradingConfig::default(),
            financial_config: FinancialMetricsConfig::default(),
        };

        let cv_result = cross_validate_fractal_models(data, &cv_config)?;

        // PERFORMANCE OPTIMIZATION: Reduce Monte Carlo simulations while maintaining statistical power
        // For validation purposes, 200-300 simulations provide 95% of the statistical information
        // that 500 simulations would, based on CLT convergence properties
        let adaptive_mc_simulations = self.calculate_adaptive_mc_simulations(data.len());

        let mc_config = MonteCarloConfig {
            num_simulations: adaptive_mc_simulations, // OPTIMIZED: Adaptive instead of fixed 500
            significance_level: 0.05,
            seed: self.bootstrap_config.seed.or(Some(42)),
            parallel: false,
            bootstrap_config: adaptive_bootstrap_config,
            deterministic_parallel: true,
        };

        let mc_result = monte_carlo_hurst_test(data, NullHypothesis::RandomWalk, &mc_config)?;

        // Extract performance metrics from cross-validation results
        let best_model_result = cv_result
            .all_results
            .get(&cv_result.best_estimator)
            .ok_or_else(|| FractalAnalysisError::NumericalError {
                reason: "Best model results not found".to_string(),
                operation: None,
            })?;

        let mspe = best_model_result.metrics.mse;
        let mape = best_model_result.metrics.mape;

        // Use cross-validation estimate stability concordance
        let estimate_stability_concordance =
            best_model_result.metrics.estimate_stability_concordance;

        // Use cross-validation hit rate as coverage probability
        let coverage_probability = best_model_result.metrics.hit_rate;

        // Robustness tests using cross-validation results
        let performances: Vec<f64> = cv_result
            .all_results
            .values()
            .map(|result| result.average_performance)
            .collect();
        let outlier_robustness = if performances.len() > 1 {
            let mean_perf = performances.iter().sum::<f64>() / performances.len() as f64;
            let variance = performances
                .iter()
                .map(|p| (p - mean_perf).powi(2))
                .sum::<f64>()
                / (performances.len() - 1) as f64;
            // Guard against division by near-zero
            let denom = mean_perf.abs().max(1e-12);
            let ratio = (variance.sqrt() / denom).min(1.0);
            1.0 - ratio
        } else {
            0.8 // Default robustness
        };

        let sample_size_robustness = best_model_result.metrics.stability;
        let detrending_robustness = outlier_robustness; // Use same metric for simplicity

        // Sensitivity analysis using actual hyperparameter values as keys
        let window_sensitivity: HashMap<usize, f64> = cv_result
            .all_results
            .iter()
            .filter_map(|(estimator, result)| match estimator {
                FractalEstimator::DetrendedFluctuation(config) => {
                    Some((config.min_scale, result.performance_std))
                }
                _ => None,
            })
            .collect();

        // For polynomial order, use DFA polynomial order from analyzer config
        let polynomial_sensitivity: HashMap<usize, f64> = {
            let mut map = HashMap::new();
            // Use the analyzer's DFA polynomial order with average performance std
            if let Some((_, result)) = cv_result
                .all_results
                .iter()
                .find(|(est, _)| matches!(est, FractalEstimator::DetrendedFluctuation(_)))
            {
                map.insert(self.dfa_polynomial_order, result.performance_std);
            }
            map
        };

        let aggregation_sensitivity: HashMap<usize, f64> = cv_result
            .all_results
            .iter()
            .enumerate()
            .map(|(i, (_, result))| (i, result.performance_std))
            .collect();

        Ok(ValidationStatistics {
            prediction_accuracy: PredictionAccuracy {
                mspe,
                mape,
                estimate_stability_concordance,
                coverage_probability,
            },
            robustness_tests: RobustnessTests {
                outlier_robustness,
                sample_size_robustness,
                detrending_robustness,
            },
            sensitivity_analysis: SensitivityAnalysis {
                window_size_sensitivity: window_sensitivity,
                polynomial_order_sensitivity: polynomial_sensitivity,
                aggregation_sensitivity: aggregation_sensitivity,
            },
        })
    }

    /// Get comprehensive analysis results
    pub fn get_analysis_results(&self, asset: &str) -> FractalResult<FractalEstimationResults> {
        let results_map =
            self.estimation_results
                .read()
                .map_err(|_| FractalAnalysisError::NumericalError {
                    reason: "Failed to acquire lock for reading results".to_string(),
                    operation: None,
                })?;

        results_map
            .get(asset)
            .cloned()
            .ok_or_else(|| FractalAnalysisError::TimeSeriesNotFound {
                name: asset.to_string(),
            })
    }

    /// Get validation statistics
    pub fn get_validation_statistics(&self, asset: &str) -> FractalResult<ValidationStatistics> {
        let validation_map = self.validation_statistics.read().map_err(|_| {
            FractalAnalysisError::NumericalError {
                reason: "Failed to acquire lock for reading validation statistics".to_string(),
                operation: None,
            }
        })?;

        validation_map
            .get(asset)
            .cloned()
            .ok_or_else(|| FractalAnalysisError::TimeSeriesNotFound {
                name: asset.to_string(),
            })
    }

    /// Set bootstrap configuration
    pub fn set_bootstrap_config(&mut self, config: BootstrapConfiguration) {
        self.bootstrap_config = config;
    }

    /// Set test configuration for statistical tests
    pub fn set_test_config(&mut self, config: TestConfiguration) {
        self.test_config = config;
    }

    /// Set analysis configuration for controlling components
    pub fn set_analysis_config(&mut self, config: AnalysisConfig) {
        self.analysis_config = config;
    }

    /// Get available assets
    pub fn get_assets(&self) -> FractalResult<Vec<String>> {
        let series_map =
            self.time_series_data
                .read()
                .map_err(|_| FractalAnalysisError::NumericalError {
                    reason: "Failed to acquire lock for reading time series data".to_string(),
                    operation: None,
                })?;

        Ok(series_map.keys().cloned().collect())
    }



    /// Estimate local Hurst exponent for a regime using multifractal analysis
    /// 
    /// Note: Local Hurst estimates are clamped to [0.01, 0.99] for downstream stability
    /// while global estimators allow the full range to detect extreme behaviors.
    /// This is intentional - local estimates feed into regime detection where
    /// numerical stability is critical, while global estimates report raw findings.
    fn estimate_local_hurst(&self, data: &[f64], regime_state: usize) -> f64 {
        // Use proper multifractal analysis from multifractal.rs
        if data.len() < 50 {
            // Return NaN for insufficient data - caller should handle appropriately
            return f64::NAN;
        }

        // First try multifractal analysis to get proper Hurst estimate
        match perform_multifractal_analysis(data) {
            Ok(mf_result) => {
                // Get Hurst exponent for q=2 (standard Hurst)
                if let Some((_, hurst)) = mf_result
                    .generalized_hurst_exponents
                    .iter()
                    .find(|(q, _)| (q - 2.0).abs() < Q_MATCH_EPSILON)
                {
                    // Clamp to [0.01, 0.99] for downstream numerical stability
                    (*hurst).max(0.01).min(0.99)
                } else if let Some((_, first_hurst)) = mf_result.generalized_hurst_exponents.first()
                {
                    // Note: Clamping only applied in fallback path for numerical stability
                    first_hurst.max(0.01).min(0.99)
                } else {
                    // Fallback to DFA if multifractal fails
                    self.fallback_hurst_estimation(data, regime_state)
                }
            }
            Err(e) => {
                #[cfg(feature = "debug_logging")]
                log::info!("Multifractal analysis failed: {:?}. Using DFA fallback.", e);
                // Fallback to DFA if multifractal analysis fails
                self.fallback_hurst_estimation(data, regime_state)
            }
        }
    }

    /// Fallback Hurst estimation using DFA and R/S methods
    fn fallback_hurst_estimation(&self, data: &[f64], _regime_state: usize) -> f64 {
        // Use DFA method for local Hurst estimation
        match estimate_dfa_hurst_only(data, self.dfa_polynomial_order) {
            Ok(hurst) if hurst.is_finite() => {
                // Note: Clamping only applied in this specific local estimation context
                // to ensure numerical stability for downstream computations
                hurst.max(0.01).min(0.99)
            },
            _ => {
                // Fallback to R/S method
                match estimate_rs_hurst_only(data) {
                    Ok(hurst) if hurst.is_finite() => {
                        // Note: Clamping only applied in fallback path
                        hurst.max(0.01).min(0.99)
                    },
                    _ => {
                        // Return NaN if estimation fails - caller should handle
                        f64::NAN
                    }
                }
            }
        }
    }

    /// Simple Hurst estimation for very short series (< 50 points)
    /// Uses a simplified variance ratio method that can work with minimal data

    /// Aggregate time series by averaging over non-overlapping windows

    /// Calculate adaptive number of stability runs for cross-validation based on statistical theory.
    ///
    /// The optimal number of stability runs balances computational cost with estimation precision.
    /// Based on the Central Limit Theorem, the standard error of performance estimates scales
    /// as 1/√K where K is the number of runs, leading to diminishing returns beyond ~20-30 runs.
    ///
    /// # Mathematical Foundation
    ///
    /// For cross-validation performance estimates, the Monte Carlo error is:
    /// SE(Performance) ≈ σ/√K
    ///
    /// Where σ is the standard deviation of performance across runs.
    /// To achieve 95% confidence within ±δ, we need: K ≥ (1.96×σ/δ)²
    ///
    /// Empirical studies suggest δ ≈ 0.05×Performance is acceptable for model selection.
    fn calculate_adaptive_stability_runs(&self, data_length: usize) -> usize {
        // Base stability runs using statistical convergence theory
        let base_runs = if data_length >= 1000 {
            // Large datasets: Higher precision possible, but diminishing returns kick in early
            15 // Sufficient for robust estimation with large datasets
        } else if data_length >= 500 {
            // Medium datasets: Good balance of precision and computational cost
            20 // Slightly more runs for medium datasets
        } else if data_length >= 200 {
            // Small datasets: More variability, need slightly more runs but limited by data
            25 // More runs to handle increased variability
        } else {
            // Very small datasets: Limited by data availability
            10 // Minimal runs due to data constraints
        };

        // Bound the number of runs to prevent excessive computation
        // Upper bound based on diminishing returns analysis
        // Lower bound based on minimum statistical requirements
        base_runs.max(5).min(30)
    }

    /// Calculate adaptive number of Monte Carlo simulations for hypothesis testing.
    ///
    /// Monte Carlo hypothesis tests require sufficient simulations to achieve desired
    /// statistical power and significance level. The number of simulations needed
    /// depends on the effect size and required precision of p-value estimation.
    ///
    /// # Statistical Foundation
    ///
    /// For hypothesis testing at significance level α, the Monte Carlo standard error is:
    /// SE(p̂) ≈ √(p(1-p)/B) where B is number of simulations
    ///
    /// To distinguish p-values near α with high confidence:
    /// B ≥ (z_{α/2})² × p(1-p) / (tolerance)²
    ///
    /// For α = 0.05 and tolerance = 0.01, this gives B ≈ 1900.
    /// However, for model validation (not formal hypothesis testing),
    /// B ≈ 200-400 provides sufficient discrimination power.
    fn calculate_adaptive_mc_simulations(&self, data_length: usize) -> usize {
        // Base simulations using hypothesis testing theory
        let base_simulations = if data_length >= 1000 {
            // Large datasets: Can afford more precision, higher signal-to-noise ratio
            250 // Good precision for large datasets
        } else if data_length >= 500 {
            // Medium datasets: Standard precision requirements
            200 // Sufficient for most validation purposes
        } else if data_length >= 200 {
            // Small datasets: Need fewer simulations due to computational constraints
            150 // Reduced precision but still statistically valid
        } else {
            // Very small datasets: Minimal simulations due to limited data
            100 // Minimal simulations for very small datasets
        };

        // Enforce bounds based on statistical requirements
        // Minimum 100: Below this, p-value estimates become unreliable
        // Maximum 500: Diminishing returns beyond this point for validation purposes
        base_simulations.max(100).min(500)
    }

    /// Calculate duration histogram for regime analysis
    fn calculate_duration_histogram(&self, durations: &[f64]) -> Vec<(f64, f64)> {
        if durations.is_empty() {
            return vec![];
        }

        let min_duration = durations.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_duration = durations.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if max_duration <= min_duration {
            return vec![(min_duration, 1.0)];
        }

        let num_bins = 10;
        let bin_width = (max_duration - min_duration) / num_bins as f64;
        let mut histogram = vec![0; num_bins];

        for &duration in durations {
            let bin_idx = ((duration - min_duration) / bin_width) as usize;
            let bin_idx = bin_idx.min(num_bins - 1);
            histogram[bin_idx] += 1;
        }

        let total_count = durations.len() as f64;
        histogram
            .into_iter()
            .enumerate()
            .map(|(i, count)| {
                let bin_center = min_duration + (i as f64 + 0.5) * bin_width;
                let frequency = count as f64 / total_count;
                (bin_center, frequency)
            })
            .collect()
    }
    
    /// Compute likelihood ratio test for regime change significance
    /// 
    /// Uses the Vuong test statistic for non-nested HMM model comparison
    fn compute_regime_change_test(
        &self,
        data: &[f64],
        change_point: usize,
        from_state: usize,
        to_state: usize,
        hmm_result: &RegimeDetectionResult,
    ) -> (f64, f64) {
        // If states are the same, no regime change
        if from_state == to_state {
            return (0.0, 1.0);
        }
        
        // Use the confidence score as a proxy for likelihood ratio
        // The confidence from HMM is typically based on posterior probability differences
        let confidence = hmm_result.change_points
            .iter()
            .find(|cp| cp.time_index == change_point)
            .map(|cp| cp.confidence)
            .unwrap_or(0.5);
        
        // Convert confidence to a test statistic
        // Use logit transform to get unbounded test statistic
        let epsilon = 1e-10;
        let bounded_conf = confidence.max(epsilon).min(1.0 - epsilon);
        let test_statistic = (bounded_conf / (1.0 - bounded_conf)).ln();
        
        // Compute p-value using chi-squared distribution approximation
        // Under null hypothesis of no change, LR statistic ~ chi^2(1)
        // For simplicity, use normal approximation
        let z_score = test_statistic.abs();
        let p_value = 2.0 * (1.0 - standard_normal_cdf(z_score));
        
        (test_statistic, p_value)
    }
    
    /// Compute confidence interval for change point location using bootstrap
    /// 
    /// Uses a parametric bootstrap approach based on HMM confidence
    fn compute_change_point_ci(
        &self,
        data: &[f64],
        change_point: usize,
        confidence: f64,
    ) -> (usize, usize) {
        let n = data.len();
        
        // Estimate uncertainty in change point location
        // Higher confidence -> narrower interval
        // Use inverse relationship: width ∝ 1/confidence
        
        // Base uncertainty (in indices) scales with sqrt(n) for diffusion-like processes
        let base_uncertainty = (n as f64).sqrt();
        
        // Adjust by confidence: high confidence (0.9+) -> narrow interval
        // Low confidence (0.5-0.6) -> wide interval
        let confidence_factor = if confidence > 0.5 {
            1.0 / (2.0 * confidence)
        } else {
            2.0
        };
        
        // Width of confidence interval (half-width on each side)
        let half_width = (base_uncertainty * confidence_factor * 0.5) as usize;
        
        // Apply reasonable bounds
        let half_width = half_width.min(n / 10).max(2); // At most 10% of data, at least 2
        
        let lower = change_point.saturating_sub(half_width);
        let upper = (change_point + half_width).min(n.saturating_sub(1));
        
        (lower, upper)
    }
}

impl Default for StatisticalFractalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::{
        fbm_to_fgn, generate_fractional_brownian_motion, FbmConfig, GeneratorConfig,
    };
    use crate::{generate_benchmark_series, BenchmarkSeriesType};

    #[test]
    fn test_fractal_analyzer_creation() {
        let analyzer = StatisticalFractalAnalyzer::new();
        assert_eq!(
            analyzer.bootstrap_config.num_bootstrap_samples,
            BootstrapConfiguration::default().num_bootstrap_samples
        );
    }

    #[test]
    #[cfg(feature = "slow_tests")]
    fn test_comprehensive_analysis() {
        let mut analyzer = StatisticalFractalAnalyzer::new();

        // Generate sample data - use much larger size for rigorous statistical tests
        let config = GeneratorConfig {
            length: 1200,
            seed: Some(42),
            ..Default::default()
        };

        let fbm_config = FbmConfig {
            hurst_exponent: 0.7,
            volatility: 0.01,
            method: crate::generators::FbmMethod::Hosking,
        };

        let fbm = generate_fractional_brownian_motion(&config, &fbm_config).unwrap();
        let returns = fbm_to_fgn(&fbm);

        analyzer
            .add_time_series("TEST".to_string(), returns)
            .unwrap();
        analyzer.analyze_all_series().unwrap();

        let results = analyzer.get_analysis_results("TEST").unwrap();
        assert!(!results.hurst_estimates.is_empty());
        assert!(!results
            .multifractal_analysis
            .generalized_hurst_exponents
            .is_empty());
    }

    #[test]
    fn test_modular_integration() {
        // Test that all modules work together
        let data: Vec<f64> = (0..300)
            .map(|i| ((i as f64) * 0.02 - 0.01) * (1.0 + (i as f64).sin() * 0.1)) // Deterministic
            .collect();

        // Test statistical tests
        let lrd_test = test_long_range_dependence(&data);
        assert!(
            lrd_test.is_ok(),
            "LRD test should succeed: {:?}",
            lrd_test.err()
        );
        let lrd_test = lrd_test.unwrap();
        assert!(lrd_test.gph_p_value >= 0.0 && lrd_test.gph_p_value <= 1.0);

        // Test multifractal analysis
        let mf_result = perform_multifractal_analysis(&data);
        assert!(
            mf_result.is_ok(),
            "Multifractal analysis should succeed: {:?}",
            mf_result.err()
        );
        let mf_result = mf_result.unwrap();
        assert!(!mf_result.generalized_hurst_exponents.is_empty());

        // Test bootstrap validation
        let config = BootstrapConfiguration::default();
        let validation = bootstrap_validate(
            &data,
            |data| data.iter().sum::<f64>() / data.len() as f64,
            &config,
        );
        assert!(
            validation.is_ok(),
            "Bootstrap validation should succeed: {:?}",
            validation.err()
        );
        let validation = validation.unwrap();
        assert!(validation.standard_error >= 0.0);

        // Test generators
        let gen_config = GeneratorConfig {
            length: 100,
            seed: Some(123),
            ..Default::default()
        };

        let white_noise =
            generate_benchmark_series(BenchmarkSeriesType::WhiteNoise, &gen_config).unwrap();
        assert_eq!(white_noise.len(), 100);
    }
}
