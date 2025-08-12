//! # Financial Fractal Analysis
//!
//! Enterprise-grade rigorous fractal analysis for quantitative finance applications.
//!
//! This crate provides statistically rigorous tools for analyzing long-range dependence,
//! multifractality, and regime changes in financial time series. All estimators include
//! confidence intervals, with optional finite-sample adjustments. Statistical inference uses
//! asymptotic theory and bootstrap methods with appropriate caveats for small samples and heavy-tailed data.
//!
//! ## Key Features
//!
//! - **Statistical Rigor**: All estimators include confidence intervals; optional bias-reduction heuristics are provided
//! - **Multiple Methods**: Hurst exponent estimation via R/S, DFA, GPH, and wavelet methods
//! - **Multifractal Analysis**: Complete MF-DFA implementation with singularity spectrum
//! - **Regime Detection**: HMM-based detection of structural breaks and fractal regimes
//! - **Bootstrap Validation**: Comprehensive resampling and statistical validation
//! - **Cross-Validation**: Walk-forward validation and model selection
//! - **Monte Carlo Testing**: Hypothesis testing with surrogate data methods
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use fractal_finance::{StatisticalFractalAnalyzer, EstimationMethod};
//!
//! let mut analyzer = StatisticalFractalAnalyzer::new();
//!
//! // Add your financial time series data
//! let returns = vec![0.01, -0.02, 0.015, -0.008, 0.012]; // Your data here
//! analyzer.add_time_series("ASSET".to_string(), returns);
//!
//! // Perform comprehensive analysis
//! analyzer.analyze_all_series().unwrap();
//!
//! // Get results
//! let results = analyzer.get_analysis_results("ASSET").unwrap();
//! for (method, estimate) in &results.hurst_estimates {
//!     println!("{:?}: H = {:.3} ± {:.3}", method,
//!         estimate.estimate, estimate.standard_error);
//! }
//! ```
//!
//! ## Analysis Methods
//!
//! ### Hurst Exponent Estimation
//! - **Rescaled Range (R/S)**: Classical method with finite-sample adjustment
//! - **Detrended Fluctuation Analysis (DFA)**: Robust to non-stationarity
//! - **GPH Periodogram**: Frequency domain estimation with HAC standard errors
//! - **Wavelet Methods**: Multi-resolution analysis using dyadic scales
//!
//! ### Advanced Analysis
//! - **Multifractal Spectrum**: Complete f(α) characterization
//! - **Regime Detection**: Hidden Markov model for structural breaks
//! - **Statistical Testing**: Comprehensive hypothesis tests for long-range dependence
//!
//! ## Architecture
//!
//! The crate is organized around the [`StatisticalFractalAnalyzer`] which orchestrates
//! all analysis methods and provides a unified interface. Individual analysis methods
//! can also be used directly for specialized applications.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(clippy::all)]

// Core modules
pub mod analyzer;
pub mod audit;
pub mod collections;
pub mod computation_cache;
pub mod config;
pub mod decimal_finance;
pub mod deterministic_config;
pub mod errors;
pub mod results;
pub mod fft_ops;
pub mod hurst_estimators;
pub mod linear_algebra;
pub mod math_utils;
pub mod memory_pool;
pub mod preprocessing;
pub mod secure_rng;

// Analysis methods
pub mod batch_testing;
pub mod block_size;
pub mod bootstrap;
pub mod bootstrap_config;
pub mod bootstrap_sampling;
pub mod confidence_intervals;
pub mod cross_validation;
pub mod diagnostics;
pub mod generators;
pub mod monte_carlo;
pub mod multifractal;
pub mod regime_detection;
// Regime detection sub-modules
pub mod emission_models;
pub mod hmm_core;
pub mod regime_config;
pub mod regime_results;
pub mod statistical_tests;
pub mod wavelet;

// Re-exports for convenience - main public API
pub use analyzer::*;
pub use config::{AnalysisConfig, AnalysisDepth};
pub use errors::{FractalAnalysisError, FractalResult};
pub use results::{
    AssumptionValidation, DataQualityMetrics, FractalEstimationResults,
    HeteroskedasticityTests, MethodValidity, ModelSelectionCriteria, NormalityTests,
    PredictionAccuracy, RegimeAnalysis, RegimeChange, RegimeDurationStatistics,
    RobustnessTests, SensitivityAnalysis, SerialCorrelationTests, StationarityTests,
    StatisticalTestResults, ValidationStatistics,
};

// Hurst estimation exports
pub use hurst_estimators::{
    EstimationMethod, HurstEstimate, HurstEstimationConfig,
    estimate_hurst_multiple_methods, estimate_hurst_by_method,
    estimate_hurst_rescaled_range, estimate_hurst_dfa,
    estimate_hurst_periodogram, estimate_hurst_wavelet,
    estimate_hurst_simple_short_series, estimate_local_hurst,
};

// Statistical test exports
pub use statistical_tests::{
    gph_test, ljung_box_test, ljung_box_test_with_config, portmanteau_test, robinson_test, 
    test_goodness_of_fit, test_long_range_dependence, test_short_range_dependence, 
    test_structural_breaks, GoodnessOfFitTests, LongRangeDependenceTest, ShortRangeDependenceTest, 
    StructuralBreakTest, StructuralBreakTestType, TestConfiguration, TestResult, PValueMethod, 
    LjungBoxDenominator,
};

// Multifractal analysis exports
pub use multifractal::{
    calculate_asymmetry_parameter, calculate_generalized_hurst_exponent,
    calculate_multifractality_degree, calculate_singularity_spectrum,
    fit_quadratic_and_get_residuals, perform_multifractal_analysis,
    perform_multifractal_analysis_with_config, perform_wtmm_analysis,
    perform_wtmm_analysis_with_config, MultifractalAnalysis, MultifractalConfig,
    MultifractalityTest, WtmmAnalysis, WtmmConfig,
};

// Bootstrap validation exports
pub use bootstrap::{
    bootstrap_validate, bootstrap_validate_pairs, calculate_bca_confidence_interval,
    calculate_bootstrap_confidence_interval, calculate_normal_confidence_interval,
    generate_bootstrap_sample, BootstrapConfiguration, BootstrapMethod, BootstrapValidation,
    ConfidenceInterval, ConfidenceIntervalMethod, EstimatorComplexity,
};

// Data generation exports
pub use generators::{
    fbm_to_fgn, generate_arfima, generate_benchmark_series, generate_fractional_brownian_motion,
    generate_multifractal_cascade, generate_regime_switching_series, ArfimaConfig,
    BenchmarkSeriesType, FbmConfig, FbmMethod, GeneratorConfig, MultifractalCascadeConfig,
};

// Monte Carlo testing exports
pub use monte_carlo::{
    fourier_surrogate, get_power_spectrum, monte_carlo_hurst_test, monte_carlo_multifractal_test,
    power_analysis_hurst_estimator, surrogate_data_test, MonteCarloConfig, MonteCarloTestResult,
    NullHypothesis, PowerAnalysisResult, SurrogateMethod,
};

// Cross-validation exports
pub use cross_validation::{
    cross_validate_fractal_models, CrossValidationConfig, CrossValidationMethod,
    CrossValidationResult, FoldResult, FractalEstimator, ModelSelectionResult, PerformanceMetrics,
    SelectionCriterion,
};

// Regime detection exports
pub use regime_detection::{
    detect_fractal_regimes, detect_fractal_regimes_with_hmm, EmissionParameters,
    FeatureExtractionMethod, FractalHMM, HMMParameters, HMMRegimeDetectionConfig, ModelCriteria,
    MultifractalRegimeParams, ObservationFeatures, RegimeChangePoint, RegimeDetectionConfig,
    RegimeDetectionResult, RegimeStatistics, ValidationMethod, VolatilityStatistics,
};

// FFT operations exports
pub use fft_ops::{
    calculate_periodogram_fft, clear_fft_cache, fft_autocorrelation, get_cached_fft_forward,
    get_cached_fft_inverse, get_fft_cache_stats,
};

// Mathematical utilities exports
pub use math_utils::{
    calculate_autocorrelations,
    calculate_segment_fluctuation,
    calculate_variance,
    calculate_wald_statistic,
    erf,
    // Safe arithmetic operations
    float_ops::{
        approx_eq, approx_eq_eps, approx_zero, approx_zero_eps, safe_div, safe_ln, safe_sqrt,
    },
    generate_window_sizes,
    integrate_series,
    local_whittle_estimate,
    // Core mathematical functions
    ols_regression,
    standard_normal_cdf,
};

// Diagnostic utilities exports
pub use diagnostics::{
    analyze_circulant_eigenvalues, deep_gph_debug, diagnose_gph_variance_bias,
    run_comprehensive_diagnostics, test_gph_corrections, validate_gph_correction,
};

// Python module exports (when python feature is enabled)
// Note: The Python module is automatically exported by PyO3's #[pymodule] attribute
