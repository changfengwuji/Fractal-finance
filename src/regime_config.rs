//! Configuration structures for regime detection.
//!
//! This module contains all configuration structures used in HMM-based
//! fractal regime detection for financial time series analysis.

use crate::{
    bootstrap::BootstrapConfiguration,
    hmm_core::FractalHMM,
};

/// Configuration for regime detection
#[derive(Debug, Clone)]
pub struct RegimeDetectionConfig {
    /// Window size for local feature extraction
    pub window_size: usize,
    /// Step size for sliding window
    pub step_size: usize,
    /// Range of number of states to test
    pub num_states_range: (usize, usize),
    /// Use model selection to choose optimal number of states
    pub auto_select_states: bool,
    /// Minimum regime duration (to avoid spurious switches)
    pub min_regime_duration: usize,
    /// Bootstrap configuration for confidence intervals
    pub bootstrap_config: BootstrapConfiguration,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for RegimeDetectionConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            step_size: 10,
            num_states_range: (2, 5),
            auto_select_states: true,
            min_regime_duration: 5,
            bootstrap_config: BootstrapConfiguration::default(),
            seed: None,
        }
    }
}

/// Feature extraction method for regime detection
#[derive(Debug, Clone)]
pub enum FeatureExtractionMethod {
    SlidingWindow,
    NonOverlapping,
    Adaptive,
}

/// Validation method for regime detection
#[derive(Debug, Clone)]
pub enum ValidationMethod {
    CrossValidation { folds: usize },
    HoldOut { test_ratio: f64 },
    Bootstrap { samples: usize },
}

/// Extended HMM regime detection configuration
pub struct HMMRegimeDetectionConfig {
    /// Base regime detection configuration
    pub base_config: RegimeDetectionConfig,
    /// HMM configuration
    pub hmm: FractalHMM,
    /// Overlap ratio for sliding windows
    pub overlap_ratio: f64,
    /// Feature extraction method
    pub feature_extraction_method: FeatureExtractionMethod,
    /// Validation method
    pub validation_method: ValidationMethod,
    /// Significance level for tests
    pub significance_level: f64,
    /// Random seed
    pub random_seed: Option<u64>,
}

impl Default for HMMRegimeDetectionConfig {
    fn default() -> Self {
        Self {
            base_config: RegimeDetectionConfig::default(),
            hmm: FractalHMM::new(3),
            overlap_ratio: 0.5,
            feature_extraction_method: FeatureExtractionMethod::SlidingWindow,
            validation_method: ValidationMethod::CrossValidation { folds: 5 },
            significance_level: 0.05,
            random_seed: Some(42),
        }
    }
}