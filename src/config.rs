//! # Analysis Configuration
//!
//! This module contains configuration structures for controlling the behavior
//! of fractal analysis, including which components to enable and the depth
//! of analysis to perform.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for controlling which analysis components to run
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AnalysisConfig {
    /// Enable multifractal analysis (MF-DFA)
    pub enable_multifractal: bool,
    /// Enable HMM-based regime detection
    pub enable_regime_detection: bool,
    /// Enable walk-forward cross-validation
    pub enable_cross_validation: bool,
    /// Enable Monte Carlo hypothesis testing
    pub enable_monte_carlo: bool,
    /// Analysis depth preset
    pub depth: AnalysisDepth,
}

/// Analysis depth presets for different use cases
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AnalysisDepth {
    /// Light: Core Hurst estimation only
    Light,
    /// Standard: Core + statistical tests (default)
    Standard,
    /// Deep: Full analysis with all components
    Deep,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self::standard()
    }
}

impl AnalysisConfig {
    /// Light configuration: Core analysis only
    pub fn light() -> Self {
        Self {
            enable_multifractal: false,
            enable_regime_detection: false,
            enable_cross_validation: false,
            enable_monte_carlo: false,
            depth: AnalysisDepth::Light,
        }
    }

    /// Standard configuration: Core + statistical tests
    pub fn standard() -> Self {
        Self {
            enable_multifractal: false,
            enable_regime_detection: false,
            enable_cross_validation: false,
            enable_monte_carlo: false,
            depth: AnalysisDepth::Standard,
        }
    }

    /// Deep configuration: Full analysis
    pub fn deep() -> Self {
        Self {
            enable_multifractal: true,
            enable_regime_detection: true,
            enable_cross_validation: true,
            enable_monte_carlo: true,
            depth: AnalysisDepth::Deep,
        }
    }
    
    /// Check if multifractal analysis is enabled
    pub fn is_multifractal_enabled(&self) -> bool {
        self.enable_multifractal
    }
    
    /// Check if regime detection is enabled
    pub fn is_regime_detection_enabled(&self) -> bool {
        self.enable_regime_detection
    }
    
    /// Check if cross-validation is enabled
    pub fn is_cross_validation_enabled(&self) -> bool {
        self.enable_cross_validation
    }
    
    /// Check if Monte Carlo testing is enabled
    pub fn is_monte_carlo_enabled(&self) -> bool {
        self.enable_monte_carlo
    }
    
    /// Get the analysis depth
    pub fn depth(&self) -> AnalysisDepth {
        self.depth
    }
}