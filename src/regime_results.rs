//! Result structures for regime detection.
//!
//! This module contains all result and statistics structures used in HMM-based
//! fractal regime detection for financial time series analysis.

use crate::emission_models::{EmissionParameters, MultifractalRegimeParams};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Regime detection result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RegimeDetectionResult {
    /// Most likely sequence of regime states
    pub regime_sequence: Vec<usize>,
    /// Posterior probabilities for each state at each time
    pub state_probabilities: Vec<Vec<f64>>,
    /// Detected regime change points
    pub change_points: Vec<RegimeChangePoint>,
    /// Estimated HMM parameters
    pub hmm_params: HMMParameters,
    /// Log-likelihood of the model
    pub log_likelihood: f64,
    /// Model selection criteria
    pub model_criteria: ModelCriteria,
    /// Regime statistics
    pub regime_statistics: Vec<RegimeStatistics>,
}

/// Information about a detected regime change
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RegimeChangePoint {
    /// Time index of change
    pub time_index: usize,
    /// Previous regime state
    pub from_state: usize,
    /// New regime state
    pub to_state: usize,
    /// Confidence of the change point
    pub confidence: f64,
    /// Change in Hurst exponent
    pub hurst_change: f64,
    /// Change in volatility
    pub volatility_change: f64,
    /// Duration of previous regime
    pub previous_regime_duration: usize,
}

/// Serializable HMM parameters for results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HMMParameters {
    /// Number of states
    pub num_states: usize,
    /// Initial probabilities
    pub initial_probs: Vec<f64>,
    /// Transition matrix
    pub transition_matrix: Vec<Vec<f64>>,
    /// Emission parameters
    pub emission_params: Vec<EmissionParameters>,
}

/// Model selection criteria for HMM
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ModelCriteria {
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Hannan-Quinn Information Criterion
    pub hqic: f64,
    /// Number of parameters
    pub num_parameters: usize,
}

/// Statistics for each detected regime
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RegimeStatistics {
    /// Regime state index
    pub state_index: usize,
    /// Regime identifier
    pub regime_id: usize,
    /// First occurrence time
    pub first_occurrence_time: usize,
    /// Last occurrence time
    pub last_occurrence_time: usize,
    /// Average duration in this regime
    pub average_duration: f64,
    /// Total time spent in this regime
    pub total_duration: usize,
    /// Number of times this regime occurred
    pub occurrence_count: usize,
    /// Average Hurst exponent in this regime
    pub average_hurst: f64,
    /// Mean Hurst exponent in this regime
    pub mean_hurst_exponent: f64,
    /// Average volatility in this regime
    pub average_volatility: f64,
    /// Regime persistence probability
    pub persistence_probability: f64,
    /// Volatility statistics for this regime
    pub volatility_statistics: VolatilityStatistics,
    /// Multifractal characteristics for this regime
    pub multifractal_characteristics: MultifractalRegimeParams,
}

/// Volatility statistics for regime analysis
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VolatilityStatistics {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

/// Calculate number of HMM parameters
pub fn calculate_hmm_parameters(num_states: usize) -> usize {
    let initial_params = num_states - 1; // Initial probabilities (sum to 1)
    let transition_params = num_states * (num_states - 1); // Transition matrix (rows sum to 1)
    let emission_params = num_states * 6; // 6 emission parameters per state

    initial_params + transition_params + emission_params
}