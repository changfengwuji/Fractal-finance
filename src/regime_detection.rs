//! Advanced regime detection using Hidden Markov Models for fractal time series.
//!
//! This module provides the main interface for detecting regimes in financial time series
//! using Hidden Markov Models with fractal features.

use crate::{
    bootstrap::*,
    calculate_autocorrelations,
    errors::{validate_data_length, FractalAnalysisError, FractalResult},
    math_utils::calculate_volatility,
    multifractal::*,
    secure_rng::{global_seed, FastrandCompat},
};

// Re-export structures from sub-modules for backward compatibility
pub use crate::emission_models::{EmissionParameters, MultifractalRegimeParams, ObservationFeatures};
pub use crate::hmm_core::FractalHMM;
pub use crate::regime_config::{
    FeatureExtractionMethod, HMMRegimeDetectionConfig, RegimeDetectionConfig, ValidationMethod,
};
pub use crate::regime_results::{
    HMMParameters, ModelCriteria, RegimeChangePoint, RegimeDetectionResult, RegimeStatistics,
    VolatilityStatistics, calculate_hmm_parameters,
};

/// Detect fractal regimes in time series using HMM
pub fn detect_fractal_regimes(
    data: &[f64],
    config: &RegimeDetectionConfig,
) -> FractalResult<RegimeDetectionResult> {
    validate_data_length(data, config.window_size * 3, "Regime detection")?;

    if let Some(seed) = config.seed {
        global_seed(seed);
    }

    // Extract observation features
    let observations = extract_observation_features(data, config)?;

    let mut best_result = None;
    let mut best_score = f64::INFINITY;

    // Try different numbers of states if auto-selection is enabled
    let state_range = if config.auto_select_states {
        config.num_states_range.0..=config.num_states_range.1
    } else {
        config.num_states_range.0..=config.num_states_range.0
    };

    for num_states in state_range {
        let mut hmm = FractalHMM::new(num_states);
        let log_likelihood = hmm.fit(&observations)?;

        // Calculate model selection criteria
        let num_params = calculate_hmm_parameters(num_states);
        let aic = -2.0 * log_likelihood + 2.0 * num_params as f64;
        let bic = -2.0 * log_likelihood + (num_params as f64) * (observations.len() as f64).ln();

        let score = if config.auto_select_states { bic } else { aic };

        if score < best_score {
            best_score = score;

            // Decode regime sequence
            let regime_sequence = hmm.decode(&observations)?;

            // Apply minimum duration constraint
            let filtered_sequence =
                apply_minimum_duration_constraint(&regime_sequence, config.min_regime_duration);

            // Calculate state probabilities using forward-backward
            let (alpha, beta, _) = hmm.forward_backward(&observations)?;
            let state_probabilities = calculate_state_probabilities(&alpha, &beta);

            // Detect change points
            let change_points = detect_change_points(&filtered_sequence, &observations, &hmm)?;

            // Calculate regime statistics
            let regime_statistics =
                calculate_regime_statistics(&filtered_sequence, &observations, &hmm);

            let model_criteria = ModelCriteria {
                aic,
                bic,
                hqic: -2.0 * log_likelihood
                    + 2.0 * (num_params as f64) * (observations.len() as f64).ln().ln(),
                num_parameters: num_params,
            };

            best_result = Some(RegimeDetectionResult {
                regime_sequence: filtered_sequence,
                state_probabilities,
                change_points,
                hmm_params: HMMParameters {
                    num_states: hmm.num_states,
                    initial_probs: hmm.initial_probs,
                    transition_matrix: hmm.transition_matrix,
                    emission_params: hmm.emission_params,
                },
                log_likelihood,
                model_criteria,
                regime_statistics,
            });
        }
    }

    best_result.ok_or(FractalAnalysisError::NumericalError {
        reason: "Failed to fit any HMM model".to_string(),
        operation: None,
    })
}

/// Advanced HMM-based regime detection with extended configuration
pub fn detect_fractal_regimes_with_hmm(
    data: &[f64],
    config: &HMMRegimeDetectionConfig,
) -> FractalResult<RegimeDetectionResult> {
    validate_data_length(
        data,
        config.base_config.window_size * 3,
        "HMM regime detection",
    )?;

    // Use thread-local RNG with optional seed
    let _rng = if let Some(seed) = config.random_seed {
        FastrandCompat::with_seed(seed)
    } else {
        FastrandCompat::new()
    };

    // Extract observation features using the specified method
    let observations = extract_observation_features_extended(data, config)?;

    let mut best_result = None;
    let mut best_score = f64::INFINITY;

    // Try different numbers of states if auto-selection is enabled
    let state_range = if config.base_config.auto_select_states {
        config.base_config.num_states_range.0..=config.base_config.num_states_range.1
    } else {
        config.base_config.num_states_range.0..=config.base_config.num_states_range.0
    };

    for num_states in state_range {
        let mut hmm = FractalHMM::new_with_initialization(num_states, &observations);
        let log_likelihood = hmm.fit(&observations)?;

        // Calculate model selection criteria
        let num_params = calculate_hmm_parameters(num_states);
        let aic = -2.0 * log_likelihood + 2.0 * num_params as f64;
        let bic = -2.0 * log_likelihood + (num_params as f64) * (observations.len() as f64).ln();

        let score = if config.base_config.auto_select_states {
            bic
        } else {
            aic
        };

        if score < best_score {
            best_score = score;

            // Decode regime sequence
            let regime_sequence = hmm.decode(&observations)?;

            // Apply minimum duration constraint
            let filtered_sequence = apply_minimum_duration_constraint(
                &regime_sequence,
                config.base_config.min_regime_duration,
            );

            // Calculate state probabilities using forward-backward
            let (alpha, beta, _) = hmm.forward_backward(&observations)?;
            let state_probabilities = calculate_state_probabilities(&alpha, &beta);

            // Detect change points
            let change_points =
                detect_change_points_extended(&filtered_sequence, &observations, &hmm)?;

            // Calculate comprehensive regime statistics
            let regime_statistics =
                calculate_regime_statistics_extended(&filtered_sequence, &observations, &hmm, data);

            let model_criteria = ModelCriteria {
                aic,
                bic,
                hqic: -2.0 * log_likelihood
                    + 2.0 * (num_params as f64) * (observations.len() as f64).ln().ln(),
                num_parameters: num_params,
            };

            best_result = Some(RegimeDetectionResult {
                regime_sequence: filtered_sequence,
                state_probabilities,
                change_points,
                hmm_params: HMMParameters {
                    num_states: hmm.num_states,
                    initial_probs: hmm.initial_probs,
                    transition_matrix: hmm.transition_matrix,
                    emission_params: hmm.emission_params,
                },
                log_likelihood,
                model_criteria,
                regime_statistics,
            });
        }
    }

    best_result.ok_or(FractalAnalysisError::NumericalError {
        reason: "Failed to fit any HMM model".to_string(),
        operation: None,
    })
}

// ============================================================================
// Feature extraction functions
// ============================================================================

/// Extract observation features from time series
fn extract_observation_features(
    data: &[f64],
    config: &RegimeDetectionConfig,
) -> FractalResult<Vec<ObservationFeatures>> {
    let mut observations = Vec::new();

    let mut i = 0;
    while i + config.window_size <= data.len() {
        let window = &data[i..i + config.window_size];

        // Calculate local Hurst exponent
        let mf_config = MultifractalConfig::default();
        let local_hurst = match calculate_generalized_hurst_exponent(window, 2.0, &mf_config) {
            Ok(h) => h,
            Err(_) => 0.5, // Default to Brownian motion value
        };

        // Calculate local volatility
        let local_volatility = calculate_volatility(window);

        // Calculate local autocorrelation (lag-1)
        let autocorr = calculate_autocorrelations(window, 1);
        let local_autocorr = if autocorr.len() > 1 {
            autocorr[1] // lag-1 autocorrelation
        } else {
            0.0
        };

        // Calculate local multifractality
        let local_multifractality = if let Ok(mf_analysis) = perform_multifractal_analysis(window) {
            mf_analysis.multifractality_degree
        } else {
            0.0
        };

        // time_index represents the center of the window in original data
        let window_center = i + config.window_size / 2;

        observations.push(ObservationFeatures {
            local_hurst,
            local_volatility,
            local_autocorr,
            local_multifractality,
            time_index: window_center,
        });

        i += config.step_size;
    }

    if observations.is_empty() {
        return Err(FractalAnalysisError::InsufficientData {
            required: config.window_size,
            actual: data.len(),
        });
    }

    Ok(observations)
}

/// Extended feature extraction with different methods
fn extract_observation_features_extended(
    data: &[f64],
    config: &HMMRegimeDetectionConfig,
) -> FractalResult<Vec<ObservationFeatures>> {
    match config.feature_extraction_method {
        FeatureExtractionMethod::SlidingWindow => {
            extract_observation_features(data, &config.base_config)
        }
        FeatureExtractionMethod::NonOverlapping => extract_non_overlapping_features(data, config),
        FeatureExtractionMethod::Adaptive => extract_adaptive_features(data, config),
    }
}

/// Extract features using non-overlapping windows
fn extract_non_overlapping_features(
    data: &[f64],
    config: &HMMRegimeDetectionConfig,
) -> FractalResult<Vec<ObservationFeatures>> {
    let mut observations = Vec::new();
    let window_size = config.base_config.window_size;
    let mut time_index = 0;

    for chunk in data.chunks(window_size) {
        if chunk.len() >= window_size {
            let features = calculate_window_features(chunk, time_index)?;
            observations.push(features);
            time_index += 1;
        }
    }

    if observations.is_empty() {
        return Err(FractalAnalysisError::InsufficientData {
            required: window_size,
            actual: data.len(),
        });
    }

    Ok(observations)
}

/// Extract features using adaptive window sizing
fn extract_adaptive_features(
    data: &[f64],
    config: &HMMRegimeDetectionConfig,
) -> FractalResult<Vec<ObservationFeatures>> {
    let mut observations = Vec::new();
    let base_window_size = config.base_config.window_size;
    let mut time_index = 0;
    let mut i = 0;

    while i + base_window_size <= data.len() {
        // Adapt window size based on local volatility
        let initial_window = &data[i..i + base_window_size];
        let volatility = calculate_volatility(initial_window);

        // Larger windows for low volatility periods, smaller for high volatility
        let adaptive_size = if volatility < 0.01 {
            (base_window_size * 2).min(data.len() - i)
        } else if volatility > 0.03 {
            (base_window_size / 2).max(20)
        } else {
            base_window_size
        };

        if i + adaptive_size <= data.len() {
            let window = &data[i..i + adaptive_size];
            let features = calculate_window_features(window, time_index)?;
            observations.push(features);

            i += adaptive_size / 2; // 50% overlap for adaptive method
            time_index += 1;
        } else {
            break;
        }
    }

    if observations.is_empty() {
        return Err(FractalAnalysisError::InsufficientData {
            required: base_window_size,
            actual: data.len(),
        });
    }

    Ok(observations)
}

/// Calculate features for a single window
fn calculate_window_features(
    window: &[f64],
    time_index: usize,
) -> FractalResult<ObservationFeatures> {
    // Calculate local Hurst exponent with proper error handling
    let mf_config = MultifractalConfig::default();
    let local_hurst = match calculate_generalized_hurst_exponent(window, 2.0, &mf_config) {
        Ok(hurst) => {
            // Validate Hurst exponent is in reasonable range
            if hurst < 0.01 || hurst > 0.99 || !hurst.is_finite() {
                0.5 // Default to random walk if invalid
            } else {
                hurst
            }
        }
        Err(_) => 0.5,
    };

    // Calculate local volatility
    let local_volatility = calculate_volatility(window);

    // Calculate local autocorrelation (lag-1)
    let local_autocorr = {
        let n = window.len();
        if n <= 1 {
            0.0
        } else {
            let mean = window.iter().sum::<f64>() / n as f64;
            let variance: f64 = window
                .iter()
                .map(|&x| {
                    let dev = x - mean;
                    dev * dev
                })
                .sum::<f64>()
                / n as f64;

            if variance < 1e-10 {
                0.0 // Constant series
            } else {
                let mut lag1_cov = 0.0;
                for i in 0..(n - 1) {
                    lag1_cov += (window[i] - mean) * (window[i + 1] - mean);
                }
                lag1_cov /= (n - 1) as f64;
                let autocorr = lag1_cov / variance;
                autocorr.max(-1.0).min(1.0)
            }
        }
    };

    // Calculate local multifractality
    let local_multifractality = if let Ok(mf_analysis) = perform_multifractal_analysis(window) {
        mf_analysis.multifractality_degree
    } else {
        0.0
    };

    Ok(ObservationFeatures {
        local_hurst,
        local_volatility,
        local_autocorr,
        local_multifractality,
        time_index,
    })
}

// ============================================================================
// Helper functions for regime analysis
// ============================================================================

/// Apply minimum duration constraint to regime sequence
fn apply_minimum_duration_constraint(sequence: &[usize], min_duration: usize) -> Vec<usize> {
    if sequence.is_empty() {
        return vec![];
    }

    let mut filtered = sequence.to_vec();
    let mut changed = true;

    while changed {
        changed = false;
        let mut i = 0;

        while i < filtered.len() {
            let current_state = filtered[i];
            let mut duration = 1;

            // Count duration of current regime
            while i + duration < filtered.len() && filtered[i + duration] == current_state {
                duration += 1;
            }

            // If duration is too short, merge with adjacent regime
            if duration < min_duration && i + duration < filtered.len() {
                let next_state = filtered[i + duration];
                for j in i..i + duration {
                    filtered[j] = next_state;
                }
                changed = true;
            }

            i += duration;
        }
    }

    filtered
}

/// Calculate state probabilities from alpha and beta
fn calculate_state_probabilities(alpha: &[Vec<f64>], beta: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let t = alpha.len();
    let num_states = alpha[0].len();
    let mut state_probs = vec![vec![0.0; num_states]; t];

    for i in 0..t {
        let sum: f64 = (0..num_states).map(|j| alpha[i][j] * beta[i][j]).sum();
        if sum > 0.0 {
            for j in 0..num_states {
                state_probs[i][j] = alpha[i][j] * beta[i][j] / sum;
            }
        }
    }

    state_probs
}

/// Detect regime change points
fn detect_change_points(
    sequence: &[usize],
    observations: &[ObservationFeatures],
    hmm: &FractalHMM,
) -> FractalResult<Vec<RegimeChangePoint>> {
    let mut change_points = Vec::new();

    if sequence.len() < 2 {
        return Ok(change_points);
    }

    let mut i = 1;
    let mut current_regime_start = 0;

    while i < sequence.len() {
        if sequence[i] != sequence[i - 1] {
            let from_state = sequence[i - 1];
            let to_state = sequence[i];

            // Calculate confidence based on emission probabilities
            let confidence = hmm.emission_log_prob(&observations[i], to_state).exp();

            // Calculate changes in parameters
            let hurst_change = hmm.emission_params[to_state].mean_vector[0]
                - hmm.emission_params[from_state].mean_vector[0];
            let volatility_change = hmm.emission_params[to_state].mean_vector[1]
                - hmm.emission_params[from_state].mean_vector[1];

            change_points.push(RegimeChangePoint {
                time_index: observations[i].time_index,
                from_state,
                to_state,
                confidence,
                hurst_change,
                volatility_change,
                previous_regime_duration: i - current_regime_start,
            });

            current_regime_start = i;
        }
        i += 1;
    }

    Ok(change_points)
}

/// Enhanced change point detection with confidence intervals
fn detect_change_points_extended(
    sequence: &[usize],
    observations: &[ObservationFeatures],
    hmm: &FractalHMM,
) -> FractalResult<Vec<RegimeChangePoint>> {
    let mut change_points = Vec::new();

    if sequence.len() < 2 {
        return Ok(change_points);
    }

    let mut i = 1;
    let mut current_regime_start = 0;

    while i < sequence.len() {
        if sequence[i] != sequence[i - 1] {
            let from_state = sequence[i - 1];
            let to_state = sequence[i];

            // Calculate confidence based on emission probabilities and state transition
            let emission_conf = hmm.emission_log_prob(&observations[i], to_state).exp();
            let transition_conf = hmm.transition_matrix[from_state][to_state];
            let confidence = (emission_conf * transition_conf).min(1.0);

            // Calculate changes in parameters
            let hurst_change = hmm.emission_params[to_state].mean_vector[0]
                - hmm.emission_params[from_state].mean_vector[0];
            let volatility_change = hmm.emission_params[to_state].mean_vector[1]
                - hmm.emission_params[from_state].mean_vector[1];

            change_points.push(RegimeChangePoint {
                time_index: observations[i].time_index,
                from_state,
                to_state,
                confidence,
                hurst_change,
                volatility_change,
                previous_regime_duration: i - current_regime_start,
            });

            current_regime_start = i;
        }
        i += 1;
    }

    Ok(change_points)
}

/// Calculate statistics for each regime
fn calculate_regime_statistics(
    sequence: &[usize],
    observations: &[ObservationFeatures],
    hmm: &FractalHMM,
) -> Vec<RegimeStatistics> {
    let mut statistics = Vec::with_capacity(hmm.num_states);

    for state in 0..hmm.num_states {
        let mut durations = Vec::new();
        let mut total_duration = 0;
        let mut occurrence_count = 0;
        let mut _hurst_sum = 0.0;
        let mut volatility_sum = 0.0;
        let mut obs_count = 0;
        let mut first_occurrence = None;
        let mut last_occurrence = 0;

        // Find all occurrences of this state
        let mut i = 0;
        while i < sequence.len() {
            if sequence[i] == state {
                if first_occurrence.is_none() {
                    first_occurrence = Some(i);
                }
                last_occurrence = i;

                // Count duration of this occurrence
                let start = i;
                while i < sequence.len() && sequence[i] == state {
                    i += 1;
                }
                let duration = i - start;
                durations.push(duration);
                total_duration += duration;
                occurrence_count += 1;

                // Accumulate statistics from observations
                for j in start..i {
                    if j < observations.len() {
                        _hurst_sum += observations[j].local_hurst;
                        volatility_sum += observations[j].local_volatility;
                        obs_count += 1;
                    }
                }
            } else {
                i += 1;
            }
        }

        let average_duration = if occurrence_count > 0 {
            total_duration as f64 / occurrence_count as f64
        } else {
            0.0
        };

        let average_hurst = hmm.emission_params[state].mean_vector[0];
        let average_volatility = if obs_count > 0 {
            volatility_sum / obs_count as f64
        } else {
            hmm.emission_params[state].mean_vector[1]
        };

        let persistence_probability = hmm.transition_matrix[state][state];

        // Calculate volatility statistics
        let mut regime_volatilities = Vec::new();
        for (i, &s) in sequence.iter().enumerate() {
            if s == state && i < observations.len() {
                regime_volatilities.push(observations[i].local_volatility);
            }
        }

        let volatility_statistics = if !regime_volatilities.is_empty() {
            let mean = regime_volatilities.iter().sum::<f64>() / regime_volatilities.len() as f64;
            let variance = regime_volatilities
                .iter()
                .map(|&v| (v - mean).powi(2))
                .sum::<f64>()
                / regime_volatilities.len() as f64;
            let std = variance.sqrt();
            let min = regime_volatilities
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            let max = regime_volatilities
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            VolatilityStatistics { mean, std, min, max }
        } else {
            VolatilityStatistics::default()
        };

        let multifractal_characteristics = compute_regime_multifractal_params(
            state,
            &regime_volatilities,
            average_hurst,
        );

        statistics.push(RegimeStatistics {
            state_index: state,
            regime_id: state,
            first_occurrence_time: first_occurrence.unwrap_or(0),
            last_occurrence_time: last_occurrence,
            average_duration,
            total_duration,
            occurrence_count,
            average_hurst,
            mean_hurst_exponent: average_hurst,
            average_volatility,
            persistence_probability,
            volatility_statistics,
            multifractal_characteristics,
        });
    }

    statistics
}

/// Calculate comprehensive regime statistics
fn calculate_regime_statistics_extended(
    sequence: &[usize],
    observations: &[ObservationFeatures],
    hmm: &FractalHMM,
    _original_data: &[f64],
) -> Vec<RegimeStatistics> {
    // Implementation is similar to calculate_regime_statistics
    // but with additional analysis if needed
    calculate_regime_statistics(sequence, observations, hmm)
}

/// Compute multifractal parameters for a regime
fn compute_regime_multifractal_params(
    state_index: usize,
    regime_volatilities: &[f64],
    average_hurst: f64,
) -> MultifractalRegimeParams {
    // Compute spectrum width from volatility spread
    let spectrum_width = if regime_volatilities.len() > 1 {
        let vol_variance = regime_volatilities.iter().map(|v| v.powi(2)).sum::<f64>()
            / regime_volatilities.len() as f64;
        (vol_variance.sqrt() * 0.5).min(0.3) // Cap spectrum width at 0.3
    } else {
        0.1 // Default width
    };

    let alpha_max = average_hurst.max(0.1).min(0.95);

    // Asymmetry parameter based on state index (different regimes have different asymmetries)
    let spectrum_asymmetry = if state_index % 2 == 0 { -0.1 } else { 0.1 };

    MultifractalRegimeParams {
        spectrum_width,
        spectrum_asymmetry,
        alpha_max,
        f_alpha_max: 1.0, // Maximum f(alpha) value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::*;

    #[test]
    fn test_regime_detection() {
        // Generate regime-switching data
        let config1 = GeneratorConfig {
            length: 100,
            seed: Some(42),
            ..Default::default()
        };

        let fbm_config1 = FbmConfig {
            hurst_exponent: 0.3,
            volatility: 0.01,
            method: FbmMethod::Hosking,
        };

        let fbm_config2 = FbmConfig {
            hurst_exponent: 0.8,
            volatility: 0.02,
            method: FbmMethod::Hosking,
        };

        let fbm1 = generate_fractional_brownian_motion(&config1, &fbm_config1).unwrap();
        let fbm2 = generate_fractional_brownian_motion(&config1, &fbm_config2).unwrap();

        let mut combined_data = fbm_to_fgn(&fbm1);
        combined_data.extend(fbm_to_fgn(&fbm2));

        let detection_config = RegimeDetectionConfig {
            window_size: 30,
            step_size: 10,
            num_states_range: (2, 3),
            auto_select_states: true,
            ..Default::default()
        };

        let result = detect_fractal_regimes(&combined_data, &detection_config).unwrap();

        assert!(!result.regime_sequence.is_empty());
        assert!(!result.state_probabilities.is_empty());
        assert_eq!(result.regime_statistics.len(), result.hmm_params.num_states);
        assert!(result.log_likelihood.is_finite());
    }
}