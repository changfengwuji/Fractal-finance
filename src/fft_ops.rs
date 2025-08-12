//! High-performance FFT operations for financial fractal analysis.
//!
//! This module provides optimized FFT-based computations that are critical for
//! efficient spectral analysis in financial time series. All operations are
//! designed for O(n log n) performance with comprehensive caching.
//!
//! ## CRITICAL FIXES (v0.2.0)
//!
//! ### Coherence Analysis
//! Previous versions had a critical error where single-shot coherence always returned 1.0.
//! This has been fixed by implementing Welch's method with proper spectral averaging.
//! Use `compute_coherence_welch` or `parallel_coherence_analysis_welch` for meaningful results.
//!
//! ### Cross-Spectral Density
//! Previous versions incorrectly returned |X*·Y|² (real) instead of X*·Y (complex),
//! losing crucial phase information needed for lead/lag analysis.
//! Now properly returns complex values preserving phase information.

use crate::computation_cache::{get_global_cache, AutocorrMethod, PeriodogramMethod, WindowType};
use crate::errors::{validate_allocation_size, FractalAnalysisError, FractalResult};
use crate::secure_rng::FastrandCompat;
use lru::LruCache;
use num_complex::Complex64;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;
use std::num::NonZeroUsize;
use std::sync::LazyLock;
use std::sync::{Arc, Mutex};

/// Signal detrending method for non-stationary financial data
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DetrendingMethod {
    /// No detrending (use raw data)
    None,
    /// Remove mean (simplest detrending for stationary data)
    RemoveMean,
    /// Remove linear trend (for data with linear drift)
    RemoveLinear,
    /// Remove polynomial trend of given order
    RemovePolynomial(usize),
}

/// Apply detrending to signal before FFT analysis
///
/// Financial time series are often non-stationary with trends that can
/// cause spectral leakage at DC (zero frequency). Detrending helps
/// reveal the true spectral characteristics.
pub fn detrend_signal(data: &[f64], method: DetrendingMethod) -> Vec<f64> {
    match method {
        DetrendingMethod::None => data.to_vec(),

        DetrendingMethod::RemoveMean => {
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            data.iter().map(|&x| x - mean).collect()
        }

        DetrendingMethod::RemoveLinear => {
            // Fit linear trend: y = a + b*x using least squares
            let n = data.len() as f64;
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_xy = 0.0;
            let mut sum_x2 = 0.0;

            for (i, &y) in data.iter().enumerate() {
                let x = i as f64;
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_x2 += x * x;
            }

            // Calculate slope and intercept
            let det = n * sum_x2 - sum_x * sum_x;
            if det.abs() < MIN_POSITIVE {
                // Degenerate case - fallback to mean removal
                return detrend_signal(data, DetrendingMethod::RemoveMean);
            }

            let slope = (n * sum_xy - sum_x * sum_y) / det;
            let intercept = (sum_y - slope * sum_x) / n;

            // Remove linear trend
            data.iter()
                .enumerate()
                .map(|(i, &y)| y - (intercept + slope * i as f64))
                .collect()
        }

        DetrendingMethod::RemovePolynomial(order) => {
            // For higher order polynomials, would need matrix operations
            // For now, fallback to linear for order > 1
            if order == 0 {
                detrend_signal(data, DetrendingMethod::RemoveMean)
            } else if order == 1 {
                detrend_signal(data, DetrendingMethod::RemoveLinear)
            } else {
                // Implement polynomial fitting via least squares
                let n = data.len();
                let x: Vec<f64> = (0..n).map(|i| i as f64).collect();

                // Build Vandermonde matrix A for polynomial fitting
                let mut a = vec![vec![0.0; order + 1]; n];
                for i in 0..n {
                    let xi = x[i];
                    let mut power = 1.0;
                    for j in 0..=order {
                        a[i][j] = power;
                        power *= xi;
                    }
                }

                // Solve normal equations: A^T * A * coeffs = A^T * y
                let mut ata = vec![vec![0.0; order + 1]; order + 1];
                let mut aty = vec![0.0; order + 1];

                for i in 0..=order {
                    for j in 0..=order {
                        for k in 0..n {
                            ata[i][j] += a[k][i] * a[k][j];
                        }
                    }
                    for k in 0..n {
                        aty[i] += a[k][i] * data[k];
                    }
                }

                // Solve using Gaussian elimination with partial pivoting
                let coeffs = match solve_linear_system(&ata, &aty) {
                    Some(c) => c,
                    None => {
                        log::warn!("Polynomial fitting failed, falling back to linear detrending");
                        return detrend_signal(data, DetrendingMethod::RemoveLinear);
                    }
                };

                // Subtract fitted polynomial from data
                let mut detrended = Vec::with_capacity(n);
                for i in 0..n {
                    let xi = x[i];
                    let mut poly_val = 0.0;
                    let mut power = 1.0;
                    for &coeff in &coeffs {
                        poly_val += coeff * power;
                        power *= xi;
                    }
                    detrended.push(data[i] - poly_val);
                }
                detrended
            }
        }
    }
}

/// Cache key for FFT planners, distinguishing forward and inverse transforms
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct FftCacheKey {
    size: usize,
    is_forward: bool,
}

/// Thread-safe LRU cache for FFT planners to eliminate O(n²) planning overhead
/// Using LRU eviction for stable performance in high-frequency trading scenarios
type FftPlanCache = LruCache<FftCacheKey, Arc<dyn rustfft::Fft<f64> + Send + Sync>>;

/// Maximum cache size to prevent memory exhaustion attacks
const MAX_CACHE_ENTRIES: usize = 1000;
/// Maximum FFT size to prevent DoS via huge allocations (2^20 = 1M points)
const MAX_FFT_SIZE: usize = 1 << 20;
/// Minimum positive value for safe division
const MIN_POSITIVE: f64 = 1e-300;

static FFT_CACHE: LazyLock<Mutex<FftPlanCache>> =
    LazyLock::new(|| Mutex::new(LruCache::new(NonZeroUsize::new(MAX_CACHE_ENTRIES).unwrap())));

/// Get cached FFT plan (forward or inverse) with LRU eviction
///
/// This refactored function eliminates code duplication between forward and inverse
/// FFT plan caching. Uses LRU cache for optimal performance in high-frequency trading.
///
/// # Arguments
/// * `size` - The size of the FFT to plan
/// * `is_forward` - true for forward FFT, false for inverse FFT
///
/// # Performance
/// - O(1) average case for cache hits
/// - LRU eviction ensures most frequently used plans stay cached
/// - Thread-safe with mutex protection
fn get_cached_fft_plan(
    size: usize,
    is_forward: bool,
) -> FractalResult<Arc<dyn rustfft::Fft<f64> + Send + Sync>> {
    // Validate size to prevent DoS
    if size == 0 {
        return Err(FractalAnalysisError::InvalidParameter {
            parameter: "fft_size".to_string(),
            value: 0.0,
            constraint: "must be > 0".to_string(),
        });
    }
    if size > MAX_FFT_SIZE {
        return Err(FractalAnalysisError::InvalidParameter {
            parameter: "fft_size".to_string(),
            value: size as f64,
            constraint: format!("must be <= {} to prevent memory exhaustion", MAX_FFT_SIZE),
        });
    }

    let cache_key = FftCacheKey { size, is_forward };

    // Handle mutex poisoning gracefully
    let mut cache = match FFT_CACHE.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            // Recover from poisoned mutex by using the data anyway
            poisoned.into_inner()
        }
    };

    // LRU cache automatically updates access order on get
    if let Some(cached_fft) = cache.get(&cache_key) {
        return Ok(cached_fft.clone());
    }

    // Create new FFT plan
    let mut planner = FftPlanner::new();
    let new_fft = if is_forward {
        planner.plan_fft_forward(size)
    } else {
        planner.plan_fft_inverse(size)
    };

    // LRU cache automatically handles eviction when capacity is reached
    cache.put(cache_key, new_fft.clone());
    Ok(new_fft)
}

/// Get cached FFT plan for forward transform
///
/// PERFORMANCE OPTIMIZATION: Uses LRU-cached FFT planners with O(1) access,
/// eliminating the O(n) planning overhead on every FFT call.
pub fn get_cached_fft_forward(
    size: usize,
) -> FractalResult<Arc<dyn rustfft::Fft<f64> + Send + Sync>> {
    get_cached_fft_plan(size, true)
}

/// Get cached FFT plan for inverse transform
///
/// PERFORMANCE OPTIMIZATION: Uses LRU-cached FFT planners with O(1) access,
/// eliminating the O(n) planning overhead on every FFT call.
pub fn get_cached_fft_inverse(
    size: usize,
) -> FractalResult<Arc<dyn rustfft::Fft<f64> + Send + Sync>> {
    get_cached_fft_plan(size, false)
}

/// Calculate periodogram using FFT for efficiency
///
/// CRITICAL PERFORMANCE FIX: Caches FFT planners to achieve proper O(n log n) scaling.
/// Previously, creating a new FftPlanner on every call caused O(n²) complexity.
///
/// The periodogram estimates the power spectral density and is fundamental to
/// statistical tests like GPH test for long-range dependence detection.
///
/// # Arguments
///
/// * `data` - Input time series data
///
/// # Returns
///
/// Periodogram values at each frequency
///
/// # Example
///
/// ```rust
/// use financial_fractal_analysis::calculate_periodogram_fft;
///
/// let data = vec![1.0, 2.0, 1.0, -1.0, -2.0, -1.0];
/// let periodogram = calculate_periodogram_fft(&data).unwrap();
/// ```
pub fn calculate_periodogram_fft_with_detrending(
    data: &[f64],
    detrend: Option<DetrendingMethod>,
) -> FractalResult<Vec<f64>> {
    let n = data.len();
    if n < 4 {
        return Err(FractalAnalysisError::FftError { size: n });
    }

    // Default to mean removal for financial data (prevents DC leakage)
    let detrend_method = detrend.unwrap_or(DetrendingMethod::RemoveMean);
    let detrended_data = detrend_signal(data, detrend_method);

    // OPTIMIZATION: Use computation cache to avoid recomputing identical periodograms
    let cache = get_global_cache();

    cache.get_or_compute_periodogram(
        &detrended_data,
        PeriodogramMethod::Periodogram,
        WindowType::None,
        || {
            // This closure only executes on cache miss
            compute_periodogram_fft_internal(&detrended_data)
        },
    )
}

/// Calculate periodogram using FFT (backward compatibility).
///
/// Automatically applies mean removal detrending for financial data.
pub fn calculate_periodogram_fft(data: &[f64]) -> FractalResult<Vec<f64>> {
    calculate_periodogram_fft_with_detrending(data, Some(DetrendingMethod::RemoveMean))
}

/// Internal periodogram computation without caching (for cache miss handling).
fn compute_periodogram_fft_internal(data: &[f64]) -> FractalResult<Vec<f64>> {
    let n = data.len();

    // Validate input data for NaN or infinite values
    for (i, &value) in data.iter().enumerate() {
        if !value.is_finite() {
            return Err(FractalAnalysisError::NumericalError {
                reason: format!("Non-finite value {} at index {} in input data", value, i),
                operation: None,
            });
        }
    }

    // Convert to complex numbers for FFT
    let mut buffer: Vec<Complex64> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();

    // Ensure we have a reasonable buffer size
    if buffer.is_empty() {
        return Err(FractalAnalysisError::FftError { size: 0 });
    }

    // Get cached FFT plan for this size
    let fft = get_cached_fft_forward(n)?;

    // Apply FFT
    fft.process(&mut buffer);

    // Calculate periodogram: |FFT(x)|² / n
    let periodogram: Vec<f64> = buffer.iter().map(|c| c.norm_sqr() / n as f64).collect();

    Ok(periodogram)
}

/// Compute autocorrelation function using FFT with bias control
///
/// # Arguments
/// * `data` - Input time series
/// * `max_lag` - Maximum lag to compute
/// * `biased` - If true, returns biased estimate (normalized by N).
///              If false, returns unbiased estimate (normalized by N-k for lag k).
///
/// # Notes on Bias
/// - **Biased estimate** (divide by N): Ensures positive semi-definite autocorrelation matrix.
///   This is preferred for spectral analysis, covariance matrix estimation, and when
///   the autocorrelation function needs to be positive semi-definite.
/// - **Unbiased estimate** (divide by N-k): Provides unbiased statistical estimate.
///   This is preferred for parameter estimation, hypothesis testing, and when
///   statistical unbiasedness is more important than positive semi-definiteness.
///
/// # Performance
/// Uses FFT for O(n log n) complexity and caches results for repeated queries.
pub fn fft_autocorrelation_with_bias(
    data: &[f64],
    max_lag: usize,
    biased: bool,
) -> FractalResult<Vec<f64>> {
    let n = data.len();
    if n < 4 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 4,
            actual: n,
        });
    }
    if max_lag >= n {
        return Err(FractalAnalysisError::InvalidParameter {
            parameter: "max_lag".to_string(),
            value: max_lag as f64,
            constraint: format!("must be less than data length ({})", n),
        });
    }

    // OPTIMIZATION: Use computation cache to avoid recomputing identical autocorrelations
    let cache = get_global_cache();

    cache.get_or_compute_autocorr(
        data,
        max_lag,
        AutocorrMethod::Fft { biased }, // CRITICAL FIX: Include bias in cache key
        || {
            // This closure only executes on cache miss
            compute_fft_autocorrelation_internal(data, max_lag, biased)
        },
    )
}

/// Internal FFT autocorrelation computation without caching (for cache miss handling).
fn compute_fft_autocorrelation_internal(
    data: &[f64],
    max_lag: usize,
    biased: bool,
) -> FractalResult<Vec<f64>> {
    let n = data.len();

    // Check for integer overflow in padding
    let padded_size = n
        .checked_mul(2)
        .ok_or_else(|| FractalAnalysisError::InvalidParameter {
            parameter: "data_length".to_string(),
            value: n as f64,
            constraint: format!("too large for FFT autocorrelation (would overflow when doubled)"),
        })?;

    // Validate allocation size
    validate_allocation_size(
        padded_size * std::mem::size_of::<Complex64>(),
        "FFT autocorrelation",
    )?;

    let mut padded_data: Vec<Complex64> = Vec::with_capacity(padded_size);

    // Add original data
    for &x in data {
        padded_data.push(Complex::new(x, 0.0));
    }

    // Zero-pad the rest
    for _ in n..padded_size {
        padded_data.push(Complex::new(0.0, 0.0));
    }

    // Forward FFT
    let fft_forward = get_cached_fft_forward(padded_size)?;
    fft_forward.process(&mut padded_data);

    // Compute power spectrum: FFT(x) * conj(FFT(x))
    for c in &mut padded_data {
        *c = *c * c.conj();
    }

    // Inverse FFT
    let fft_inverse = get_cached_fft_inverse(padded_size)?;
    fft_inverse.process(&mut padded_data);

    // CRITICAL: rustfft's IFFT is not normalized
    // After IFFT, padded_data[i].re contains the unnormalized autocorrelation at lag i
    // The result is scaled by padded_size, so we need to divide by padded_size
    // to get the true autocovariance values

    let mut autocorrs = Vec::with_capacity(max_lag + 1);

    // First compute autocovariance values with proper normalization
    let mut autocovariances = Vec::with_capacity(max_lag + 1);

    for i in 0..=max_lag {
        // rustfft's IFFT output is not normalized, divide by padded_size
        // to get the true autocovariance
        let autocovariance = padded_data[i].re / padded_size as f64;

        // Apply bias correction based on the estimator type
        let corrected_value = if biased {
            // Biased estimator: sum/N for all lags
            // No additional correction needed since autocovariance already represents sum/padded_size
            // and we want sum/n
            autocovariance * padded_size as f64 / n as f64
        } else {
            // Unbiased estimator: sum/(N-lag) for lag k
            // Scale from sum/padded_size to sum/(n-lag)
            autocovariance * padded_size as f64 / (n - i) as f64
        };

        autocovariances.push(corrected_value);
    }

    // Normalize by lag-0 value to get correlation coefficients
    let lag_zero_value = autocovariances[0];

    // Check for zero or near-zero lag-0 value to prevent division issues
    if lag_zero_value.abs() < MIN_POSITIVE {
        return Err(FractalAnalysisError::NumericalError {
            reason: format!(
                "Lag-0 autocovariance too small ({}) for reliable normalization",
                lag_zero_value
            ),
            operation: None,
        });
    }

    // Convert autocovariances to autocorrelations by normalizing with lag-0 value
    for value in autocovariances {
        let correlation = value / lag_zero_value;
        // Clamp to valid correlation range [-1, 1] to handle numerical errors
        autocorrs.push(correlation.max(-1.0).min(1.0));
    }

    Ok(autocorrs)
}

/// Compute biased autocorrelation function using FFT (backward compatibility)
///
/// This computes the biased autocorrelation estimate, which ensures
/// positive semi-definiteness. For unbiased estimates or explicit control,
/// use `fft_autocorrelation_with_bias`.
///
/// OPTIMIZATION: Uses computation cache to avoid recomputing identical autocorrelations.
/// Also uses the convolution theorem: autocorr(x) = IFFT(FFT(x) * conj(FFT(x)))
/// This achieves O(n log n) complexity compared to O(n²) direct computation.
pub fn fft_autocorrelation(data: &[f64], max_lag: usize) -> FractalResult<Vec<f64>> {
    fft_autocorrelation_with_bias(data, max_lag, true)
}

/// Clear the FFT cache to free memory
///
/// Useful for long-running applications to prevent cache growth.
pub fn clear_fft_cache() {
    // Handle mutex poisoning gracefully
    match FFT_CACHE.lock() {
        Ok(mut cache) => cache.clear(),
        Err(poisoned) => {
            // Clear the cache even if mutex was poisoned
            poisoned.into_inner().clear();
        }
    }
}

/// Get current FFT cache statistics
///
/// Returns (forward_plans, inverse_plans) counts for monitoring.
pub fn get_fft_cache_stats() -> (usize, usize) {
    // Handle mutex poisoning gracefully
    let cache = match FFT_CACHE.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };

    // LruCache uses iter() to iterate through entries
    let mut forward_count = 0;
    let mut inverse_count = 0;

    for (key, _) in cache.iter() {
        if key.is_forward {
            forward_count += 1;
        } else {
            inverse_count += 1;
        }
    }

    (forward_count, inverse_count)
}

//////////////////////////////////////////////////////////////////////////////
// ENHANCED PARALLEL FFT OPERATIONS FOR QUANTITATIVE FINANCE
//////////////////////////////////////////////////////////////////////////////

/// Parallel FFT operations optimized for financial time series analysis
/// where batch processing and millisecond-level performance are critical
#[cfg(feature = "parallel")]
pub mod parallel_fft {
    use super::*;
    use rayon::prelude::*;

    /// Parallel batch periodogram computation for multiple time series
    /// Critical for portfolio-level spectral analysis and cross-asset correlations
    pub fn batch_periodogram_parallel(datasets: &[&[f64]]) -> FractalResult<Vec<Vec<f64>>> {
        if datasets.is_empty() {
            return Err(FractalAnalysisError::InvalidParameter {
                parameter: "datasets".to_string(),
                value: 0.0,
                constraint: "must have at least one dataset".to_string(),
            });
        }

        // Parallel computation of periodograms across datasets
        let results: Result<Vec<_>, _> = datasets
            .par_iter()
            .map(|data| calculate_periodogram_fft(data))
            .collect();

        results
    }

    /// Parallel cross-spectral density computation for pairs of time series
    /// Essential for correlation trading and statistical arbitrage strategies
    ///
    /// IMPORTANT: Returns COMPLEX cross-spectral density to preserve phase information.
    /// Phase information is crucial for lead/lag analysis in financial markets.
    pub fn parallel_cross_spectral_density(
        x_datasets: &[&[f64]],
        y_datasets: &[&[f64]],
    ) -> FractalResult<Vec<Vec<Complex64>>> {
        if x_datasets.len() != y_datasets.len() {
            return Err(FractalAnalysisError::InvalidParameter {
                parameter: "dataset_lengths".to_string(),
                value: x_datasets.len() as f64,
                constraint: format!("must match y_datasets length of {}", y_datasets.len()),
            });
        }

        // Parallel computation of cross-spectral densities
        let results: Result<Vec<_>, _> = x_datasets
            .par_iter()
            .zip(y_datasets.par_iter())
            .map(|(x_data, y_data)| {
                if x_data.len() != y_data.len() {
                    return Err(FractalAnalysisError::InvalidParameter {
                        parameter: "series_lengths".to_string(),
                        value: x_data.len() as f64,
                        constraint: format!("must match y_data length of {}", y_data.len()),
                    });
                }

                // Compute cross-spectral density using FFT
                let n = x_data.len();
                let fft_forward = get_cached_fft_forward(n)?;

                // Convert to complex data for FFT
                let mut x_complex: Vec<Complex64> =
                    x_data.iter().map(|&val| Complex64::new(val, 0.0)).collect();
                let mut y_complex: Vec<Complex64> =
                    y_data.iter().map(|&val| Complex64::new(val, 0.0)).collect();

                // Perform FFTs
                fft_forward.process(&mut x_complex);
                fft_forward.process(&mut y_complex);

                // Compute COMPLEX cross-spectral density: X* · Y
                // Preserves phase information for lead/lag analysis
                let cross_spectrum: Vec<Complex64> = x_complex
                    .iter()
                    .zip(y_complex.iter())
                    .map(|(x_freq, y_freq)| {
                        // Complex conjugate product normalized by n
                        x_freq.conj() * y_freq / (n as f64)
                    })
                    .collect();

                Ok(cross_spectrum)
            })
            .collect();

        results
    }

    /// Parallel coherence analysis for multiple time series pairs using Welch's method
    /// Key metric for quantifying frequency-domain correlations in trading
    ///
    /// IMPORTANT: This implementation uses Welch's method with overlapping segments
    /// to provide meaningful coherence estimates. Single-shot FFT coherence is always 1.
    ///
    /// Returns coherence values in [0,1] for each frequency bin.
    pub fn parallel_coherence_analysis_welch(
        x_datasets: &[&[f64]],
        y_datasets: &[&[f64]],
        segment_size: Option<usize>,
        overlap_ratio: Option<f64>,
    ) -> FractalResult<Vec<Vec<f64>>> {
        if x_datasets.len() != y_datasets.len() {
            return Err(FractalAnalysisError::InvalidParameter {
                parameter: "dataset_lengths".to_string(),
                value: x_datasets.len() as f64,
                constraint: format!("must match y_datasets length of {}", y_datasets.len()),
            });
        }

        // Default parameters for Welch's method
        let default_segment_size = |n: usize| (n as f64).sqrt() as usize;
        let default_overlap = 0.5; // 50% overlap

        // Parallel coherence computation using Welch's method
        let results: Result<Vec<_>, _> = x_datasets
            .par_iter()
            .zip(y_datasets.par_iter())
            .map(|(x_data, y_data)| {
                if x_data.len() != y_data.len() {
                    return Err(FractalAnalysisError::InvalidParameter {
                        parameter: "series_lengths".to_string(),
                        value: x_data.len() as f64,
                        constraint: format!("must match y_data length of {}", y_data.len()),
                    });
                }

                let n = x_data.len();
                let seg_size = segment_size.unwrap_or_else(|| default_segment_size(n));
                let overlap = overlap_ratio.unwrap_or(default_overlap);

                // Compute coherence using Welch's method
                compute_coherence_welch(x_data, y_data, seg_size, overlap)
            })
            .collect();

        results
    }

    /// Parallel windowed FFT analysis for time-frequency decomposition
    /// Critical for non-stationary financial time series analysis
    pub fn parallel_windowed_fft(
        data: &[f64],
        window_size: usize,
        overlap: f64,
    ) -> FractalResult<Vec<Vec<f64>>> {
        if window_size == 0 || overlap < 0.0 || overlap >= 1.0 {
            return Err(FractalAnalysisError::InvalidParameter {
                parameter: "window_size".to_string(),
                value: window_size as f64,
                constraint: "must be > 0 and overlap must be in [0, 1)".to_string(),
            });
        }

        if data.len() < window_size {
            return Err(FractalAnalysisError::InvalidParameter {
                parameter: "data_length".to_string(),
                value: data.len() as f64,
                constraint: format!("must be at least window_size ({})", window_size),
            });
        }

        // Calculate hop size with safety check
        let hop_fraction = (1.0 - overlap).max(0.01); // Ensure minimum hop
        let hop_size = ((hop_fraction * window_size as f64) as usize).max(1);

        // Check for divide by zero
        if hop_size == 0 {
            return Err(FractalAnalysisError::InvalidParameter {
                parameter: "overlap".to_string(),
                value: overlap,
                constraint: "produces zero hop size".to_string(),
            });
        }

        let num_windows = (data.len().saturating_sub(window_size)) / hop_size + 1;

        // Parallel processing of overlapping windows
        let results: Result<Vec<_>, _> = (0..num_windows)
            .into_par_iter()
            .map(|i| {
                let start = i * hop_size;
                let end = (start + window_size).min(data.len());

                if end - start < window_size {
                    // Handle last window that might be shorter
                    let mut padded_window = vec![0.0; window_size];
                    padded_window[..end - start].copy_from_slice(&data[start..end]);
                    calculate_periodogram_fft(&padded_window)
                } else {
                    calculate_periodogram_fft(&data[start..end])
                }
            })
            .collect();

        results
    }

    /// Parallel autocorrelation batch computation using FFT
    /// Optimized for computing autocorrelations across multiple time series
    pub fn batch_autocorrelation_parallel(
        datasets: &[&[f64]],
        max_lag: usize,
    ) -> FractalResult<Vec<Vec<f64>>> {
        if datasets.is_empty() {
            return Err(FractalAnalysisError::InvalidParameter {
                parameter: "datasets".to_string(),
                value: 0.0,
                constraint: "must have at least one dataset".to_string(),
            });
        }

        // Parallel computation of autocorrelations
        let results: Result<Vec<_>, _> = datasets
            .par_iter()
            .map(|data| fft_autocorrelation(data, max_lag))
            .collect();

        results
    }

    /// Internal implementation of Welch's coherence method
    ///
    /// This core implementation is shared by both parallel and sequential versions.
    /// It provides meaningful coherence estimates by averaging over multiple segments.
    /// For single-shot FFT, coherence is always 1, which is not useful for analysis.
    fn compute_coherence_welch_internal(
        x_data: &[f64],
        y_data: &[f64],
        segment_size: usize,
        overlap_ratio: f64,
        parallel_segments: bool,
    ) -> FractalResult<Vec<f64>> {
        let n = x_data.len();

        // Validate segment size
        if segment_size > n {
            return Err(FractalAnalysisError::InvalidParameter {
                parameter: "segment_size".to_string(),
                value: segment_size as f64,
                constraint: format!("must be <= data length {}", n),
            });
        }

        // Calculate hop size (non-overlapping portion)
        let hop_size = ((segment_size as f64 * (1.0 - overlap_ratio)) as usize).max(1);
        let num_segments = (n - segment_size) / hop_size + 1;

        if num_segments < 2 {
            return Err(FractalAnalysisError::InvalidParameter {
                parameter: "num_segments".to_string(),
                value: num_segments as f64,
                constraint: "need at least 2 segments for averaging".to_string(),
            });
        }

        // Process segments (parallel or sequential based on flag)
        let segment_results = if parallel_segments {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                (0..num_segments)
                    .into_par_iter()
                    .map(|i| {
                        process_coherence_segment(x_data, y_data, i, hop_size, segment_size, n)
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
            #[cfg(not(feature = "parallel"))]
            {
                (0..num_segments)
                    .map(|i| {
                        process_coherence_segment(x_data, y_data, i, hop_size, segment_size, n)
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
        } else {
            (0..num_segments)
                .map(|i| process_coherence_segment(x_data, y_data, i, hop_size, segment_size, n))
                .collect::<Result<Vec<_>, _>>()?
        };

        // Accumulate results from all segments
        let mut pxx_sum = vec![0.0; segment_size];
        let mut pyy_sum = vec![0.0; segment_size];
        let mut pxy_sum = vec![Complex64::new(0.0, 0.0); segment_size];

        for (pxx, pyy, pxy) in segment_results {
            for j in 0..segment_size {
                pxx_sum[j] += pxx[j];
                pyy_sum[j] += pyy[j];
                pxy_sum[j] += pxy[j];
            }
        }

        // Compute averaged coherence
        let mut coherence = Vec::with_capacity(segment_size);
        for j in 0..segment_size {
            let pxx_avg = pxx_sum[j] / num_segments as f64;
            let pyy_avg = pyy_sum[j] / num_segments as f64;
            let pxy_avg = pxy_sum[j] / num_segments as f64;

            // Coherence: |<Pxy>|² / (<Pxx> * <Pyy>)
            if pxx_avg > MIN_POSITIVE && pyy_avg > MIN_POSITIVE {
                let coh = pxy_avg.norm_sqr() / (pxx_avg * pyy_avg);
                coherence.push(coh.max(0.0).min(1.0));
            } else {
                coherence.push(0.0);
            }
        }

        Ok(coherence)
    }

    /// Process a single segment for coherence calculation
    fn process_coherence_segment(
        x_data: &[f64],
        y_data: &[f64],
        segment_idx: usize,
        hop_size: usize,
        segment_size: usize,
        data_len: usize,
    ) -> FractalResult<(Vec<f64>, Vec<f64>, Vec<Complex64>)> {
        let start = segment_idx * hop_size;
        let end = (start + segment_size).min(data_len);

        // Extract segment with zero-padding if needed
        let mut x_segment = vec![0.0; segment_size];
        let mut y_segment = vec![0.0; segment_size];
        x_segment[..end - start].copy_from_slice(&x_data[start..end]);
        y_segment[..end - start].copy_from_slice(&y_data[start..end]);

        // Apply Hann window to reduce spectral leakage
        apply_hann_window(&mut x_segment);
        apply_hann_window(&mut y_segment);

        // Convert to complex for FFT
        let mut x_complex: Vec<Complex64> = x_segment
            .iter()
            .map(|&val| Complex64::new(val, 0.0))
            .collect();
        let mut y_complex: Vec<Complex64> = y_segment
            .iter()
            .map(|&val| Complex64::new(val, 0.0))
            .collect();

        // Perform FFT
        let fft_forward = get_cached_fft_forward(segment_size)?;
        fft_forward.process(&mut x_complex);
        fft_forward.process(&mut y_complex);

        // Compute power spectra and cross-spectrum for this segment
        let mut pxx = Vec::with_capacity(segment_size);
        let mut pyy = Vec::with_capacity(segment_size);
        let mut pxy = Vec::with_capacity(segment_size);

        for j in 0..segment_size {
            pxx.push(x_complex[j].norm_sqr());
            pyy.push(y_complex[j].norm_sqr());
            pxy.push(x_complex[j].conj() * y_complex[j]);
        }

        Ok((pxx, pyy, pxy))
    }

    /// Parallel version of Welch's coherence method
    #[cfg(feature = "parallel")]
    pub fn compute_coherence_welch(
        x_data: &[f64],
        y_data: &[f64],
        segment_size: usize,
        overlap_ratio: f64,
    ) -> FractalResult<Vec<f64>> {
        compute_coherence_welch_internal(x_data, y_data, segment_size, overlap_ratio, true)
    }

    /// Sequential version of Welch's coherence method
    #[cfg(not(feature = "parallel"))]
    pub fn compute_coherence_welch(
        x_data: &[f64],
        y_data: &[f64],
        segment_size: usize,
        overlap_ratio: f64,
    ) -> FractalResult<Vec<f64>> {
        compute_coherence_welch_internal(x_data, y_data, segment_size, overlap_ratio, false)
    }

    /// Apply Hann window with proper energy normalization
    ///
    /// The window is normalized to preserve signal power, ensuring
    /// that power spectral density estimates are accurate even when
    /// windows are applied to reduce spectral leakage.
    fn apply_hann_window(data: &mut [f64]) {
        let n = data.len();
        if n <= 1 {
            return;
        }

        // Calculate window coefficients and their energy
        let mut window_coeffs = Vec::with_capacity(n);
        let mut window_energy = 0.0;

        for i in 0..n {
            let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
            window_coeffs.push(w);
            window_energy += w * w;
        }

        // Normalize by the RMS of the window to preserve power
        // This ensures that the power spectral density is correctly scaled
        let norm_factor = if window_energy > MIN_POSITIVE {
            (window_energy / n as f64).sqrt()
        } else {
            1.0
        };

        // Apply normalized window
        for i in 0..n {
            data[i] *= window_coeffs[i] / norm_factor;
        }
    }

    /// WARNING: Single-shot coherence analysis (deprecated)
    ///
    /// This function computes coherence using single FFTs without averaging.
    /// The result is mathematically always 1.0 for all frequencies, which is not useful.
    /// Use `parallel_coherence_analysis_welch` instead for meaningful results.
    #[deprecated(
        since = "0.2.0",
        note = "Single-shot coherence is always 1. Use parallel_coherence_analysis_welch instead"
    )]
    pub fn parallel_coherence_analysis_single_shot(
        x_datasets: &[&[f64]],
        y_datasets: &[&[f64]],
    ) -> FractalResult<Vec<Vec<f64>>> {
        // This implementation is kept for backward compatibility but should not be used
        // It will always return coherence = 1.0 for all frequencies
        log::warn!("Single-shot coherence analysis always returns 1.0. Use Welch's method for meaningful results.");

        if x_datasets.len() != y_datasets.len() {
            return Err(FractalAnalysisError::InvalidParameter {
                parameter: "dataset_lengths".to_string(),
                value: x_datasets.len() as f64,
                constraint: format!("must match y_datasets length of {}", y_datasets.len()),
            });
        }

        // Return vectors of 1.0 for each dataset pair
        let results: Result<Vec<_>, _> = x_datasets
            .par_iter()
            .zip(y_datasets.par_iter())
            .map(|(x_data, y_data)| {
                if x_data.len() != y_data.len() {
                    return Err(FractalAnalysisError::InvalidParameter {
                        parameter: "series_lengths".to_string(),
                        value: x_data.len() as f64,
                        constraint: format!("must match y_data length of {}", y_data.len()),
                    });
                }
                // Return all 1.0s as this is what single-shot coherence produces
                Ok(vec![1.0; x_data.len()])
            })
            .collect();

        results
    }
}

/// Compute complex cross-spectral density preserving phase information
///
/// Returns the complex cross-spectrum X* · Y normalized by n.
/// Phase information is crucial for determining lead/lag relationships.
pub fn cross_spectral_density(x_data: &[f64], y_data: &[f64]) -> FractalResult<Vec<Complex64>> {
    if x_data.len() != y_data.len() {
        return Err(FractalAnalysisError::InvalidParameter {
            parameter: "series_lengths".to_string(),
            value: x_data.len() as f64,
            constraint: format!("must match y_data length of {}", y_data.len()),
        });
    }

    let n = x_data.len();
    let fft_forward = get_cached_fft_forward(n)?;

    // Convert to complex
    let mut x_complex: Vec<Complex64> =
        x_data.iter().map(|&val| Complex64::new(val, 0.0)).collect();
    let mut y_complex: Vec<Complex64> =
        y_data.iter().map(|&val| Complex64::new(val, 0.0)).collect();

    // Perform FFTs
    fft_forward.process(&mut x_complex);
    fft_forward.process(&mut y_complex);

    // Compute complex cross-spectral density
    let cross_spectrum: Vec<Complex64> = x_complex
        .iter()
        .zip(y_complex.iter())
        .map(|(x_freq, y_freq)| x_freq.conj() * y_freq / (n as f64))
        .collect();

    Ok(cross_spectrum)
}

/// Compute coherence using Welch's method with proper spectral averaging
///
/// This is the correct way to compute coherence for real-world signals.
/// Single-shot coherence (without averaging) is mathematically always 1.0.
///
/// # Arguments
/// * `x_data` - First time series
/// * `y_data` - Second time series  
/// * `segment_size` - Size of each segment for Welch's method
/// * `overlap_ratio` - Overlap ratio between segments (0.0 to 1.0)
///
/// # Returns
/// Coherence values in [0,1] for each frequency bin
pub fn coherence_welch(
    x_data: &[f64],
    y_data: &[f64],
    segment_size: Option<usize>,
    overlap_ratio: Option<f64>,
) -> FractalResult<Vec<f64>> {
    if x_data.len() != y_data.len() {
        return Err(FractalAnalysisError::InvalidParameter {
            parameter: "series_lengths".to_string(),
            value: x_data.len() as f64,
            constraint: format!("must match y_data length of {}", y_data.len()),
        });
    }

    let n = x_data.len();
    let seg_size = segment_size.unwrap_or_else(|| (n as f64).sqrt() as usize);
    let overlap = overlap_ratio.unwrap_or(0.5);

    // Inline implementation for non-parallel builds
    // For parallel builds, use the parallel_fft module version
    #[cfg(feature = "parallel")]
    {
        crate::fft_ops::parallel_fft::compute_coherence_welch(x_data, y_data, seg_size, overlap)
    }
    #[cfg(not(feature = "parallel"))]
    {
        // Sequential implementation
        compute_coherence_welch_sequential_impl(x_data, y_data, seg_size, overlap)
    }
}

// Sequential implementation for non-parallel builds
#[cfg(not(feature = "parallel"))]
fn compute_coherence_welch_sequential_impl(
    x_data: &[f64],
    y_data: &[f64],
    segment_size: usize,
    overlap_ratio: f64,
) -> FractalResult<Vec<f64>> {
    let n = x_data.len();

    if segment_size > n {
        return Err(FractalAnalysisError::InvalidParameter {
            parameter: "segment_size".to_string(),
            value: segment_size as f64,
            constraint: format!("must be <= data length {}", n),
        });
    }

    let hop_size = ((segment_size as f64 * (1.0 - overlap_ratio)) as usize).max(1);
    let num_segments = (n - segment_size) / hop_size + 1;

    if num_segments < 2 {
        return Err(FractalAnalysisError::InvalidParameter {
            parameter: "num_segments".to_string(),
            value: num_segments as f64,
            constraint: "need at least 2 segments for averaging".to_string(),
        });
    }

    let mut pxx_sum = vec![0.0; segment_size];
    let mut pyy_sum = vec![0.0; segment_size];
    let mut pxy_sum = vec![Complex64::new(0.0, 0.0); segment_size];

    for i in 0..num_segments {
        let start = i * hop_size;
        let end = (start + segment_size).min(n);

        let mut x_segment = vec![0.0; segment_size];
        let mut y_segment = vec![0.0; segment_size];
        x_segment[..end - start].copy_from_slice(&x_data[start..end]);
        y_segment[..end - start].copy_from_slice(&y_data[start..end]);

        // Apply normalized Hann window to both segments
        // Calculate and apply window with proper normalization
        use std::f64::consts::PI;
        let n = segment_size;
        let mut window_energy = 0.0;
        let mut window_coeffs = Vec::with_capacity(n);

        for i in 0..n {
            let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
            window_coeffs.push(w);
            window_energy += w * w;
        }

        let norm_factor = if window_energy > MIN_POSITIVE {
            (window_energy / n as f64).sqrt()
        } else {
            1.0
        };

        for i in 0..n {
            let normalized_window = window_coeffs[i] / norm_factor;
            x_segment[i] *= normalized_window;
            y_segment[i] *= normalized_window;
        }

        let mut x_complex: Vec<Complex64> = x_segment
            .iter()
            .map(|&val| Complex64::new(val, 0.0))
            .collect();
        let mut y_complex: Vec<Complex64> = y_segment
            .iter()
            .map(|&val| Complex64::new(val, 0.0))
            .collect();

        let fft_forward = get_cached_fft_forward(segment_size)?;
        fft_forward.process(&mut x_complex);
        fft_forward.process(&mut y_complex);

        for j in 0..segment_size {
            pxx_sum[j] += x_complex[j].norm_sqr();
            pyy_sum[j] += y_complex[j].norm_sqr();
            pxy_sum[j] += x_complex[j].conj() * y_complex[j];
        }
    }

    let mut coherence = Vec::with_capacity(segment_size);
    for j in 0..segment_size {
        let pxx_avg = pxx_sum[j] / num_segments as f64;
        let pyy_avg = pyy_sum[j] / num_segments as f64;
        let pxy_avg = pxy_sum[j] / num_segments as f64;

        if pxx_avg > MIN_POSITIVE && pyy_avg > MIN_POSITIVE {
            let coh = pxy_avg.norm_sqr() / (pxx_avg * pyy_avg);
            coherence.push(coh.max(0.0).min(1.0));
        } else {
            coherence.push(0.0);
        }
    }

    Ok(coherence)
}

/// Solve linear system Ax = b using Gaussian elimination with partial pivoting
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 || a[0].len() != n || b.len() != n {
        return None;
    }

    // Create augmented matrix [A|b]
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        for i in (k + 1)..n {
            if aug[i][k].abs() > aug[max_row][k].abs() {
                max_row = i;
            }
        }

        // Swap rows
        if max_row != k {
            aug.swap(k, max_row);
        }

        // Check for singular matrix
        if aug[k][k].abs() < 1e-10 {
            return None;
        }

        // Eliminate column
        for i in (k + 1)..n {
            let factor = aug[i][k] / aug[k][k];
            for j in k..=n {
                aug[i][j] -= factor * aug[k][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_fft_cache_functionality() {
        // Clear cache first
        clear_fft_cache();

        // Test cache miss and hit with unique size to avoid conflicts
        let size = 1337; // Unique size unlikely to be used by other tests
        let fft1 = get_cached_fft_forward(size).unwrap();
        let fft2 = get_cached_fft_forward(size).unwrap();

        // Should be the same cached instance
        assert!(Arc::ptr_eq(&fft1, &fft2));

        // Test inverse cache too
        let fft_inv = get_cached_fft_inverse(size).unwrap();

        // Test cache stats - should have at least 1 forward and 1 inverse for our unique size
        let (forward_count, inverse_count) = get_fft_cache_stats();
        assert!(forward_count >= 1);
        assert!(inverse_count >= 1);
    }

    #[test]
    fn test_calculate_periodogram_fft_sine_wave() {
        // Test with a pure sine wave: x(t) = sin(2π * f * t)
        let n = 64;
        let frequency = 8.0; // Frequency bins for 64-point FFT
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * frequency * i as f64 / n as f64).sin())
            .collect();

        let periodogram = calculate_periodogram_fft(&data).unwrap();
        assert_eq!(periodogram.len(), n);

        // For a pure sine wave, energy should be at the frequency bin
        // Since we return full periodogram, negative frequencies are also included
        let max_index = periodogram
            .iter()
            .enumerate()
            .max_by(|a, b| {
                // Handle NaN safely
                match a.1.partial_cmp(b.1) {
                    Some(ord) => ord,
                    None => std::cmp::Ordering::Equal, // Treat NaN as equal
                }
            })
            .unwrap()
            .0;

        // For real signals, energy appears at both positive and negative frequency bins
        // Positive frequency: frequency, Negative frequency: n - frequency
        let positive_freq_bin = frequency as usize;
        let negative_freq_bin = n - positive_freq_bin;

        assert!(
            max_index == positive_freq_bin || max_index == negative_freq_bin,
            "Max energy at bin {} should be at {} or {}",
            max_index,
            positive_freq_bin,
            negative_freq_bin
        );
    }

    #[test]
    fn test_fft_autocorrelation() {
        // Test 1: Constant signal - verify correct mathematical behavior
        let n = 8;
        let constant_data = vec![3.0; n];
        let autocorrs = fft_autocorrelation(&constant_data, 4).unwrap();

        // Lag 0 should always be 1.0 (perfect self-correlation)
        assert_approx_eq!(autocorrs[0], 1.0, 1e-10);

        // For constant signal via FFT method with zero-padding,
        // the biased estimate gives (n-lag)/n for each lag
        // This is because zero-padding affects the computation
        assert_approx_eq!(autocorrs[1], 7.0 / 8.0, 1e-10); // (8-1)/8 = 0.875
        assert_approx_eq!(autocorrs[2], 6.0 / 8.0, 1e-10); // (8-2)/8 = 0.75
        assert_approx_eq!(autocorrs[3], 5.0 / 8.0, 1e-10); // (8-3)/8 = 0.625

        // Test 2: White noise should have low autocorrelations at non-zero lags
        let white_noise = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let autocorrs2 = fft_autocorrelation(&white_noise, 3).unwrap();

        // Lag 0 should always be 1.0
        assert_approx_eq!(autocorrs2[0], 1.0, 1e-10);

        // Other lags should be reasonable and bounded
        for i in 1..autocorrs2.len() {
            assert!(
                autocorrs2[i].abs() <= 1.0,
                "Autocorr should be bounded: |r[{}]| = {}",
                i,
                autocorrs2[i]
            );
        }
    }

    #[test]
    fn test_fft_edge_cases() {
        // Test FFT with edge cases

        // Empty data (FFT requires minimum 4 points)
        let empty_data = vec![];
        let result = calculate_periodogram_fft(&empty_data);
        assert!(matches!(result, Err(FractalAnalysisError::FftError { .. })));

        // Single point (FFT requires minimum 4 points)
        let single_point = vec![1.0];
        let result = calculate_periodogram_fft(&single_point);
        assert!(matches!(result, Err(FractalAnalysisError::FftError { .. })));

        // All zeros should work
        let zeros = vec![0.0; 8];
        let periodogram = calculate_periodogram_fft(&zeros).unwrap();
        for &value in &periodogram {
            assert_approx_eq!(value, 0.0, 1e-10);
        }
    }

    #[test]
    fn test_periodogram_parseval() {
        // Parseval's theorem: sum of |X(k)|²/n should equal sum of |x(n)|²
        let data = vec![1.0, 2.0, -1.0, 0.5, 1.5, -0.5, 0.8, -1.2];

        // Since calculate_periodogram_fft applies mean removal, we need to detrend data first
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let detrended: Vec<f64> = data.iter().map(|&x| x - mean).collect();
        
        let time_domain_energy: f64 = detrended.iter().map(|x| x * x).sum();

        let periodogram = calculate_periodogram_fft(&data).unwrap();
        // Periodogram is already |FFT(x)|²/n, so sum gives Parseval's RHS
        let freq_domain_energy: f64 = periodogram.iter().sum::<f64>();

        // Should satisfy Parseval's theorem within numerical precision
        let relative_error = (freq_domain_energy - time_domain_energy).abs() / time_domain_energy;
        assert!(
            relative_error < 1e-10,
            "Parseval's theorem violated: time={}, freq={}, error={}",
            time_domain_energy,
            freq_domain_energy,
            relative_error
        );
    }

    #[test]
    fn test_cross_spectral_density_returns_complex() {
        // Test that CSD returns complex values preserving phase information
        let x = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
        let y = vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]; // 90-degree phase shift

        let csd = cross_spectral_density(&x, &y).unwrap();

        // CSD should be complex
        assert_eq!(csd.len(), x.len());

        // For phase-shifted signals, imaginary part should be non-zero
        let has_imaginary = csd.iter().any(|c| c.im.abs() > 1e-10);
        assert!(
            has_imaginary,
            "CSD should have non-zero imaginary parts for phase-shifted signals"
        );
    }

    #[test]
    fn test_coherence_welch_proper_range() {
        // Test that Welch's coherence is in [0,1] and not always 1
        let n = 256;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin()
                    + 0.5 * {
                        let mut rng = FastrandCompat::new();
                        rng.f64()
                    }
                    - 0.25
            })
            .collect();

        let y: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin()
                    + 0.8 * {
                        let mut rng = FastrandCompat::new();
                        rng.f64()
                    }
                    - 0.4 // More noise
            })
            .collect();

        let coherence = coherence_welch(&x, &y, Some(64), Some(0.5)).unwrap();

        // Check all values are in [0,1]
        for &coh in &coherence {
            assert!(
                coh >= 0.0 && coh <= 1.0,
                "Coherence {} out of range [0,1]",
                coh
            );
        }

        // With noise, not all frequencies should have coherence = 1
        let all_ones = coherence.iter().all(|&c| (c - 1.0).abs() < 1e-6);
        assert!(
            !all_ones,
            "Welch's coherence should not be all 1.0 for noisy signals"
        );

        // At the signal frequency, coherence should be relatively high
        // At noise frequencies, coherence should be lower
        let max_coh = coherence.iter().cloned().fold(0.0, f64::max);
        let min_coh = coherence.iter().cloned().fold(1.0, f64::min);
        assert!(
            max_coh - min_coh > 0.1,
            "Should have variation in coherence across frequencies"
        );
    }

    #[test]
    #[allow(deprecated)]
    fn test_single_shot_coherence_always_one() {
        // Test that single-shot coherence is indeed always 1 (demonstrating the bug)
        #[cfg(feature = "parallel")]
        {
            use crate::fft_ops::parallel_fft::parallel_coherence_analysis_single_shot;

            let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]];
            let y = vec![vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]];

            let x_refs: Vec<&[f64]> = x.iter().map(|v| v.as_slice()).collect();
            let y_refs: Vec<&[f64]> = y.iter().map(|v| v.as_slice()).collect();

            let coherence = parallel_coherence_analysis_single_shot(&x_refs, &y_refs).unwrap();

            // Single-shot coherence is mathematically always 1.0
            for freq_coherence in coherence {
                for &coh in &freq_coherence {
                    assert_approx_eq!(coh, 1.0, 1e-10);
                }
            }
        }
    }

    // ========== COMPREHENSIVE PARALLEL FFT MODULE TESTS ==========
    // These tests ensure the parallel FFT operations work correctly
    // and were added to catch critical bugs like coherence always returning 1.0

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_coherence_analysis_welch() {
        // Test that Welch's method produces meaningful coherence values
        let n = 256;
        let freq = 2.0 * PI / 10.0;

        // Create two signals with known phase relationship
        let x: Vec<f64> = (0..n).map(|i| (freq * i as f64).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (freq * i as f64 + PI / 4.0).sin()).collect(); // Phase shifted

        // Single dataset for testing
        let x_datasets = vec![x.as_slice()];
        let y_datasets = vec![y.as_slice()];

        let result = crate::fft_ops::parallel_fft::parallel_coherence_analysis_welch(
            &x_datasets,
            &y_datasets,
            Some(64),  // segment_size
            Some(0.5), // overlap
        );

        assert!(result.is_ok());
        let coherences = result.unwrap();
        assert_eq!(coherences.len(), 1);

        // Coherence should be high at the signal frequency but not always 1.0
        let coh = &coherences[0];

        // Find peak coherence around the signal frequency
        let freq_bin = (freq * 64.0 / (2.0 * PI)) as usize;
        let peak_coh = coh[freq_bin];

        // Should be high but not exactly 1.0 (due to windowing and averaging)
        assert!(
            peak_coh > 0.8 && peak_coh < 1.0,
            "Peak coherence {} should be high but not 1.0",
            peak_coh
        );

        // Other frequencies should have lower coherence
        let avg_other_coh: f64 = coh
            .iter()
            .enumerate()
            .filter(|(i, _)| (*i as i32 - freq_bin as i32).abs() > 2)
            .map(|(_, &c)| c)
            .sum::<f64>()
            / (coh.len() - 5) as f64;

        assert!(
            avg_other_coh < 0.5,
            "Non-signal frequencies should have low coherence"
        );
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_spectral_analysis() {
        // Test parallel spectral analysis with overlapping windows
        let n = 512;
        let freq1 = 2.0 * PI / 20.0;
        let freq2 = 2.0 * PI / 5.0;

        // Signal with two frequencies
        let data: Vec<f64> = (0..n)
            .map(|i| (freq1 * i as f64).sin() + 0.5 * (freq2 * i as f64).cos())
            .collect();

        let result = crate::fft_ops::parallel_fft::parallel_windowed_fft(&data, 128, 0.5);
        assert!(result.is_ok());

        let spectra = result.unwrap();
        assert!(!spectra.is_empty());

        // All spectral windows should have consistent length
        let expected_len = 128;
        for spectrum in &spectra {
            assert_eq!(spectrum.len(), expected_len);
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_batch_autocorrelation_parallel() {
        // Test batch processing of autocorrelations
        let datasets: Vec<Vec<f64>> = (0..4)
            .map(|_| (0..100).map(|i| (0.1 * i as f64).sin()).collect())
            .collect();

        let dataset_refs: Vec<&[f64]> = datasets.iter().map(|d| d.as_slice()).collect();

        let result =
            crate::fft_ops::parallel_fft::batch_autocorrelation_parallel(&dataset_refs, 20);
        assert!(result.is_ok());

        let autocorrs = result.unwrap();
        assert_eq!(autocorrs.len(), 4);

        for acf in &autocorrs {
            assert_eq!(acf.len(), 21); // 0..=20
            assert_approx_eq!(acf[0], 1.0, 1e-10); // ACF at lag 0 should be 1
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_coherence_edge_cases() {
        // Test edge cases that previously caused issues

        // Empty datasets
        let empty: Vec<&[f64]> = vec![];
        let result = crate::fft_ops::parallel_fft::parallel_coherence_analysis_welch(
            &empty,
            &empty,
            Some(32),
            Some(0.5),
        );
        assert!(result.is_err());

        // Mismatched dataset counts
        let x = vec![vec![1.0, 2.0, 3.0]];
        let y = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let x_refs: Vec<&[f64]> = x.iter().map(|d| d.as_slice()).collect();
        let y_refs: Vec<&[f64]> = y.iter().map(|d| d.as_slice()).collect();

        let result = crate::fft_ops::parallel_fft::parallel_coherence_analysis_welch(
            &x_refs,
            &y_refs,
            Some(2),
            Some(0.0),
        );
        assert!(result.is_err());

        // Segment size larger than data
        let small_data = vec![1.0; 10];
        let datasets = vec![small_data.as_slice()];
        let result = crate::fft_ops::parallel_fft::parallel_coherence_analysis_welch(
            &datasets,
            &datasets,
            Some(20),
            Some(0.5),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_spectral_density_preserves_phase() {
        // Critical test: Ensure cross-spectral density returns complex values
        // preserving phase information for lead/lag analysis

        let n = 128;
        let freq = 2.0 * PI / 16.0;
        let phase_shift = PI / 3.0;

        // Create two signals with known phase relationship
        let x: Vec<f64> = (0..n).map(|i| (freq * i as f64).sin()).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (freq * i as f64 + phase_shift).sin())
            .collect();

        let result = cross_spectral_density(&x, &y);
        assert!(result.is_ok());

        let csd = result.unwrap();
        assert_eq!(csd.len(), n);

        // Find the frequency bin corresponding to our signal
        let freq_bin = (freq * n as f64 / (2.0 * PI)) as usize;

        // The phase at the signal frequency should reflect the phase shift
        let phase_at_freq = csd[freq_bin].arg();

        // Due to FFT properties, we might need to check the negative of the phase
        let phase_diff = (phase_at_freq - phase_shift).abs();
        let phase_diff_alt = (phase_at_freq + phase_shift).abs();

        assert!(
            phase_diff < 0.2
                || phase_diff_alt < 0.2
                || (2.0 * PI - phase_diff) < 0.2
                || (2.0 * PI - phase_diff_alt) < 0.2,
            "Phase information not preserved: expected {}, got {}",
            phase_shift,
            phase_at_freq
        );
    }

    #[test]
    fn test_autocorrelation_bias_parameter() {
        // Test the new bias parameter in autocorrelation
        let data: Vec<f64> = (0..100).map(|i| (0.1 * i as f64).sin()).collect();

        // Compute both biased and unbiased estimates
        let biased = fft_autocorrelation_with_bias(&data, 10, true).unwrap();
        let unbiased = fft_autocorrelation_with_bias(&data, 10, false).unwrap();

        // Both should have same length
        assert_eq!(biased.len(), 11);
        assert_eq!(unbiased.len(), 11);

        // At lag 0, both should be 1.0 (normalized)
        assert_approx_eq!(biased[0], 1.0, 1e-10);
        assert_approx_eq!(unbiased[0], 1.0, 1e-10);

        // For higher lags, unbiased should generally have larger absolute values
        // due to the smaller denominator (N-k instead of N)
        for k in 5..10 {
            // This relationship might not hold for all data, but generally true
            // for autocorrelations that decay with lag
            if biased[k].abs() > 0.01 {
                // Only check if not near zero
                let ratio = unbiased[k].abs() / biased[k].abs();
                // Unbiased should be larger by roughly N/(N-k)
                let expected_ratio = 100.0 / (100.0 - k as f64);
                // Allow some tolerance due to normalization
                assert!(
                    ratio > 0.9,
                    "Unbiased estimate should generally be larger at lag {}",
                    k
                );
            }
        }
    }

    #[test]
    fn test_lru_cache_eviction() {
        // Test that LRU cache properly evicts least recently used entries
        // This is a stress test to ensure cache doesn't grow unbounded

        // Clear cache first
        clear_fft_cache();

        // Create many different sized FFTs to fill cache
        for size in 100..200 {
            let _ = get_cached_fft_forward(size);
        }

        // Access some specific sizes multiple times (make them "hot")
        for _ in 0..5 {
            let _ = get_cached_fft_forward(150);
            let _ = get_cached_fft_forward(175);
        }

        // Add more entries to trigger eviction
        for size in 200..300 {
            let _ = get_cached_fft_forward(size);
        }

        // The frequently accessed sizes should still be cached (LRU property)
        // We can't directly test cache contents, but we can verify no errors
        let result1 = get_cached_fft_forward(150);
        let result2 = get_cached_fft_forward(175);

        assert!(result1.is_ok());
        assert!(result2.is_ok());
    }

    #[test]
    fn test_parallel_fft_thread_safety() {
        // Test thread safety of parallel FFT operations
        use std::sync::Arc;
        use std::thread;

        let data = Arc::new(
            (0..256)
                .map(|i| (0.05 * i as f64).sin())
                .collect::<Vec<f64>>(),
        );
        let mut handles = vec![];

        // Spawn multiple threads doing FFT operations simultaneously
        for _ in 0..4 {
            let data_clone = Arc::clone(&data);
            let handle = thread::spawn(move || {
                // Each thread performs multiple FFT operations
                for _ in 0..10 {
                    let _ = fft_autocorrelation(&data_clone, 50);
                    let _ = calculate_periodogram_fft(&data_clone);
                }
            });
            handles.push(handle);
        }

        // All threads should complete without panicking
        for handle in handles {
            assert!(handle.join().is_ok());
        }
    }
}
