//! Automatic block size selection for block bootstrap methods.
//!
//! This module implements the Politis & White (2004) automatic block size
//! selection algorithm and related methods for determining optimal block
//! sizes in block bootstrap procedures for time series data.

use crate::bootstrap_config::MAX_BLOCK_SIZE_RATIO;

/// Multivariate Politis & White automatic block size selection.
///
/// Extends the Politis & White (2004) method to paired time series data
/// by considering the autocorrelation structure of both series.
///
/// # Arguments
/// * `data` - Paired time series data as (x, y) tuples
///
/// # Returns
/// Optimal block size that accounts for both series' temporal dependencies
pub fn multivariate_politis_white_block_size(data: &[(f64, f64)]) -> usize {
    let n = data.len();

    // For very small samples, use simple heuristic
    if n < 50 {
        return ((n as f64).powf(1.0 / 3.0) as usize).max(2);
    }

    // Extract both series
    let first_series: Vec<f64> = data.iter().map(|(x, _)| *x).collect();
    let second_series: Vec<f64> = data.iter().map(|(_, y)| *y).collect();

    // Calculate optimal block sizes for each series
    let block_size_x = politis_white_block_size(&first_series);
    let block_size_y = politis_white_block_size(&second_series);

    // Strategy 1: Conservative approach - use the larger block size
    // This ensures both series' dependencies are adequately captured
    let conservative_size = block_size_x.max(block_size_y);

    // Strategy 2: Consider the strength of autocorrelation in each series
    // Calculate first-order autocorrelations as a proxy for dependency strength
    let mean_x = first_series.iter().sum::<f64>() / n as f64;
    let mean_y = second_series.iter().sum::<f64>() / n as f64;

    let var_x = first_series
        .iter()
        .map(|&x| (x - mean_x).powi(2))
        .sum::<f64>()
        / n as f64;
    let var_y = second_series
        .iter()
        .map(|&y| (y - mean_y).powi(2))
        .sum::<f64>()
        / n as f64;

    // Calculate first-order autocorrelations
    let mut autocorr_x = 0.0;
    let mut autocorr_y = 0.0;

    if var_x > 1e-10 {
        for i in 1..n {
            autocorr_x += (first_series[i] - mean_x) * (first_series[i - 1] - mean_x);
        }
        autocorr_x /= (n - 1) as f64 * var_x;
    }

    if var_y > 1e-10 {
        for i in 1..n {
            autocorr_y += (second_series[i] - mean_y) * (second_series[i - 1] - mean_y);
        }
        autocorr_y /= (n - 1) as f64 * var_y;
    }

    // Weighted average based on autocorrelation strength
    let weight_x = autocorr_x.abs();
    let weight_y = autocorr_y.abs();

    // If the series are very similar (identical or near-identical), return the same as single series
    if (block_size_x as i32 - block_size_y as i32).abs() <= 1 {
        // For identical or nearly identical series, use the average (which should be almost the same)
        return ((block_size_x + block_size_y) / 2)
            .max(2) // Ensure minimum block size of 2
            .min(n / 2); // Cap at 1/2 of data length, same as single series
    }

    let optimal_size = if weight_x + weight_y > 1e-10 {
        // Weighted average when both series have dependencies
        // Give more weight to the series with stronger autocorrelation
        let weighted_avg = (block_size_x as f64 * weight_x + block_size_y as f64 * weight_y)
            / (weight_x + weight_y);
        weighted_avg.round() as usize
    } else {
        // If no significant autocorrelation, use conservative approach
        conservative_size
    };

    // For safety, ensure we're not underestimating: use at least the conservative size
    // but not more than necessary (cap at reasonable maximum)
    // Use n/2 cap to be consistent with single series politis_white_block_size
    optimal_size
        .max(conservative_size)
        .max(2) // Ensure minimum block size of 2
        .min(n / 2) // Cap at 1/2 of data length, same as single series
}

/// Politis & White (2004) automatic block size selection for block bootstrap.
///
/// This method provides data-driven selection of the optimal block size
/// based on the spectral density at frequency zero and the autocorrelation structure.
///
/// Reference: Politis, D.N. and White, H. (2004). "Automatic block-length selection
/// for the dependent bootstrap." Econometric Reviews, 23(1), 53-70.
pub fn politis_white_block_size(data: &[f64]) -> usize {
    let n = data.len();

    // For very small samples, use simple heuristic
    if n < 50 {
        return ((n as f64).powf(1.0 / 3.0) as usize).max(2);
    }

    // Calculate mean
    let mean = data.iter().sum::<f64>() / n as f64;

    // Calculate autocorrelations up to lag K_n
    let k_max = (10.0 * (n as f64).ln() / (2.0_f64).ln()).min(n as f64 / 4.0) as usize;
    let mut autocorr = vec![0.0; k_max + 1];
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if variance < 1e-10 {
        // Fast-exit for near-constant series
        return 2; // Minimum meaningful block size
    }

    // Calculate autocorrelations
    for k in 0..=k_max {
        let mut sum = 0.0;
        for i in k..n {
            sum += (data[i] - mean) * (data[i - k] - mean);
        }
        autocorr[k] = sum / (n as f64 * variance);
    }

    // Apply the flat-top kernel to autocorrelations
    let mut weighted_sum = autocorr[0]; // rho(0) = 1
    let bandwidth = select_bandwidth(&autocorr, n);

    for k in 1..=k_max {
        let lambda = k as f64 / bandwidth;
        let weight = flat_top_kernel(lambda);
        weighted_sum += 2.0 * weight * autocorr[k];
    }

    // Convert autocorrelations to autocovariances and estimate spectral density at zero
    // Spectral density = sum of weighted autocovariances, not autocorrelations
    // Note: avoid shadowing the variance variable computed above
    let var = data.iter().map(|&x| x * x).sum::<f64>() / n as f64 - mean * mean;
    let spectral_density_zero = (weighted_sum * var).max(1e-12);

    // Calculate optimal block size using Politis & White formula
    // For circular block bootstrap: b_opt = (2 * pi * f(0) * n)^(1/3)
    // For non-circular: multiply by 1.5
    let optimal_block_size =
        (2.0 * std::f64::consts::PI * spectral_density_zero * n as f64).powf(1.0 / 3.0);

    // Apply factor for non-circular block bootstrap
    let adjusted_block_size = (1.5 * optimal_block_size) as usize;

    // Ensure reasonable bounds and enforce MAX_BLOCK_SIZE_RATIO
    let max_block = ((n as f64 * MAX_BLOCK_SIZE_RATIO) as usize).max(2);
    adjusted_block_size.max(2).min(max_block)
}

/// Select bandwidth for spectral density estimation using data-driven method.
fn select_bandwidth(autocorr: &[f64], n: usize) -> f64 {
    let k_max = autocorr.len() - 1;

    // Find first insignificant autocorrelation (simple rule)
    let threshold = 2.0 / (n as f64).sqrt();

    for k in 1..=k_max {
        if autocorr[k].abs() < threshold {
            // Use 1.5 times the first insignificant lag as bandwidth
            return (1.5 * k as f64).min(k_max as f64);
        }
    }

    // If all are significant, use sqrt(n) rule
    (n as f64).sqrt().min(k_max as f64)
}

/// Flat-top kernel for spectral density estimation.
fn flat_top_kernel(x: f64) -> f64 {
    let abs_x = x.abs();

    if abs_x <= 0.5 {
        1.0
    } else if abs_x <= 1.0 {
        2.0 * (1.0 - abs_x)
    } else {
        0.0
    }
}