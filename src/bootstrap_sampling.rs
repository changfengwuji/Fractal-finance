//! Bootstrap sample generation methods.
//!
//! This module provides various bootstrap resampling techniques for generating
//! bootstrap samples from time series data, including standard (iid), block,
//! stationary, and circular bootstrap methods.

use crate::{
    bootstrap_config::{BootstrapConfiguration, BootstrapMethod, MAX_BLOCK_SIZE_RATIO},
    block_size::politis_white_block_size,
    errors::{FractalAnalysisError, FractalResult},
    secure_rng::{SecureRng, ThreadLocalRng},
};

// Golden ratio constant for seed mixing to ensure good distribution
const GOLDEN_RATIO_SEED_MIX: u64 = 0x9E3779B97F4A7C15;

/// Mix seed with iteration index for deterministic, decorrelated random streams.
/// 
/// Uses golden ratio multiplication and bit rotation to ensure good distribution
/// and avoid correlation between adjacent seeds.
///
/// # Arguments
/// * `base_seed` - Base seed value
/// * `index` - Iteration index
///
/// # Returns
/// Mixed seed value
pub fn mix_seed(base_seed: u64, index: usize) -> u64 {
    base_seed
        .wrapping_mul(GOLDEN_RATIO_SEED_MIX)
        .wrapping_add(index as u64)
        .rotate_left(17)
}

/// Generate bootstrap sample according to the specified method.
///
/// Creates a new bootstrap sample using the configured resampling method.
/// This allocates a new vector for the sample.
///
/// # Seeding Behavior
/// 
/// When a seed is provided in the configuration:
/// - **Single call**: Returns the same sample every time when called with the same seed
/// - **Bootstrap loops**: The main `bootstrap_validate` function mixes the seed with the 
///   iteration index using `mix_seed(base_seed, iteration)` to generate different but
///   deterministic samples for each bootstrap iteration
/// - **Reproducibility**: Using the same seed will produce identical results across runs,
///   both in parallel and sequential execution modes
/// 
/// This design allows both:
/// 1. Generating a single reproducible sample (when called directly)
/// 2. Generating many different samples deterministically (when used in bootstrap loops)
///
/// If no seed is provided, a cryptographically secure random generator is used.
pub fn generate_bootstrap_sample(
    data: &[f64],
    config: &BootstrapConfiguration,
) -> FractalResult<Vec<f64>> {
    // Validate block size if explicitly provided
    if let Some(block_size) = config.block_size {
        let max_allowed = (data.len() as f64 * MAX_BLOCK_SIZE_RATIO) as usize;
        if block_size > max_allowed {
            return Err(FractalAnalysisError::InvalidParameter {
                parameter: "block_size".to_string(),
                value: block_size as f64,
                constraint: format!("Must not exceed {} ({}% of data length {})", 
                    max_allowed, (MAX_BLOCK_SIZE_RATIO * 100.0) as i32, data.len()),
            });
        }
    }
    
    if let Some(seed) = config.seed {
        // Use seeded RNG for deterministic behavior
        let mut rng = SecureRng::with_seed(seed);
        match config.bootstrap_method {
            BootstrapMethod::Standard => Ok(standard_bootstrap_sample_with_rng(data, &mut rng)),
            BootstrapMethod::Block => {
                let block_size = config
                    .block_size
                    .unwrap_or_else(|| politis_white_block_size(data));
                Ok(block_bootstrap_sample_with_rng(data, block_size, &mut rng))
            }
            BootstrapMethod::Stationary => {
                let block_size = config.block_size.unwrap_or_else(|| politis_white_block_size(data));
                Ok(stationary_bootstrap_sample_with_rng(
                    data, block_size, &mut rng,
                ))
            }
            BootstrapMethod::Circular => Ok(circular_bootstrap_sample_with_rng(data, &mut rng)),
        }
    } else {
        // Use thread-local RNG for performance
        match config.bootstrap_method {
            BootstrapMethod::Standard => Ok(standard_bootstrap_sample(data)),
            BootstrapMethod::Block => {
                let block_size = config
                    .block_size
                    .unwrap_or_else(|| politis_white_block_size(data));
                Ok(block_bootstrap_sample(data, block_size))
            }
            BootstrapMethod::Stationary => {
                let block_size = config.block_size.unwrap_or_else(|| politis_white_block_size(data));
                Ok(stationary_bootstrap_sample(data, block_size))
            }
            BootstrapMethod::Circular => Ok(circular_bootstrap_sample(data)),
        }
    }
}

/// Generate a bootstrap sample in-place to avoid memory allocation.
///
/// More efficient version that reuses an existing buffer to avoid
/// repeated memory allocation during bootstrap validation.
///
/// # Seeding Behavior
/// 
/// Same as `generate_bootstrap_sample` - when called with a fixed seed, returns
/// the same sample every time. Bootstrap loops should mix the seed with iteration
/// index for different samples. See `generate_bootstrap_sample` for details.
pub fn generate_bootstrap_sample_inplace(
    data: &[f64],
    config: &BootstrapConfiguration,
    buffer: &mut [f64],
) -> FractalResult<()> {
    if buffer.len() != data.len() {
        return Err(FractalAnalysisError::BootstrapError {
            reason: format!(
                "Buffer length {} does not match data length {}",
                buffer.len(),
                data.len()
            ),
        });
    }

    // If a seed is provided, use deterministic RNG
    if let Some(seed) = config.seed {
        let mut rng = SecureRng::with_seed(seed);
        match config.bootstrap_method {
            BootstrapMethod::Standard => {
                standard_bootstrap_sample_inplace_with_rng(data, buffer, &mut rng);
            }
            BootstrapMethod::Block => {
                let block_size = config
                    .block_size
                    .unwrap_or_else(|| politis_white_block_size(data));
                block_bootstrap_sample_inplace_with_rng(data, buffer, block_size, &mut rng);
            }
            BootstrapMethod::Stationary => {
                let block_size = config.block_size.unwrap_or_else(|| politis_white_block_size(data));
                stationary_bootstrap_sample_inplace_with_rng(data, buffer, block_size, &mut rng);
            }
            BootstrapMethod::Circular => {
                circular_bootstrap_sample_inplace_with_rng(data, buffer, &mut rng);
            }
        }
        return Ok(());
    }

    // Unseeded path uses ThreadLocalRng for performance
    match config.bootstrap_method {
        BootstrapMethod::Standard => {
            standard_bootstrap_sample_inplace(data, buffer);
            Ok(())
        }
        BootstrapMethod::Block => {
            let block_size = config
                .block_size
                .unwrap_or_else(|| politis_white_block_size(data));
            block_bootstrap_sample_inplace(data, block_size, buffer);
            Ok(())
        }
        BootstrapMethod::Stationary => {
            let block_size = config.block_size.unwrap_or_else(|| politis_white_block_size(data));
            stationary_bootstrap_sample_inplace(data, block_size, buffer);
            Ok(())
        }
        BootstrapMethod::Circular => {
            circular_bootstrap_sample_inplace(data, buffer);
            Ok(())
        }
    }
}

/// Generate a bootstrap sample in-place using a provided RNG.
///
/// This version uses the provided RNG for deterministic sampling.
pub fn generate_bootstrap_sample_inplace_with_rng(
    data: &[f64],
    config: &BootstrapConfiguration,
    buffer: &mut [f64],
    rng: &mut SecureRng,
) -> FractalResult<()> {
    if buffer.len() != data.len() {
        return Err(FractalAnalysisError::BootstrapError {
            reason: format!(
                "Buffer length {} does not match data length {}",
                buffer.len(),
                data.len()
            ),
        });
    }

    match config.bootstrap_method {
        BootstrapMethod::Standard => {
            standard_bootstrap_sample_inplace_with_rng(data, buffer, rng);
            Ok(())
        }
        BootstrapMethod::Block => {
            let block_size = config
                .block_size
                .unwrap_or_else(|| politis_white_block_size(data));
            block_bootstrap_sample_inplace_with_rng(data, buffer, block_size, rng);
            Ok(())
        }
        BootstrapMethod::Stationary => {
            let block_size = config.block_size.unwrap_or_else(|| politis_white_block_size(data));
            stationary_bootstrap_sample_inplace_with_rng(data, buffer, block_size, rng);
            Ok(())
        }
        BootstrapMethod::Circular => {
            circular_bootstrap_sample_inplace_with_rng(data, buffer, rng);
            Ok(())
        }
    }
}

/// Standard bootstrap (iid resampling).
///
/// Classic bootstrap that samples observations independently with replacement.
/// Appropriate for independent data but may not preserve dependence structure.
fn standard_bootstrap_sample(data: &[f64]) -> Vec<f64> {
    (0..data.len())
        .map(|_| data[ThreadLocalRng::usize(0..data.len())])
        .collect()
}

/// In-place standard bootstrap sampling.
fn standard_bootstrap_sample_inplace(data: &[f64], buffer: &mut [f64]) {
    for i in 0..buffer.len() {
        buffer[i] = data[ThreadLocalRng::usize(0..data.len())];
    }
}

/// Block bootstrap for dependent data.
///
/// Samples blocks of consecutive observations to preserve short-range
/// dependence structure. Suitable for time series with autocorrelation.
fn block_bootstrap_sample(data: &[f64], block_size: usize) -> Vec<f64> {
    let n = data.len();
    let block_size = block_size.min(n).max(2); // Ensure minimum block size of 2
    // OPTIMIZATION: Pre-allocate with capacity since we'll fill to size n
    let mut sample = Vec::with_capacity(n);

    while sample.len() < n {
        let start = ThreadLocalRng::usize(0..n.saturating_sub(block_size).saturating_add(1));
        let end = (start + block_size).min(n);
        sample.extend_from_slice(&data[start..end]);
    }

    sample.truncate(n);
    sample
}

/// Stationary bootstrap with geometric block lengths.
///
/// Uses geometrically distributed block lengths to create a stationary
/// bootstrap sample. Provides better coverage of the dependence structure.
fn stationary_bootstrap_sample(data: &[f64], expected_block_size: usize) -> Vec<f64> {
    let n = data.len();
    // OPTIMIZATION: Pre-allocate with capacity since we'll fill to size n
    let mut sample = Vec::with_capacity(n);
    // Clamp expected_block_size to reasonable bounds
    let expected_block_size = expected_block_size.max(2).min(n);
    let p = 1.0 / expected_block_size as f64; // Probability of ending a block

    while sample.len() < n {
        let start = ThreadLocalRng::usize(0..n);
        let mut length = 1;

        // Generate geometric block length
        while ThreadLocalRng::f64() > p && sample.len() + length < n {
            length += 1;
        }

        for i in 0..length {
            if sample.len() >= n {
                break;
            }
            sample.push(data[(start + i) % n]);
        }
    }

    sample.truncate(n);
    sample
}

/// Circular block bootstrap.
///
/// Creates a bootstrap sample by sampling blocks with wrap-around.
/// This properly preserves local dependence structure while allowing variation.
fn circular_bootstrap_sample(data: &[f64]) -> Vec<f64> {
    circular_block_bootstrap_sample(data, None)
}

/// Circular block bootstrap with specified block size.
///
/// Samples blocks of data with wrap-around at boundaries, preserving
/// local dependence structure while allowing proper variation.
fn circular_block_bootstrap_sample(data: &[f64], block_size: Option<usize>) -> Vec<f64> {
    let n = data.len();
    let block_size = block_size.unwrap_or_else(|| {
        // Use Politis-White rule if not specified
        politis_white_block_size(data).max(2).min((n as f64 * MAX_BLOCK_SIZE_RATIO) as usize)
    });
    let block_size = block_size.max(2).min(n);
    
    let mut sample = Vec::with_capacity(n);
    
    while sample.len() < n {
        let start = ThreadLocalRng::usize(0..n);
        let take = (n - sample.len()).min(block_size);
        
        for i in 0..take {
            sample.push(data[(start + i) % n]);
        }
    }
    
    sample.truncate(n);
    sample
}

/// In-place block bootstrap sampling.
fn block_bootstrap_sample_inplace(data: &[f64], block_size: usize, buffer: &mut [f64]) {
    let n = data.len();
    let block_size = block_size.min(n).max(2); // Ensure minimum block size of 2
    let mut pos = 0;

    while pos < n {
        let start = ThreadLocalRng::usize(0..n.saturating_sub(block_size).saturating_add(1));
        let end = (start + block_size).min(n);
        let block_len = end - start;
        let copy_len = (n - pos).min(block_len);

        // Debug assertion removed for production performance

        buffer[pos..pos + copy_len].copy_from_slice(&data[start..start + copy_len]);
        pos += copy_len;
    }
}

/// In-place stationary bootstrap sampling.
fn stationary_bootstrap_sample_inplace(
    data: &[f64],
    expected_block_size: usize,
    buffer: &mut [f64],
) {
    let n = data.len();
    // Clamp expected_block_size to reasonable bounds
    let expected_block_size = expected_block_size.max(2).min(n);
    let p = 1.0 / expected_block_size as f64;
    let mut pos = 0;

    while pos < n {
        let start = ThreadLocalRng::usize(0..n);
        let mut length = 1;

        // Generate geometric block length
        while ThreadLocalRng::f64() > p && pos + length < n {
            length += 1;
        }

        for i in 0..length {
            if pos >= n {
                break;
            }
            buffer[pos] = data[(start + i) % n];
            pos += 1;
        }
    }
}

/// In-place circular block bootstrap sampling.
fn circular_bootstrap_sample_inplace(data: &[f64], buffer: &mut [f64]) {
    circular_block_bootstrap_sample_inplace(data, None, buffer);
}

/// In-place circular block bootstrap with specified block size.
fn circular_block_bootstrap_sample_inplace(data: &[f64], block_size: Option<usize>, buffer: &mut [f64]) {
    let n = data.len();
    let block_size = block_size.unwrap_or_else(|| {
        politis_white_block_size(data).max(2).min((n as f64 * MAX_BLOCK_SIZE_RATIO) as usize)
    });
    let block_size = block_size.max(2).min(n);
    
    let mut pos = 0;
    
    while pos < n {
        let start = ThreadLocalRng::usize(0..n);
        let take = (n - pos).min(block_size);
        
        for i in 0..take {
            buffer[pos] = data[(start + i) % n];
            pos += 1;
        }
    }
}

/// Seeded versions of bootstrap functions for deterministic behavior

pub(crate) fn standard_bootstrap_sample_with_rng(data: &[f64], rng: &mut SecureRng) -> Vec<f64> {
    (0..data.len())
        .map(|_| data[rng.usize(0..data.len())])
        .collect()
}

pub(crate) fn block_bootstrap_sample_with_rng(
    data: &[f64],
    block_size: usize,
    rng: &mut SecureRng,
) -> Vec<f64> {
    let n = data.len();
    let mut sample = Vec::with_capacity(n);
    let block_size = block_size.min(n).max(2); // Minimum block size of 2 for dependent data

    while sample.len() < n {
        let start_max = n.saturating_sub(block_size);
        let start = if start_max == 0 {
            0
        } else {
            rng.usize(0..start_max + 1)
        };
        let end = (start + block_size).min(n);
        sample.extend_from_slice(&data[start..end]);
    }

    sample.truncate(n);
    sample
}

pub(crate) fn stationary_bootstrap_sample_with_rng(
    data: &[f64],
    block_size: usize,
    rng: &mut SecureRng,
) -> Vec<f64> {
    let n = data.len();
    let mut sample = Vec::with_capacity(n);
    // Clamp block_size to reasonable bounds
    let block_size = block_size.max(2).min(n);
    let p = 1.0 / block_size as f64;

    while sample.len() < n {
        let start = rng.usize(0..n);
        let mut length = 1;

        // Generate geometric block length
        while rng.f64() > p && sample.len() + length < n {
            length += 1;
        }

        // Add block with wrap-around
        for i in 0..length {
            if sample.len() < n {
                sample.push(data[(start + i) % n]);
            }
        }
    }

    sample.truncate(n);
    sample
}

pub(crate) fn circular_bootstrap_sample_with_rng(data: &[f64], rng: &mut SecureRng) -> Vec<f64> {
    circular_block_bootstrap_sample_with_rng(data, None, rng)
}

fn circular_block_bootstrap_sample_with_rng(data: &[f64], block_size: Option<usize>, rng: &mut SecureRng) -> Vec<f64> {
    let n = data.len();
    let block_size = block_size.unwrap_or_else(|| {
        politis_white_block_size(data).max(2).min((n as f64 * MAX_BLOCK_SIZE_RATIO) as usize)
    });
    let block_size = block_size.max(2).min(n);
    
    let mut sample = Vec::with_capacity(n);
    
    while sample.len() < n {
        let start = rng.usize(0..n);
        let take = (n - sample.len()).min(block_size);
        
        for i in 0..take {
            sample.push(data[(start + i) % n]);
        }
    }
    
    sample.truncate(n);
    sample
}

/// Seeded versions of in-place bootstrap functions for deterministic behavior

pub(crate) fn standard_bootstrap_sample_inplace_with_rng(
    data: &[f64],
    buffer: &mut [f64],
    rng: &mut SecureRng,
) {
    if data.is_empty() || buffer.is_empty() {
        return;
    }
    for i in 0..buffer.len() {
        buffer[i] = data[rng.usize(0..data.len())];
    }
}

pub(crate) fn block_bootstrap_sample_inplace_with_rng(
    data: &[f64],
    buffer: &mut [f64],
    block_size: usize,
    rng: &mut SecureRng,
) {
    if data.is_empty() || buffer.is_empty() {
        return;
    }
    let n = data.len();
    let block_size = block_size.min(n).max(1); // Allow block size of 1 for single element
    let mut pos = 0;

    while pos < n {
        let start = rng.usize(0..n.saturating_sub(block_size).saturating_add(1));
        let end = (start + block_size).min(n);
        let block_len = end - start;

        let copy_len = block_len.min(n - pos);
        buffer[pos..pos + copy_len].copy_from_slice(&data[start..start + copy_len]);
        pos += copy_len;
    }
}

pub(crate) fn stationary_bootstrap_sample_inplace_with_rng(
    data: &[f64],
    buffer: &mut [f64],
    block_size: usize,
    rng: &mut SecureRng,
) {
    if data.is_empty() || buffer.is_empty() {
        return;
    }
    let n = data.len();
    // Clamp block_size to reasonable bounds
    let block_size = block_size.max(1).min(n);
    let p = 1.0 / block_size as f64;
    let mut pos = 0;

    while pos < n {
        let start = rng.usize(0..n);
        let mut length = 1;

        // Generate geometric block length
        while rng.f64() > p && pos + length < n {
            length += 1;
        }

        // Add block with wrap-around
        for i in 0..length {
            if pos < n {
                buffer[pos] = data[(start + i) % n];
                pos += 1;
            }
        }
    }
}

pub(crate) fn circular_bootstrap_sample_inplace_with_rng(
    data: &[f64],
    buffer: &mut [f64],
    rng: &mut SecureRng,
) {
    circular_block_bootstrap_sample_inplace_with_rng(data, None, buffer, rng);
}

fn circular_block_bootstrap_sample_inplace_with_rng(
    data: &[f64],
    block_size: Option<usize>,
    buffer: &mut [f64],
    rng: &mut SecureRng,
) {
    if data.is_empty() || buffer.is_empty() {
        return;
    }
    let n = data.len();
    let block_size = block_size.unwrap_or_else(|| {
        politis_white_block_size(data).max(1).min((n as f64 * MAX_BLOCK_SIZE_RATIO) as usize)
    });
    let block_size = block_size.max(1).min(n);
    
    let mut pos = 0;
    
    while pos < n {
        let start = rng.usize(0..n);
        let take = (n - pos).min(block_size);
        
        for i in 0..take {
            buffer[pos] = data[(start + i) % n];
            pos += 1;
        }
    }
}