//! Wavelet analysis methods for fractal characterization
//!
//! This module provides wavelet-based methods for analyzing fractal properties
//! of time series using wavelet variance calculations at dyadic scales
//! with Daubechies D(4) wavelets.

use crate::errors::{FractalAnalysisError, FractalResult};
use crate::math_utils::{float_ops, mad, median, ols_regression};

/// Daubechies-4 filter length constant
const FILTER_LEN: usize = 4;

/// Calculate wavelet variance at a given dyadic scale
///
/// Computes wavelet variance using a MODWT-inspired coefficient calculation
/// with Daubechies D(4) wavelets. The variance is computed at dyadic scales
/// (powers of 2) for use in Hurst exponent estimation via regression.
///
/// Note: This implementation computes wavelet detail coefficients using
/// MODWT-scaled filters but is primarily designed for variance estimation
/// at specific scales rather than full multiresolution analysis.
pub fn calculate_wavelet_variance(data: &[f64], scale: usize) -> f64 {
    if scale == 0 || !scale.is_power_of_two() || scale >= data.len() {
        return 0.0;
    }

    // Convert scale to level: scale = 2^j, so j = log2(scale)
    // The scale parameter should be a power of 2 (2, 4, 8, 16...)
    let level = scale.trailing_zeros() as usize;
    if level == 0 {
        return 0.0;
    }

    // Compute MODWT coefficients at level j
    let modwt_coeffs = compute_modwt_level(data, level);
    let n = modwt_coeffs.len();

    // Calculate wavelet variance using ALL coefficients (no boundary exclusion)
    // This is to debug the variance scaling issue
    let sum_squared: f64 = modwt_coeffs
        .iter()
        .map(|&c| c * c)
        .sum();
    
    sum_squared / (n as f64)
}

/// Compute MODWT coefficients at a specific decomposition level
///
/// IMPORTANT: This function returns DETAIL (wavelet) coefficients W_j at level j,
/// NOT the smooth (scaling) coefficients V_j. The detail coefficients W_j capture
/// variations at scale 2^j, which is what we need for the wavelet variance method.
///
/// Uses circular boundary conditions with MODWT-scaled filters (DWT filters divided by √2).
///
/// # Parameters
/// * `data` - Input time series
/// * `level` - Decomposition level j ≥ 1 (not scale!), where scale = 2^j
///
/// # Returns
/// Detail wavelet coefficients W_j at the specified level
pub fn compute_modwt_level(data: &[f64], level: usize) -> Vec<f64> {
    // Guard against level == 0
    if level == 0 {
        return data.to_vec();
    }
    
    let n = data.len();

    // MODWT Daubechies D(4) coefficients
    // These coefficients are already normalized for MODWT (sum to sqrt(2), energy = 1)
    // They incorporate the 1/sqrt(2) scaling needed for MODWT
    const H0: f64 = 0.4829629131445341;
    const H1: f64 = 0.8365163037378079;
    const H2: f64 = 0.2241438680420134;
    const H3: f64 = -0.1294095225512604;

    // MODWT wavelet filter: g[k] = (-1)^k * h[L-1-k]
    const G0: f64 = -0.1294095225512604;
    const G1: f64 = -0.2241438680420134;
    const G2: f64 = 0.8365163037378079;
    const G3: f64 = -0.4829629131445341;

    // Initialize with data
    let mut current = data.to_vec();

    // Apply MODWT pyramid algorithm up to desired level
    for j in 1..=level {
        let mut wavelet_coeffs = vec![0.0; n];
        let dilation = 2_usize.pow((j - 1) as u32);

        // Circular convolution with dilated MODWT wavelet filter
        // NO additional level-dependent normalization - filters are already MODWT normalized
        for t in 0..n {
            // Standard MODWT uses backward indexing in time
            wavelet_coeffs[t] = 
                G0 * current[t]
                + G1 * current[(t + n - dilation) % n]     
                + G2 * current[(t + n - 2 * dilation) % n] 
                + G3 * current[(t + n - 3 * dilation) % n];
        }

        // For next level, use scaling coefficients
        if j < level {
            let mut scaling_coeffs = vec![0.0; n];
            for t in 0..n {
                // Apply dilated MODWT scaling filter
                scaling_coeffs[t] = 
                    H0 * current[t]
                    + H1 * current[(t + n - dilation) % n]
                    + H2 * current[(t + n - 2 * dilation) % n]
                    + H3 * current[(t + n - 3 * dilation) % n];
            }
            current = scaling_coeffs;
        } else {
            current = wavelet_coeffs;
        }
    }

    current
}

/// Haar wavelet variance calculation using block means
///
/// Computes true Haar wavelet detail coefficients as the difference between
/// adjacent block means of size s.
pub fn calculate_haar_wavelet_variance(data: &[f64], s: usize) -> f64 {
    if s == 0 || 2 * s > data.len() {
        return 0.0;
    }

    let n = data.len();
    let mut coefficients = Vec::with_capacity(n - 2 * s + 1);

    // Compute prefix sums for O(1) block mean calculation
    let mut prefix_sum = vec![0.0; n + 1];
    for i in 0..n {
        prefix_sum[i + 1] = prefix_sum[i] + data[i];
    }

    // Compute Haar detail coefficients: d_i = (mean1 - mean2) / sqrt(2)
    // where mean1 is over [i, i+s) and mean2 is over [i+s, i+2s)
    const SQRT2_INV: f64 = std::f64::consts::FRAC_1_SQRT_2;
    
    for i in 0..=n - 2 * s {
        let mean1 = (prefix_sum[i + s] - prefix_sum[i]) / s as f64;
        let mean2 = (prefix_sum[i + 2 * s] - prefix_sum[i + s]) / s as f64;
        let coeff = (mean1 - mean2) * SQRT2_INV;
        if coeff.is_finite() {
            coefficients.push(coeff);
        }
    }

    if coefficients.len() < 2 {
        return 0.0;
    }

    // Calculate variance of wavelet coefficients
    let mean = coefficients.iter().sum::<f64>() / coefficients.len() as f64;
    let variance = coefficients
        .iter()
        .map(|&c| (c - mean).powi(2))
        .sum::<f64>()
        / coefficients.len() as f64;

    if variance.is_finite() && variance >= 0.0 {
        variance
    } else {
        0.0
    }
}

/// Estimate wavelet Hurst exponent only (for bootstrap) - core logic without bootstrap CI
pub fn estimate_wavelet_hurst_only(data: &[f64]) -> FractalResult<f64> {
    // Validation check
    if data.len() < 64 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 64,
            actual: data.len(),
        });
    }

    let n = data.len();
    let mut scale_variances = Vec::new();
    let mut scales = Vec::new();

    // Use the same L_j logic as calculate_wavelet_variance to avoid including unusable scales
    let mut scale = 2usize;
    while (scale - 1) * (FILTER_LEN - 1) + 1 <= n {
        // Check if we have enough effective coefficients at this scale
        let effective_n = n - ((scale - 1) * (FILTER_LEN - 1) + 1) + 1;
        if effective_n < 8 {
            break; // Too few coefficients for reliable estimation
        }
        
        let variance = calculate_wavelet_variance(data, scale);
        if variance > 0.0 && variance.is_finite() {
            // Safe logarithm with proper error handling
            if let Some(log_val) = float_ops::safe_ln(variance) {
                scale_variances.push(log_val);
                scales.push((scale as f64).ln());
            }
        }
        scale <<= 1; // Equivalent to scale *= 2 but more idiomatic for powers of 2
    }

    if scale_variances.len() < 3 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 3,
            actual: scale_variances.len(),
        });
    }

    let (slope, _, _) = ols_regression(&scales, &scale_variances)?;
    // For FGN (fractional Gaussian noise): log(var) ~ (2H - 1) * log(scale)
    // So slope = 2H - 1, therefore H = (slope + 1) / 2
    // Clamp to valid range [0, 1] as Hurst exponent must be in this range
    Ok(((slope + 1.0) / 2.0).max(0.0).min(1.0))
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modwt_level() {
        // Simple test with synthetic data
        let data: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();
        let coeffs = compute_modwt_level(&data, 1);
        
        assert_eq!(coeffs.len(), data.len());
        
        // Check that coefficients have reasonable values (not all zero, not infinite)
        let non_zero_count = coeffs.iter().filter(|&&x| x.abs() > 1e-10).count();
        assert!(non_zero_count > 0);
        
        let all_finite = coeffs.iter().all(|x| x.is_finite());
        assert!(all_finite);
    }

    #[test]
    fn test_wavelet_variance() {
        let data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.05).sin() + 0.1 * i as f64).collect();
        
        // Test at different scales
        let var2 = calculate_wavelet_variance(&data, 2);
        let var4 = calculate_wavelet_variance(&data, 4);
        let var8 = calculate_wavelet_variance(&data, 8);
        
        // Variances should be positive and finite
        assert!(var2 > 0.0 && var2.is_finite());
        assert!(var4 > 0.0 && var4.is_finite());
        assert!(var8 > 0.0 && var8.is_finite());
        
        // Check that log-log slope is positive (indicating increasing variance with scale)
        let xs = vec![(2f64).ln(), (4f64).ln(), (8f64).ln()];
        let ys = vec![var2.ln(), var4.ln(), var8.ln()];
        let (slope, _, _) = ols_regression(&xs, &ys).unwrap();
        assert!(slope > 0.0, "Log-log slope should be positive for trending signal");
    }

    #[test]
    fn test_estimate_wavelet_hurst() {
        use crate::generators::{generate_fractional_brownian_motion, fbm_to_fgn, FbmConfig, FbmMethod, GeneratorConfig};
        
        // Generate proper FBM with known Hurst
        let target_hurst = 0.7;
        let config = GeneratorConfig {
            length: 256,
            seed: Some(789),
            ..Default::default()
        };
        let fbm_config = FbmConfig {
            hurst_exponent: target_hurst,
            volatility: 1.0,
            method: FbmMethod::CirculantEmbedding,
        };
        
        let fbm = generate_fractional_brownian_motion(&config, &fbm_config).unwrap();
        let fgn = fbm_to_fgn(&fbm); // Convert to increments for analysis
        
        let result = estimate_wavelet_hurst_only(&fgn);
        assert!(result.is_ok());
        
        let h = result.unwrap();
        // Should be in valid range
        assert!(h > 0.0 && h <= 1.0);
        
        // Should recover the Hurst exponent within reasonable tolerance  
        assert!(
            (h - target_hurst).abs() < 0.2,
            "Wavelet estimate {} should be close to true Hurst {}",
            h, target_hurst
        );
    }
    
    #[test]
    fn test_modwt_scaling_validation() {
        use crate::generators::{generate_fractional_brownian_motion, fbm_to_fgn, FbmConfig, FbmMethod, GeneratorConfig};
        
        // Test MODWT scaling for known Hurst exponents
        let test_hursts = vec![0.3, 0.5, 0.7, 0.9];
        
        for target_h in test_hursts {
            let config = GeneratorConfig {
                length: 512,
                seed: Some(12345),
                ..Default::default()
            };
            let fbm_config = FbmConfig {
                hurst_exponent: target_h,
                volatility: 1.0,
                method: FbmMethod::CirculantEmbedding,
            };
            
            let fbm = generate_fractional_brownian_motion(&config, &fbm_config).unwrap();
            let fgn = fbm_to_fgn(&fbm);
            
            // Compute wavelet variances at different scales
            let mut log_scales = Vec::new();
            let mut log_variances = Vec::new();
            
            let mut scale = 2;
            while scale < fgn.len() / 4 {
                let variance = calculate_wavelet_variance(&fgn, scale);
                if variance > 0.0 {
                    eprintln!("H={}, scale={}, variance={}", target_h, scale, variance);
                    log_scales.push((scale as f64).ln());
                    log_variances.push(variance.ln());
                }
                scale *= 2;
            }
            
            // For FGN: log(var) ~ (2H - 1) * log(scale) + const
            // Perform linear regression to estimate slope
            if log_scales.len() >= 3 {
                let (slope, _, _) = crate::math_utils::ols_regression(&log_scales, &log_variances).unwrap();
                let estimated_h = (slope + 1.0) / 2.0;
                
                eprintln!("MODWT scaling test: H={}, estimated={}, slope={}", target_h, estimated_h, slope);
                
                // Should recover Hurst within reasonable tolerance
                assert!(
                    (estimated_h - target_h).abs() < 0.25,
                    "MODWT scaling test: H={}, estimated={}, slope={}",
                    target_h, estimated_h, slope
                );
            }
        }
    }

    #[test]
    fn test_median() {
        assert_eq!(median(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median(&[5.0, 1.0, 3.0, 2.0, 4.0]), 3.0);
        assert!(median(&[]).is_nan());
    }

    #[test]
    fn test_mad() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let med = median(&data);
        let m = mad(&data, med);
        assert_eq!(m, 1.0); // MAD of [1,2,3,4,5] with median 3 is 1
    }
}