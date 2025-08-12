//! # Diagnostic and Debugging Tools
//!
//! Advanced diagnostic utilities for debugging and validating fractal analysis methods.
//! This module provides sophisticated tools for analyzing numerical stability, method accuracy,
//! and identifying systematic biases in fractal estimators.
//!
//! ## Key Diagnostic Categories
//!
//! ### Eigenvalue Analysis
//! - Circulant embedding matrix conditioning
//! - Numerical precision analysis for high Hurst exponents
//! - Variance preservation validation
//!
//! ### GPH Test Diagnostics  
//! - Standard error bias detection and correction
//! - Periodogram implementation validation
//! - Theoretical vs empirical comparison
//!
//! ### Numerical Stability
//! - Catastrophic cancellation detection
//! - Matrix conditioning analysis
//! - Floating-point precision validation
//!
//! ## Usage Example
//!
//! ```rust
//! use financial_fractal_analysis::diagnostics::{analyze_circulant_eigenvalues, diagnose_gph_variance_bias};
//!
//! # fn example() {
//! // Analyze eigenvalue conditioning for high Hurst exponents
//! analyze_circulant_eigenvalues(1000, 0.9, 1.0);
//!
//! // Diagnose GPH test variance bias
//! diagnose_gph_variance_bias();
//! # }
//! ```

use crate::{
    errors::{FractalAnalysisError, FractalResult},
    fft_ops::calculate_periodogram_fft,
    generators::{generate_benchmark_series, BenchmarkSeriesType, GeneratorConfig},
    math_utils::ols_regression,
    statistical_tests::gph_test,
};
use rustfft::{num_complex::Complex64, FftPlanner};

// OPTIMIZATION: Precomputed mathematical constants for performance
const PI_SQUARED: f64 = std::f64::consts::PI * std::f64::consts::PI;
const PI_SQUARED_OVER_6: f64 = PI_SQUARED / 6.0;

/// Analyze circulant embedding eigenvalue conditioning for FBM generation
///
/// This function investigates the mathematical root cause of systematic Hurst underestimation
/// for H > 0.7 in CirculantEmbedding method by analyzing eigenvalue properties.
pub fn analyze_circulant_eigenvalues(n: usize, h: f64, sigma2: f64) {
    // println!("\n=== EIGENVALUE ANALYSIS: n={}, H={:.2} ===", n, h);

    // CRITICAL SAFETY CHECK: Prevent massive memory allocation
    const MAX_FFT_SIZE: usize = 1 << 26; // 2^26 = 67,108,864
    const MAX_INPUT_SIZE: usize = MAX_FFT_SIZE / 4; // Conservative limit

    if n > MAX_INPUT_SIZE {
        // println!(
        //     "ERROR: Input size {} too large for eigenvalue analysis (max: {})",
        //     n, MAX_INPUT_SIZE
        // );
        return;
    }

    let m = (2 * n).next_power_of_two().min(MAX_FFT_SIZE);
    let mut gamma = vec![0.0; m];

    // Compute autocovariances exactly as in CirculantEmbedding
    gamma[0] = sigma2;

    for k in 1..n {
        let k_f64 = k as f64;
        gamma[k] = 0.5
            * sigma2
            * ((k_f64 + 1.0).powf(2.0 * h) + (k_f64 - 1.0).powf(2.0 * h)
                - 2.0 * k_f64.powf(2.0 * h));
    }

    // Complete circulant matrix
    for k in 1..n {
        if m - k < m {
            gamma[m - k] = gamma[k];
        }
    }

    // Convert to complex for FFT
    let mut fft_buffer = Vec::with_capacity(m);
    for &gamma_val in gamma.iter() {
        fft_buffer.push(Complex64::new(gamma_val, 0.0));
    }

    // Compute eigenvalues via FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(m);
    fft.process(&mut fft_buffer);

    // Analyze eigenvalues
    let eigenvals: Vec<f64> = fft_buffer.iter().map(|x| x.re).collect();

    let min_eigenval = eigenvals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_eigenval = eigenvals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let negative_count = eigenvals.iter().filter(|&&x| x < 0.0).count();
    let near_zero_count = eigenvals.iter().filter(|&&x| x.abs() < 1e-12).count();

    // println!("Min eigenvalue: {:.2e}", min_eigenval);
    // println!("Max eigenvalue: {:.2e}", max_eigenval);
    // println!(
    //     "Condition number: {:.2e}",
    //     max_eigenval / min_eigenval.max(1e-16)
    // );
    // println!("Negative eigenvalues: {}/{}", negative_count, m);
    // println!("Near-zero eigenvalues: {}/{}", near_zero_count, m);

    if negative_count > 0 {
        // println!("⚠️  NEGATIVE EIGENVALUES DETECTED - Matrix not positive definite!");
        // println!("   This causes systematic bias in FBM generation.");
    }

    if max_eigenval / min_eigenval.max(1e-16) > 1e12 {
        // println!("⚠️  SEVERE ILL-CONDITIONING - Numerical precision issues!");
        // println!("   Small eigenvalues become unreliable in f64 precision.");
    }

    // Check covariance values for numerical precision issues
    // println!("\nCovariance Analysis:");
    // println!("γ(0) = {:.6e}", gamma[0]);
    // println!("γ(1) = {:.6e}", gamma[1]);
    // if n > 2 {
    //     println!("γ(2) = {:.6e}", gamma[2]);
    // }

    // Check for catastrophic cancellation in covariance computation
    if h > 0.8 {
        let k = 1.0_f64;
        let term1 = (k + 1.0).powf(2.0 * h);
        let term2 = (k - 1.0).powf(2.0 * h); // This is 0^(2H) = 0 for k=1
        let term3 = 2.0 * k.powf(2.0 * h);

        // println!("\nCatastrophic Cancellation Check (k=1):");
        // println!("  (k+1)^(2H) = {:.6e}", term1);
        // println!("  (k-1)^(2H) = {:.6e}", term2);
        // println!("  2k^(2H) = {:.6e}", term3);
        // println!("  Sum = {:.6e}", term1 + term2 - term3);

        // For k=1: (2^(2H) + 0 - 2*1^(2H)) = 2^(2H) - 2 = 2(2^(2H-1) - 1)
        let analytical = 2.0 * (2.0_f64.powf(2.0 * h - 1.0) - 1.0);
        // println!("  Analytical = {:.6e}", analytical);
        // println!(
        //     "  Relative error = {:.2e}",
        //     ((term1 + term2 - term3) - analytical).abs() / analytical.abs()
        // );
    }

    // Check variance preservation
    let total_variance: f64 = eigenvals.iter().filter(|&&x| x > 0.0).sum();
    let expected_variance = sigma2; // This should equal σ²
    println!("\nVariance Check:");
    println!("Expected total variance: {:.6e}", expected_variance);
    println!("Actual eigenvalue sum: {:.6e}", total_variance);
    println!("Variance ratio: {:.6}", total_variance / expected_variance);

    if (total_variance / expected_variance - 1.0).abs() > 0.01 {
        println!("⚠️  VARIANCE MISMATCH - Scaling error detected!");
    }
}

/// Diagnose GPH test variance bias
///
/// This investigates why GPH test statistics have variance 1.83 instead of 1.0
/// and provides comprehensive analysis of the bias sources.
pub fn diagnose_gph_variance_bias() {
    println!("\n=== GPH VARIANCE BIAS ANALYSIS ===");

    let mut all_statistics = Vec::new();
    let mut all_d_estimates = Vec::new();
    let mut all_std_errors = Vec::new();
    let mut sample_sizes = Vec::new();

    // Test with different sample sizes to see finite-sample effects
    let test_sizes = vec![256, 512, 1024, 2048];

    for &n in &test_sizes {
        println!("\n--- Sample Size n = {} ---", n);

        let mut size_statistics = Vec::new();
        let mut size_d_estimates = Vec::new();
        let mut size_std_errors = Vec::new();

        for i in 0..100 {
            let config = GeneratorConfig {
                length: n,
                seed: Some(i as u64),
                sampling_frequency: 1.0,
            };

            let white_noise =
                generate_benchmark_series(BenchmarkSeriesType::WhiteNoise, &config).unwrap();

            if let Ok((test_stat, _, _)) = gph_test(&white_noise) {
                size_statistics.push(test_stat);

                // Manual calculation to understand the components
                if let Ok((d_est, std_err)) = manual_gph_calculation(&white_noise) {
                    size_d_estimates.push(d_est);
                    size_std_errors.push(std_err);
                }
            }
        }

        if !size_statistics.is_empty() {
            let mean = size_statistics.iter().sum::<f64>() / size_statistics.len() as f64;
            let variance = size_statistics
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .sum::<f64>()
                / (size_statistics.len() - 1) as f64;

            let d_mean = size_d_estimates.iter().sum::<f64>() / size_d_estimates.len() as f64;
            let d_var = size_d_estimates
                .iter()
                .map(|&x| {
                    let diff = x - d_mean;
                    diff * diff
                })
                .sum::<f64>()
                / (size_d_estimates.len() - 1) as f64;

            let se_mean = size_std_errors.iter().sum::<f64>() / size_std_errors.len() as f64;

            println!(
                "Test statistics - Mean: {:.4}, Variance: {:.4}",
                mean, variance
            );
            println!("d estimates - Mean: {:.4}, Variance: {:.4}", d_mean, d_var);
            println!("Standard errors - Mean: {:.4}", se_mean);
            println!(
                "Theoretical se for white noise: {:.4}",
                theoretical_gph_se(n)
            );

            all_statistics.extend(size_statistics);
            all_d_estimates.extend(size_d_estimates);
            all_std_errors.extend(size_std_errors);
            sample_sizes.push(n);
        }
    }

    // Overall analysis
    if !all_statistics.is_empty() {
        let overall_mean = all_statistics.iter().sum::<f64>() / all_statistics.len() as f64;
        let overall_variance = all_statistics
            .iter()
            .map(|&x| {
                let diff = x - overall_mean;
                diff * diff
            })
            .sum::<f64>()
            / (all_statistics.len() - 1) as f64;

        println!("\n=== OVERALL RESULTS ===");
        println!("Overall Mean: {:.6}", overall_mean);
        println!("Overall Variance: {:.6}", overall_variance);
        println!("Expected Variance: 1.000000");
        println!("Variance Ratio: {:.6}", overall_variance);

        if overall_variance > 1.2 {
            println!("⚠️  SIGNIFICANT VARIANCE BIAS DETECTED!");

            // Check if it's a systematic bias in standard error estimation
            let se_mean = all_std_errors.iter().sum::<f64>() / all_std_errors.len() as f64;
            let theoretical_se = theoretical_gph_se(1024); // Approximate
            println!("Average estimated SE: {:.6}", se_mean);
            println!("Theoretical SE: {:.6}", theoretical_se);
            println!("SE Ratio: {:.6}", se_mean / theoretical_se);

            if (se_mean / theoretical_se - 1.0).abs() > 0.1 {
                println!("🚨 STANDARD ERROR ESTIMATION BIAS DETECTED!");
            }
        }
    }
}

/// Deep GPH implementation debug
///
/// Compares our implementation with theoretical GPH method step-by-step,
/// providing detailed analysis of each computational component.
pub fn deep_gph_debug() {
    // println!("\n=== DEEP GPH MATHEMATICAL DEBUG ===");

    let config = GeneratorConfig {
        length: 1024,
        seed: Some(42),
        sampling_frequency: 1.0,
    };

    let white_noise = match generate_benchmark_series(BenchmarkSeriesType::WhiteNoise, &config) {
        Ok(data) => data,
        Err(e) => {
            println!("Failed to generate white noise: {:?}", e);
            return;
        }
    };

    println!("White noise sample: {:?}", &white_noise[0..10]);
    println!(
        "White noise mean: {:.6}",
        white_noise.iter().sum::<f64>() / white_noise.len() as f64
    );
    println!("White noise variance: {:.6}", {
        let mean = white_noise.iter().sum::<f64>() / white_noise.len() as f64;
        white_noise
            .iter()
            .map(|x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / (white_noise.len() - 1) as f64
    });

    // Step 1: Compare periodogram implementations
    println!("\n--- STEP 1: PERIODOGRAM COMPARISON ---");

    let n = white_noise.len();
    let m = (n as f64).powf(0.5) as usize;
    println!("n = {}, m = {}", n, m);

    // Our FFT-based periodogram
    match calculate_periodogram_fft(&white_noise) {
        Ok(periodogram_fft) => {
            println!("FFT periodogram length: {}", periodogram_fft.len());
            println!(
                "FFT periodogram first 10: {:?}",
                &periodogram_fft[0..10.min(periodogram_fft.len())]
            );

            // Manual DFT periodogram for comparison
            let periodogram_manual = calculate_manual_periodogram(&white_noise, m);
            println!("Manual periodogram length: {}", periodogram_manual.len());
            println!(
                "Manual periodogram first 10: {:?}",
                &periodogram_manual[0..10.min(periodogram_manual.len())]
            );

            // Compare the two
            let diff: Vec<f64> = periodogram_fft
                .iter()
                .take(periodogram_manual.len())
                .zip(&periodogram_manual)
                .map(|(fft, manual)| (fft - manual).abs())
                .collect();
            println!(
                "Max difference FFT vs Manual: {:.2e}",
                diff.iter().fold(0.0f64, |a, &b| a.max(b))
            );
        }
        Err(e) => println!("FFT periodogram error: {:?}", e),
    }

    // Step 2: Compare regression setups
    println!("\n--- STEP 2: REGRESSION SETUP COMPARISON ---");

    match gph_test(&white_noise) {
        Ok((t_stat, p_val, h_est)) => {
            println!(
                "Original GPH: t={:.6}, p={:.6}, H={:.6}",
                t_stat, p_val, h_est
            );
        }
        Err(e) => println!("Original GPH error: {:?}", e),
    }

    // Manual GPH with detailed steps
    match manual_gph_detailed(&white_noise) {
        Ok((t_stat, se, d_est, details)) => {
            println!("Manual GPH: t={:.6}, se={:.6}, d={:.6}", t_stat, se, d_est);
            println!("Manual GPH details: {}", details);
        }
        Err(e) => println!("Manual GPH error: {:?}", e),
    }

    // Step 3: Theoretical calculations
    println!("\n--- STEP 3: THEORETICAL ANALYSIS ---");
    theoretical_gph_analysis(n, m);
}

/// Test GPH standard error corrections
///
/// Investigates and tests fixes for the 32% standard error underestimation in GPH test.
pub fn test_gph_corrections() {
    println!("\n=== GPH STANDARD ERROR CORRECTION TEST ===");

    let config = GeneratorConfig {
        length: 1024,
        seed: Some(42),
        sampling_frequency: 1.0,
    };

    let white_noise = match generate_benchmark_series(BenchmarkSeriesType::WhiteNoise, &config) {
        Ok(data) => data,
        Err(e) => {
            println!("Failed to generate white noise: {:?}", e);
            return;
        }
    };

    // Compare different approaches
    println!("\n--- Original GPH Implementation ---");
    match gph_test(&white_noise) {
        Ok((t_stat, p_value, h_est)) => {
            println!("t-statistic: {:.6}", t_stat);
            println!("p-value: {:.6}", p_value);
            println!("H estimate: {:.6}", h_est);
        }
        Err(e) => println!("Error: {:?}", e),
    }

    println!("\n--- Manual GPH with Corrections ---");

    // Test different correction factors
    let correction_factors = vec![1.0, 1.2, 1.3, 1.4, 1.5, 1.47]; // 1.47 ≈ 1/0.68

    for &correction in &correction_factors {
        match corrected_gph_test(&white_noise, correction) {
            Ok((t_stat, std_err, d_est)) => {
                println!(
                    "Correction {:.2}: t={:.4}, se={:.4}, d={:.4}",
                    correction, t_stat, std_err, d_est
                );
            }
            Err(e) => println!("Correction {:.2}: Error {:?}", correction, e),
        }
    }

    // Test the theoretical correction based on log-periodogram properties
    println!("\n--- Theoretical Analysis ---");
    if let Ok(theoretical_se) = calculate_theoretical_gph_se(&white_noise) {
        println!("Theoretical SE: {:.6}", theoretical_se);
    }
}

/// Validate GPH correction with Monte Carlo simulation
///
/// Runs extensive Monte Carlo simulations to validate the effectiveness of
/// standard error corrections for the GPH test.
pub fn validate_gph_correction() {
    println!("\n=== GPH CORRECTION VALIDATION ===");

    let mut corrected_statistics = Vec::new();
    let correction_factor = 1.47; // Based on diagnostic analysis

    for i in 0..100 {
        let config = GeneratorConfig {
            length: 1024,
            seed: Some(i as u64),
            sampling_frequency: 1.0,
        };

        let white_noise =
            generate_benchmark_series(BenchmarkSeriesType::WhiteNoise, &config).unwrap();

        if let Ok((t_stat, _, _)) = corrected_gph_test(&white_noise, correction_factor) {
            corrected_statistics.push(t_stat);
        }
    }

    if !corrected_statistics.is_empty() {
        let mean = corrected_statistics.iter().sum::<f64>() / corrected_statistics.len() as f64;
        let variance = corrected_statistics
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / (corrected_statistics.len() - 1) as f64;

        println!("Corrected statistics:");
        println!("Mean: {:.6}", mean);
        println!("Variance: {:.6}", variance);
        println!("Expected variance: 1.000000");
        println!("Improvement: {:.1}%", (1.83 - variance) / 1.83 * 100.0);

        if (variance - 1.0).abs() < 0.2 {
            println!("✅ CORRECTION SUCCESSFUL!");
        } else {
            println!("⚠️  Correction needs refinement");
        }
    }
}

/// Run comprehensive diagnostic test suite
///
/// Executes all available diagnostic tests to provide a complete analysis
/// of numerical stability and method accuracy.
pub fn run_comprehensive_diagnostics() {
    println!("===========================================");
    println!("     COMPREHENSIVE DIAGNOSTIC SUITE");
    println!("===========================================");

    // Eigenvalue conditioning analysis
    println!("\n🔍 EIGENVALUE CONDITIONING ANALYSIS");
    let n = 1000;
    let sigma2 = 1.0;
    let h_values = vec![0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95];

    for h in h_values {
        analyze_circulant_eigenvalues(n, h, sigma2);
    }

    // GPH test diagnostics
    println!("\n🔍 GPH TEST DIAGNOSTICS");
    diagnose_gph_variance_bias();
    deep_gph_debug();
    test_gph_corrections();
    validate_gph_correction();

    println!("\n===========================================");
    println!("     DIAGNOSTIC SUITE COMPLETE");
    println!("===========================================");
}

// Helper functions

fn manual_gph_calculation(data: &[f64]) -> Result<(f64, f64), &'static str> {
    let n = data.len();

    // Calculate periodogram - simplified version
    let mut periodogram = Vec::new();
    let mut frequencies = Vec::new();

    // Use the same bandwidth as the actual GPH test
    let m = (n as f64).powf(0.5) as usize;

    for k in 1..=m.min(n / 2) {
        let freq = 2.0 * std::f64::consts::PI * k as f64 / n as f64;

        // Simple DFT calculation for diagnostic purposes
        let mut real_sum = 0.0;
        let mut imag_sum = 0.0;

        for (j, &x) in data.iter().enumerate() {
            let angle = freq * j as f64;
            real_sum += x * angle.cos();
            imag_sum += x * angle.sin();
        }

        let power = (real_sum * real_sum + imag_sum * imag_sum) / n as f64;
        if power > 0.0 {
            periodogram.push(power.ln());
            frequencies.push(freq.ln());
        }
    }

    if periodogram.len() < 5 {
        return Err("Insufficient data points");
    }

    // OLS regression
    match ols_regression(&frequencies, &periodogram) {
        Ok((slope, std_error, _)) => {
            let d_estimate = -slope;
            Ok((d_estimate, std_error))
        }
        Err(_) => Err("Regression failed"),
    }
}

fn theoretical_gph_se(n: usize) -> f64 {
    let m = (n as f64).powf(0.5);
    // Theoretical standard error for GPH test
    // Based on GPH (1983) asymptotic theory
    (std::f64::consts::PI * std::f64::consts::PI / 6.0 / m).sqrt()
}

fn calculate_manual_periodogram(data: &[f64], max_freq: usize) -> Vec<f64> {
    let n = data.len();
    let mut periodogram = Vec::new();

    // Calculate periodogram I(λ_j) = (1/2πn) |∑ x_t e^{-iλ_j t}|²
    // where λ_j = 2πj/n for j = 1, 2, ..., max_freq

    for j in 1..=max_freq.min(n / 2) {
        let lambda = 2.0 * std::f64::consts::PI * j as f64 / n as f64;

        let mut real_sum = 0.0;
        let mut imag_sum = 0.0;

        for (t, &x) in data.iter().enumerate() {
            let angle = lambda * t as f64;
            real_sum += x * angle.cos();
            imag_sum -= x * angle.sin(); // Note: negative for e^{-iλt}
        }

        // I(λ) = (1/2πn) |DFT|²
        let power =
            (real_sum * real_sum + imag_sum * imag_sum) / (2.0 * std::f64::consts::PI * n as f64);
        periodogram.push(power);
    }

    periodogram
}

fn manual_gph_detailed(data: &[f64]) -> FractalResult<(f64, f64, f64, String)> {
    let n = data.len();
    let m = (n as f64).powf(0.5) as usize;

    let mut details = String::new();
    details.push_str(&format!("n={}, m={}\n", n, m));

    // Calculate periodogram manually
    let periodogram = calculate_manual_periodogram(data, m);

    details.push_str(&format!(
        "Periodogram computed for {} frequencies\n",
        periodogram.len()
    ));

    // Set up regression: log I(λ_j) = α - d log λ_j + ε_j
    let mut log_periodogram = Vec::new();
    let mut log_frequencies = Vec::new();

    for (j, &I_lambda) in periodogram.iter().enumerate() {
        let lambda = 2.0 * std::f64::consts::PI * (j + 1) as f64 / n as f64;

        if I_lambda > 0.0 {
            log_periodogram.push(I_lambda.ln());
            log_frequencies.push(lambda.ln());
        }
    }

    details.push_str(&format!("Regression points: {}\n", log_periodogram.len()));
    details.push_str(&format!(
        "Frequency range: {:.6} to {:.6}\n",
        log_frequencies.first().unwrap_or(&0.0).exp(),
        log_frequencies.last().unwrap_or(&0.0).exp()
    ));

    if log_periodogram.len() < 5 {
        return Err(FractalAnalysisError::StatisticalTestError {
            test_name: "Manual GPH".to_string(),
        });
    }

    // OLS regression
    let (slope, std_error, residuals) = ols_regression(&log_frequencies, &log_periodogram)?;

    details.push_str(&format!(
        "OLS slope: {:.6}, std_error: {:.6}\n",
        slope, std_error
    ));
    details.push_str(&format!(
        "RSS: {:.6}\n",
        residuals.iter().map(|r| r * r).sum::<f64>()
    ));

    // GPH estimates
    let d_estimate = -slope; // d = -β where log I(λ) = α + β log(4sin²(λ/2)) + ε
    let d_std_error = std_error;
    let t_statistic = d_estimate / d_std_error;

    details.push_str(&format!(
        "d estimate: {:.6}, d std_error: {:.6}\n",
        d_estimate, d_std_error
    ));

    Ok((t_statistic, std_error, d_estimate, details))
}

fn theoretical_gph_analysis(n: usize, m: usize) {
    println!("=== THEORETICAL GPH ANALYSIS ===");

    // GPH (1983) theoretical results for Gaussian white noise
    println!("Sample size n: {}", n);
    println!("Bandwidth m: {}", m);

    // For white noise: E[log I(λ_j)] = log(σ²/2π) + γ
    // where γ ≈ 0.5772 is Euler's constant
    let euler_gamma = 0.5772156649015329;
    let theoretical_intercept = (1.0 / (2.0 * std::f64::consts::PI)).ln() + euler_gamma;
    println!("Theoretical intercept: {:.6}", theoretical_intercept);

    // For white noise: slope should be 0 (no long-range dependence)
    println!("Theoretical slope: 0.000000");

    // Standard error of d parameter: SE(d) = sqrt(π²/6m)
    let theoretical_se_d = (PI_SQUARED_OVER_6 / m as f64).sqrt();
    println!("Theoretical SE(d): {:.6}", theoretical_se_d);

    // Since d = -slope, SE(slope) = SE(d)
    let theoretical_se_slope = theoretical_se_d;
    println!("Theoretical SE(slope): {:.6}", theoretical_se_slope);

    // Under H0: d = 0, the t-statistic should be N(0,1)
    println!("Under H0: t ~ N(0,1)");

    // Theoretical variance of log-periodogram
    let theoretical_var_log_I = PI_SQUARED_OVER_6;
    println!("Theoretical Var[log I(λ)]: {:.6}", theoretical_var_log_I);
}

fn corrected_gph_test(data: &[f64], correction_factor: f64) -> FractalResult<(f64, f64, f64)> {
    let n = data.len();

    // Calculate periodogram
    let periodogram = calculate_periodogram_fft(data)?;

    // Use same bandwidth as original GPH
    let m = (n as f64).powf(0.5) as usize;
    let mut log_periodogram = Vec::new();
    let mut log_frequencies = Vec::new();

    for (i, _) in periodogram.iter().enumerate().take(m).skip(1) {
        let freq = 2.0 * std::f64::consts::PI * (i + 1) as f64 / n as f64;
        if periodogram[i] > 0.0 {
            log_periodogram.push(periodogram[i].max(1e-15).ln());
            log_frequencies.push(freq.ln());
        }
    }

    if log_periodogram.len() < 5 {
        return Err(FractalAnalysisError::StatisticalTestError {
            test_name: "Corrected GPH".to_string(),
        });
    }

    // OLS regression
    let (slope, std_error, _) = ols_regression(&log_frequencies, &log_periodogram)?;

    // Apply correction to standard error
    let corrected_std_error = std_error * correction_factor;

    let d_estimate = -slope;
    let corrected_t_statistic = d_estimate / corrected_std_error;

    Ok((corrected_t_statistic, corrected_std_error, d_estimate))
}

fn calculate_theoretical_gph_se(data: &[f64]) -> FractalResult<f64> {
    let n = data.len();
    let m = (n as f64).powf(0.5);

    // GPH (1983) theoretical standard error for d parameter
    // SE(d) = sqrt(π²/6 / m) for Gaussian white noise
    let theoretical_se_d = (PI_SQUARED_OVER_6 / m).sqrt();

    // Since we compute t = d / (SE_slope / 2), we need SE_slope = 2 * SE_d
    let theoretical_se_slope = 2.0 * theoretical_se_d;

    println!("Sample size: {}", n);
    println!("Bandwidth m: {:.1}", m);
    println!("Theoretical SE(d): {:.6}", theoretical_se_d);
    println!("Theoretical SE(slope): {:.6}", theoretical_se_slope);

    Ok(theoretical_se_slope)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eigenvalue_conditioning() {
        println!("EIGENVALUE CONDITIONING ANALYSIS");
        println!("=================================");

        let n = 1000;
        let sigma2 = 1.0;

        // Test different H values to see where conditioning breaks down
        let h_values = vec![0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95];

        for h in h_values {
            analyze_circulant_eigenvalues(n, h, sigma2);
        }
    }

    #[test]
    fn test_gph_variance_diagnostic() {
        diagnose_gph_variance_bias();
    }

    #[test]
    fn test_deep_gph_debug() {
        deep_gph_debug();
    }

    #[test]
    fn test_gph_correction_analysis() {
        test_gph_corrections();
        validate_gph_correction();
    }

    #[test]
    fn test_comprehensive_diagnostics() {
        run_comprehensive_diagnostics();
    }
}
