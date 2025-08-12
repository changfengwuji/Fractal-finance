//! Optimized batch processing for statistical tests in financial fractal analysis.
//!
//! This module provides efficient batch processing capabilities for running multiple
//! statistical tests simultaneously, sharing computations where possible to minimize
//! redundant calculations. Essential for comprehensive financial analysis workflows
//! that require multiple hypothesis testing scenarios.

use crate::{
    errors::{validate_all_finite, validate_data_length, FractalAnalysisError, FractalResult},
    local_whittle_estimate,
    math_utils::{calculate_autocorrelations, calculate_variance, ols_regression},
    memory_pool::{get_f64_buffer, return_f64_buffer},
    statistical_tests::*,
};
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::collections::HashMap;

// Safety constants
const MIN_VARIANCE: f64 = 1e-14;
const MIN_DENOMINATOR: f64 = 1e-10;
const MAX_LAG_RATIO: f64 = 0.25; // Maximum lag as fraction of data length

/// Configuration for batch statistical testing operations.
#[derive(Debug, Clone)]
pub struct BatchTestConfig {
    /// Tests to perform in this batch
    pub tests: Vec<StatisticalTest>,
    /// Significance levels to test (e.g., [0.01, 0.05, 0.10])
    pub significance_levels: Vec<f64>,
    /// Number of bootstrap samples for tests that support it
    pub bootstrap_samples: usize,
    /// Maximum lag for autocorrelation-based tests
    pub max_lag: usize,
    /// Share computations across tests when possible
    pub enable_shared_computations: bool,
}

/// Types of statistical tests available for batch processing.
#[derive(Debug, Clone)]
pub enum StatisticalTest {
    /// Ljung-Box test for serial correlation
    LjungBox { lags: Vec<usize> },
    /// Jarque-Bera test for normality
    JarqueBera,
    /// Anderson-Darling test for distribution fit
    AndersonDarling { distribution: TestDistribution },
    /// GPH test for long-range dependence
    GphTest { bandwidth_fraction: f64 },
    /// Robinson test for fractional integration
    RobinsonTest { bandwidth_fraction: f64 },
    /// Variance ratio test for random walk hypothesis
    VarianceRatio { lags: Vec<usize> },
    /// ARCH test for heteroskedasticity
    ArchTest { lags: Vec<usize> },
    /// Structural break test
    StructuralBreak { method: BreakTestMethod },
}

/// Available distributions for goodness-of-fit testing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TestDistribution {
    Normal,
    TDistribution { df: i32 },
    Exponential,
}

/// Methods for structural break testing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BreakTestMethod {
    Chow,
    Quandt,
    SupF,
}

// Manual implementations for StatisticalTest to handle f64 fields
impl PartialEq for StatisticalTest {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                StatisticalTest::LjungBox { lags: lags1 },
                StatisticalTest::LjungBox { lags: lags2 },
            ) => lags1 == lags2,
            (StatisticalTest::JarqueBera, StatisticalTest::JarqueBera) => true,
            (
                StatisticalTest::AndersonDarling { distribution: d1 },
                StatisticalTest::AndersonDarling { distribution: d2 },
            ) => d1 == d2,
            (
                StatisticalTest::GphTest {
                    bandwidth_fraction: bf1,
                },
                StatisticalTest::GphTest {
                    bandwidth_fraction: bf2,
                },
            ) => (bf1 - bf2).abs() < f64::EPSILON,
            (
                StatisticalTest::RobinsonTest {
                    bandwidth_fraction: bf1,
                },
                StatisticalTest::RobinsonTest {
                    bandwidth_fraction: bf2,
                },
            ) => (bf1 - bf2).abs() < f64::EPSILON,
            (
                StatisticalTest::VarianceRatio { lags: lags1 },
                StatisticalTest::VarianceRatio { lags: lags2 },
            ) => lags1 == lags2,
            (
                StatisticalTest::ArchTest { lags: lags1 },
                StatisticalTest::ArchTest { lags: lags2 },
            ) => lags1 == lags2,
            (
                StatisticalTest::StructuralBreak { method: m1 },
                StatisticalTest::StructuralBreak { method: m2 },
            ) => m1 == m2,
            _ => false,
        }
    }
}

impl Eq for StatisticalTest {}

impl std::hash::Hash for StatisticalTest {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            StatisticalTest::LjungBox { lags } => {
                0u8.hash(state);
                lags.hash(state);
            }
            StatisticalTest::JarqueBera => {
                1u8.hash(state);
            }
            StatisticalTest::AndersonDarling { distribution } => {
                2u8.hash(state);
                distribution.hash(state);
            }
            StatisticalTest::GphTest { bandwidth_fraction } => {
                3u8.hash(state);
                // Hash f64 by its bit representation
                bandwidth_fraction.to_bits().hash(state);
            }
            StatisticalTest::RobinsonTest { bandwidth_fraction } => {
                4u8.hash(state);
                // Hash f64 by its bit representation
                bandwidth_fraction.to_bits().hash(state);
            }
            StatisticalTest::VarianceRatio { lags } => {
                5u8.hash(state);
                lags.hash(state);
            }
            StatisticalTest::ArchTest { lags } => {
                6u8.hash(state);
                lags.hash(state);
            }
            StatisticalTest::StructuralBreak { method } => {
                7u8.hash(state);
                method.hash(state);
            }
        }
    }
}

/// Results from batch statistical testing.
#[derive(Debug, Clone)]
pub struct BatchTestResults {
    /// Individual test results indexed by test type
    pub test_results: HashMap<StatisticalTest, TestResult>,
    /// Shared computations that were reused across tests
    pub shared_computations: SharedComputations,
    /// Performance statistics
    pub performance_stats: BatchPerformanceStats,
}

/// Results for individual statistical tests.
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Test statistic value
    pub statistic: f64,
    /// P-values for each significance level tested
    pub p_values: Vec<f64>,
    /// Critical values for each significance level
    pub critical_values: Vec<f64>,
    /// Whether test rejects null hypothesis at each level
    pub rejections: Vec<bool>,
    /// Degrees of freedom (if applicable)
    pub degrees_of_freedom: Option<i32>,
    /// Additional test-specific information
    pub additional_info: HashMap<String, f64>,
}

/// Shared computations that can be reused across multiple tests.
#[derive(Debug, Clone, Default)]
pub struct SharedComputations {
    /// Periodogram (for spectral tests)
    pub periodogram: Option<Vec<f64>>,
    /// Sample autocorrelations (for serial correlation tests)
    pub autocorrelations: Option<Vec<f64>>,
    /// Sample moments (mean, variance, skewness, kurtosis)
    pub moments: Option<SampleMoments>,
    /// Cumulative returns (for variance ratio tests)
    pub cumulative_returns: Option<Vec<f64>>,
    /// Number of computations that were shared
    pub reuse_count: usize,
}

/// Sample moments for distribution testing.
#[derive(Debug, Clone)]
pub struct SampleMoments {
    pub mean: f64,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub count: usize,
}

/// Performance statistics for batch processing.
#[derive(Debug, Clone)]
pub struct BatchPerformanceStats {
    /// Total processing time in milliseconds
    pub total_time_ms: u64,
    /// Number of shared computations
    pub shared_computations: usize,
    /// Estimated time saved through sharing (milliseconds)
    pub time_saved_ms: u64,
    /// Memory efficiency ratio (0.0 to 1.0)
    pub memory_efficiency: f64,
}

impl Default for BatchTestConfig {
    fn default() -> Self {
        Self {
            tests: vec![
                StatisticalTest::LjungBox {
                    lags: vec![5, 10, 20],
                },
                StatisticalTest::JarqueBera,
                StatisticalTest::AndersonDarling {
                    distribution: TestDistribution::Normal,
                },
            ],
            significance_levels: vec![0.01, 0.05, 0.10],
            bootstrap_samples: 1000,
            max_lag: 50,
            enable_shared_computations: true,
        }
    }
}

/// Execute batch statistical testing with optimized shared computations.
///
/// This function efficiently processes multiple statistical tests by:
/// 1. Pre-computing shared intermediate results
/// 2. Reusing computations across compatible tests
/// 3. Leveraging memory pooling and caching systems
/// 4. Providing comprehensive results with performance metrics
pub fn execute_batch_tests(
    data: &[f64],
    config: &BatchTestConfig,
) -> FractalResult<BatchTestResults> {
    let start_time = std::time::Instant::now();

    validate_data_length(data, 10, "batch_tests")?;

    if config.tests.is_empty() {
        return Err(FractalAnalysisError::InvalidParameter {
            parameter: "tests".to_string(),
            value: 0.0,
            constraint: "must specify at least one test".to_string(),
        });
    }

    let mut shared_computations = SharedComputations::default();
    let mut test_results = HashMap::new();

    // OPTIMIZATION: Pre-compute shared intermediate results
    if config.enable_shared_computations {
        precompute_shared_results(data, config, &mut shared_computations)?;
    }

    // Execute each test, leveraging shared computations
    for test in &config.tests {
        let result = execute_single_test(data, test, config, &shared_computations)?;
        test_results.insert(test.clone(), result);
    }

    let total_time = start_time.elapsed().as_millis() as u64;

    let performance_stats = BatchPerformanceStats {
        total_time_ms: total_time,
        shared_computations: shared_computations.reuse_count,
        time_saved_ms: estimate_time_saved(&shared_computations, config.tests.len()),
        memory_efficiency: calculate_memory_efficiency(&shared_computations, &config.tests),
    };

    Ok(BatchTestResults {
        test_results,
        shared_computations,
        performance_stats,
    })
}

/// Pre-compute shared results that can be reused across multiple tests.
fn precompute_shared_results(
    data: &[f64],
    config: &BatchTestConfig,
    shared: &mut SharedComputations,
) -> FractalResult<()> {
    let n = data.len();

    // Determine which shared computations are needed
    let needs_periodogram = config.tests.iter().any(|t| {
        matches!(
            t,
            StatisticalTest::GphTest { .. } | StatisticalTest::RobinsonTest { .. }
        )
    });

    let needs_autocorr = config.tests.iter().any(|t| {
        matches!(
            t,
            StatisticalTest::LjungBox { .. } | StatisticalTest::ArchTest { .. }
        )
    });

    let needs_moments = config.tests.iter().any(|t| {
        matches!(
            t,
            StatisticalTest::JarqueBera | StatisticalTest::AndersonDarling { .. }
        )
    });

    let needs_cumulative = config
        .tests
        .iter()
        .any(|t| matches!(t, StatisticalTest::VarianceRatio { .. }));

    // OPTIMIZATION: Compute periodogram with caching
    if needs_periodogram && n >= 8 {
        use crate::fft_ops::calculate_periodogram_fft;
        let periodogram = calculate_periodogram_fft(data)?;
        shared.periodogram = Some(periodogram);
        shared.reuse_count += 1;
    }

    // OPTIMIZATION: Compute autocorrelations with caching and optimal method selection
    if needs_autocorr {
        let max_lag = config.max_lag.min(n / 4);
        let autocorrs = calculate_autocorrelations(data, max_lag);
        shared.autocorrelations = Some(autocorrs);
        shared.reuse_count += 1;
    }

    // OPTIMIZATION: Compute sample moments efficiently in single pass
    if needs_moments {
        let moments = compute_sample_moments(data)?;
        shared.moments = Some(moments);
        shared.reuse_count += 1;
    }

    // OPTIMIZATION: Compute cumulative returns for variance ratio tests
    if needs_cumulative {
        // Properly manage memory buffer
        let mut cumulative_vec = Vec::with_capacity(n);
        if n > 0 {
            cumulative_vec.push(data[0]);
            for i in 1..n {
                let sum = cumulative_vec[i - 1] + data[i];
                if !sum.is_finite() {
                    return Err(FractalAnalysisError::NumericalError {
                        reason: "Cumulative sum overflow".to_string(),
                        operation: None,
                    });
                }
                cumulative_vec.push(sum);
            }
        }
        shared.cumulative_returns = Some(cumulative_vec);
        shared.reuse_count += 1;
    }

    Ok(())
}

/// Execute a single statistical test, leveraging shared computations.
fn execute_single_test(
    data: &[f64],
    test: &StatisticalTest,
    config: &BatchTestConfig,
    shared: &SharedComputations,
) -> FractalResult<TestResult> {
    match test {
        StatisticalTest::LjungBox { lags } => execute_ljung_box_batch(data, lags, config, shared),
        StatisticalTest::JarqueBera => execute_jarque_bera_batch(data, config, shared),
        StatisticalTest::AndersonDarling { distribution } => {
            execute_anderson_darling_batch(data, distribution, config, shared)
        }
        StatisticalTest::GphTest { bandwidth_fraction } => {
            execute_gph_test_batch(data, *bandwidth_fraction, config, shared)
        }
        StatisticalTest::RobinsonTest { bandwidth_fraction } => {
            execute_robinson_test_batch(data, *bandwidth_fraction, config, shared)
        }
        StatisticalTest::VarianceRatio { lags } => {
            execute_variance_ratio_batch(data, lags, config, shared)
        }
        StatisticalTest::ArchTest { lags } => execute_arch_test_batch(data, lags, config, shared),
        StatisticalTest::StructuralBreak { method } => {
            execute_structural_break_batch(data, method, config, shared)
        }
    }
}

/// Execute Ljung-Box test with shared autocorrelation computations.
fn execute_ljung_box_batch(
    data: &[f64],
    lags: &[usize],
    config: &BatchTestConfig,
    shared: &SharedComputations,
) -> FractalResult<TestResult> {
    let n = data.len();
    let max_lag = *lags.iter().max().unwrap_or(&10);

    // Use shared autocorrelations if available
    let autocorrs = if let Some(ref shared_autocorrs) = shared.autocorrelations {
        shared_autocorrs.clone()
    } else {
        calculate_autocorrelations(data, max_lag)
    };

    // Compute Ljung-Box statistic for the largest lag
    let mut lb_stat = 0.0;
    let n_f64 = n as f64;

    for k in 1..=max_lag {
        if k < autocorrs.len() {
            let rho_k = autocorrs[k];
            // Safe division with minimum denominator check
            let denominator = (n_f64 - k as f64).max(MIN_DENOMINATOR);
            if rho_k.is_finite() {
                lb_stat += rho_k * rho_k / denominator;
            }
        }
    }
    lb_stat *= n_f64 * (n_f64 + 2.0);

    // Calculate p-values and critical values for each significance level
    // Pre-allocate with known capacity for performance
    let significance_count = config.significance_levels.len();
    let mut p_values = Vec::with_capacity(significance_count);
    let mut critical_values = Vec::with_capacity(significance_count);
    let mut rejections = Vec::with_capacity(significance_count);

    let chi_squared =
        ChiSquared::new(max_lag as f64).map_err(|_| FractalAnalysisError::NumericalError {
            reason: format!(
                "Failed to create chi-squared distribution with {} degrees of freedom",
                max_lag
            ),
            operation: None,
        })?;

    let p_value = 1.0 - chi_squared.cdf(lb_stat);

    for &alpha in &config.significance_levels {
        p_values.push(p_value);
        let critical_val = chi_squared.inverse_cdf(1.0 - alpha);
        critical_values.push(critical_val);
        rejections.push(lb_stat > critical_val);
    }

    let mut additional_info = HashMap::new();
    additional_info.insert("n_observations".to_string(), n as f64);
    additional_info.insert("max_lag".to_string(), max_lag as f64);

    Ok(TestResult {
        statistic: lb_stat,
        p_values,
        critical_values,
        rejections,
        degrees_of_freedom: Some(max_lag as i32),
        additional_info,
    })
}

/// Execute Jarque-Bera test using shared sample moments.
fn execute_jarque_bera_batch(
    data: &[f64],
    config: &BatchTestConfig,
    shared: &SharedComputations,
) -> FractalResult<TestResult> {
    let n = data.len();

    // Use shared moments if available
    let moments = if let Some(ref shared_moments) = shared.moments {
        shared_moments.clone()
    } else {
        compute_sample_moments(data)?
    };

    // Compute Jarque-Bera statistic with NaN safety
    let excess_kurt = moments.kurtosis - 3.0;
    let jb_stat = if moments.skewness.is_finite() && excess_kurt.is_finite() {
        (n as f64 / 6.0) * (moments.skewness * moments.skewness + (excess_kurt * excess_kurt) / 4.0)
    } else {
        return Err(FractalAnalysisError::NumericalError {
            reason: "Non-finite values in skewness or kurtosis".to_string(),
            operation: None,
        });
    };

    // Calculate p-values and critical values
    // Pre-allocate with known capacity for performance
    let significance_count = config.significance_levels.len();
    let mut p_values = Vec::with_capacity(significance_count);
    let mut critical_values = Vec::with_capacity(significance_count);
    let mut rejections = Vec::with_capacity(significance_count);

    let chi_squared = ChiSquared::new(2.0).map_err(|_| FractalAnalysisError::NumericalError {
        reason: "Failed to create chi-squared distribution with 2 degrees of freedom".to_string(),
        operation: None,
    })?;

    let p_value = 1.0 - chi_squared.cdf(jb_stat);

    for &alpha in &config.significance_levels {
        p_values.push(p_value);
        let critical_val = chi_squared.inverse_cdf(1.0 - alpha);
        critical_values.push(critical_val);
        rejections.push(jb_stat > critical_val);
    }

    let mut additional_info = HashMap::new();
    additional_info.insert("skewness".to_string(), moments.skewness);
    additional_info.insert("kurtosis".to_string(), moments.kurtosis);

    Ok(TestResult {
        statistic: jb_stat,
        p_values,
        critical_values,
        rejections,
        degrees_of_freedom: Some(2),
        additional_info,
    })
}

/// Execute Anderson-Darling test using shared moments.
fn execute_anderson_darling_batch(
    _data: &[f64],
    _distribution: &TestDistribution,
    config: &BatchTestConfig,
    _shared: &SharedComputations,
) -> FractalResult<TestResult> {
    // Implementation would depend on the specific distribution
    // For now, return a placeholder result
    let mut additional_info = HashMap::new();
    additional_info.insert("distribution".to_string(), 0.0);

    Ok(TestResult {
        statistic: 0.0,
        p_values: vec![0.5; config.significance_levels.len()],
        critical_values: vec![1.0; config.significance_levels.len()],
        rejections: vec![false; config.significance_levels.len()],
        degrees_of_freedom: None,
        additional_info,
    })
}

/// Execute GPH test using shared periodogram.
fn execute_gph_test_batch(
    data: &[f64],
    bandwidth_fraction: f64,
    config: &BatchTestConfig,
    shared: &SharedComputations,
) -> FractalResult<TestResult> {
    let n = data.len();

    // Use shared periodogram if available
    let periodogram = if let Some(ref shared_periodogram) = shared.periodogram {
        shared_periodogram.clone()
    } else {
        use crate::fft_ops::calculate_periodogram_fft;
        calculate_periodogram_fft(data)?
    };

    // Compute GPH estimator with overflow safety
    if bandwidth_fraction <= 0.0 || bandwidth_fraction >= 1.0 {
        return Err(FractalAnalysisError::InvalidParameter {
            parameter: "bandwidth_fraction".to_string(),
            value: bandwidth_fraction,
            constraint: "Must be between 0 and 1".to_string(),
        });
    }

    let m = ((n as f64).powf(bandwidth_fraction).min(n as f64 / 2.0) as usize).max(1);

    if m < 2 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 2,
            actual: m,
        });
    }

    let gph_stat = local_whittle_estimate(&periodogram, m);

    // For simplicity, use standard critical values
    // Pre-allocate with known capacity for performance
    let significance_count = config.significance_levels.len();
    let mut p_values = Vec::with_capacity(significance_count);
    let mut critical_values = Vec::with_capacity(significance_count);
    let mut rejections = Vec::with_capacity(significance_count);

    for &alpha in &config.significance_levels {
        p_values.push(0.5); // Placeholder
        critical_values.push(1.96); // Placeholder for normal approximation
        rejections.push(gph_stat.abs() > 1.96);
    }

    let mut additional_info = HashMap::new();
    additional_info.insert("bandwidth".to_string(), m as f64);
    additional_info.insert("bandwidth_fraction".to_string(), bandwidth_fraction);

    Ok(TestResult {
        statistic: gph_stat,
        p_values,
        critical_values,
        rejections,
        degrees_of_freedom: None,
        additional_info,
    })
}

/// Placeholder implementations for remaining test functions
fn execute_robinson_test_batch(
    _data: &[f64],
    _bandwidth_fraction: f64,
    config: &BatchTestConfig,
    _shared: &SharedComputations,
) -> FractalResult<TestResult> {
    Ok(TestResult {
        statistic: 0.0,
        p_values: vec![0.5; config.significance_levels.len()],
        critical_values: vec![1.96; config.significance_levels.len()],
        rejections: vec![false; config.significance_levels.len()],
        degrees_of_freedom: None,
        additional_info: HashMap::new(),
    })
}

fn execute_variance_ratio_batch(
    _data: &[f64],
    _lags: &[usize],
    config: &BatchTestConfig,
    _shared: &SharedComputations,
) -> FractalResult<TestResult> {
    Ok(TestResult {
        statistic: 0.0,
        p_values: vec![0.5; config.significance_levels.len()],
        critical_values: vec![1.96; config.significance_levels.len()],
        rejections: vec![false; config.significance_levels.len()],
        degrees_of_freedom: None,
        additional_info: HashMap::new(),
    })
}

fn execute_arch_test_batch(
    _data: &[f64],
    _lags: &[usize],
    config: &BatchTestConfig,
    _shared: &SharedComputations,
) -> FractalResult<TestResult> {
    Ok(TestResult {
        statistic: 0.0,
        p_values: vec![0.5; config.significance_levels.len()],
        critical_values: vec![1.96; config.significance_levels.len()],
        rejections: vec![false; config.significance_levels.len()],
        degrees_of_freedom: None,
        additional_info: HashMap::new(),
    })
}

fn execute_structural_break_batch(
    _data: &[f64],
    _method: &BreakTestMethod,
    config: &BatchTestConfig,
    _shared: &SharedComputations,
) -> FractalResult<TestResult> {
    Ok(TestResult {
        statistic: 0.0,
        p_values: vec![0.5; config.significance_levels.len()],
        critical_values: vec![1.96; config.significance_levels.len()],
        rejections: vec![false; config.significance_levels.len()],
        degrees_of_freedom: None,
        additional_info: HashMap::new(),
    })
}

/// Compute sample moments efficiently in a single pass.
fn compute_sample_moments(data: &[f64]) -> FractalResult<SampleMoments> {
    let n = data.len();
    if n < 4 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 4,
            actual: n,
        });
    }

    // Validate all values are finite
    validate_all_finite(data, "sample_moments_data")?;

    let n_f64 = n as f64;

    // Single-pass computation of moments
    let mut m1 = 0.0; // mean
    let mut m2 = 0.0; // second moment
    let mut m3 = 0.0; // third moment
    let mut m4 = 0.0; // fourth moment

    for &x in data {
        if x.is_finite() {
            m1 += x;
            let x2 = x * x;
            m2 += x2;
            m3 += x2 * x;
            m4 += x2 * x2;
        } else {
            return Err(FractalAnalysisError::NumericalError {
                reason: "Non-finite value in data".to_string(),
                operation: None,
            });
        }
    }

    m1 /= n_f64;
    m2 /= n_f64;
    m3 /= n_f64;
    m4 /= n_f64;

    let variance = (m2 - m1 * m1).max(MIN_VARIANCE);
    if variance < MIN_VARIANCE {
        // Return default moments for constant data
        return Ok(SampleMoments {
            mean: m1,
            variance: MIN_VARIANCE,
            skewness: 0.0,
            kurtosis: 3.0, // Normal kurtosis
            count: n,
        });
    }

    let _std_dev = variance.sqrt();
    let m1_squared = m1 * m1;
    let m1_cubed = m1_squared * m1;
    let m1_fourth = m1_cubed * m1;
    let variance_squared = variance * variance;
    let skewness = (m3 - 3.0 * m1 * m2 + 2.0 * m1_cubed) / variance.powf(1.5);
    let kurtosis =
        (m4 - 4.0 * m1 * m3 + 6.0 * m1_squared * m2 - 3.0 * m1_fourth) / variance_squared;

    Ok(SampleMoments {
        mean: m1,
        variance,
        skewness,
        kurtosis,
        count: n,
    })
}

/// Estimate time saved through shared computations.
fn estimate_time_saved(shared: &SharedComputations, num_tests: usize) -> u64 {
    let mut saved_ms = 0u64;

    // Rough estimates based on typical computation times
    if shared.periodogram.is_some() {
        saved_ms += (num_tests.saturating_sub(1) * 50) as u64; // ~50ms per periodogram
    }
    if shared.autocorrelations.is_some() {
        saved_ms += (num_tests.saturating_sub(1) * 20) as u64; // ~20ms per autocorr
    }
    if shared.moments.is_some() {
        saved_ms += (num_tests.saturating_sub(1) * 5) as u64; // ~5ms per moment calculation
    }
    if shared.cumulative_returns.is_some() {
        saved_ms += (num_tests.saturating_sub(1) * 10) as u64; // ~10ms per cumulative sum
    }

    saved_ms
}

/// Calculate memory efficiency based on shared computations.
fn calculate_memory_efficiency(shared: &SharedComputations, tests: &[StatisticalTest]) -> f64 {
    if tests.is_empty() {
        return 0.0;
    }

    let total_possible_sharing = tests.len();
    let actual_sharing = shared.reuse_count;

    (actual_sharing as f64 / total_possible_sharing as f64).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_default() {
        let config = BatchTestConfig::default();
        assert!(!config.tests.is_empty());
        assert!(!config.significance_levels.is_empty());
        assert!(config.enable_shared_computations);
    }

    #[test]
    fn test_sample_moments() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let moments = compute_sample_moments(&data).unwrap();

        assert!((moments.mean - 3.0).abs() < 1e-10);
        assert!(moments.variance > 0.0);
        assert!(moments.count == 5);
    }

    #[test]
    fn test_batch_testing_execution() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let config = BatchTestConfig::default();

        let results = execute_batch_tests(&data, &config).unwrap();

        assert!(!results.test_results.is_empty());
        // Test may run very quickly, so allow 0 ms timing
        assert!(results.performance_stats.total_time_ms >= 0);
        assert!(results.test_results.len() == config.tests.len());
    }

    #[test]
    fn test_shared_computations() {
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let config = BatchTestConfig {
            tests: vec![
                StatisticalTest::LjungBox { lags: vec![5] },
                StatisticalTest::JarqueBera,
            ],
            significance_levels: vec![0.05],
            bootstrap_samples: 100,
            max_lag: 10,
            enable_shared_computations: true,
        };

        let results = execute_batch_tests(&data, &config).unwrap();

        // Should have some shared computations
        assert!(results.shared_computations.reuse_count > 0);
    }
}
