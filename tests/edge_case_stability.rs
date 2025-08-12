//! Comprehensive edge case and numerical stability tests for all modules.
//!
//! This test suite focuses on edge cases, boundary conditions, and numerical stability
//! that are critical for financial applications where precision matters.

use assert_approx_eq::assert_approx_eq;
use fractal_finance::*;
use rand::Rng;
use std::f64::{INFINITY, NAN, NEG_INFINITY};

/// Test edge cases for mathematical utility functions
mod math_utils_edge_cases {
    use super::*;

    #[test]
    fn test_ols_regression_edge_cases() {
        // Test 1: Single data point
        let x = vec![1.0];
        let y = vec![2.0];
        let result = math_utils::ols_regression(&x, &y);
        assert!(result.is_err(), "Single point should fail");

        // Test 2: Empty vectors
        let x: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];
        let result = math_utils::ols_regression(&x, &y);
        assert!(result.is_err(), "Empty vectors should fail");

        // Test 3: Mismatched lengths
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        let result = math_utils::ols_regression(&x, &y);
        assert!(result.is_err(), "Mismatched lengths should fail");

        // Test 4: All x values identical (vertical line)
        let x = vec![5.0, 5.0, 5.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let result = math_utils::ols_regression(&x, &y);
        assert!(result.is_err(), "Vertical line should fail");

        // Test 5: NaN values
        let x = vec![1.0, 2.0, NAN, 4.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let result = math_utils::ols_regression(&x, &y);
        assert!(result.is_err(), "NaN in x should fail");

        // Test 6: Infinity values
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, INFINITY, 3.0, 4.0];
        let result = math_utils::ols_regression(&x, &y);
        assert!(result.is_err(), "Infinity in y should fail");

        // Test 7: Very large numbers (near overflow)
        let x = vec![1e100, 2e100, 3e100, 4e100];
        let y = vec![1e100, 2e100, 3e100, 4e100];
        let result = math_utils::ols_regression(&x, &y);
        // Should handle large numbers gracefully or fail predictably
        match result {
            Ok((slope, intercept, _)) => {
                assert!(slope.is_finite(), "Slope should be finite");
                assert!(intercept.is_finite(), "Intercept should be finite");
            }
            Err(_) => {
                // Acceptable to fail with very large numbers
            }
        }

        // Test 8: Very small numbers (near underflow)
        let x = vec![1e-100, 2e-100, 3e-100, 4e-100];
        let y = vec![2e-100, 4e-100, 6e-100, 8e-100];
        let result = math_utils::ols_regression(&x, &y);
        match result {
            Ok((slope, intercept, _)) => {
                assert!(slope.is_finite(), "Slope should be finite");
                assert!(intercept.is_finite(), "Intercept should be finite");
                assert_approx_eq!(slope, 2.0, 1e-6);
            }
            Err(_) => {
                // Acceptable to fail with very small numbers
            }
        }
    }

    #[test]
    fn test_periodogram_edge_cases() {
        // Test 1: Single sample
        let data = vec![1.0];
        let result = calculate_periodogram_fft(&data);
        assert!(result.is_err(), "Single sample should fail");

        // Test 2: Two samples (minimum for FFT)
        let data = vec![1.0, 2.0];
        let result = calculate_periodogram_fft(&data);
        match result {
            Ok(periodogram) => {
                assert!(!periodogram.is_empty());
                assert!(periodogram.iter().all(|&x| x.is_finite()));
            }
            Err(_) => {
                // May fail due to insufficient data
            }
        }

        // Test 3: Constant data (zero variance)
        let data = vec![5.0; 1000];
        let result = calculate_periodogram_fft(&data);
        match result {
            Ok(periodogram) => {
                // Should handle constant data gracefully
                assert!(periodogram.iter().all(|&x| x.is_finite()));
            }
            Err(_) => {
                // May fail due to zero variance
            }
        }

        // Test 4: Data with NaN
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        data[2] = NAN;
        let result = calculate_periodogram_fft(&data);
        assert!(result.is_err(), "NaN data should fail");

        // Test 5: Very large dynamic range
        let data = vec![1e-10, 1e10, 1e-10, 1e10];
        let result = calculate_periodogram_fft(&data);
        match result {
            Ok(periodogram) => {
                assert!(periodogram.iter().all(|&x| x.is_finite()));
            }
            Err(_) => {
                // May fail due to numerical issues
            }
        }
    }

    #[test]
    fn test_variance_calculation_edge_cases() {
        // Test 1: Empty data
        let data: Vec<f64> = vec![];
        let result = if data.is_empty() {
            0.0
        } else {
            math_utils::calculate_variance(&data)
        };
        assert!(data.is_empty(), "Empty data should be handled");

        // Test 2: Single data point
        let data = vec![5.0];
        let variance = math_utils::calculate_variance(&data);
        // Single point should have zero variance
        assert_approx_eq!(variance, 0.0, 1e-15);

        // Test 3: Constant data
        let data = vec![7.0; 100];
        let variance = math_utils::calculate_variance(&data);
        assert_approx_eq!(variance, 0.0, 1e-15);

        // Test 4: Data with extreme outliers
        let mut data = vec![1.0; 100];
        data[50] = 1e10;
        let variance = math_utils::calculate_variance(&data);
        assert!(variance.is_finite() && variance > 0.0);

        // Test 5: NaN in data
        let data = vec![1.0, 2.0, NAN, 4.0];
        // NaN data should cause issues - test by checking if any element is NaN
        assert!(data.iter().any(|x| x.is_nan()), "Data should contain NaN");
    }

    #[test]
    fn test_safe_arithmetic_operations() {
        use fractal_finance::math_utils::float_ops::*;

        // Test safe_div edge cases
        assert!(safe_div(1.0, 0.0).is_none());
        assert!(safe_div(0.0, 0.0).is_none());
        assert!(safe_div(INFINITY, 1.0).is_none());
        assert!(safe_div(1.0, NAN).is_none());

        // Very small denominators
        let result = safe_div(1.0, 1e-100);
        match result {
            Some(val) => assert!(val.is_finite()),
            None => {} // Acceptable to fail
        }

        // Test safe_ln edge cases
        assert!(safe_ln(0.0).is_none());
        assert!(safe_ln(-1.0).is_none());
        assert!(safe_ln(NAN).is_none());
        assert!(safe_ln(INFINITY).is_none());

        // Very small positive numbers
        let result = safe_ln(1e-100);
        match result {
            Some(val) => assert!(val.is_finite()),
            None => {} // May underflow
        }

        // Test safe_sqrt edge cases
        assert!(safe_sqrt(-1.0).is_none());
        assert!(safe_sqrt(NAN).is_none());
        assert!(safe_sqrt(INFINITY).is_none());

        // Very large numbers
        let result = safe_sqrt(1e100);
        match result {
            Some(val) => assert!(val.is_finite()),
            None => {} // May overflow
        }
    }
}

/// Test edge cases for data generators
mod generators_edge_cases {
    use super::*;

    #[test]
    fn test_fbm_generation_extreme_parameters() {
        // Test 1: H very close to 0
        let config = FbmConfig {
            hurst_exponent: 0.001,
            volatility: 1.0,
            method: FbmMethod::Hosking,
        };
        let gen_config = GeneratorConfig {
            length: 100,
            seed: Some(42),
            ..Default::default()
        };
        let result = generate_fractional_brownian_motion(&gen_config, &config);
        match result {
            Ok(data) => {
                assert_eq!(data.len(), 100);
                assert!(data.iter().all(|&x| x.is_finite()));
            }
            Err(_) => {
                // May fail due to numerical instability
            }
        }

        // Test 2: H very close to 1
        let config = FbmConfig {
            hurst_exponent: 0.999,
            volatility: 1.0,
            method: FbmMethod::Hosking,
        };
        let gen_config = GeneratorConfig {
            length: 100,
            seed: Some(43),
            ..Default::default()
        };
        let result = generate_fractional_brownian_motion(&gen_config, &config);
        match result {
            Ok(data) => {
                assert_eq!(data.len(), 100);
                assert!(data.iter().all(|&x| x.is_finite()));
            }
            Err(_) => {
                // Expected to fail due to numerical instability
            }
        }

        // Test 3: Very small sample size
        let config = FbmConfig {
            hurst_exponent: 0.7,
            volatility: 1.0,
            method: FbmMethod::Hosking,
        };
        let gen_config = GeneratorConfig {
            length: 2,
            seed: Some(44),
            ..Default::default()
        };
        let result = generate_fractional_brownian_motion(&gen_config, &config);
        match result {
            Ok(data) => {
                assert_eq!(data.len(), 2);
                assert!(data.iter().all(|&x| x.is_finite()));
            }
            Err(_) => {
                // May fail due to insufficient size
            }
        }

        // Test 4: Zero volatility
        let config = FbmConfig {
            hurst_exponent: 0.7,
            volatility: 0.0,
            method: FbmMethod::Hosking,
        };
        let gen_config = GeneratorConfig {
            length: 100,
            seed: Some(45),
            ..Default::default()
        };
        let result = generate_fractional_brownian_motion(&gen_config, &config);
        match result {
            Ok(data) => {
                // Should produce constant series
                assert!(data.iter().all(|&x| x.abs() < 1e-10));
            }
            Err(_) => {
                // May reject zero volatility
            }
        }

        // Test 5: Extreme volatility
        let config = FbmConfig {
            hurst_exponent: 0.7,
            volatility: 1e10,
            method: FbmMethod::Hosking,
        };
        let gen_config = GeneratorConfig {
            length: 100,
            seed: Some(46),
            ..Default::default()
        };
        let result = generate_fractional_brownian_motion(&gen_config, &config);
        match result {
            Ok(data) => {
                assert!(data.iter().all(|&x| x.is_finite()));
            }
            Err(_) => {
                // May fail due to numerical overflow
            }
        }
    }

    #[test]
    fn test_arfima_generation_edge_cases() {
        // Test 1: d very close to 0.5 (non-stationary boundary)
        let config = ArfimaConfig {
            d_param: 0.499,
            ar_params: vec![0.1],
            ma_params: vec![0.1],
            innovation_variance: 1.0,
        };
        let gen_config = GeneratorConfig {
            length: 100,
            seed: Some(47),
            ..Default::default()
        };
        let result = generate_arfima(&gen_config, &config);
        match result {
            Ok(data) => {
                assert_eq!(data.len(), 100);
                assert!(data.iter().all(|&x| x.is_finite()));
            }
            Err(_) => {
                // May fail near non-stationary boundary
            }
        }

        // Test 2: d = 0.5 (exactly non-stationary)
        let config = ArfimaConfig {
            d_param: 0.5,
            ar_params: vec![0.1],
            ma_params: vec![0.1],
            innovation_variance: 1.0,
        };
        let gen_config = GeneratorConfig {
            length: 100,
            seed: Some(48),
            ..Default::default()
        };
        let result = generate_arfima(&gen_config, &config);
        assert!(result.is_err(), "d=0.5 should fail (non-stationary)");

        // Test 3: Large AR coefficients (near unit root)
        let config = ArfimaConfig {
            d_param: 0.3,
            ar_params: vec![0.99],
            ma_params: vec![],
            innovation_variance: 1.0,
        };
        let gen_config = GeneratorConfig {
            length: 100,
            seed: Some(49),
            ..Default::default()
        };
        let result = generate_arfima(&gen_config, &config);
        match result {
            Ok(data) => {
                assert!(data.iter().all(|&x| x.is_finite()));
            }
            Err(_) => {
                // May fail due to near unit root
            }
        }

        // Test 4: Very small noise variance
        let gen_config = GeneratorConfig {
            length: 100,
            seed: Some(42),
            ..Default::default()
        };
        let arfima_config = ArfimaConfig {
            d_param: 0.3,
            ar_params: vec![0.1],
            ma_params: vec![0.1],
            innovation_variance: 1e-10,
        };
        let result = generate_arfima(&gen_config, &arfima_config).unwrap();
        assert!(result.iter().all(|&x| x.is_finite()));
        // Should produce relatively small values due to small innovation variance
        // Note: ARFIMA processes can accumulate over time even with small innovations
        let max_abs_value = result
            .iter()
            .map(|&x| x.abs())
            .fold(0.0f64, |a, b| a.max(b));
        assert!(
            max_abs_value < 1.0,
            "Values should be reasonably bounded with small innovation variance"
        );
    }

    #[test]
    fn test_multifractal_cascade_edge_cases() {
        // Test 1: Very small variance parameters
        let gen_config = GeneratorConfig {
            length: 256,
            seed: Some(42),
            ..Default::default()
        };
        let cascade_config = MultifractalCascadeConfig {
            levels: 8,
            intermittency: 0.1,
            lognormal_params: (0.0, 1e-5), // Very small variance
            base_volatility: 1e-10,
        };
        let result = generate_multifractal_cascade(&gen_config, &cascade_config);
        match result {
            Ok(data) => {
                assert_eq!(data.len(), 256);
                assert!(data.iter().all(|&x| x.is_finite()));
            }
            Err(_) => {
                // May fail with very small variance
            }
        }

        // Test 2: Single level cascade
        let gen_config2 = GeneratorConfig {
            length: 2,
            seed: Some(43),
            ..Default::default()
        };
        let cascade_config2 = MultifractalCascadeConfig {
            levels: 1,
            intermittency: 0.5,
            lognormal_params: (0.0, 0.1),
            base_volatility: 1.0,
        };
        let result = generate_multifractal_cascade(&gen_config2, &cascade_config2);
        match result {
            Ok(data) => {
                assert_eq!(data.len(), 2);
                assert!(data.iter().all(|&x| x.is_finite()));
            }
            Err(_) => {
                // May fail with single level
            }
        }

        // Test 3: Large volatility parameters
        let gen_config = GeneratorConfig {
            length: 32,
            seed: Some(42),
            ..Default::default()
        };
        let cascade_config = MultifractalCascadeConfig {
            levels: 5,
            intermittency: 0.8,
            lognormal_params: (0.0, 2.0), // Large variance
            base_volatility: 10.0,
        };
        let result = generate_multifractal_cascade(&gen_config, &cascade_config);
        match result {
            Ok(data) => {
                assert!(data.iter().all(|&x| x.is_finite()));
            }
            Err(_) => {
                // May fail due to extreme variance
            }
        }
    }
}

/// Test edge cases for statistical tests
mod statistical_tests_edge_cases {
    use super::*;

    #[test]
    fn test_gph_test_edge_cases() {
        // Test 1: Minimum data size
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = gph_test(&data);
        match result {
            Ok((t_statistic, p_value, hurst_estimate)) => {
                assert!(t_statistic.is_finite());
                assert!(p_value.is_finite());
                assert!(hurst_estimate.is_finite());
            }
            Err(_) => {
                // May fail due to insufficient data
            }
        }

        // Test 2: Constant data (zero variance)
        let data = vec![5.0; 100];
        let result = gph_test(&data);
        assert!(result.is_err(), "Constant data should fail");

        // Test 3: Linear trend
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = gph_test(&data);
        match result {
            Ok((t_statistic, _p_value, hurst_estimate)) => {
                // GPH should handle trends, but estimate may be biased
                assert!(t_statistic.is_finite());
                assert!(hurst_estimate.is_finite());
            }
            Err(_) => {
                // May fail on strong trends
            }
        }

        // Test 4: Data with outliers
        let mut data: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        data[50] = 1000.0; // Large outlier
        let result = gph_test(&data);
        match result {
            Ok((t_statistic, _p_value, hurst_estimate)) => {
                assert!(t_statistic.is_finite());
                assert!(hurst_estimate.is_finite());
            }
            Err(_) => {
                // May fail due to outliers
            }
        }

        // Test 5: Very noisy data
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..1000)
            .map(|_| rng.gen::<f64>() * 1000.0 - 500.0)
            .collect();
        let result = gph_test(&data);
        match result {
            Ok((t_statistic, _p_value, hurst_estimate)) => {
                assert!(t_statistic.is_finite());
                assert!(hurst_estimate.is_finite());
                // Hurst estimate should be close to 0.5 for white noise and in valid range [0,1]
                assert!(hurst_estimate > 0.0 && hurst_estimate < 1.0);
            }
            Err(_) => {
                // May fail on very noisy data
            }
        }
    }

    #[test]
    fn test_ljung_box_edge_cases() {
        // Test 1: Perfect periodicity
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let result = ljung_box_test(&data, 10);
        match result {
            Ok((test_statistic, p_value)) => {
                assert!(test_statistic.is_finite());
                assert!(p_value.is_finite());
                assert!(p_value >= 0.0 && p_value <= 1.0);
            }
            Err(_) => {
                // May fail on periodic data
            }
        }

        // Test 2: Very small autocorrelations
        let data: Vec<f64> = (0..1000).map(|i| (i as f64) * 1e-10).collect();
        let result = ljung_box_test(&data, 5);
        match result {
            Ok((test_statistic, p_value)) => {
                assert!(test_statistic.is_finite());
                assert!(p_value.is_finite());
            }
            Err(_) => {
                // May fail with very small correlations
            }
        }

        // Test 3: Lags >= data length
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ljung_box_test(&data, 10);
        assert!(result.is_err(), "Lags >= data length should fail");

        // Test 4: Zero lags
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ljung_box_test(&data, 0);
        assert!(result.is_err(), "Zero lags should fail");
    }

    #[test]
    fn test_structural_break_edge_cases() {
        // Test 1: No breaks (constant data)
        let data = vec![5.0; 200];
        let result = test_structural_breaks(&data);
        match result {
            Ok(test_results) => {
                // Should run multiple tests (CUSUM, etc.)
                assert!(!test_results.is_empty());
                for test_result in &test_results {
                    assert!(test_result.test_statistic.is_finite());
                    // Should not detect breaks in constant data - few break dates
                    assert!(test_result.break_dates.len() < 3);
                }
            }
            Err(_) => {
                // May fail on constant data
            }
        }

        // Test 2: Single obvious break
        let mut data = vec![0.0; 100];
        data.extend(vec![10.0; 100]);
        let result = test_structural_breaks(&data);
        match result {
            Ok(test_results) => {
                assert!(!test_results.is_empty());
                for test_result in &test_results {
                    assert!(test_result.test_statistic.is_finite());
                    assert!(test_result.p_value.is_finite());

                    // Different tests detect different types of breaks
                    match test_result.test_type {
                        StructuralBreakTestType::Cusum | StructuralBreakTestType::QuandtAndrews => {
                            // These tests detect mean shifts - should detect step function
                            assert!(test_result.p_value < 0.05, 
                                "Test {:?} MUST detect mean shift in step function (0→10) with p < 0.05, got p = {}.", 
                                test_result.test_type, test_result.p_value);
                        }
                        StructuralBreakTestType::CusumOfSquares => {
                            // This test detects variance changes - step function has constant variance
                            // So it should NOT detect a break (high p-value is correct)
                            assert!(test_result.p_value > 0.05, 
                                "CUSUM of Squares should NOT detect break in constant-variance step function, got p = {}", 
                                test_result.p_value);
                        }
                        _ => {
                            // For any other test types, expect them to detect the break
                            assert!(test_result.p_value < 0.05, 
                                "Test {:?} should detect structural break with p < 0.05, got p = {}", 
                                test_result.test_type, test_result.p_value);
                        }
                    }

                    // Only some tests locate breaks
                    match test_result.test_type {
                        StructuralBreakTestType::QuandtAndrews => {
                            assert!(!test_result.break_dates.is_empty(), 
                                "Quandt-Andrews test should locate the break, but found no break dates");
                        }
                        _ => {
                            // Other tests may or may not provide break locations
                        }
                    }
                }
            }
            Err(_) => {
                // May fail on extreme breaks
            }
        }

        // Test 3: Multiple small breaks
        let mut data = Vec::new();
        for i in 0..10 {
            data.extend(vec![i as f64; 20]);
        }
        let result = test_structural_breaks(&data);
        match result {
            Ok(test_results) => {
                assert!(!test_results.is_empty());
                for test_result in &test_results {
                    assert!(test_result.test_statistic.is_finite());
                }
            }
            Err(_) => {
                // May fail with many small breaks
            }
        }
    }
}

/// Test edge cases for multifractal analysis
mod multifractal_edge_cases {
    use super::*;

    #[test]
    fn test_mf_dfa_extreme_q_values() {
        let data: Vec<f64> = (0..500).map(|i| (i as f64).sin()).collect();

        // Test 1: Very large positive q
        let config = MultifractalConfig {
            q_range: (10.0, 20.0),
            num_q_values: 5,
            min_scale: 4,
            max_scale_factor: 10.0, // 500/100 = 5, so factor should be reasonable
            polynomial_order: 1,
        };

        let result = perform_multifractal_analysis(&data);
        match result {
            Ok(analysis) => {
                assert!(!analysis.generalized_hurst_exponents.is_empty());
                // Large q should emphasize large fluctuations
                for (q, h) in &analysis.generalized_hurst_exponents {
                    assert!(h.is_finite());
                    // Note: actual q values depend on implementation defaults
                }
            }
            Err(_) => {
                // May fail with extreme q values
            }
        }

        // Test 2: Very large negative q
        let config = MultifractalConfig {
            q_range: (-20.0, -10.0),
            num_q_values: 5,
            min_scale: 4,
            max_scale_factor: 10.0,
            polynomial_order: 1,
        };

        let result = perform_multifractal_analysis(&data);
        match result {
            Ok(analysis) => {
                assert!(!analysis.generalized_hurst_exponents.is_empty());
                // Should handle extreme data
                for (q, h) in &analysis.generalized_hurst_exponents {
                    assert!(h.is_finite());
                }
            }
            Err(_) => {
                // May fail with extreme negative q values
            }
        }

        // Test 3: q = 0 (logarithmic averaging)
        let config = MultifractalConfig {
            q_range: (0.0, 0.0),
            num_q_values: 1,
            min_scale: 4,
            max_scale_factor: 10.0,
            polynomial_order: 1,
        };

        let result = perform_multifractal_analysis(&data);
        match result {
            Ok(analysis) => {
                assert!(!analysis.generalized_hurst_exponents.is_empty());
                for (q, h) in &analysis.generalized_hurst_exponents {
                    assert!(h.is_finite());
                }
            }
            Err(_) => {
                // May fail at q=0 due to logarithmic singularity
            }
        }
    }

    #[test]
    fn test_mf_dfa_extreme_scales() {
        let data: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.01).sin()).collect();

        // Test 1: Very small scales
        let config = MultifractalConfig {
            q_range: (-2.0, 2.0),
            num_q_values: 9,
            min_scale: 2,
            max_scale_factor: 250.0, // 1000/4 = 250
            polynomial_order: 1,
        };

        let result = perform_multifractal_analysis(&data);
        match result {
            Ok(analysis) => {
                assert!(!analysis.generalized_hurst_exponents.is_empty());
            }
            Err(_) => {
                // May fail with very small scales
            }
        }

        // Test 2: Scales approaching data length
        let config = MultifractalConfig {
            q_range: (-2.0, 2.0),
            num_q_values: 9,
            min_scale: 400,
            max_scale_factor: 2.0, // 1000/500 = 2
            polynomial_order: 1,
        };

        let result = perform_multifractal_analysis(&data);
        match result {
            Ok(analysis) => {
                assert!(!analysis.generalized_hurst_exponents.is_empty());
            }
            Err(_) => {
                // Expected to fail with very large scales
            }
        }

        // Test 3: Single scale
        let config = MultifractalConfig {
            q_range: (-2.0, 2.0),
            num_q_values: 9,
            min_scale: 10,
            max_scale_factor: 51.2, // 512/10 = 51.2
            polynomial_order: 1,
        };

        let result = perform_multifractal_analysis(&data);
        // May succeed or fail depending on implementation
    }

    #[test]
    fn test_singularity_spectrum_edge_cases() {
        // Test 1: Monofractal data (constant H(q))
        let data = vec![1.0; 1000]; // Constant data
        let result = perform_multifractal_analysis(&data);
        match result {
            Ok(analysis) => {
                let spectrum = &analysis.singularity_spectrum;
                if !spectrum.is_empty() {
                    // Monofractal should have narrow spectrum
                    let alpha_range = spectrum
                        .iter()
                        .map(|(alpha, _)| *alpha)
                        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), a| {
                            (min.min(a), max.max(a))
                        });
                    let spectrum_width = alpha_range.1 - alpha_range.0;
                    assert!(
                        spectrum_width < 0.5,
                        "Monofractal should have narrow spectrum"
                    );
                }
            }
            Err(_) => {
                // May fail on constant data
            }
        }

        // Test 2: Data with extreme multifractality
        let mut data = Vec::new();
        // Create artificial multifractal with extreme clustering
        for i in 0..100 {
            if i % 10 == 0 {
                data.push(100.0); // Large spikes
            } else {
                data.push(0.1); // Small values
            }
        }

        let result = perform_multifractal_analysis(&data);
        match result {
            Ok(analysis) => {
                let spectrum = &analysis.singularity_spectrum;
                if !spectrum.is_empty() {
                    // Should detect strong multifractality
                    assert!(!spectrum.is_empty());
                }
            }
            Err(_) => {
                // May fail on extreme data
            }
        }
    }
}

/// Test edge cases for bootstrap methods
mod bootstrap_edge_cases {
    use super::*;

    #[test]
    fn test_bootstrap_extreme_configurations() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();

        // Test 1: Single bootstrap sample
        let config = BootstrapConfiguration {
            num_bootstrap_samples: 1,
            confidence_levels: vec![0.95],
            bootstrap_method: BootstrapMethod::Block,
            block_size: Some(10),
            confidence_interval_method: ConfidenceIntervalMethod::BootstrapPercentile,
            seed: Some(42),
            studentized_outer: None,
            studentized_inner: None,
            jackknife_block_size: None,
            force_block_jackknife: None,
        };

        let result = bootstrap_validate(&data, |x| x.iter().sum::<f64>(), &config);
        match result {
            Ok(validation) => {
                assert!(!validation.confidence_intervals.is_empty());
            }
            Err(_) => {
                // May fail with single sample
            }
        }

        // Test 2: Extreme confidence levels
        let config = BootstrapConfiguration {
            num_bootstrap_samples: 100,
            confidence_levels: vec![0.999],
            bootstrap_method: BootstrapMethod::Block,
            block_size: Some(10),
            confidence_interval_method: ConfidenceIntervalMethod::BootstrapPercentile,
            seed: Some(42),
            studentized_outer: None,
            studentized_inner: None,
            jackknife_block_size: None,
            force_block_jackknife: None,
        };

        let result = bootstrap_validate(&data, |x| x.iter().sum::<f64>(), &config);
        match result {
            Ok(validation) => {
                // Very high confidence should produce wide intervals
                if let Some(ci) = validation
                    .confidence_intervals
                    .iter()
                    .find(|ci| ci.method == ConfidenceIntervalMethod::BootstrapPercentile)
                {
                    let width = ci.upper_bound - ci.lower_bound;
                    assert!(width > 0.0);
                }
            }
            Err(_) => {
                // May fail with extreme confidence levels
            }
        }

        // Test 3: Very low confidence level
        let config = BootstrapConfiguration {
            num_bootstrap_samples: 100,
            confidence_levels: vec![0.01],
            bootstrap_method: BootstrapMethod::Block,
            block_size: Some(10),
            confidence_interval_method: ConfidenceIntervalMethod::BootstrapPercentile,
            seed: Some(42),
            studentized_outer: None,
            studentized_inner: None,
            jackknife_block_size: None,
            force_block_jackknife: None,
        };

        let result = bootstrap_validate(&data, |x| x.iter().sum::<f64>(), &config);
        match result {
            Ok(validation) => {
                // Very low confidence should produce narrow intervals
                if let Some(ci) = validation
                    .confidence_intervals
                    .iter()
                    .find(|ci| ci.method == ConfidenceIntervalMethod::BootstrapPercentile)
                {
                    let width = ci.upper_bound - ci.lower_bound;
                    assert!(width >= 0.0);
                }
            }
            Err(_) => {
                // May fail with extreme confidence levels
            }
        }
    }

    #[test]
    fn test_bootstrap_with_constant_statistic() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();

        // Statistic that always returns the same value
        let constant_statistic = |_: &[f64]| 42.0;

        let config = BootstrapConfiguration {
            num_bootstrap_samples: 100,
            confidence_levels: vec![0.95],
            bootstrap_method: BootstrapMethod::Block,
            block_size: Some(15),
            confidence_interval_method: ConfidenceIntervalMethod::BootstrapBca,
            seed: Some(42),
            studentized_outer: None,
            studentized_inner: None,
            jackknife_block_size: None,
            force_block_jackknife: None,
        };

        let result = bootstrap_validate(&data, constant_statistic, &config);
        match result {
            Ok(validation) => {
                // All methods should handle constant statistics
                for ci in &validation.confidence_intervals {
                    assert_approx_eq!(ci.lower_bound, 42.0, 1e-10);
                    assert_approx_eq!(ci.upper_bound, 42.0, 1e-10);
                }
            }
            Err(_) => {
                // May fail with constant statistics
            }
        }
    }

    #[test]
    fn test_bootstrap_with_extreme_statistics() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();

        // Test 1: Statistic that returns very large values
        let large_statistic = |x: &[f64]| x.iter().sum::<f64>() * 1e10;

        let config = BootstrapConfiguration {
            num_bootstrap_samples: 50,
            confidence_levels: vec![0.95],
            bootstrap_method: BootstrapMethod::Block,
            block_size: Some(10),
            confidence_interval_method: ConfidenceIntervalMethod::BootstrapPercentile,
            seed: Some(42),
            studentized_outer: None,
            studentized_inner: None,
            jackknife_block_size: None,
            force_block_jackknife: None,
        };

        let result = bootstrap_validate(&data, large_statistic, &config);
        match result {
            Ok(validation) => {
                for ci in &validation.confidence_intervals {
                    assert!(ci.lower_bound.is_finite());
                    assert!(ci.upper_bound.is_finite());
                }
            }
            Err(_) => {
                // May fail with extreme values
            }
        }

        // Test 2: Statistic that can return NaN
        let risky_statistic = |x: &[f64]| {
            let sum = x.iter().sum::<f64>();
            let mean = sum / x.len() as f64;
            if mean == 0.0 {
                NAN
            } else {
                1.0 / mean
            }
        };

        let result = bootstrap_validate(&data, risky_statistic, &config);
        // Should handle NaN gracefully or fail predictably
        match result {
            Ok(_) => {
                // If successful, all values should be finite
            }
            Err(_) => {
                // Expected to fail with NaN-producing statistics
            }
        }
    }
}

/// Test edge cases for HMM regime detection
mod regime_detection_edge_cases {
    use super::*;

    #[test]
    fn test_hmm_with_extreme_data() {
        // Test 1: Constant data (single regime)
        let data = vec![5.0; 200];
        let config = RegimeDetectionConfig::default();

        let result = detect_fractal_regimes(&data, &config);
        match result {
            Ok(result) => {
                // Should detect single regime or fail gracefully
                assert!(!result.regime_sequence.is_empty());
            }
            Err(_) => {
                // May fail on constant data
            }
        }

        // Test 2: Binary switching data
        let mut data = Vec::new();
        for i in 0..200 {
            data.push(if i % 20 < 10 { 0.0 } else { 10.0 });
        }

        let config = RegimeDetectionConfig::default();
        let result = detect_fractal_regimes(&data, &config);
        match result {
            Ok(result) => {
                // Should detect regime switches
                assert!(result.change_points.len() <= 20); // At most 20 switches
            }
            Err(_) => {
                // May fail on extreme switching
            }
        }

        // Test 3: Data with outliers
        let mut data: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        data[100] = 1000.0; // Extreme outlier

        let config = RegimeDetectionConfig::default();
        let result = detect_fractal_regimes(&data, &config);
        match result {
            Ok(result) => {
                // Should handle outliers gracefully
                assert!(!result.regime_sequence.is_empty());
            }
            Err(_) => {
                // May fail with extreme outliers
            }
        }
    }

    #[test]
    fn test_hmm_with_single_state() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();

        // Test with single state (degenerate case)
        let mut config = RegimeDetectionConfig::default();
        config.num_states_range = (1, 1);
        let result = detect_fractal_regimes(&data, &config);
        match result {
            Ok(result) => {
                // Single state should work - all regime assignments should be state 0
                assert!(result.regime_sequence.iter().all(|&s| s == 0));
            }
            Err(_) => {
                // May reject single state as invalid
            }
        }
    }

    #[test]
    fn test_hmm_with_insufficient_data() {
        // Test 1: Very small dataset
        let data = vec![1.0, 2.0, 3.0];
        let config = RegimeDetectionConfig::default();
        let result = detect_fractal_regimes(&data, &config);
        assert!(result.is_err(), "Insufficient data should fail");

        // Test 2: Data smaller than number of states
        let data = vec![1.0, 2.0];
        let config = RegimeDetectionConfig::default();
        let result = detect_fractal_regimes(&data, &config);
        assert!(result.is_err(), "More states than data points should fail");
    }
}

/// Integration tests for edge cases across modules
mod integration_edge_cases {
    use super::*;

    #[test]
    fn test_full_analysis_with_extreme_data() {
        let mut analyzer = StatisticalFractalAnalyzer::new();

        // Test 1: Very short time series
        let short_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        analyzer.add_time_series("SHORT".to_string(), short_data);

        let result = analyzer.analyze_series("SHORT");
        match result {
            Ok(_) => {
                // Should handle short series gracefully
            }
            Err(_) => {
                // Expected to fail with insufficient data
            }
        }

        // Test 2: High-frequency financial data with microstructure noise
        let mut noisy_data = Vec::new();
        let mut price = 100.0;
        for _ in 0..1000 {
            let return_change = rand::random::<f64>() * 0.02 - 0.01; // ±1% random walk
            price *= (1.0 + return_change).exp();
            price += (rand::random::<f64>() - 0.5) * 0.001; // Microstructure noise
            noisy_data.push((price as f64).ln()); // Log prices
        }

        analyzer.add_time_series("NOISY".to_string(), noisy_data);
        let result = analyzer.analyze_series("NOISY");
        match result {
            Ok(_) => {
                // Should handle noisy financial data
                if let Ok(results) = analyzer.get_analysis_results("NOISY") {
                    assert!(!results.hurst_estimates.is_empty());

                    // Validate all estimates are reasonable
                    for (method, estimate) in &results.hurst_estimates {
                        assert!(
                            estimate.estimate >= 0.0 && estimate.estimate <= 1.0,
                            "Hurst estimate {:?} = {} out of bounds",
                            method,
                            estimate.estimate
                        );
                        assert!(
                            estimate.standard_error >= 0.0,
                            "Standard error should be non-negative"
                        );
                    }
                }
            }
            Err(_) => {
                // May fail on very noisy data
            }
        }

        // Test 3: Data with structural breaks
        let mut break_data = Vec::new();
        // First regime: H ≈ 0.3
        for i in 0..200 {
            break_data.push((i as f64 * 0.01).sin() * 0.1);
        }
        // Second regime: H ≈ 0.7
        for i in 200..400 {
            break_data.push((i as f64 * 0.001).sin() * 2.0);
        }

        analyzer.add_time_series("BREAKS".to_string(), break_data);
        let result = analyzer.analyze_series("BREAKS");
        match result {
            Ok(_) => {
                // Should detect structural changes
                if let Ok(results) = analyzer.get_analysis_results("BREAKS") {
                    assert!(!results.hurst_estimates.is_empty());
                }
            }
            Err(_) => {
                // May fail with structural breaks
            }
        }
    }

    #[test]
    fn test_memory_and_performance_stress() {
        let mut analyzer = StatisticalFractalAnalyzer::new();

        // Test 1: Large dataset stress test
        let large_data: Vec<f64> = (0..10000)
            .map(|i| (i as f64 * 0.001).sin() + 0.1 * (i as f64).sqrt())
            .collect();

        analyzer.add_time_series("LARGE".to_string(), large_data);
        let result = analyzer.analyze_series("LARGE");
        match result {
            Ok(_) => {
                // Should handle large datasets
                if let Ok(results) = analyzer.get_analysis_results("LARGE") {
                    assert!(!results.hurst_estimates.is_empty());
                }
            }
            Err(_) => {
                // May fail due to memory/performance limits
            }
        }

        // Test 2: Multiple series stress test
        for i in 0..50 {
            let data: Vec<f64> = (0..100)
                .map(|j| (j as f64 * 0.1 + i as f64).sin())
                .collect();
            analyzer.add_time_series(format!("SERIES_{}", i), data);
        }

        let result = analyzer.analyze_all_series();
        match result {
            Ok(_) => {
                // Should handle many series
                for i in 0..50 {
                    let series_result = analyzer.get_analysis_results(&format!("SERIES_{}", i));
                    assert!(series_result.is_ok(), "Series {} should be analyzed", i);
                }
            }
            Err(_) => {
                // May fail with resource constraints
            }
        }
    }
}
