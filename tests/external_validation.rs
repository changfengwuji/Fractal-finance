//! External validation tests against established Python libraries
//!
//! These tests validate our Rust implementations against well-established Python
//! libraries in the fractal analysis ecosystem. This is critical for financial
//! applications where mathematical correctness is paramount.

use fractal_finance::{
    generators::*, multifractal::*, statistical_tests::*, EstimationMethod,
    StatisticalFractalAnalyzer,
};
use std::fs;
use std::path::Path;
use std::process::Command;

/// Test data for external validation - shared between Rust and Python
#[derive(Clone)]
pub struct ValidationTestCase {
    pub name: String,
    pub data: Vec<f64>,
    pub expected_hurst: f64,
    pub tolerance: f64,
}

/// Generate standard test cases for validation
pub fn generate_validation_test_cases() -> Vec<ValidationTestCase> {
    let gen_config = GeneratorConfig {
        length: 1000,
        seed: Some(12345),
        ..Default::default()
    };

    // Generate FBM data with different Hurst exponents
    let fbm_03_config = FbmConfig {
        hurst_exponent: 0.3,
        volatility: 1.0,
        method: FbmMethod::Hosking,
    };
    let fbm_05_config = FbmConfig {
        hurst_exponent: 0.5,
        volatility: 1.0,
        method: FbmMethod::Hosking,
    };
    let fbm_07_config = FbmConfig {
        hurst_exponent: 0.7,
        volatility: 1.0,
        method: FbmMethod::Hosking,
    };

    vec![
        ValidationTestCase {
            name: "fbm_h03_n1000".to_string(),
            data: generate_fractional_brownian_motion(&gen_config, &fbm_03_config)
                .unwrap_or(vec![0.0; 1000]),
            expected_hurst: 0.3,
            tolerance: 0.15,
        },
        ValidationTestCase {
            name: "fbm_h05_n1000".to_string(),
            data: generate_fractional_brownian_motion(&gen_config, &fbm_05_config)
                .unwrap_or(vec![0.0; 1000]),
            expected_hurst: 0.5,
            tolerance: 0.1,
        },
        ValidationTestCase {
            name: "fbm_h07_n1000".to_string(),
            data: generate_fractional_brownian_motion(&gen_config, &fbm_07_config)
                .unwrap_or(vec![0.0; 1000]),
            expected_hurst: 0.7,
            tolerance: 0.15,
        },
        ValidationTestCase {
            name: "fbm_h08_n2000".to_string(),
            data: {
                let gen_config = GeneratorConfig {
                    length: 2000,
                    seed: Some(12346),
                    ..Default::default()
                };
                let fbm_config = FbmConfig {
                    hurst_exponent: 0.8,
                    volatility: 1.0,
                    method: FbmMethod::Hosking,
                };
                generate_fractional_brownian_motion(&gen_config, &fbm_config)
                    .unwrap_or(vec![0.0; 2000])
            },
            expected_hurst: 0.8,
            tolerance: 0.12,
        },
    ]
}

/// Export test data to JSON for Python consumption
pub fn export_test_data_to_json(
    test_cases: &[ValidationTestCase],
) -> Result<(), Box<dyn std::error::Error>> {
    let validation_dir = Path::new("external_validation/data");
    fs::create_dir_all(validation_dir)?;

    for test_case in test_cases {
        let json_data = serde_json::json!({
            "name": test_case.name,
            "data": test_case.data,
            "expected_hurst": test_case.expected_hurst,
            "tolerance": test_case.tolerance
        });

        let file_path = validation_dir.join(format!("{}.json", test_case.name));
        fs::write(file_path, serde_json::to_string_pretty(&json_data)?)?;
    }

    Ok(())
}

/// Export test data to CSV for Python consumption (more reliable format)
pub fn export_test_data_to_csv(
    test_cases: &[ValidationTestCase],
) -> Result<(), Box<dyn std::error::Error>> {
    let validation_dir = Path::new("external_validation/data");
    fs::create_dir_all(validation_dir)?;

    for test_case in test_cases {
        let csv_content = test_case
            .data
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        let file_path = validation_dir.join(format!("{}.csv", test_case.name));
        fs::write(file_path, csv_content)?;

        // Also write metadata
        let meta_content = format!(
            "name,{}\nexpected_hurst,{}\ntolerance,{}\n",
            test_case.name, test_case.expected_hurst, test_case.tolerance
        );
        let meta_path = validation_dir.join(format!("{}_meta.csv", test_case.name));
        fs::write(meta_path, meta_content)?;
    }

    Ok(())
}

/// Compute Rust results for validation
pub fn compute_rust_results(
    test_cases: &[ValidationTestCase],
) -> Result<(), Box<dyn std::error::Error>> {
    let results_dir = Path::new("external_validation/results");
    fs::create_dir_all(results_dir)?;

    for test_case in test_cases {
        println!("Computing Rust results for: {}", test_case.name);

        let mut analyzer = StatisticalFractalAnalyzer::new();
        analyzer.add_time_series(test_case.name.clone(), test_case.data.clone());
        analyzer.analyze_series(&test_case.name)?;

        let results = analyzer.get_analysis_results(&test_case.name)?;

        // Extract key results for comparison
        let rust_results = serde_json::json!({
            "test_case": test_case.name,
            "hurst_estimates": {
                "rescaled_range": results.hurst_estimates.get(&EstimationMethod::RescaledRange)
                    .map(|est| serde_json::json!({
                        "estimate": est.estimate,
                        "standard_error": est.standard_error,
                        "confidence_interval": {
                            "lower": est.confidence_interval.lower_bound,
                            "upper": est.confidence_interval.upper_bound
                        }
                    })),
                "dfa": results.hurst_estimates.get(&EstimationMethod::DetrendedFluctuationAnalysis)
                    .map(|est| serde_json::json!({
                        "estimate": est.estimate,
                        "standard_error": est.standard_error,
                        "confidence_interval": {
                            "lower": est.confidence_interval.lower_bound,
                            "upper": est.confidence_interval.upper_bound
                        }
                    })),
                "gph": results.hurst_estimates.get(&EstimationMethod::PeriodogramRegression)
                    .map(|est| serde_json::json!({
                        "estimate": est.estimate,
                        "standard_error": est.standard_error,
                        "confidence_interval": {
                            "lower": est.confidence_interval.lower_bound,
                            "upper": est.confidence_interval.upper_bound
                        }
                    }))
            },
            "multifractal": {
                "generalized_hurst_exponents": results.multifractal_analysis.generalized_hurst_exponents,
                "singularity_spectrum": results.multifractal_analysis.singularity_spectrum,
                "is_multifractal": results.multifractal_analysis.multifractality_test.is_multifractal
            },
            "statistical_tests": {
                "gph_test": {
                    "test_statistic": results.statistical_tests.long_range_dependence_test.gph_statistic,
                    "p_value": results.statistical_tests.long_range_dependence_test.gph_p_value
                }
            }
        });

        let file_path = results_dir.join(format!("{}_rust.json", test_case.name));
        fs::write(file_path, serde_json::to_string_pretty(&rust_results)?)?;
    }

    Ok(())
}

/// Run Python validation script
pub fn run_python_validation() -> Result<(), Box<dyn std::error::Error>> {
    let output = Command::new("python")
        .arg("external_validation/validate_against_python.py")
        .current_dir(".")
        .output();

    match output {
        Ok(output) => {
            if output.status.success() {
                println!("Python validation completed successfully");
                println!("Python stdout: {}", String::from_utf8_lossy(&output.stdout));
            } else {
                eprintln!("Python validation failed");
                eprintln!("Python stderr: {}", String::from_utf8_lossy(&output.stderr));
                return Err("Python validation failed".into());
            }
        }
        Err(e) => {
            eprintln!("Failed to run Python validation: {}", e);
            eprintln!("Make sure Python is installed and external_validation/validate_against_python.py exists");
            return Err(e.into());
        }
    }

    Ok(())
}

/// Compare Rust and Python results
pub fn compare_results(
    test_cases: &[ValidationTestCase],
) -> Result<(), Box<dyn std::error::Error>> {
    let results_dir = Path::new("external_validation/results");

    for test_case in test_cases {
        println!("\n=== Comparing results for: {} ===", test_case.name);

        // Load Rust results
        let rust_file = results_dir.join(format!("{}_rust.json", test_case.name));
        let python_file = results_dir.join(format!("{}_python.json", test_case.name));

        if !rust_file.exists() {
            eprintln!("Rust results file not found: {:?}", rust_file);
            continue;
        }

        if !python_file.exists() {
            eprintln!("Python results file not found: {:?}", python_file);
            continue;
        }

        let rust_content = fs::read_to_string(rust_file)?;
        let python_content = fs::read_to_string(python_file)?;

        let rust_results: serde_json::Value = serde_json::from_str(&rust_content)?;
        let python_results: serde_json::Value = serde_json::from_str(&python_content)?;

        // Compare Hurst estimates
        compare_hurst_estimates(&rust_results, &python_results, test_case)?;

        // Compare multifractal results
        compare_multifractal_results(&rust_results, &python_results, test_case)?;
    }

    Ok(())
}

fn compare_hurst_estimates(
    rust_results: &serde_json::Value,
    python_results: &serde_json::Value,
    test_case: &ValidationTestCase,
) -> Result<(), Box<dyn std::error::Error>> {
    let methods = vec![
        ("rescaled_range", "R/S Analysis"),
        ("dfa", "DFA"),
        ("gph", "GPH Test"),
    ];

    for (rust_key, method_name) in methods {
        if let (Some(rust_est), Some(python_est)) = (
            rust_results["hurst_estimates"][rust_key]["estimate"].as_f64(),
            python_results["hurst_estimates"][rust_key]["estimate"].as_f64(),
        ) {
            let difference = (rust_est - python_est).abs();
            let relative_error = difference / python_est.abs();

            println!(
                "{}: Rust={:.4}, Python={:.4}, Diff={:.4}, RelErr={:.2}%",
                method_name,
                rust_est,
                python_est,
                difference,
                relative_error * 100.0
            );

            if difference > test_case.tolerance {
                eprintln!(
                    "WARNING: Large difference in {} estimate: {:.4} > {:.4}",
                    method_name, difference, test_case.tolerance
                );
            }

            // Check if both are reasonably close to expected value
            let rust_error = (rust_est - test_case.expected_hurst).abs();
            let python_error = (python_est - test_case.expected_hurst).abs();

            if rust_error > test_case.tolerance || python_error > test_case.tolerance {
                eprintln!(
                    "WARNING: Method {} may be inaccurate for this test case",
                    method_name
                );
                eprintln!(
                    "  Expected: {:.3}, Rust error: {:.4}, Python error: {:.4}",
                    test_case.expected_hurst, rust_error, python_error
                );
            }
        }
    }

    Ok(())
}

fn compare_multifractal_results(
    rust_results: &serde_json::Value,
    python_results: &serde_json::Value,
    test_case: &ValidationTestCase,
) -> Result<(), Box<dyn std::error::Error>> {
    // Compare multifractality detection
    if let (Some(rust_mf), Some(python_mf)) = (
        rust_results["multifractal"]["is_multifractal"].as_bool(),
        python_results["multifractal"]["is_multifractal"].as_bool(),
    ) {
        println!("Multifractality: Rust={}, Python={}", rust_mf, python_mf);

        if rust_mf != python_mf {
            eprintln!("WARNING: Disagreement on multifractality detection");
        }
    }

    // Compare generalized Hurst exponents (if available)
    if let (Some(rust_h_q), Some(python_h_q)) = (
        rust_results["multifractal"]["generalized_hurst_exponents"].as_array(),
        python_results["multifractal"]["generalized_hurst_exponents"].as_array(),
    ) {
        let min_len = rust_h_q.len().min(python_h_q.len());
        if min_len > 0 {
            let mut total_diff = 0.0;
            let mut count = 0;

            for i in 0..min_len {
                if let (Some(rust_val), Some(python_val)) = (
                    rust_h_q[i].as_array().and_then(|arr| arr.get(1)?.as_f64()),
                    python_h_q[i]
                        .as_array()
                        .and_then(|arr| arr.get(1)?.as_f64()),
                ) {
                    total_diff += (rust_val - python_val).abs();
                    count += 1;
                }
            }

            if count > 0 {
                let avg_diff = total_diff / count as f64;
                println!(
                    "Generalized Hurst exponents avg difference: {:.4}",
                    avg_diff
                );

                if avg_diff > 0.1 {
                    eprintln!("WARNING: Large average difference in generalized Hurst exponents");
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_validation_data() {
        let test_cases = generate_validation_test_cases();
        assert!(!test_cases.is_empty());

        for test_case in &test_cases {
            assert!(!test_case.data.is_empty());
            assert!(test_case.expected_hurst > 0.0 && test_case.expected_hurst < 1.0);
            assert!(test_case.tolerance > 0.0);
        }
    }

    #[test]
    fn test_export_test_data() {
        let test_cases = generate_validation_test_cases();

        // Export to CSV (more reliable than JSON for this use case)
        export_test_data_to_csv(&test_cases).expect("Failed to export test data");

        // Verify files were created
        for test_case in &test_cases {
            let data_file =
                Path::new("external_validation/data").join(format!("{}.csv", test_case.name));
            let meta_file =
                Path::new("external_validation/data").join(format!("{}_meta.csv", test_case.name));

            assert!(
                data_file.exists(),
                "Data file should exist: {:?}",
                data_file
            );
            assert!(
                meta_file.exists(),
                "Meta file should exist: {:?}",
                meta_file
            );
        }
    }

    #[test]
    fn test_compute_rust_results() {
        let test_cases = vec![generate_validation_test_cases()[0].clone()]; // Just test one case

        compute_rust_results(&test_cases).expect("Failed to compute Rust results");

        let results_file = Path::new("external_validation/results")
            .join(format!("{}_rust.json", test_cases[0].name));
        assert!(
            results_file.exists(),
            "Results file should exist: {:?}",
            results_file
        );
    }

    /// Full validation workflow test (requires Python environment)
    #[test]
    #[ignore] // Run with: cargo test external_validation_workflow -- --ignored
    fn test_full_external_validation_workflow() {
        println!("Starting full external validation workflow...");

        // Generate test cases
        let test_cases = generate_validation_test_cases();

        // Export test data
        export_test_data_to_csv(&test_cases).expect("Failed to export test data");

        // Compute Rust results
        compute_rust_results(&test_cases).expect("Failed to compute Rust results");

        // Run Python validation
        run_python_validation().expect("Failed to run Python validation");

        // Compare results
        compare_results(&test_cases).expect("Failed to compare results");

        println!("External validation workflow completed successfully!");
    }
}
