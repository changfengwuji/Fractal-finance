//! Error types and validation functions for fractal analysis.
//!
//! This module provides comprehensive error handling for all fractal analysis operations,
//! including data validation, numerical stability checks, and operation-specific errors.

use std::sync::Arc;
use thiserror::Error;

/// Comprehensive error types for fractal analysis operations.
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum FractalAnalysisError {
    /// Insufficient data for the requested analysis method.
    #[error("Insufficient data: need at least {required} points, got {actual}")]
    InsufficientData {
        /// Minimum required data points
        required: usize,
        /// Actual number of data points provided
        actual: usize,
    },

    /// Invalid parameter value for analysis configuration.
    #[error("Invalid parameter: {parameter} = {value}, expected {constraint}")]
    InvalidParameter {
        /// Parameter name
        parameter: String,
        /// Invalid value provided
        value: f64,
        /// Valid range or constraint description
        constraint: String,
    },

    /// Numerical computation error due to instability or convergence failure.
    #[error("Numerical computation failed: {reason}")]
    NumericalError {
        /// Detailed reason for numerical failure
        reason: String,
        /// Operation that failed (optional for backward compatibility)
        #[cfg_attr(feature = "serde", serde(skip))]
        operation: Option<String>,
    },

    /// Model selection failed.
    #[error("Model selection failed: {reason}")]
    ModelSelectionFailed {
        /// Reason for failure
        reason: String,
    },

    /// Numerical instability detected in computation.
    #[error("Numerical instability: {message}")]
    NumericalInstability {
        /// Description of the numerical instability
        message: String,
    },

    /// FFT computation error for spectral analysis.
    #[error("FFT computation failed: input size {size} not supported")]
    FftError {
        /// Input size that caused the FFT failure
        size: usize,
    },

    /// Statistical test computation failure.
    #[error("Statistical test failed: {test_name} could not be computed")]
    StatisticalTestError {
        /// Name of the statistical test that failed
        test_name: String,
    },

    /// Bootstrap resampling error.
    #[error("Bootstrap resampling failed: {reason}")]
    BootstrapError {
        /// Reason for bootstrap failure
        reason: String,
    },

    /// Regime detection algorithm failure.
    #[error("Regime detection failed: {algorithm} could not identify regimes")]
    RegimeDetectionError {
        /// Algorithm name that failed
        algorithm: String,
    },

    /// Model validation error.
    #[error("Model validation failed: {validation_type} returned invalid results")]
    ValidationError {
        /// Type of validation that failed
        validation_type: String,
    },

    /// Time series not found error.
    #[error("Time series not found: {name}")]
    TimeSeriesNotFound {
        /// Name of the time series that was not found
        name: String,
    },

    /// Concurrent access error.
    #[error("Concurrent access failed: {resource}")]
    ConcurrencyError {
        /// Resource that couldn't be accessed
        resource: String,
    },

    /// I/O operation error.
    #[error("I/O operation failed: {operation}")]
    IoError {
        /// I/O operation that failed
        operation: String,
        /// Underlying error if available
        #[source]
        source: Option<Arc<std::io::Error>>,
    },

    /// Feature not yet implemented.
    #[error("Feature not implemented: {feature}")]
    NotImplemented {
        /// Feature that is not yet implemented
        feature: String,
    },

    /// Serialization/deserialization error.
    #[error("Serialization failed: {format}")]
    SerializationError {
        /// Format that failed (JSON, CBOR, etc)
        format: String,
    },
}

/// Result type for fractal analysis operations.
///
/// This is a convenience type alias for operations that may fail with [`FractalAnalysisError`].
pub type FractalResult<T> = Result<T, FractalAnalysisError>;

/// Validates that data has sufficient length for analysis.
///
/// # Arguments
/// * `data` - Input time series data
/// * `min_required` - Minimum number of data points required
/// * `operation` - Name of the operation requiring the data
///
/// # Returns
/// * `Ok(())` if data length is sufficient
/// * `Err(FractalAnalysisError::InsufficientData)` if data is too short
///
/// # Example
/// ```rust
/// use financial_fractal_analysis::errors::validate_data_length;
///
/// let data = vec![1.0, 2.0, 3.0];
/// assert!(validate_data_length(&data, 2, "test").is_ok());
/// assert!(validate_data_length(&data, 5, "test").is_err());
/// ```
pub fn validate_data_length(
    data: &[f64],
    min_required: usize,
    _operation: &str,
) -> FractalResult<()> {
    // Check for empty data first
    if data.is_empty() && min_required > 0 {
        return Err(FractalAnalysisError::InsufficientData {
            required: min_required,
            actual: 0,
        });
    }

    if data.len() < min_required {
        Err(FractalAnalysisError::InsufficientData {
            required: min_required,
            actual: data.len(),
        })
    } else {
        Ok(())
    }
}

/// Validates that a parameter is within expected bounds.
///
/// # Arguments
/// * `value` - Parameter value to validate
/// * `min` - Minimum acceptable value (inclusive)
/// * `max` - Maximum acceptable value (inclusive)
/// * `name` - Parameter name for error reporting
///
/// # Returns
/// * `Ok(())` if value is within bounds
/// * `Err(FractalAnalysisError::InvalidParameter)` if value is out of bounds
///
/// # Example
/// ```rust
/// use financial_fractal_analysis::errors::validate_parameter;
///
/// assert!(validate_parameter(0.5, 0.0, 1.0, "hurst").is_ok());
/// assert!(validate_parameter(1.5, 0.0, 1.0, "hurst").is_err());
/// ```
pub fn validate_parameter(value: f64, min: f64, max: f64, name: &str) -> FractalResult<()> {
    // First check if any input is NaN
    if value.is_nan() {
        return Err(FractalAnalysisError::InvalidParameter {
            parameter: name.to_string(),
            value,
            constraint: "must not be NaN".to_string(),
        });
    }

    if min.is_nan() || max.is_nan() {
        return Err(FractalAnalysisError::NumericalError {
            reason: format!(
                "Invalid bounds for parameter {}: min={}, max={}",
                name, min, max
            ),
            operation: None,
        });
    }

    // Check if min <= max
    if min > max {
        return Err(FractalAnalysisError::NumericalError {
            reason: format!(
                "Invalid bounds for parameter {}: min ({}) > max ({})",
                name, min, max
            ),
            operation: None,
        });
    }

    if value < min || value > max {
        Err(FractalAnalysisError::InvalidParameter {
            parameter: name.to_string(),
            value,
            constraint: format!("[{}, {}]", min, max),
        })
    } else {
        Ok(())
    }
}

/// Validates that a value is finite and not NaN.
///
/// This is crucial for numerical stability in fractal analysis algorithms.
///
/// # Arguments
/// * `value` - Value to validate
/// * `name` - Variable name for error reporting
///
/// # Returns
/// * `Ok(())` if value is finite
/// * `Err(FractalAnalysisError::NumericalError)` if value is infinite or NaN
///
/// # Example
/// ```rust
/// use financial_fractal_analysis::errors::validate_finite;
///
/// assert!(validate_finite(1.0, "test").is_ok());
/// assert!(validate_finite(f64::NAN, "test").is_err());
/// assert!(validate_finite(f64::INFINITY, "test").is_err());
/// ```
pub fn validate_finite(value: f64, name: &str) -> FractalResult<()> {
    if !value.is_finite() {
        Err(FractalAnalysisError::NumericalError {
            reason: format!("{} is not finite: {}", name, value),
            operation: None,
        })
    } else {
        Ok(())
    }
}

/// Validates that all values in a slice are finite.
///
/// This function checks an entire array for numerical validity, which is
/// essential before performing any fractal analysis operations.
///
/// For performance, this function returns immediately on the first non-finite value.
///
/// # Arguments
/// * `data` - Array of values to validate
/// * `name` - Array name for error reporting
///
/// # Returns
/// * `Ok(())` if all values are finite
/// * `Err(FractalAnalysisError::NumericalError)` if any value is infinite or NaN
///
/// # Example
/// ```rust
/// use financial_fractal_analysis::errors::validate_all_finite;
///
/// let good_data = vec![1.0, 2.0, 3.0];
/// let bad_data = vec![1.0, f64::NAN, 3.0];
///
/// assert!(validate_all_finite(&good_data, "test").is_ok());
/// assert!(validate_all_finite(&bad_data, "test").is_err());
/// ```
pub fn validate_all_finite(data: &[f64], name: &str) -> FractalResult<()> {
    // Early return for empty data
    if data.is_empty() {
        return Ok(());
    }

    // Find first non-finite value
    if let Some((i, &value)) = data.iter().enumerate().find(|(_, &v)| !v.is_finite()) {
        let value_desc = if value.is_nan() {
            "NaN".to_string()
        } else if value.is_infinite() {
            if value.is_sign_positive() {
                "Infinity".to_string()
            } else {
                "-Infinity".to_string()
            }
        } else {
            format!("{}", value)
        };

        return Err(FractalAnalysisError::NumericalError {
            reason: format!(
                "{} contains non-finite value at index {}: {}",
                name, i, value_desc
            ),
            operation: None,
        });
    }

    Ok(())
}

/// CRITICAL SAFETY: Validate memory allocation size to prevent system crashes
///
/// This function prevents allocation requests that could cause out-of-memory errors
/// or system instability. Maximum allocation is set to 1GB for safety.
///
/// # Arguments
/// * `size` - Number of bytes to allocate
/// * `operation` - Name of the operation requesting allocation
///
/// # Returns
/// * `Ok(())` if allocation size is safe
/// * `Err(FractalAnalysisError::NumericalError)` if allocation is too large
///
/// # Example
/// ```rust
/// use financial_fractal_analysis::errors::validate_allocation_size;
///
/// assert!(validate_allocation_size(1000, "test").is_ok());
/// assert!(validate_allocation_size(2_000_000_000, "test").is_err());
/// ```
pub fn validate_allocation_size(size: usize, operation: &str) -> FractalResult<()> {
    // Maximum safe allocation: 1GB = 2^30 bytes
    const MAX_SAFE_ALLOCATION: usize = 1 << 30; // 1,073,741,824 bytes

    if size > MAX_SAFE_ALLOCATION {
        return Err(FractalAnalysisError::NumericalError {
            reason: format!(
                "Attempted allocation of {} bytes ({:.2} GB) in '{}' exceeds safety limit of {} bytes (1.0 GB). This likely indicates a calculation error or integer overflow.",
                size,
                (size as f64).min(f64::MAX) / (1024.0 * 1024.0 * 1024.0),
                operation,
                MAX_SAFE_ALLOCATION
            ),
            operation: None
        });
    }

    // Additional check for suspicious power-of-2 sizes that might indicate overflow
    // Check size > 0 to prevent underflow in (size - 1)
    if size > 0 && size >= (1 << 28) && (size & (size - 1)) == 0 {
        // Log warning only in debug mode to prevent information leakage
        #[cfg(debug_assertions)]
        log::warn!(
            "Large power-of-2 allocation detected: {} bytes ({:.2} GB) in '{}'. This might indicate an integer overflow.",
            size,
            size as f64 / (1024.0 * 1024.0 * 1024.0),
            operation
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_data_length_sufficient() {
        // Test Case 1: Sufficient data length
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = validate_data_length(&data, 3, "test_operation");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_data_length_insufficient() {
        // Test Case 2: Insufficient data length
        let data = vec![1.0, 2.0];
        let result = validate_data_length(&data, 5, "test_operation");

        match result {
            Err(FractalAnalysisError::InsufficientData { required, actual }) => {
                assert_eq!(required, 5);
                assert_eq!(actual, 2);
            }
            _ => panic!("Expected InsufficientData error"),
        }
    }

    #[test]
    fn test_validate_data_length_exact_minimum() {
        // Test Case 3: Exact minimum required
        let data = vec![1.0, 2.0, 3.0];
        let result = validate_data_length(&data, 3, "test_operation");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_parameter_valid_range() {
        // Test Case 1: Valid parameter within range
        let result = validate_parameter(0.5, 0.0, 1.0, "hurst_exponent");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_parameter_out_of_range_high() {
        // Test Case 2: Parameter above maximum
        let result = validate_parameter(1.5, 0.0, 1.0, "hurst_exponent");

        match result {
            Err(FractalAnalysisError::InvalidParameter {
                parameter,
                value,
                constraint,
            }) => {
                assert_eq!(parameter, "hurst_exponent");
                assert_eq!(value, 1.5);
                assert_eq!(constraint, "[0, 1]");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_validate_parameter_out_of_range_low() {
        // Test Case 3: Parameter below minimum
        let result = validate_parameter(-0.5, 0.0, 1.0, "hurst_exponent");

        match result {
            Err(FractalAnalysisError::InvalidParameter {
                parameter,
                value,
                constraint,
            }) => {
                assert_eq!(parameter, "hurst_exponent");
                assert_eq!(value, -0.5);
                assert_eq!(constraint, "[0, 1]");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_validate_parameter_boundary_values() {
        // Test Case 4: Boundary values should be valid
        assert!(validate_parameter(0.0, 0.0, 1.0, "test").is_ok());
        assert!(validate_parameter(1.0, 0.0, 1.0, "test").is_ok());
    }

    #[test]
    fn test_validate_parameter_nan_inputs() {
        // Test NaN value
        let result = validate_parameter(f64::NAN, 0.0, 1.0, "test");
        assert!(matches!(
            result,
            Err(FractalAnalysisError::InvalidParameter { .. })
        ));

        // Test NaN bounds
        let result = validate_parameter(0.5, f64::NAN, 1.0, "test");
        assert!(matches!(result, Err(FractalAnalysisError::NumericalError { operation: None, .. })));

        let result = validate_parameter(0.5, 0.0, f64::NAN, "test");
        assert!(matches!(result, Err(FractalAnalysisError::NumericalError { operation: None, .. })));

        // Test invalid bounds (min > max)
        let result = validate_parameter(0.5, 1.0, 0.0, "test");
        assert!(matches!(result, Err(FractalAnalysisError::NumericalError { operation: None, .. })));
    }

    #[test]
    fn test_validate_finite_valid_values() {
        // Test Case 1: Valid finite values
        assert!(validate_finite(1.0, "test_value").is_ok());
        assert!(validate_finite(-1.0, "test_value").is_ok());
        assert!(validate_finite(0.0, "test_value").is_ok());
        assert!(validate_finite(1e-10, "test_value").is_ok());
        assert!(validate_finite(1e10, "test_value").is_ok());
    }

    #[test]
    fn test_validate_finite_nan() {
        // Test Case 2: NaN should be invalid
        let result = validate_finite(f64::NAN, "test_value");

        match result {
            Err(FractalAnalysisError::NumericalError {
                reason,
                operation: None,
            }) => {
                assert!(reason.contains("test_value"));
                assert!(reason.contains("not finite"));
            }
            _ => panic!("Expected NumericalError for NaN"),
        }
    }

    #[test]
    fn test_validate_finite_infinity() {
        // Test Case 3: Infinity should be invalid
        let result = validate_finite(f64::INFINITY, "test_value");

        match result {
            Err(FractalAnalysisError::NumericalError {
                reason,
                operation: None,
            }) => {
                assert!(reason.contains("test_value"));
                assert!(reason.contains("not finite"));
            }
            _ => panic!("Expected NumericalError for INFINITY"),
        }

        let result = validate_finite(f64::NEG_INFINITY, "test_value");
        assert!(matches!(result, Err(FractalAnalysisError::NumericalError { operation: None, .. })));
    }

    #[test]
    fn test_validate_all_finite_valid_array() {
        // Test Case 1: All finite values
        let good_data = vec![1.0, 2.0, 3.0, -1.0, 0.0, 1e-10, 1e10];
        let result = validate_all_finite(&good_data, "test_array");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_all_finite_empty_array() {
        // Test Case 2: Empty array should be valid
        let empty_data: Vec<f64> = vec![];
        let result = validate_all_finite(&empty_data, "test_array");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_all_finite_with_nan() {
        // Test Case 3: Array containing NaN
        let bad_data = vec![1.0, 2.0, f64::NAN, 4.0];
        let result = validate_all_finite(&bad_data, "test_array");

        match result {
            Err(FractalAnalysisError::NumericalError {
                reason,
                operation: None,
            }) => {
                assert!(reason.contains("test_array"));
                assert!(reason.contains("index 2"));
                assert!(reason.contains("non-finite"));
            }
            _ => panic!("Expected NumericalError for array with NaN"),
        }
    }

    #[test]
    fn test_validate_all_finite_with_infinity() {
        // Test Case 4: Array containing infinity
        let bad_data = vec![1.0, 2.0, 3.0, f64::INFINITY];
        let result = validate_all_finite(&bad_data, "test_array");

        match result {
            Err(FractalAnalysisError::NumericalError {
                reason,
                operation: None,
            }) => {
                assert!(reason.contains("test_array"));
                assert!(reason.contains("index 3"));
                assert!(reason.contains("non-finite"));
            }
            _ => panic!("Expected NumericalError for array with infinity"),
        }
    }

    #[test]
    fn test_validate_all_finite_first_element_bad() {
        // Test Case 5: First element is non-finite
        let bad_data = vec![f64::NAN, 2.0, 3.0];
        let result = validate_all_finite(&bad_data, "test_array");

        match result {
            Err(FractalAnalysisError::NumericalError {
                reason,
                operation: None,
            }) => {
                assert!(reason.contains("index 0"));
            }
            _ => panic!("Expected NumericalError for first element NaN"),
        }
    }

    #[test]
    fn test_validate_all_finite_last_element_bad() {
        // Test Case 6: Last element is non-finite
        let bad_data = vec![1.0, 2.0, f64::NEG_INFINITY];
        let result = validate_all_finite(&bad_data, "test_array");

        match result {
            Err(FractalAnalysisError::NumericalError {
                reason,
                operation: None,
            }) => {
                assert!(reason.contains("index 2"));
            }
            _ => panic!("Expected NumericalError for last element -infinity"),
        }
    }

    #[test]
    fn test_error_display_formatting() {
        // Test that error messages are properly formatted
        let insufficient_data_error = FractalAnalysisError::InsufficientData {
            required: 100,
            actual: 50,
        };
        let error_string = format!("{}", insufficient_data_error);
        assert!(error_string.contains("Insufficient data"));
        assert!(error_string.contains("100"));
        assert!(error_string.contains("50"));

        let invalid_param_error = FractalAnalysisError::InvalidParameter {
            parameter: "hurst".to_string(),
            value: 1.5,
            constraint: "[0, 1]".to_string(),
        };
        let error_string = format!("{}", invalid_param_error);
        assert!(error_string.contains("Invalid parameter"));
        assert!(error_string.contains("hurst"));
        assert!(error_string.contains("1.5"));
        assert!(error_string.contains("[0, 1]"));

        let numerical_error = FractalAnalysisError::NumericalError {
            reason: "Matrix singular".to_string(),
            operation: None,
        };
        let error_string = format!("{}", numerical_error);
        assert!(error_string.contains("Numerical computation failed"));
        assert!(error_string.contains("Matrix singular"));
    }

    #[test]
    fn test_fractal_result_type_alias() {
        // Test that FractalResult type alias works correctly
        fn example_function() -> FractalResult<f64> {
            Ok(1.0)
        }

        fn failing_function() -> FractalResult<f64> {
            Err(FractalAnalysisError::NumericalError {
                reason: "Test error".to_string(),
                operation: None,
            })
        }

        assert!(example_function().is_ok());
        assert!(failing_function().is_err());

        // Test unwrap and pattern matching
        let success_result = example_function().unwrap();
        assert_eq!(success_result, 1.0);

        let failure_result = failing_function();
        match failure_result {
            Err(FractalAnalysisError::NumericalError {
                reason,
                operation: None,
            }) => {
                assert_eq!(reason, "Test error");
            }
            _ => panic!("Expected NumericalError"),
        }
    }

    #[test]
    fn test_comprehensive_error_scenarios() {
        // Test realistic error scenarios that might occur in fractal analysis

        // Scenario 1: DFA requires minimum 100 points
        let short_series = vec![1.0; 50];
        let result = validate_data_length(&short_series, 100, "Detrended Fluctuation Analysis");
        assert!(matches!(
            result,
            Err(FractalAnalysisError::InsufficientData { .. })
        ));

        // Scenario 2: Hurst exponent must be in (0, 1)
        let result = validate_parameter(0.0, 0.0, 1.0, "hurst_exponent");
        assert!(result.is_ok()); // Boundary case

        let result = validate_parameter(-0.1, 0.0, 1.0, "hurst_exponent");
        assert!(matches!(
            result,
            Err(FractalAnalysisError::InvalidParameter { .. })
        ));

        // Scenario 3: Financial data with corrupt values
        let corrupt_data = vec![1.0, 2.0, f64::NAN, 4.0, f64::INFINITY];
        let result = validate_all_finite(&corrupt_data, "stock_returns");
        assert!(matches!(result, Err(FractalAnalysisError::NumericalError { operation: None, .. })));
    }
}
