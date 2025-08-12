//! Emission models and parameters for Hidden Markov Model regime detection.
//!
//! This module contains the emission parameter structures and observation features
//! used in HMM-based fractal regime detection for financial time series.

use crate::{
    errors::{FractalAnalysisError, FractalResult},
    math_utils::constants,
};
use nalgebra::{Cholesky, Matrix4, Vector4};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Emission parameters for each regime state
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EmissionParameters {
    /// Mean vector [hurst, volatility, autocorr, multifractality]
    pub mean_vector: [f64; 4],
    /// Covariance matrix (4x4) for multivariate Gaussian
    pub covariance_matrix: [[f64; 4]; 4],
    /// Inverse covariance matrix (cached for efficiency)
    pub precision_matrix: [[f64; 4]; 4],
    /// Log determinant of covariance matrix (cached)
    pub log_det_cov: f64,
    /// Regime-specific multifractal parameters
    pub multifractal_params: MultifractalRegimeParams,
}

/// Multifractal parameters specific to a regime
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MultifractalRegimeParams {
    /// Width of multifractal spectrum
    pub spectrum_width: f64,
    /// Asymmetry of spectrum
    pub spectrum_asymmetry: f64,
    /// Holder exponent at maximum
    pub alpha_max: f64,
    /// Maximum value of f(alpha)
    pub f_alpha_max: f64,
}

/// Feature vector for HMM observations
#[derive(Debug, Clone)]
pub struct ObservationFeatures {
    /// Local Hurst exponent estimate
    pub local_hurst: f64,
    /// Local volatility estimate
    pub local_volatility: f64,
    /// Local autocorrelation
    pub local_autocorr: f64,
    /// Local multifractality measure
    pub local_multifractality: f64,
    /// Time window index
    pub time_index: usize,
}

impl Default for EmissionParameters {
    fn default() -> Self {
        let mean_vector = [0.5, 0.01, 0.1, 0.05]; // hurst, vol, autocorr, multifractality
        let mut covariance_matrix = [[0.0; 4]; 4];

        // Initialize with diagonal covariance (independence assumption as starting point)
        // CRITICAL FIX: Increased variances to prevent singularity after Hosking fix
        covariance_matrix[0][0] = 0.05; // hurst variance (increased from 0.04)
        covariance_matrix[1][1] = 0.01; // volatility variance (increased from 0.0001)
        covariance_matrix[2][2] = 0.02; // autocorr variance (increased from 0.01)
        covariance_matrix[3][3] = 0.01; // multifractality variance (increased from 0.0025)

        let precision_matrix = covariance_matrix; // Will be properly computed later
        let log_det_cov = 0.0; // Will be computed later

        Self {
            mean_vector,
            covariance_matrix,
            precision_matrix,
            log_det_cov,
            multifractal_params: MultifractalRegimeParams::default(),
        }
    }
}

impl EmissionParameters {
    /// Update precision matrix and log determinant from covariance matrix
    pub fn update_cached_values(&mut self) -> FractalResult<()> {
        // CRITICAL FIX: Add regularization to prevent singularity
        const REGULARIZATION_STRENGTH: f64 = 1e-6;

        // Apply regularization to diagonal elements
        for i in 0..4 {
            self.covariance_matrix[i][i] += REGULARIZATION_STRENGTH;
        }

        // Convert to nalgebra matrix
        let cov_matrix = Matrix4::from_row_slice(&[
            self.covariance_matrix[0][0],
            self.covariance_matrix[0][1],
            self.covariance_matrix[0][2],
            self.covariance_matrix[0][3],
            self.covariance_matrix[1][0],
            self.covariance_matrix[1][1],
            self.covariance_matrix[1][2],
            self.covariance_matrix[1][3],
            self.covariance_matrix[2][0],
            self.covariance_matrix[2][1],
            self.covariance_matrix[2][2],
            self.covariance_matrix[2][3],
            self.covariance_matrix[3][0],
            self.covariance_matrix[3][1],
            self.covariance_matrix[3][2],
            self.covariance_matrix[3][3],
        ]);

        // Comprehensive matrix conditioning checks
        let det = cov_matrix.determinant();

        // Check for near-singular matrix with more tolerance after regularization
        if det.abs() < constants::MATRIX_CONDITION_EPSILON * 1e-3 {
            return Err(FractalAnalysisError::NumericalError {
                reason: format!(
                    "Covariance matrix is near-singular after regularization (det = {:.2e})",
                    det
                ),
                operation: None,
            });
        }

        // Check condition number via eigenvalues for better numerical assessment
        let eigenvalues = match cov_matrix.eigenvalues() {
            Some(eigs) => eigs,
            None => {
                return Err(FractalAnalysisError::NumericalError {
                    reason: "Failed to compute eigenvalues of covariance matrix".to_string(),
                    operation: None,
                });
            }
        };
        if eigenvalues.len() == 4 {
            let min_eigenval = eigenvalues.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_eigenval = eigenvalues.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            // Check for positive definiteness
            if min_eigenval <= 0.0 {
                return Err(FractalAnalysisError::NumericalError {
                    reason: format!(
                        "Matrix is not positive definite (min eigenvalue: {:.2e})",
                        min_eigenval
                    ),
                    operation: None,
                });
            }

            // Check condition number
            let condition_number = max_eigenval / min_eigenval;
            if condition_number > 1e12 {
                return Err(FractalAnalysisError::NumericalError {
                    reason: format!(
                        "Matrix is ill-conditioned (condition number: {:.2e})",
                        condition_number
                    ),
                    operation: None,
                });
            }
        }

        // Attempt Cholesky decomposition with progressive regularization
        let cholesky = match Cholesky::new(cov_matrix) {
            Some(chol) => chol,
            None => {
                // Try progressive regularization levels
                let regularization_levels = [
                    constants::MATRIX_REGULARIZATION,
                    constants::MATRIX_REGULARIZATION * 10.0,
                    constants::MATRIX_REGULARIZATION * 100.0,
                    1e-6,
                    1e-4,
                ];

                let mut successful_chol = None;
                for &reg_level in &regularization_levels {
                    let regularized = cov_matrix + Matrix4::identity() * reg_level;
                    if let Some(chol) = Cholesky::new(regularized) {
                        successful_chol = Some(chol);
                        break;
                    }
                }

                successful_chol.ok_or_else(|| FractalAnalysisError::NumericalError {
                    reason:
                        "Cannot compute Cholesky decomposition even with progressive regularization"
                            .to_string(),
                    operation: None,
                })?
            }
        };

        // Compute inverse (precision matrix)
        let precision = cholesky.inverse();

        // Copy back to array format
        for i in 0..4 {
            for j in 0..4 {
                self.precision_matrix[i][j] = precision[(i, j)];
            }
        }

        // Compute log determinant efficiently using Cholesky decomposition
        self.log_det_cov = 2.0 * cholesky.l().diagonal().iter().map(|x| x.ln()).sum::<f64>();

        Ok(())
    }

    /// Extract feature vector from observation
    pub fn extract_features(&self, obs: &ObservationFeatures) -> Vector4<f64> {
        Vector4::new(
            obs.local_hurst,
            obs.local_volatility,
            obs.local_autocorr,
            obs.local_multifractality,
        )
    }
}

impl Default for MultifractalRegimeParams {
    fn default() -> Self {
        Self {
            spectrum_width: 0.1,
            spectrum_asymmetry: 0.0,
            alpha_max: 0.5, // Must be in range (0,1) for financial time series
            f_alpha_max: 1.0,
        }
    }
}