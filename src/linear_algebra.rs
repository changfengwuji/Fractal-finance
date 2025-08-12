//! Linear algebra operations for financial fractal analysis
//!
//! This module provides numerical linear algebra routines optimized for
//! the statistical computations required in fractal analysis, including
//! QR decomposition, least squares solving, and regression operations.

use crate::errors::{FractalAnalysisError, FractalResult};

/// Validates that input contains no NaN or Inf values
fn ensure_finite_matrix(a: &[Vec<f64>], operation: &str) -> FractalResult<()> {
    for (i, row) in a.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if !val.is_finite() {
                return Err(FractalAnalysisError::NumericalError {
                    reason: format!("Non-finite value ({}) at position [{},{}]", val, i, j),
                    operation: Some(operation.to_string()),
                });
            }
        }
    }
    Ok(())
}

/// Validates that a vector contains no NaN or Inf values
fn ensure_finite_vector(v: &[f64], operation: &str) -> FractalResult<()> {
    for (i, &val) in v.iter().enumerate() {
        if !val.is_finite() {
            return Err(FractalAnalysisError::NumericalError {
                reason: format!("Non-finite value ({}) at position [{}]", val, i),
                operation: Some(operation.to_string()),
            });
        }
    }
    Ok(())
}

/// Validates that a matrix is rectangular (not ragged) and non-empty
fn ensure_rectangular_matrix(a: &[Vec<f64>]) -> FractalResult<(usize, usize)> {
    if a.is_empty() {
        return Err(FractalAnalysisError::NumericalError {
            reason: "Empty matrix provided".to_string(),
            operation: Some("matrix_validation".to_string()),
        });
    }
    
    let n = a[0].len();
    if n == 0 {
        return Err(FractalAnalysisError::NumericalError {
            reason: "Zero-width matrix (no columns)".to_string(),
            operation: Some("matrix_validation".to_string()),
        });
    }
    
    if !a.iter().all(|row| row.len() == n) {
        return Err(FractalAnalysisError::NumericalError {
            reason: "Ragged matrix (inconsistent row lengths)".to_string(),
            operation: Some("matrix_validation".to_string()),
        });
    }
    
    Ok((a.len(), n))
}

/// QR decomposition for regression residuals
///
/// More efficient and numerically stable than forming normal equations.
/// 
/// # Arguments
/// * `y` - Response vector of length n
/// * `x` - Predictors as k vectors, each of length n (predictor-by-observation format)
///         x[predictor_idx][observation_idx]
/// 
/// Internally transposes to n×k matrix for QR decomposition.
/// 
/// Returns: (residuals, coefficients)
pub fn qr_regression_residuals(
    y: &[f64],
    x: &[Vec<f64>],
) -> FractalResult<(Vec<f64>, Vec<f64>)> {
    let n = y.len();
    let k = x.len();
    
    // Validate inputs
    ensure_finite_vector(y, "qr_regression_residuals")?;
    
    // Validate that all predictor columns have the same length and are finite
    for (i, col) in x.iter().enumerate() {
        if col.len() != n {
            return Err(FractalAnalysisError::NumericalError {
                reason: format!("Predictor column {} has length {} but expected {}", i, col.len(), n),
                operation: Some("qr_regression_residuals".to_string()),
            });
        }
        ensure_finite_vector(col, "qr_regression_residuals")?;
    }

    // Create X matrix (n x k)
    let mut x_mat = vec![vec![0.0; k]; n];
    for i in 0..n {
        for j in 0..k {
            x_mat[i][j] = x[j][i];
        }
    }

    // Use economy QR solve for efficiency
    let beta = economy_qr_solve(&x_mat, y)?;

    // Compute residuals
    let mut residuals = Vec::with_capacity(n);
    for i in 0..n {
        let mut fitted = 0.0;
        for j in 0..k {
            fitted += x_mat[i][j] * beta[j];
        }
        residuals.push(y[i] - fitted);
    }

    Ok((residuals, beta))
}

/// Householder QR decomposition with adaptive rank deficiency detection
///
/// Uses machine-epsilon scaled tolerance that adapts to matrix norm
/// for better numerical stability across different data scales.
///
/// Returns (Q, R) matrices for backward compatibility.
/// Use economy_qr_solve for more efficient solving without forming Q.
pub fn householder_qr(a: &[Vec<f64>]) -> FractalResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    // Validate input matrix
    let (m, n) = ensure_rectangular_matrix(a)?;
    ensure_finite_matrix(a, "householder_qr")?;

    // Compute Frobenius norm of A for adaptive tolerance
    let mut matrix_norm = 0.0;
    for row in a {
        for &val in row {
            matrix_norm += val * val;
        }
    }
    matrix_norm = matrix_norm.sqrt();

    // Adaptive tolerance: eps * max(m,n) * ||A||_F * factor
    // This scales with both matrix size and magnitude
    // Using factor of 10 for more practical rank detection
    let rank_tol = 10.0 * f64::EPSILON * (m.max(n) as f64) * matrix_norm;

    let mut r = a.to_vec();
    let mut q = vec![vec![0.0; m]; m];
    for i in 0..m {
        q[i][i] = 1.0;
    }

    // Process up to min(n, m-1) columns (last row doesn't need a reflector)
    let steps = n.min(m.saturating_sub(1));
    for k in 0..steps {
        // Compute Householder vector
        let mut x = vec![0.0; m - k];
        for i in k..m {
            x[i - k] = r[i][k];
        }

        let mut norm_x = 0.0;
        for &xi in &x {
            norm_x += xi * xi;
        }
        norm_x = norm_x.sqrt();

        // Use adaptive tolerance for rank deficiency detection
        if norm_x < rank_tol {
            // Mark this column as rank-deficient by zeroing it out
            for i in k..m {
                r[i][k] = 0.0;
            }
            continue;
        }

        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        x[0] += sign * norm_x;

        // Normalize
        let mut norm_v = 0.0;
        for &xi in &x {
            norm_v += xi * xi;
        }
        norm_v = norm_v.sqrt();

        // Use same adaptive tolerance for consistency
        if norm_v < rank_tol {
            continue;
        }

        for xi in &mut x {
            *xi /= norm_v;
        }

        // Apply Householder transformation to R
        for j in k..n {
            let mut dot = 0.0;
            for i in k..m {
                dot += x[i - k] * r[i][j];
            }
            for i in k..m {
                r[i][j] -= 2.0 * x[i - k] * dot;
            }
        }

        // Apply to Q (right-multiply: Q = Q * H_k)
        // Note: Right-multiplication works because Householder reflectors are symmetric (H = H^T)
        // and we apply them to R on the left in the same sequence, giving us A = Q*R
        for j in 0..m {  // row j of Q
            let mut dot = 0.0;
            for i in k..m {
                dot += q[j][i] * x[i - k];
            }
            for i in k..m {
                q[j][i] -= 2.0 * dot * x[i - k];
            }
        }
    }

    Ok((q, r))
}

/// QR decomposition with column pivoting for improved numerical stability
/// 
/// Returns (Q, R, P, rank) where:
/// - Q is orthogonal (m x m)
/// - R is upper triangular/trapezoidal (m x n)
/// - P is the column permutation (n-element vector)
/// - rank is the numerical rank of the matrix
/// 
/// Such that A*P = Q*R, where P is the permutation matrix derived from the P vector.
/// This handles rank-deficient matrices better than standard QR.
pub fn qr_with_pivoting(a: &[Vec<f64>]) -> FractalResult<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<usize>, usize)> {
    // Validate input
    let (m, n) = ensure_rectangular_matrix(a)?;
    ensure_finite_matrix(a, "qr_with_pivoting")?;
    
    // Initialize R as copy of A, Q as identity
    let mut r = a.to_vec();
    let mut q = vec![vec![0.0; m]; m];
    for i in 0..m {
        q[i][i] = 1.0;
    }
    
    // Initialize permutation as identity
    let mut perm: Vec<usize> = (0..n).collect();
    
    // Compute Frobenius norm of matrix for global rank tolerance
    let mut frobenius_norm = 0.0;
    for row in &r {
        for &val in row {
            frobenius_norm += val * val;
        }
    }
    frobenius_norm = frobenius_norm.sqrt();
    let rank_tol = 10.0 * f64::EPSILON * (m.max(n) as f64) * frobenius_norm;
    
    // Compute column norms for pivoting (store actual norms, not squared)
    let mut col_norms = vec![0.0; n];
    for j in 0..n {
        let mut norm_sq = 0.0;
        for i in 0..m {
            norm_sq += r[i][j] * r[i][j];
        }
        col_norms[j] = norm_sq.sqrt();
    }
    
    let steps = n.min(m);
    let mut rank = 0;
    
    for k in 0..steps {
        // Find pivot column (largest remaining column norm)
        let mut max_norm = 0.0;
        let mut pivot_col = k;
        
        for j in k..n {
            if col_norms[j] > max_norm {
                max_norm = col_norms[j];
                pivot_col = j;
            }
        }
        
        // Check for rank deficiency using global tolerance
        if max_norm < rank_tol {
            // Matrix is rank-deficient at column k
            break;
        }
        rank += 1;
        
        // Swap columns if needed
        if pivot_col != k {
            // Swap columns in R
            for i in 0..m {
                let temp = r[i][k];
                r[i][k] = r[i][pivot_col];
                r[i][pivot_col] = temp;
            }
            // Swap column norms
            col_norms.swap(k, pivot_col);
            // Update permutation
            perm.swap(k, pivot_col);
        }
        
        // Compute Householder vector for column k
        let mut v = vec![0.0; m - k];
        for i in k..m {
            v[i - k] = r[i][k];
        }
        
        // Compute norm directly from the active subvector (not cached)
        // This ensures correctness even after column swaps
        let mut norm_v = 0.0;
        for &vi in &v {
            norm_v += vi * vi;
        }
        norm_v = norm_v.sqrt();
        
        if norm_v < rank_tol {
            continue;
        }
        
        let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
        v[0] += sign * norm_v;
        
        // Normalize Householder vector
        let mut norm_v2 = 0.0;
        for &vi in &v {
            norm_v2 += vi * vi;
        }
        norm_v2 = norm_v2.sqrt();
        
        if norm_v2 < rank_tol {
            continue;
        }
        
        for vi in &mut v {
            *vi /= norm_v2;
        }
        
        // Apply Householder transformation to R
        for j in k..n {
            let mut dot = 0.0;
            for i in k..m {
                dot += v[i - k] * r[i][j];
            }
            for i in k..m {
                r[i][j] -= 2.0 * v[i - k] * dot;
            }
        }
        
        // Update column norms for remaining columns
        // Note: We intentionally recompute from scratch rather than using downdating
        // to prevent loss of orthogonality from accumulated rounding errors
        if k + 1 < n {
            for j in k + 1..n {
                let mut norm_sq = 0.0;
                for i in k + 1..m {
                    norm_sq += r[i][j] * r[i][j];
                }
                col_norms[j] = norm_sq.sqrt();
            }
        }
        
        // Apply to Q (right-multiply: Q = Q * H_k)
        // Note: Right-multiplication works because Householder reflectors are symmetric (H = H^T)
        // and we apply them to R on the left in the same sequence, giving us A = Q*R
        for j in 0..m {  // row j of Q
            let mut dot = 0.0;
            for i in k..m {
                dot += q[j][i] * v[i - k];
            }
            for i in k..m {
                q[j][i] -= 2.0 * dot * v[i - k];
            }
        }
    }
    
    Ok((q, r, perm, rank))
}

/// Economy QR solve: Solve Ax = b using QR decomposition without forming Q explicitly
///
/// More efficient than householder_qr for solving linear systems as it avoids
/// materializing the full Q matrix. Uses Householder reflectors directly.
///
/// Note: This function rejects underdetermined systems (n > m). For such systems,
/// consider using QR with column pivoting for a minimum-norm least squares solution.
pub fn economy_qr_solve(a: &[Vec<f64>], b: &[f64]) -> FractalResult<Vec<f64>> {
    // Validate input matrix
    let (m, n) = ensure_rectangular_matrix(a)?;
    ensure_finite_matrix(a, "economy_qr_solve")?;
    ensure_finite_vector(b, "economy_qr_solve")?;
    
    if m != b.len() {
        return Err(FractalAnalysisError::NumericalError {
            reason: "Matrix-vector dimension mismatch in QR solve".to_string(),
            operation: Some("economy_qr_solve".to_string()),
        });
    }
    
    // Check for underdetermined system
    if n > m {
        return Err(FractalAnalysisError::NumericalError {
            reason: "Underdetermined system (more columns than rows)".to_string(),
            operation: Some("economy_qr_solve".to_string()),
        });
    }

    // Copy A to R and b to y for in-place operations
    let mut r = a.to_vec();
    let mut y = b.to_vec();

    // Compute Frobenius norm for adaptive tolerance
    let mut matrix_norm = 0.0;
    for row in &r {
        for &val in row {
            matrix_norm += val * val;
        }
    }
    matrix_norm = matrix_norm.sqrt();
    // Use more conservative rank tolerance to avoid numerical issues
    // Standard formula: tol = max(m,n) * eps * ||A||_F
    // We use a larger safety factor for robustness
    let rank_tol = 100.0 * f64::EPSILON * (m.max(n) as f64) * matrix_norm.max(1.0);

    // Perform QR factorization
    // Process up to min(n, m-1) columns (last row doesn't need a reflector)
    let steps = n.min(m.saturating_sub(1));
    for k in 0..steps {
        // Compute Householder vector for column k
        let mut v = vec![0.0; m - k];
        for i in k..m {
            v[i - k] = r[i][k];
        }

        let mut norm_v = 0.0;
        for &vi in &v {
            norm_v += vi * vi;
        }
        norm_v = norm_v.sqrt();

        if norm_v < rank_tol {
            // Rank deficient column
            for i in k..m {
                r[i][k] = 0.0;
            }
            continue;
        }

        let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
        v[0] += sign * norm_v;

        // Normalize
        let mut norm_v2 = 0.0;
        for &vi in &v {
            norm_v2 += vi * vi;
        }
        norm_v2 = norm_v2.sqrt();

        if norm_v2 < rank_tol {
            continue;
        }

        for vi in &mut v {
            *vi /= norm_v2;
        }

        // Apply Householder transformation to R
        for j in k..n {
            let mut dot = 0.0;
            for i in k..m {
                dot += v[i - k] * r[i][j];
            }
            for i in k..m {
                r[i][j] -= 2.0 * v[i - k] * dot;
            }
        }

        // Apply same transformation to y (Q'b)
        let mut dot_y = 0.0;
        for i in k..m {
            dot_y += v[i - k] * y[i];
        }
        for i in k..m {
            y[i] -= 2.0 * v[i - k] * dot_y;
        }
    }

    // Back-substitution to solve Rx = Q'b
    let mut x = vec![0.0; n];
    for i in (0..n.min(m)).rev() {
        // Use consistent tolerance for singularity check
        if r[i][i].abs() < rank_tol {
            // Matrix is numerically singular
            // Try to return a least-squares solution by setting x[i] = 0
            x[i] = 0.0;
            continue;
        }

        let mut sum = y[i];
        for j in i + 1..n {
            sum -= r[i][j] * x[j];
        }
        x[i] = sum / r[i][i];
    }

    Ok(x)
}

// Note: construct_q_from_reflectors removed as it was not connected to any producer.
// If needed in future, implement alongside a proper reflector storage mechanism.

/// Multiple regression using QR decomposition for numerical stability
/// 
/// # Arguments
/// * `x` - Predictors as k vectors, each of length n (predictor-by-observation format)
///         x[predictor_idx][observation_idx]
/// * `y` - Response vector of length n
/// 
/// Internally transposes to n×k matrix for QR decomposition.
pub fn multiple_regression(x: &[Vec<f64>], y: &[f64]) -> FractalResult<Vec<f64>> {
    let k = x.len(); // number of predictors including intercept
    let n = y.len();

    if n < k {
        return Err(FractalAnalysisError::InsufficientData {
            required: k,
            actual: n,
        });
    }
    
    // Validate inputs
    ensure_finite_vector(y, "multiple_regression")?;
    
    // Validate that all predictor columns have the same length and are finite
    for (i, col) in x.iter().enumerate() {
        if col.len() != n {
            return Err(FractalAnalysisError::NumericalError {
                reason: format!("Predictor column {} has inconsistent length", i),
                operation: Some("multiple_regression".to_string()),
            });
        }
        ensure_finite_vector(col, "multiple_regression")?;
    }

    // Transpose X to get n x k matrix for QR decomposition
    let mut x_mat = vec![vec![0.0; k]; n];
    for i in 0..n {
        for j in 0..k {
            x_mat[i][j] = x[j][i];
        }
    }

    // Use QR decomposition to solve the least squares problem
    qr_least_squares(&x_mat, y)
}

/// Solve overdetermined least squares problem using QR decomposition
/// 
/// Returns the coefficient vector for the overdetermined system (m ≥ n).
/// For underdetermined systems (n > m), returns an error.
/// 
/// # Arguments
/// * `x` - Design matrix (m rows × n columns), row-major format: x[row][col]
/// * `y` - Response vector (m observations)
pub fn qr_least_squares(x: &[Vec<f64>], y: &[f64]) -> FractalResult<Vec<f64>> {
    // Use economy QR solve for efficiency
    economy_qr_solve(x, y)
}

/// Solve least squares problem using QR with column pivoting
/// 
/// Handles rank-deficient systems by using column pivoting to identify
/// the most linearly independent columns. Returns the minimum-norm solution
/// for rank-deficient cases.
/// 
/// # Arguments
/// * `x` - Design matrix (m rows × n columns), row-major format: x[row][col]
/// * `y` - Response vector (m observations)
/// 
/// Returns: (coefficients, rank, permutation)
/// - coefficients: Solution vector (permuted order)
/// - rank: Numerical rank of the matrix
/// - permutation: Column permutation applied
pub fn qr_least_squares_pivoted(x: &[Vec<f64>], y: &[f64]) -> FractalResult<(Vec<f64>, usize, Vec<usize>)> {
    // Validate input
    let (m, n) = ensure_rectangular_matrix(x)?;
    ensure_finite_matrix(x, "qr_least_squares_pivoted")?;
    ensure_finite_vector(y, "qr_least_squares_pivoted")?;
    
    if m != y.len() {
        return Err(FractalAnalysisError::NumericalError {
            reason: "Matrix-vector dimension mismatch".to_string(),
            operation: Some("qr_least_squares_pivoted".to_string()),
        });
    }
    
    // Get QR decomposition with pivoting
    let (q, r, perm, rank) = qr_with_pivoting(x)?;
    
    // Compute Q'*y
    let mut qty = vec![0.0; m];
    for i in 0..m {
        let mut sum = 0.0;
        for j in 0..m {
            sum += q[j][i] * y[j];  // Q is stored row-major, so Q' is q[j][i]
        }
        qty[i] = sum;
    }
    
    // Back-substitution for the first 'rank' equations
    let mut x_perm = vec![0.0; n];
    
    // Solve the upper triangular system for the first 'rank' variables
    for i in (0..rank.min(n)).rev() {
        let mut sum = qty[i];
        for j in i + 1..rank.min(n) {
            sum -= r[i][j] * x_perm[j];
        }
        if r[i][i].abs() > f64::EPSILON {
            x_perm[i] = sum / r[i][i];
        } else {
            x_perm[i] = 0.0;
        }
    }
    
    // Set remaining coefficients to zero (rank-deficient case)
    for i in rank..n {
        x_perm[i] = 0.0;
    }
    
    Ok((x_perm, rank, perm))
}

/// Compute residuals from regression
/// 
/// # Arguments
/// * `x` - Predictors in row-major format (k predictors × n observations)
/// * `y` - Response vector (n observations)
/// * `coeffs` - Regression coefficients (k values)
/// 
/// # Panics
/// Panics in debug mode if dimensions don't match. In release mode, may produce
/// incorrect results if dimensions are mismatched.
pub fn compute_residuals(x: &[Vec<f64>], y: &[f64], coeffs: &[f64]) -> Vec<f64> {
    let n = y.len();
    
    // Debug assertions to catch dimension mismatches
    debug_assert_eq!(coeffs.len(), x.len(), "coefficients length must match number of predictors");
    debug_assert!(x.iter().all(|col| col.len() == n), "all predictor columns must have same length as y");
    
    let mut residuals = Vec::with_capacity(n);

    for t in 0..n {
        let mut fitted = 0.0;
        for (i, coeff) in coeffs.iter().enumerate() {
            fitted += coeff * x[i][t];
        }
        residuals.push(y[t] - fitted);
    }

    residuals
}

/// Newey-West automatic bandwidth selection
pub fn newey_west_bandwidth(residuals: &[f64]) -> usize {
    let n = residuals.len();
    if n <= 2 {
        return 1;
    }

    // Newey-West (1994) automatic bandwidth selection
    // Using formula: floor(4 * (n/100)^(2/9))
    let raw = (4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize;

    // Ensure reasonable bounds: at least 1, at most n-1
    let upper = (n / 4).max(1).min(n.saturating_sub(1));
    raw.clamp(1, upper)
}

/// Newey-West long-run variance estimator
///
/// Uses Bartlett kernel weights with automatic bandwidth selection.
/// 
/// Implementation notes:
/// - γ₀ uses (n-1) denominator for unbiased variance
/// - γₖ uses (n-k) denominator for small-sample bias correction  
/// - Maximum lag is capped at min(max_lag, n/4) for conservative bandwidth
/// - Applies pragmatic positive floor to prevent numerical issues in KPSS-type statistics
/// 
/// WARNING: The positive floor (max of 1e-12*variance or ε) will bias
/// the LRV upward for series with genuinely tiny long-run variance. This is intentional
/// to prevent numerical instability in test statistics, but may affect power for
/// detecting stationarity in near-constant series.
pub fn newey_west_lrv(residuals: &[f64], max_lag: usize) -> FractalResult<f64> {
    let n = residuals.len();
    
    // Guard against small samples
    if n < 2 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 2,
            actual: n,
        });
    }

    // Demean residuals first for proper autocovariance computation
    let mean = residuals.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = residuals.iter().map(|&r| r - mean).collect();

    // Variance at lag 0 (using n-1 for unbiased estimate)
    let variance = centered.iter().map(|&x| x * x).sum::<f64>() / (n - 1) as f64;
    let mut lrv = variance;

    // L = maximum lag for Bartlett kernel, capped at n/4 as per documentation
    let L = max_lag.min(n / 4).max(1);

    // Add weighted autocovariances with bias correction
    for k in 1..=L {
        // Bartlett kernel weight: 1 - k/(L+1) where L is max lag
        let weight = 1.0 - (k as f64) / ((L + 1) as f64);

        // Compute autocovariance at lag k with (n-k) denominator for small-sample bias correction
        let mut autocov = 0.0;
        for i in k..n {
            autocov += centered[i] * centered[i - k];
        }
        // Use (n-k) denominator for unbiased estimate in small samples
        autocov /= (n - k) as f64;

        lrv += 2.0 * weight * autocov;
    }

    // Apply pragmatic positive floor to prevent numerical issues
    // For near-white residuals, the LRV can be extremely small which causes
    // KPSS statistic to blow up. Use a simple floor based on variance.
    // WARNING: This biases LRV upward for near-constant series
    let floor = (variance * 1e-12).max(f64::EPSILON);
    Ok(lrv.max(floor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qr_decomposition() {
        let a = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        
        let result = householder_qr(&a);
        assert!(result.is_ok());
        
        let (q, r) = result.unwrap();
        
        // Check Q is orthogonal (Q'Q = I)
        for i in 0..q.len() {
            for j in 0..q.len() {
                let mut dot = 0.0;
                for k in 0..q.len() {
                    dot += q[k][i] * q[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                let tol = 100.0 * f64::EPSILON * (q.len() as f64).sqrt(); // scale by matrix dimension
                assert!((dot - expected).abs() < tol,
                    "Q'Q[{},{}] = {} but expected {}", i, j, dot, expected);
            }
        }
        
        // Check R is upper triangular
        for i in 1..r.len() {
            for j in 0..i.min(r[0].len()) {
                // Use tolerance based on matrix scale, not just r[0][0]
                let r_norm = r.iter().flat_map(|row| row.iter()).map(|&x| x*x).sum::<f64>().sqrt();
                let tol = 100.0 * f64::EPSILON * r_norm;
                assert!(r[i][j].abs() < tol,
                    "R[{},{}] = {} should be zero (tol={})", i, j, r[i][j], tol);
            }
        }
        
        // CRITICAL: Check A ≈ Q*R (reconstruction)
        for i in 0..a.len() {
            for j in 0..a[0].len() {
                let mut acc = 0.0;
                for t in 0..q.len() {
                    acc += q[i][t] * r[t][j];
                }
                // Compute matrix norm for scale-aware tolerance
                let a_norm = a.iter().flat_map(|row| row.iter()).map(|&x| x*x).sum::<f64>().sqrt();
                let tol = 100.0 * f64::EPSILON * a_norm;
                assert!((acc - a[i][j]).abs() < tol,
                    "QR[{},{}] = {} but A[{},{}] = {} (tol={})", i, j, acc, i, j, a[i][j], tol);
            }
        }
    }

    #[test]
    fn test_economy_qr_solve() {
        let a = vec![
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![1.0, 3.0],
        ];
        let b = vec![1.0, 2.0, 2.0];
        
        let result = economy_qr_solve(&a, &b);
        assert!(result.is_ok());
        
        let x = result.unwrap();
        assert_eq!(x.len(), 2);
        
        // Verify solution approximately satisfies Ax = b
        for i in 0..3 {
            let computed = a[i][0] * x[0] + a[i][1] * x[1];
            // Note: least squares solution, not exact
            assert!((computed - b[i]).abs() < 0.5);
        }
    }

    #[test]
    fn test_multiple_regression() {
        // Simple linear regression y = 2 + 3x
        let x = vec![
            vec![1.0, 1.0, 1.0, 1.0], // intercept
            vec![1.0, 2.0, 3.0, 4.0], // x values
        ];
        let y = vec![5.0, 8.0, 11.0, 14.0];
        
        let result = multiple_regression(&x, &y);
        assert!(result.is_ok());
        
        let coeffs = result.unwrap();
        // Relative tolerance based on coefficient magnitude
        let tol = 100.0 * f64::EPSILON * 3.0; // scale by largest coefficient
        assert!((coeffs[0] - 2.0).abs() < tol, "intercept error: {}", (coeffs[0] - 2.0).abs());
        assert!((coeffs[1] - 3.0).abs() < tol, "slope error: {}", (coeffs[1] - 3.0).abs());
    }

    #[test]
    fn test_compute_residuals() {
        let x = vec![
            vec![1.0, 1.0, 1.0], // intercept
            vec![1.0, 2.0, 3.0], // x values
        ];
        let y = vec![3.0, 5.0, 7.0];
        let coeffs = vec![1.0, 2.0]; // y = 1 + 2x
        
        let residuals = compute_residuals(&x, &y, &coeffs);
        
        for residual in residuals {
            // For exact fit, residuals should be near machine precision * data scale
            assert!(residual.abs() < 100.0 * f64::EPSILON * 7.0); // scale by largest y value
        }
    }
    
    #[test]
    fn test_empty_matrix() {
        let a: Vec<Vec<f64>> = vec![];
        let result = householder_qr(&a);
        assert!(result.is_err());
        if let Err(e) = result {
            match e {
                FractalAnalysisError::NumericalError { reason, .. } => {
                    assert!(reason.contains("Empty matrix"));
                }
                _ => panic!("Expected NumericalError"),
            }
        }
    }
    
    #[test]
    fn test_zero_width_matrix() {
        let a = vec![vec![], vec![], vec![]];
        let result = householder_qr(&a);
        assert!(result.is_err());
        if let Err(e) = result {
            match e {
                FractalAnalysisError::NumericalError { reason, .. } => {
                    assert!(reason.contains("Zero-width"));
                }
                _ => panic!("Expected NumericalError"),
            }
        }
    }
    
    #[test]
    fn test_ragged_matrix() {
        let a = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0, 5.0],  // Different length
            vec![6.0, 7.0],
        ];
        let result = householder_qr(&a);
        assert!(result.is_err());
        if let Err(e) = result {
            match e {
                FractalAnalysisError::NumericalError { reason, .. } => {
                    assert!(reason.contains("Ragged matrix"));
                }
                _ => panic!("Expected NumericalError"),
            }
        }
    }
    
    #[test]
    fn test_single_row_matrix() {
        // Test m = 1 edge case
        let a = vec![vec![1.0, 2.0, 3.0]];
        let result = householder_qr(&a);
        assert!(result.is_ok());
        
        let (q, r) = result.unwrap();
        assert_eq!(q.len(), 1);
        assert_eq!(r.len(), 1);
        
        // Verify Q is identity for 1x1
        // Q should be identity for single row
        assert!((q[0][0] - 1.0).abs() < 10.0 * f64::EPSILON);
        
        // Verify A = QR
        for j in 0..a[0].len() {
            // R should match A for single row, with relative tolerance
            let tol = 10.0 * f64::EPSILON * a[0][j].abs().max(1.0);
            assert!((r[0][j] - a[0][j]).abs() < tol);
        }
    }
    
    #[test]
    fn test_underdetermined_system() {
        // More columns than rows
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let b = vec![1.0, 2.0];
        
        let result = economy_qr_solve(&a, &b);
        assert!(result.is_err());
        if let Err(e) = result {
            match e {
                FractalAnalysisError::NumericalError { reason, .. } => {
                    assert!(reason.contains("Underdetermined"));
                }
                _ => panic!("Expected NumericalError"),
            }
        }
    }
    
    #[test]
    fn test_newey_west_small_sample() {
        // Test n = 1
        let residuals = vec![1.0];
        let result = newey_west_lrv(&residuals, 1);
        assert!(result.is_err());
        
        // Test n = 2 (should work)
        let residuals = vec![1.0, 2.0];
        let result = newey_west_lrv(&residuals, 1);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_mismatched_predictor_lengths() {
        let x = vec![
            vec![1.0, 1.0, 1.0, 1.0],  // length 4
            vec![1.0, 2.0, 3.0],        // length 3 - mismatch!
        ];
        let y = vec![5.0, 8.0, 11.0, 14.0];
        
        let result = multiple_regression(&x, &y);
        assert!(result.is_err());
        if let Err(e) = result {
            match e {
                FractalAnalysisError::NumericalError { reason, .. } => {
                    assert!(reason.contains("inconsistent length"));
                }
                _ => panic!("Expected NumericalError"),
            }
        }
    }
    
    #[test]
    fn test_rank_deficient_matrix() {
        // Create a rank-deficient matrix (third column is sum of first two)
        let a = vec![
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 1.0, 2.0],
            vec![2.0, 1.0, 3.0],
        ];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        
        // Should detect singularity during back-substitution
        let result = economy_qr_solve(&a, &b);
        // This should either succeed with a least squares solution
        // or fail with a singular matrix error
        if result.is_err() {
            if let Err(e) = result {
                match e {
                    FractalAnalysisError::NumericalError { reason, .. } => {
                        assert!(reason.contains("Singular") || reason.contains("singular"));
                    }
                    _ => panic!("Expected NumericalError"),
                }
            }
        }
    }
    
    #[test]
    fn test_qr_with_pivoting() {
        // Test with a rank-deficient matrix (third column is sum of first two)
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 9.0],
            vec![7.0, 8.0, 15.0],
        ];
        
        let result = qr_with_pivoting(&a);
        assert!(result.is_ok());
        
        let (q, r, perm, rank) = result.unwrap();
        assert!(rank > 0);
        
        // Check Q is orthogonal
        for i in 0..q.len() {
            for j in 0..q.len() {
                let mut dot = 0.0;
                for k in 0..q.len() {
                    dot += q[k][i] * q[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                let tol = 100.0 * f64::EPSILON * (q.len() as f64).sqrt();
                assert!((dot - expected).abs() < tol);
            }
        }
        
        // Check R is upper triangular
        for i in 1..r.len() {
            for j in 0..i.min(r[0].len()) {
                // Use matrix norm for scale-aware tolerance
                let r_norm = r.iter().flat_map(|row| row.iter()).map(|&x| x*x).sum::<f64>().sqrt();
                let tol = 100.0 * f64::EPSILON * r_norm;
                assert!(r[i][j].abs() < tol);
            }
        }
        
        // Check A*P ≈ Q*R
        for i in 0..a.len() {
            for j in 0..a[0].len() {
                let mut qr = 0.0;
                for k in 0..q[0].len() {
                    qr += q[i][k] * r[k][j];
                }
                // Apply inverse permutation: column perm[j] of original A goes to column j
                let a_permuted = a[i][perm[j]];
                // Scale tolerance by matrix norm
                let a_norm = a.iter().flat_map(|row| row.iter()).map(|&x| x*x).sum::<f64>().sqrt();
                let tol = 100.0 * f64::EPSILON * a_norm;
                assert!((qr - a_permuted).abs() < tol,
                    "QR[{},{}] = {} but A*P[{},{}] = {} (tol={})", i, j, qr, i, j, a_permuted, tol);
            }
        }
    }
    
    #[test]
    fn test_nan_inf_validation() {
        // Test NaN in matrix
        let a = vec![
            vec![1.0, 2.0],
            vec![0.0_f64 / 0.0, 4.0],  // Create NaN value
        ];
        let result = qr_with_pivoting(&a);
        assert!(result.is_err());
        
        // Test Inf in matrix
        let a = vec![
            vec![1.0, 1.0_f64 / 0.0],  // Create Inf value
            vec![3.0, 4.0],
        ];
        let result = qr_with_pivoting(&a);
        assert!(result.is_err());
        
        // Test NaN in vector for solve
        let a = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let b = vec![1.0, 0.0_f64 / 0.0];  // Create NaN value
        let result = economy_qr_solve(&a, &b);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_exactly_rank_deficient() {
        // Matrix with exactly linearly dependent columns (rank 1)
        // Second column is exactly 2 * first column
        let a = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
        ];
        
        let result = qr_with_pivoting(&a);
        assert!(result.is_ok());
        
        let (_q, r, _perm, rank) = result.unwrap();
        
        // Should detect rank 1 for exactly dependent columns
        assert_eq!(rank, 1, "Matrix with exactly dependent columns should have rank 1");
        
        // Second diagonal element should be effectively zero relative to first
        let relative_tol = 100.0 * f64::EPSILON * 3.0; // scale by matrix dimension
        assert!(r[1][1].abs() <= relative_tol * r[0][0].abs(),
            "R[1][1] = {} should be near zero relative to R[0][0] = {} for rank-1 matrix", 
            r[1][1], r[0][0]);
    }
    
    #[test]
    fn test_nearly_rank_deficient() {
        // Test rank detection using the EXACT same tolerance as the algorithm
        // The algorithm uses: rank_tol = 10 * f64::EPSILON * max(m,n) * frobenius_norm
        
        // Helper to compute Frobenius norm (same as in the algorithm)
        let frobenius_norm = |a: &[Vec<f64>]| -> f64 {
            let mut sum = 0.0;
            for row in a {
                for &val in row {
                    sum += val * val;
                }
            }
            sum.sqrt()
        };
        
        // Case 1: Small epsilon - should be detected as rank 1
        // From analysis: |R[1][1]| ≈ 0.65 * epsilon for this matrix structure
        // We need |R[1][1]| < rank_tol, so epsilon < rank_tol / 0.65
        let epsilon_small = 2e-14; // Small enough to be below tolerance
        let a1 = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0 + epsilon_small],
            vec![3.0, 6.0],
        ];
        
        // Compute the same tolerance the algorithm will use
        let m1 = a1.len();
        let n1 = a1[0].len();
        let fro1 = frobenius_norm(&a1);
        let rank_tol1 = 10.0 * f64::EPSILON * (m1.max(n1) as f64) * fro1;
        
        let result1 = qr_with_pivoting(&a1);
        assert!(result1.is_ok());
        let (_q1, r1, _perm1, rank1) = result1.unwrap();
        
        // Assert rank 1 and that R[1][1] is below the algorithm's tolerance
        assert_eq!(rank1, 1, 
            "Matrix with epsilon = {} should be detected as rank 1", epsilon_small);
        assert!(r1[1][1].abs() <= rank_tol1,
            "R[1][1] = {} should be <= rank_tol = {} for rank-1 detection", 
            r1[1][1].abs(), rank_tol1);
        
        // Case 2: Larger epsilon - should be detected as rank 2
        let epsilon_large = 1e-10; // Large enough to be above tolerance
        let a2 = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0 + epsilon_large],
            vec![3.0, 6.0],
        ];
        
        // Compute the same tolerance the algorithm will use
        let m2 = a2.len();
        let n2 = a2[0].len();
        let fro2 = frobenius_norm(&a2);
        let rank_tol2 = 10.0 * f64::EPSILON * (m2.max(n2) as f64) * fro2;
        
        let result2 = qr_with_pivoting(&a2);
        assert!(result2.is_ok());
        let (_q2, r2, _perm2, rank2) = result2.unwrap();
        
        // Assert rank 2 and that R[1][1] is above the algorithm's tolerance
        assert_eq!(rank2, 2, 
            "Matrix with epsilon = {} should be detected as rank 2", epsilon_large);
        assert!(r2[1][1].abs() > rank_tol2,
            "R[1][1] = {} should be > rank_tol = {} for rank-2 detection", 
            r2[1][1].abs(), rank_tol2);
        
        // Additional sanity check: rank_tol should be between the two R[1][1] values
        assert!(r1[1][1].abs() < rank_tol1 && rank_tol1 < r2[1][1].abs(),
            "Tolerance {} should separate the two cases: {} < tol < {}",
            rank_tol1, r1[1][1].abs(), r2[1][1].abs());
    }
    
    #[test]
    fn test_clearly_full_rank() {
        // Matrix with clearly independent columns to contrast with rank-deficient cases
        let a = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        
        let result = qr_with_pivoting(&a);
        assert!(result.is_ok());
        
        let (_q, r, _perm, rank) = result.unwrap();
        
        // Should detect full rank
        assert_eq!(rank, 2, "Matrix with independent columns should have full rank");
        
        // Both diagonal elements should be substantial
        // Check diagonal elements are not too small (relative to matrix scale)
        let tol = 100.0 * f64::EPSILON;
        assert!(r[0][0].abs() > tol, "R[0][0] = {} should be non-zero", r[0][0]);
        assert!(r[1][1].abs() > tol, "R[1][1] = {} should be non-zero", r[1][1]);
    }
    
    #[test]
    fn test_rank_deficient_detection() {
        // Test with a matrix that has an exactly zero column
        let a = vec![
            vec![1.0, 0.0, 2.0],
            vec![3.0, 0.0, 4.0],
            vec![5.0, 0.0, 6.0],
            vec![7.0, 0.0, 8.0],
        ];
        
        let result = qr_with_pivoting(&a);
        assert!(result.is_ok());
        
        let (q, r, perm, rank) = result.unwrap();
        assert_eq!(rank, 2, "Should detect rank 2 for matrix with zero column");
        
        // The zero column (originally column 1) should be among the last columns after pivoting
        // We check that column 1 is not in the first 'rank' positions
        let zero_col_position = perm.iter().position(|&x| x == 1).unwrap();
        assert!(zero_col_position >= rank, 
            "Zero column (originally 1) should be after rank columns, but is at position {}", 
            zero_col_position);
        
        // Test with nearly zero columns relative to Frobenius norm
        let epsilon = f64::EPSILON;
        let a = vec![
            vec![1.0, epsilon, 2.0],
            vec![3.0, epsilon, 4.0],
            vec![5.0, epsilon, 6.0],
        ];
        
        let result = qr_with_pivoting(&a);
        assert!(result.is_ok());
        let (_q, _r, _perm, rank) = result.unwrap();
        assert_eq!(rank, 2, "Should detect rank 2 for matrix with tiny column");
    }
    
    #[test]
    fn test_column_swap_reflector_correctness() {
        // Test that ensures reflectors are built from the trailing subvector
        // after column swaps. This guards against using stale cached norms.
        // 
        // Create a matrix where the first column has small norm initially
        // but would be selected as pivot if using full-column norm after swap
        let a = vec![
            vec![0.1, 10.0, 1.0],   // First column has small leading element
            vec![0.2, 20.0, 2.0],   // Second column dominates initially
            vec![100.0, 30.0, 5.0], // First column has large trailing element
            vec![200.0, 40.0, 7.0], // Make sure all columns are linearly independent
        ];
        
        let result = qr_with_pivoting(&a);
        assert!(result.is_ok());
        
        let (q, r, perm, rank) = result.unwrap();
        
        // Should have full rank
        assert_eq!(rank, 3, "Should detect full rank");
        
        // With this matrix, the first column (index 0) now has the largest norm due to
        // the large trailing elements (100, 200), so it should be selected first
        assert_eq!(perm[0], 0, "First column should be pivot first (largest norm)");
        
        // Verify Q is orthogonal
        for i in 0..q.len() {
            for j in 0..q.len() {
                let mut dot = 0.0;
                for k in 0..q.len() {
                    dot += q[k][i] * q[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                let tol = 100.0 * f64::EPSILON * (q.len() as f64).sqrt();
                assert!((dot - expected).abs() < tol,
                    "Q not orthogonal at [{},{}]: dot={}, expected={}", i, j, dot, expected);
            }
        }
        
        // Verify QR = A*P reconstruction
        for i in 0..a.len() {
            for j in 0..a[0].len() {
                let mut qr = 0.0;
                for k in 0..q[0].len() {
                    qr += q[i][k] * r[k][j];
                }
                let a_permuted = a[i][perm[j]];
                let a_norm = a.iter().flat_map(|row| row.iter()).map(|&x| x*x).sum::<f64>().sqrt();
                let tol = 100.0 * f64::EPSILON * a_norm;
                assert!((qr - a_permuted).abs() < tol,
                    "QR reconstruction failed at [{},{}]: QR={}, A*P={}", i, j, qr, a_permuted);
            }
        }
    }
    
    #[test]
    fn test_qr_tall_rectangular() {
        // Test m > n case
        let a = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        
        let result = qr_with_pivoting(&a);
        assert!(result.is_ok());
        
        let (q, r, perm, rank) = result.unwrap();
        assert_eq!(rank, 2, "Full rank for tall matrix");
        
        // Check Q is orthogonal (4x4)
        assert_eq!(q.len(), 4);
        assert_eq!(q[0].len(), 4);
        for i in 0..4 {
            for j in 0..4 {
                let mut dot = 0.0;
                for k in 0..4 {
                    dot += q[k][i] * q[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                let tol = 100.0 * f64::EPSILON * 2.0; // sqrt(4) = 2
                assert!((dot - expected).abs() < tol);
            }
        }
        
        // Check R is upper trapezoidal (4x2)
        assert_eq!(r.len(), 4);
        assert_eq!(r[0].len(), 2);
        for i in 2..4 {
            for j in 0..2 {
                // Use matrix norm for scale-aware tolerance
                let r_norm = r.iter().flat_map(|row| row.iter()).map(|&x| x*x).sum::<f64>().sqrt();
                let tol = 100.0 * f64::EPSILON * r_norm;
                assert!(r[i][j].abs() < tol, "R[{},{}] = {} should be zero", i, j, r[i][j]);
            }
        }
        
        // Check A*P ≈ Q*R
        for i in 0..4 {
            for j in 0..2 {
                let mut qr = 0.0;
                for k in 0..4 {
                    qr += q[i][k] * r[k][j];
                }
                let a_permuted = a[i][perm[j]];
                let a_norm = a.iter().flat_map(|row| row.iter()).map(|&x| x*x).sum::<f64>().sqrt();
                let tol = 100.0 * f64::EPSILON * a_norm;
                assert!((qr - a_permuted).abs() < tol,
                    "QR reconstruction failed at [{},{}]", i, j);
            }
        }
    }
}