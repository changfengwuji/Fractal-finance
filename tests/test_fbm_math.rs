// Mathematical test to understand the correct relationship between FGN and FBM
// This will help us fix the scaling issue

#[cfg(test)]
mod test_fbm_math {
    use fractal_finance::generators::*;

    #[test]
    fn test_fbm_variance_relationship() {
        // For FBM B_H(t) with Hurst H and scale σ:
        // Var[B_H(t)] = σ² * t^(2H)

        // For FGN X_i = B_H(i) - B_H(i-1):
        // Var[X_i] = Var[B_H(i) - B_H(i-1)]
        //          = Var[B_H(i)] + Var[B_H(i-1)] - 2*Cov[B_H(i), B_H(i-1)]

        // The covariance of FBM is:
        // Cov[B_H(s), B_H(t)] = (σ²/2) * (s^(2H) + t^(2H) - |t-s|^(2H))

        let h = 0.7;
        let sigma = 1.0;
        let sigma2 = sigma * sigma;

        // Calculate variance of first increment X_1 = B_H(1) - B_H(0)
        // Var[X_1] = Var[B_H(1)] since B_H(0) = 0
        let var_x1 = sigma2 * 1.0_f64.powf(2.0 * h);
        println!("Var[X_1] = {:.6}", var_x1);

        // Calculate variance of second increment X_2 = B_H(2) - B_H(1)
        // Using the formula above
        let var_b2 = sigma2 * 2.0_f64.powf(2.0 * h);
        let var_b1 = sigma2 * 1.0_f64.powf(2.0 * h);
        let cov_b2_b1 = (sigma2 / 2.0)
            * (2.0_f64.powf(2.0 * h) + 1.0_f64.powf(2.0 * h) - 1.0_f64.powf(2.0 * h));
        let var_x2 = var_b2 + var_b1 - 2.0 * cov_b2_b1;
        println!("Var[X_2] = {:.6}", var_x2);

        // General formula for increment variance
        // For unit increments: Var[B_H(k) - B_H(k-1)] = σ²
        // This is a key property of FBM!
        println!("\nKey insight: Var[B_H(k) - B_H(k-1)] = σ² for all k");
        println!("This means FGN increments have constant variance σ²");

        // When we integrate FGN to get FBM:
        // B_H(n) = sum_{i=1}^n X_i where X_i are FGN increments

        // The variance of B_H(n) should be:
        // Var[B_H(n)] = σ² * n^(2H)

        // But if we just sum n independent increments each with variance σ²:
        // Var[sum X_i] = n * σ² (if independent)

        // FGN increments are NOT independent! They have long-range correlations.
        // The autocovariance of FGN is:
        // γ(k) = (σ²/2) * (|k+1|^(2H) + |k-1|^(2H) - 2|k|^(2H))

        println!("\nAutocovariance of FGN at different lags:");
        for k in 0..5 {
            let gamma_k = if k == 0 {
                sigma2
            } else {
                (sigma2 / 2.0)
                    * ((k as f64 + 1.0).powf(2.0 * h) + (k as f64 - 1.0).powf(2.0 * h)
                        - 2.0 * (k as f64).powf(2.0 * h))
            };
            println!("γ({}) = {:.6}", k, gamma_k);
        }

        // Now let's verify that when we sum FGN with this autocovariance structure,
        // we get the correct FBM variance
        let n = 10;
        let mut cov_matrix = vec![vec![0.0; n]; n];

        // Build covariance matrix of FGN
        for i in 0..n {
            for j in 0..n {
                let lag = (i as i32 - j as i32).abs() as usize;
                cov_matrix[i][j] = if lag == 0 {
                    sigma2
                } else {
                    (sigma2 / 2.0)
                        * ((lag as f64 + 1.0).powf(2.0 * h) + (lag as f64 - 1.0).powf(2.0 * h)
                            - 2.0 * (lag as f64).powf(2.0 * h))
                };
            }
        }

        // Variance of B_H(n) = sum_{i=1}^n X_i
        // Var[sum X_i] = sum_i sum_j Cov[X_i, X_j] = sum of all elements in cov_matrix
        let mut var_sum = 0.0;
        for i in 0..n {
            for j in 0..n {
                var_sum += cov_matrix[i][j];
            }
        }

        let theoretical_var = sigma2 * (n as f64).powf(2.0 * h);
        println!("\nFor n={}, H={:.1}:", n, h);
        println!("Var[B_H(n)] from summing covariances = {:.6}", var_sum);
        println!(
            "Theoretical Var[B_H(n)] = σ² * n^(2H) = {:.6}",
            theoretical_var
        );
        println!("Ratio: {:.6}", var_sum / theoretical_var);

        // The ratio should be 1.0 if our formulas are correct
        assert!(
            (var_sum / theoretical_var - 1.0).abs() < 0.01,
            "Variance formulas don't match!"
        );
    }
}
