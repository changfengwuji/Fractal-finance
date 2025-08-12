#[test]
fn test_variogram_white_noise_debug() {
    use fractal_finance::hurst_estimators::{estimate_hurst_by_method, EstimationMethod, HurstEstimationConfig};
    
    // Generate deterministic white noise using simple hash
    let white_noise: Vec<f64> = (0..1000).map(|i| {
        // Simple hash function for reproducible pseudo-random values
        let mut x = i as u64;
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        x = (x >> 33) ^ x;
        x = x.wrapping_mul(0x5555555555555555);
        ((x % 1000000) as f64 / 1000000.0) - 0.5
    }).collect();
    
    println!("Testing variogram with white noise (should give H ≈ 0.5)");
    
    // Check variance stability
    let n = white_noise.len();
    let first_half = &white_noise[..n/2];
    let second_half = &white_noise[n/2..];
    
    let var1: f64 = first_half.iter().map(|x| x * x).sum::<f64>() / first_half.len() as f64;
    let var2: f64 = second_half.iter().map(|x| x * x).sum::<f64>() / second_half.len() as f64;
    
    println!("First half variance: {:.6}", var1);
    println!("Second half variance: {:.6}", var2);
    println!("Variance ratio: {:.6}", var1 / var2);
    
    let config = HurstEstimationConfig::default();
    let result = estimate_hurst_by_method(&white_noise, &EstimationMethod::VariogramMethod, &config);
    
    match result {
        Ok(estimate) => {
            println!("Variogram estimate for white noise: H = {:.4}", estimate.estimate);
            assert!(estimate.estimate > 0.3 && estimate.estimate < 0.7,
                "Variogram should detect H ≈ 0.5 for white noise, got {}", estimate.estimate);
        }
        Err(e) => {
            panic!("Error: {:?}", e);
        }
    }
}