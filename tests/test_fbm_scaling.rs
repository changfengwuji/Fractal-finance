// Test module to debug FBM scaling issue
#[cfg(test)]
mod tests {
    use fractal_finance::generators::{
        generate_fractional_brownian_motion, FbmConfig, FbmMethod, GeneratorConfig,
    };

    #[test]
    fn test_fbm_scaling_at_boundaries() {
        let test_lengths = vec![900, 1000, 1100, 1400, 1500, 1600, 2000];
        let hurst = 0.7;
        let volatility = 1.0;

        println!("\n=== Testing FBM generation at different lengths ===");
        println!("Expected variance formula: σ² * n^(2H) where σ=1.0, H=0.7");

        for length in test_lengths {
            let config = GeneratorConfig {
                length,
                seed: Some(42),
                sampling_frequency: 1.0,
            };

            // Test with Auto method (default)
            let fbm_config = FbmConfig {
                hurst_exponent: hurst,
                volatility,
                method: FbmMethod::Auto,
            };

            let fbm = generate_fractional_brownian_motion(&config, &fbm_config).unwrap();

            // Calculate actual variance
            let mean: f64 = fbm.iter().sum::<f64>() / fbm.len() as f64;
            let variance: f64 =
                fbm.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / fbm.len() as f64;
            let std_dev = variance.sqrt();

            // Expected std dev for FBM: σ * sqrt(n^(2H))
            let expected_std = volatility * (length as f64).powf(hurst);

            // For circulant embedding, calculate what m would be
            let m = (2 * length as usize).next_power_of_two();

            println!("\nLength = {}", length);
            println!(
                "  Method: {}",
                if length <= 1000 {
                    "Hosking"
                } else {
                    "CirculantEmbedding"
                }
            );
            println!("  Actual std dev:    {:.6}", std_dev);
            println!("  Expected std dev:  {:.6}", expected_std);
            println!("  Ratio actual/expected: {:.6}", std_dev / expected_std);
            println!("  CircEmbed m would be: {}", m);
            println!(
                "  1/m = {:.9}, 1/sqrt(m) = {:.9}",
                1.0 / (m as f64),
                1.0 / (m as f64).sqrt()
            );

            // Check which method is actually being used
            if length <= 1000 {
                assert!(
                    std_dev > expected_std * 0.1,
                    "Hosking method variance too low at length {}",
                    length
                );
            }
        }
    }

    #[test]
    fn test_direct_method_comparison() {
        println!("\n=== Direct comparison of methods ===");

        let lengths = vec![500, 1500];
        let hurst = 0.7;
        let volatility = 1.0;

        for length in lengths {
            let config = GeneratorConfig {
                length,
                seed: Some(42),
                sampling_frequency: 1.0,
            };

            // Test Hosking if length allows
            if length <= 1000 {
                let fbm_config = FbmConfig {
                    hurst_exponent: hurst,
                    volatility,
                    method: FbmMethod::Hosking,
                };

                let fbm = generate_fractional_brownian_motion(&config, &fbm_config).unwrap();
                let std_dev = calculate_std(&fbm);
                println!("Length {} (Hosking): std = {:.6}", length, std_dev);
            }

            // Test CirculantEmbedding
            let fbm_config = FbmConfig {
                hurst_exponent: hurst,
                volatility,
                method: FbmMethod::CirculantEmbedding,
            };

            let fbm = generate_fractional_brownian_motion(&config, &fbm_config).unwrap();
            let std_dev = calculate_std(&fbm);
            println!(
                "Length {} (CirculantEmbedding): std = {:.6}",
                length, std_dev
            );

            // Check the scaling
            let expected_std = volatility * (length as f64).powf(hurst);
            let ratio = std_dev / expected_std;
            println!("  Expected std: {:.6}, Ratio: {:.6}", expected_std, ratio);

            // For n=1500, m=4096, check if we're seeing 1/m scaling instead of 1/sqrt(m)
            if length == 1500 {
                let m = 4096;
                println!("  If using 1/m scaling: {:.9}", 1.0 / (m as f64));
                println!(
                    "  If using 1/sqrt(m) scaling: {:.9}",
                    1.0 / (m as f64).sqrt()
                );
                println!("  Actual ratio suggests scaling of: {:.9}", ratio);
            }
        }
    }

    fn calculate_std(data: &[f64]) -> f64 {
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        let variance: f64 =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        variance.sqrt()
    }
}
