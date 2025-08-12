# Financial Fractal Analysis

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-%3E%3D1.70-orange.svg)](https://www.rust-lang.org)
[![Crates.io](https://img.shields.io/crates/v/fractal_finance.svg)](https://crates.io/crates/fractal_finance)
[![Documentation](https://docs.rs/fractal_finance/badge.svg)](https://docs.rs/fractal_finance)

Enterprise-grade fractal analysis library for quantitative finance applications. This Rust library provides comprehensive tools for analyzing long-range dependence, multifractality, and regime changes in financial time series with statistical rigor and high performance.

## Key Features

- **Statistical Rigor**: All estimators include bias corrections, confidence intervals, and comprehensive hypothesis testing
- **Multiple Methods**: Hurst exponent estimation via R/S, DFA, GPH, and wavelet methods
- **Multifractal Analysis**: Complete MF-DFA implementation with singularity spectrum
- **Regime Detection**: HMM-based detection of structural breaks and fractal regimes
- **High Performance**: Written in Rust for optimal speed and memory efficiency
- **Enterprise Ready**: Comprehensive validation, testing, and documentation

## Quick Start

### Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
fractal_finance = "0.1.2"
```

### Basic Usage

```rust
use fractal_finance::{StatisticalFractalAnalyzer, EstimationMethod};
use rand::prelude::*;
use rand_distr::StandardNormal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut analyzer = StatisticalFractalAnalyzer::new();

    // Generate 500 realistic financial return data points
    let returns = generate_financial_returns(500);
    println!("Generated {} data points for analysis", returns.len());
    analyzer.add_time_series("ASSET".to_string(), returns);

    // Perform comprehensive analysis
    analyzer.analyze_all_series()?;

    // Get results
    let results = analyzer.get_analysis_results("ASSET")?;
    for (method, estimate) in &results.hurst_estimates {
        println!("{:?}: H = {:.3} ± {:.3}", method,
            estimate.estimate, estimate.standard_error);
    }
    
    Ok(())
}

fn generate_financial_returns(n: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    let mut returns = Vec::with_capacity(n);
    
    // Parameters for realistic financial returns
    let base_volatility = 0.015f64; // 1.5% daily volatility
    let mut volatility = base_volatility;
    let mut previous_return = 0.0f64;
    
    for i in 0..n {
        // Add volatility clustering (GARCH-like effect)
        let volatility_shock = rng.gen_range(-0.005..0.005);
        volatility = (base_volatility + 0.1 * previous_return.abs() + volatility_shock).max(0.005f64);
        
        // Generate return with some persistence (memory effect)
        let white_noise: f64 = rng.sample(rand_distr::StandardNormal);
        let persistence_factor = 0.15 * previous_return; // 15% persistence
        let trend_component = 0.0001 * (i as f64 / 100.0).sin(); // Small trend component
        
        let return_val = trend_component + persistence_factor + volatility * white_noise;
        
        returns.push(return_val);
        previous_return = return_val;
    }
    
    returns
}
```

### Advanced Analysis

```rust
use fractal_finance::{
    multifractal::{mf_dfa_analysis, MultifractalConfig},
    regime_detection::{detect_fractal_regimes, RegimeConfig},
    bootstrap::{BootstrapConfiguration, BootstrapMethod},
};

fn advanced_analysis(data: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    // Multifractal analysis
    let mf_config = MultifractalConfig::default();
    let mf_results = mf_dfa_analysis(data, &mf_config)?;

    // Regime detection
    let regime_config = RegimeConfig::default();
    let regimes = detect_fractal_regimes(data, 2, &regime_config)?;

    // Bootstrap validation
    let bootstrap_config = BootstrapConfiguration {
        num_bootstrap_samples: 1000,
        confidence_levels: vec![0.95, 0.99],
        bootstrap_method: BootstrapMethod::Block,
        ..Default::default()
    };

    Ok(())
}
```

## Analysis Methods

### Hurst Exponent Estimation

The library implements multiple methods for Hurst exponent estimation:

- **Rescaled Range (R/S)**: Classical method with Lo's bias correction
- **Detrended Fluctuation Analysis (DFA)**: Robust to non-stationarity
- **GPH Periodogram**: Frequency-domain estimation with HAC standard errors
- **Wavelet-based**: Using MODWT for scale-dependent analysis
- **Whittle Estimator**: Maximum likelihood in frequency domain

### Multifractal Analysis

- **MF-DFA**: Multifractal DFA for generalized Hurst exponents
- **Singularity Spectrum**: f(α) characterization
- **WTMM**: Wavelet Transform Modulus Maxima method

### Statistical Testing

- **Long-range dependence**: GPH, Robinson tests
- **Stationarity**: ADF, KPSS tests with rigorous p-values
- **Structural breaks**: CUSUM, Quandt-Andrews tests
- **Goodness-of-fit**: Anderson-Darling, Cramér-von Mises

## Performance

The library is optimized for quantitative finance applications:

- FFT caching for repeated spectral calculations
- Memory pooling for large dataset processing
- Parallel processing support via Rayon
- SIMD optimizations (when enabled)

## Test Status

- 236/236 unit tests passing
- 6 integration tests marked as ignored (edge cases, performance variations)
- Run ignored tests with: `cargo test -- --ignored`

Known test limitations:

- Performance tests may timeout on slower hardware
- Some edge cases with extreme numerical values
- Cross-method validation shows expected variations

## Documentation

Comprehensive documentation is available:

- [Live Documentation](https://www.pyquantlib.com/docs/) - Complete API documentation with examples
- [API Documentation](https://docs.rs/fractal_finance) - Official Rust documentation
- Module-level documentation in source code

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is dual-licensed under either of:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
* MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Important Notes

**P-values for ADF and KPSS tests are APPROXIMATE** based on critical value interpolation or asymptotic approximations. For regulatory compliance or critical financial decisions, use test statistics with appropriate critical value tables.

## Links

- [Repository](https://github.com/changfengwuji/fractal-finance)
- [Issues](https://github.com/changfengwuji/fractal-finance/issues)
- [Discussions](https://github.com/changfengwuji/fractal-finance/discussions)