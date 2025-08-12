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
use fractal_finance::{
    StatisticalFractalAnalyzer,
    hurst_estimators::{estimate_hurst_by_method, EstimationMethod, HurstEstimationConfig},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create analyzer
    let mut analyzer = StatisticalFractalAnalyzer::new();

    // Add your financial time series
    let returns = vec![0.01, -0.02, 0.015, /* ... your data ... */];
    analyzer.add_time_series("AAPL".to_string(), returns);

    // Perform comprehensive analysis
    let results = analyzer.analyze_series("AAPL")?;

    // Access results
    println!("Hurst exponent (DFA): {:.3}", results.hurst_estimates.dfa.estimate);
    println!("Multifractal degree: {:.3}", results.multifractal.degree_of_multifractality);

    Ok(())
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