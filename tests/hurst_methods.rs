use fractal_finance::{estimate_hurst_multiple_methods, EstimationMethod, HurstEstimationConfig};
use fractal_finance::secure_rng::FastrandCompat;

#[test]
fn test_estimate_hurst_multiple_methods_returns_all() {
    let mut rng = FastrandCompat::with_seed(7);
    let data: Vec<f64> = (0..512).map(|_| rng.f64() - 0.5).collect();
    let mut config = HurstEstimationConfig::default();
    config.bootstrap_config.num_bootstrap_samples = 200;
    config.bootstrap_config.confidence_interval_method =
        fractal_finance::bootstrap::ConfidenceIntervalMethod::Normal;
    let results = estimate_hurst_multiple_methods(&data, &config).unwrap();
    for method in [
        EstimationMethod::RescaledRange,
        EstimationMethod::DetrendedFluctuationAnalysis,
        EstimationMethod::PeriodogramRegression,
        EstimationMethod::WaveletEstimation,
        EstimationMethod::WhittleEstimator,
        EstimationMethod::VariogramMethod,
    ] {
        assert!(results.contains_key(&method));
    }
}
