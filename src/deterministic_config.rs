//! Configuration management for deterministic financial computations.
//!
//! This module provides global configuration for ensuring reproducible
//! and resource-bounded computations in production financial systems.

use crate::secure_rng::FastrandCompat;
use std::sync::{Arc, RwLock};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Global configuration for deterministic behavior.
///
/// This configuration ensures all computations are reproducible by
/// controlling random number generation, resource limits, and audit settings.
#[derive(Debug, Clone)]
pub struct DeterministicConfig {
    /// Master seed for all RNG operations.
    /// All random operations derive their seeds from this value.
    pub master_seed: u64,
    
    /// Maximum memory usage in megabytes.
    /// Operations exceeding this limit should fail gracefully.
    pub max_memory_mb: usize,
    
    /// Maximum computation time in milliseconds.
    /// Long-running operations should check this limit.
    pub max_computation_ms: u64,
    
    /// Enable comprehensive audit logging.
    /// When true, all operations should log to the audit trail.
    pub enable_audit: bool,
    
    /// Maximum iterations for iterative algorithms.
    /// Prevents infinite loops and complexity attacks.
    pub max_iterations: usize,
    
    /// Numerical tolerance for convergence checks.
    /// Used by iterative algorithms to determine convergence.
    pub convergence_tolerance: f64,
    
    /// Enable debug assertions in production.
    /// Useful for catching issues in production with minimal overhead.
    pub enable_debug_checks: bool,
}

impl Default for DeterministicConfig {
    fn default() -> Self {
        Self {
            master_seed: 42,                 // Classic deterministic seed
            max_memory_mb: 1024,             // 1GB default limit
            max_computation_ms: 60000,       // 60 second timeout
            enable_audit: true,              // Audit by default for compliance
            max_iterations: 1_000_000,       // 1M iterations max
            convergence_tolerance: 1e-10,    // Standard numerical tolerance
            enable_debug_checks: false,      // Disabled by default for performance
        }
    }
}

impl DeterministicConfig {
    /// Create a production configuration with strict limits.
    pub fn production() -> Self {
        Self {
            master_seed: 42,
            max_memory_mb: 4096,           // 4GB for production
            max_computation_ms: 300000,    // 5 minutes max
            enable_audit: true,
            max_iterations: 10_000_000,    // 10M for complex computations
            convergence_tolerance: 1e-12,  // Higher precision for production
            enable_debug_checks: false,
        }
    }
    
    /// Create a development configuration with relaxed limits.
    pub fn development() -> Self {
        Self {
            master_seed: 42,
            max_memory_mb: 512,            // Lower limit for dev
            max_computation_ms: 10000,     // 10 seconds for faster feedback
            enable_audit: false,           // No audit in dev
            max_iterations: 100_000,       // Smaller for debugging
            convergence_tolerance: 1e-8,   // Lower precision acceptable
            enable_debug_checks: true,     // Enable checks in dev
        }
    }
    
    /// Create a test configuration with minimal limits.
    pub fn test() -> Self {
        Self {
            master_seed: 12345,            // Different seed for tests
            max_memory_mb: 128,            // Minimal memory
            max_computation_ms: 1000,      // 1 second for fast tests
            enable_audit: false,
            max_iterations: 1000,          // Small for unit tests
            convergence_tolerance: 1e-6,   // Relaxed for tests
            enable_debug_checks: true,
        }
    }
    
    /// Check if a memory allocation would exceed limits.
    pub fn check_memory_limit(&self, size_bytes: usize) -> bool {
        let size_mb = size_bytes / (1024 * 1024);
        size_mb <= self.max_memory_mb
    }
    
    /// Get memory limit in bytes.
    pub fn max_memory_bytes(&self) -> usize {
        self.max_memory_mb * 1024 * 1024
    }
}

/// Thread-safe global configuration singleton.
lazy_static::lazy_static! {
    pub static ref GLOBAL_CONFIG: Arc<RwLock<DeterministicConfig>> = 
        Arc::new(RwLock::new(DeterministicConfig::default()));
}

/// Get a copy of the current global configuration.
pub fn get_config() -> DeterministicConfig {
    GLOBAL_CONFIG.read().unwrap().clone()
}

/// Update the global configuration.
pub fn set_config(config: DeterministicConfig) {
    *GLOBAL_CONFIG.write().unwrap() = config;
}

/// Create a deterministic RNG for a specific operation.
///
/// This function creates a new RNG instance seeded deterministically
/// based on the master seed and the operation name. This ensures that
/// the same operation always gets the same sequence of random numbers,
/// while different operations get different sequences.
///
/// # Mathematical Foundation
/// The seed is computed as: master_seed ⊕ hash(operation_name)
/// This provides 2^64 possible seeds while maintaining determinism.
pub fn create_deterministic_rng(operation: &str) -> FastrandCompat {
    let config = get_config();
    let seed = config.master_seed;
    
    // Hash the operation name to get a unique but deterministic value
    let mut hasher = DefaultHasher::new();
    operation.hash(&mut hasher);
    let operation_hash = hasher.finish();
    
    // Combine with master seed using XOR for good distribution
    let combined_seed = seed ^ operation_hash;
    
    FastrandCompat::with_seed(combined_seed)
}

/// Configuration builder for fluent API.
pub struct ConfigBuilder {
    config: DeterministicConfig,
}

impl ConfigBuilder {
    /// Start building a new configuration.
    pub fn new() -> Self {
        Self {
            config: DeterministicConfig::default(),
        }
    }
    
    /// Set the master seed.
    pub fn master_seed(mut self, seed: u64) -> Self {
        self.config.master_seed = seed;
        self
    }
    
    /// Set maximum memory in MB.
    pub fn max_memory_mb(mut self, mb: usize) -> Self {
        self.config.max_memory_mb = mb;
        self
    }
    
    /// Set maximum computation time in ms.
    pub fn max_computation_ms(mut self, ms: u64) -> Self {
        self.config.max_computation_ms = ms;
        self
    }
    
    /// Enable or disable audit.
    pub fn enable_audit(mut self, enable: bool) -> Self {
        self.config.enable_audit = enable;
        self
    }
    
    /// Set maximum iterations.
    pub fn max_iterations(mut self, iterations: usize) -> Self {
        self.config.max_iterations = iterations;
        self
    }
    
    /// Set convergence tolerance.
    pub fn convergence_tolerance(mut self, tolerance: f64) -> Self {
        self.config.convergence_tolerance = tolerance;
        self
    }
    
    /// Enable or disable debug checks.
    pub fn enable_debug_checks(mut self, enable: bool) -> Self {
        self.config.enable_debug_checks = enable;
        self
    }
    
    /// Build and return the configuration.
    pub fn build(self) -> DeterministicConfig {
        self.config
    }
    
    /// Build and set as global configuration.
    pub fn build_and_set_global(self) -> DeterministicConfig {
        let config = self.build();
        set_config(config.clone());
        config
    }
}

/// Environment-based configuration detection.
pub fn from_environment() -> DeterministicConfig {
    let env = std::env::var("FRACTAL_ENV").unwrap_or_else(|_| "production".to_string());
    
    match env.as_str() {
        "production" | "prod" => DeterministicConfig::production(),
        "development" | "dev" => DeterministicConfig::development(),
        "test" => DeterministicConfig::test(),
        _ => DeterministicConfig::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .master_seed(999)
            .max_memory_mb(2048)
            .max_computation_ms(30000)
            .enable_audit(false)
            .max_iterations(5000)
            .convergence_tolerance(1e-9)
            .enable_debug_checks(true)
            .build();
        
        assert_eq!(config.master_seed, 999);
        assert_eq!(config.max_memory_mb, 2048);
        assert_eq!(config.max_computation_ms, 30000);
        assert!(!config.enable_audit);
        assert_eq!(config.max_iterations, 5000);
        assert_eq!(config.convergence_tolerance, 1e-9);
        assert!(config.enable_debug_checks);
    }
    
    #[test]
    fn test_deterministic_rng() {
        // Set a known configuration
        set_config(DeterministicConfig {
            master_seed: 12345,
            ..Default::default()
        });
        
        // Same operation should produce same sequence
        let mut rng1 = create_deterministic_rng("test_operation");
        let mut rng2 = create_deterministic_rng("test_operation");
        
        // Generate some values and verify they're identical
        let vals1: Vec<u64> = (0..10).map(|_| rng1.u64(0..1000000)).collect();
        let vals2: Vec<u64> = (0..10).map(|_| rng2.u64(0..1000000)).collect();
        assert_eq!(vals1, vals2, "Same operation should produce identical sequences");
        
        // Different operations should produce different sequences
        let mut rng3 = create_deterministic_rng("different_operation");
        let vals3: Vec<u64> = (0..10).map(|_| rng3.u64(0..1000000)).collect();
        assert_ne!(vals1, vals3, "Different operations should produce different sequences");
        
        // Verify that the sequences are actually different (not just shifted)
        let overlap = vals1.iter().filter(|v| vals3.contains(v)).count();
        assert!(overlap < 5, "Sequences should have minimal overlap, found {} common values", overlap);
    }
    
    #[test]
    fn test_memory_limits() {
        let config = DeterministicConfig {
            max_memory_mb: 100,
            ..Default::default()
        };
        
        assert!(config.check_memory_limit(50 * 1024 * 1024)); // 50MB OK
        assert!(config.check_memory_limit(100 * 1024 * 1024)); // 100MB OK
        assert!(!config.check_memory_limit(101 * 1024 * 1024)); // 101MB exceeds
        
        assert_eq!(config.max_memory_bytes(), 100 * 1024 * 1024);
    }
    
    #[test]
    fn test_environment_configs() {
        let prod = DeterministicConfig::production();
        assert_eq!(prod.max_memory_mb, 4096);
        assert!(prod.enable_audit);
        
        let dev = DeterministicConfig::development();
        assert_eq!(dev.max_memory_mb, 512);
        assert!(!dev.enable_audit);
        
        let test = DeterministicConfig::test();
        assert_eq!(test.max_memory_mb, 128);
        assert_eq!(test.master_seed, 12345);
    }
}