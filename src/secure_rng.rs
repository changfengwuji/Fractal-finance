//! Cryptographically secure random number generation for financial applications.
//!
//! This module provides secure RNG suitable for regulatory compliance and
//! financial applications where predictability could lead to market manipulation.

use crate::errors::{FractalAnalysisError, FractalResult};
use once_cell::sync::Lazy;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Global operation ID counter for unique IDs across all RNG instances
static GLOBAL_OPERATION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Global seed for deterministic mode (None means use OS entropy)
static GLOBAL_SEED: Lazy<RwLock<Option<u64>>> = Lazy::new(|| RwLock::new(None));

/// Global seed generation counter to detect when seed changes
static SEED_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Global thread ID counter for deterministic thread identification
static THREAD_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Secure RNG wrapper for financial applications.
///
/// Uses ChaCha20 cryptographically secure pseudo-random number generator
/// which is suitable for financial applications requiring unpredictability.
///
/// Features:
/// - CSPRNG (Cryptographically Secure Pseudo-Random Number Generator)
/// - Deterministic seeding for reproducibility when needed
/// - Audit logging capabilities with unique IDs
/// - Thread-safe operation
#[derive(Clone)]
pub struct SecureRng {
    /// Internal ChaCha20 RNG
    rng: ChaCha20Rng,
    /// Audit trail for regulatory compliance (using VecDeque for efficient rotation)
    audit_log: Arc<Mutex<VecDeque<RngAuditEntry>>>,
    /// Whether to enable audit logging
    enable_audit: bool,
}

/// Audit entry for RNG operations
#[derive(Debug, Clone)]
pub struct RngAuditEntry {
    /// Unique operation ID (globally unique)
    pub operation_id: u64,
    /// Timestamp of operation (nanoseconds since UNIX epoch for high-frequency trading precision)
    pub timestamp: u128,
    /// Type of operation
    pub operation: String,
    /// Number of random values generated
    pub count: usize,
    /// Seed used (if deterministic)
    pub seed: Option<u64>,
}

impl SecureRng {
    /// Create a new secure RNG with entropy from the OS.
    ///
    /// This is the recommended way to create an RNG for production use.
    pub fn new() -> Self {
        Self {
            rng: ChaCha20Rng::from_entropy(),
            audit_log: Arc::new(Mutex::new(VecDeque::new())),
            enable_audit: false,
        }
    }

    /// Create a new secure RNG with a specific seed for reproducibility.
    ///
    /// Use this only when reproducibility is required (e.g., testing, debugging).
    /// For production financial applications, use `new()` instead.
    ///
    /// SECURITY NOTE: Uses cryptographic expansion to convert u64 to full 256-bit seed.
    pub fn with_seed(seed: u64) -> Self {
        let mut instance = Self {
            // CRITICAL FIX: Use seed_from_u64 for proper entropy expansion
            // This cryptographically expands the u64 to a full 256-bit seed
            rng: ChaCha20Rng::seed_from_u64(seed),
            audit_log: Arc::new(Mutex::new(VecDeque::new())),
            enable_audit: false,
        };

        // Record seed in audit log for complete traceability
        instance.log_operation_with_seed("init_with_seed", 0, Some(seed));
        instance
    }

    /// Enable audit logging for regulatory compliance.
    pub fn enable_audit(&mut self) {
        self.enable_audit = true;
    }

    /// Generate a random f64 in [0, 1).
    pub fn f64(&mut self) -> f64 {
        self.log_operation("f64", 1, None);
        self.rng.gen::<f64>()
    }

    /// Generate a random usize in the given range.
    pub fn usize(&mut self, range: std::ops::Range<usize>) -> usize {
        self.log_operation("usize", 1, None);
        self.rng.gen_range(range)
    }

    /// Generate a random u64 in the given range (no modulo bias).
    pub fn u64(&mut self, range: std::ops::Range<u64>) -> u64 {
        self.log_operation("u64", 1, None);
        self.rng.gen_range(range)
    }

    /// Generate multiple random f64 values efficiently.
    pub fn fill_f64(&mut self, buffer: &mut [f64]) {
        self.log_operation("fill_f64", buffer.len(), None);
        for value in buffer.iter_mut() {
            *value = self.rng.gen::<f64>();
        }
    }

    /// Generate a random boolean with given probability of being true.
    pub fn bool(&mut self, p: f64) -> bool {
        self.log_operation("bool", 1, None);
        self.rng.gen_bool(p)
    }

    /// Log an RNG operation for audit trail.
    fn log_operation(&self, operation: &str, count: usize, seed: Option<u64>) {
        self.log_operation_internal(operation, count, seed, false);
    }

    /// Log an RNG operation with seed (for initialization).
    fn log_operation_with_seed(&self, operation: &str, count: usize, seed: Option<u64>) {
        self.log_operation_internal(operation, count, seed, true);
    }

    /// Internal logging function with consistent error handling.
    fn log_operation_internal(
        &self,
        operation: &str,
        count: usize,
        seed: Option<u64>,
        force: bool,
    ) {
        // Only log if audit is enabled or forced (for initialization)
        if !self.enable_audit && !force {
            return;
        }

        // Consistent error handling: panic on poisoned lock in high-reliability systems
        let mut log = self
            .audit_log
            .lock()
            .expect("RNG audit log mutex poisoned - critical system failure");

        // Use nanosecond precision for high-frequency trading
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        // Use global atomic counter for unique IDs
        let operation_id = GLOBAL_OPERATION_COUNTER.fetch_add(1, Ordering::SeqCst);

        log.push_back(RngAuditEntry {
            operation_id,
            timestamp,
            operation: operation.to_string(),
            count,
            seed,
        });

        // Efficient rotation using VecDeque
        while log.len() > 10000 {
            log.pop_front();
        }
    }

    /// Get the audit log for regulatory reporting.
    pub fn get_audit_log(&self) -> FractalResult<Vec<RngAuditEntry>> {
        self.audit_log
            .lock()
            .map(|log| log.iter().cloned().collect())
            .map_err(|_| FractalAnalysisError::ConcurrencyError {
                resource: "rng_audit_log".to_string(),
            })
    }
}

/// Global secure RNG for shared state and centralized audit.
///
/// WARNING: This uses a single Mutex which can become a bottleneck in high-concurrency
/// scenarios (e.g., thousands of parallel Monte Carlo simulations). For performance-critical
/// applications, consider creating thread-local SecureRng::new() instances instead.
///
/// This provides a single, thread-safe RNG instance with centralized audit logging.
/// Use this when you need consistent randomness across threads or centralized audit.
static GLOBAL_RNG: Lazy<Mutex<SecureRng>> = Lazy::new(|| Mutex::new(SecureRng::new()));

/// Get a random f64 using the global secure RNG.
///
/// WARNING: This uses a global Mutex which may cause contention in high-concurrency scenarios.
/// For performance-critical code, consider using thread-local SecureRng instances.
///
/// This uses a Mutex-protected global instance for thread safety and centralized audit.
pub fn secure_random_f64() -> FractalResult<f64> {
    GLOBAL_RNG
        .lock()
        .map(|mut rng| rng.f64())
        .map_err(|_| FractalAnalysisError::ConcurrencyError {
            resource: "global_rng".to_string(),
        })
}

/// Get a random usize in range using the global secure RNG.
///
/// WARNING: This uses a global Mutex which may cause contention in high-concurrency scenarios.
/// For performance-critical code, consider using thread-local SecureRng instances.
///
/// This uses a Mutex-protected global instance for thread safety and centralized audit.
pub fn secure_random_usize(range: std::ops::Range<usize>) -> FractalResult<usize> {
    GLOBAL_RNG
        .lock()
        .map(|mut rng| rng.usize(range))
        .map_err(|_| FractalAnalysisError::ConcurrencyError {
            resource: "global_rng".to_string(),
        })
}

/// Get the global audit log for all RNG operations.
///
/// This provides centralized audit trail for regulatory compliance.
pub fn get_global_audit_log() -> FractalResult<Vec<RngAuditEntry>> {
    GLOBAL_RNG
        .lock()
        .map_err(|_| FractalAnalysisError::ConcurrencyError {
            resource: "global_rng".to_string(),
        })?
        .get_audit_log()
}

/// Enable audit logging for the global RNG.
pub fn enable_global_audit() -> FractalResult<()> {
    GLOBAL_RNG
        .lock()
        .map(|mut rng| rng.enable_audit())
        .map_err(|_| FractalAnalysisError::ConcurrencyError {
            resource: "global_rng".to_string(),
        })
}

/// Set a global seed for deterministic behavior.
///
/// WARNING: This should only be used for testing or when reproducibility is required.
/// In production financial applications, use OS entropy.
pub fn global_seed(seed: u64) {
    // Store the global seed and ALWAYS increment generation to force RNG reset
    // This ensures reproducibility even when the same seed is used multiple times
    if let Ok(mut global_seed) = GLOBAL_SEED.write() {
        *global_seed = Some(seed);
        // Always increment to force thread-local RNG reinitialization
        SEED_GENERATION.fetch_add(1, Ordering::SeqCst);
    }
    
    // Update the global RNG
    let _ = GLOBAL_RNG.lock().map(|mut rng| {
        *rng = SecureRng::with_seed(seed);
    });
    
    // Reset the thread ID counter to ensure consistent thread IDs across test runs
    THREAD_ID_COUNTER.store(0, Ordering::SeqCst);
}

/// Clear the global seed, returning to OS entropy mode.
pub fn clear_global_seed() {
    if let Ok(mut global_seed) = GLOBAL_SEED.write() {
        *global_seed = None;
        SEED_GENERATION.fetch_add(1, Ordering::SeqCst);
    }
}

/// Execute a function with the thread-local RNG.
///
/// This is the correct way to use thread-local RNG, ensuring state is maintained.
/// The RNG is stored in thread-local storage and reused across calls.
pub fn with_thread_local_rng<F, R>(f: F) -> R 
where
    F: FnOnce(&mut FastrandCompat) -> R,
{
    thread_local! {
        static RNG: std::cell::RefCell<Option<FastrandCompat>> = std::cell::RefCell::new(None);
        static THREAD_ID: std::cell::Cell<u64> = std::cell::Cell::new(0);
        static LAST_GENERATION: std::cell::Cell<u64> = std::cell::Cell::new(0);
    }

    RNG.with(|rng_cell| {
        let mut rng_opt = rng_cell.borrow_mut();
        
        // Check if we need to re-initialize due to seed generation change
        let current_generation = SEED_GENERATION.load(Ordering::SeqCst);
        let needs_reinit = LAST_GENERATION.with(|gen| {
            let last = gen.get();
            if last != current_generation {
                gen.set(current_generation);
                // CRITICAL: Reset thread ID when generation changes to ensure reproducibility
                THREAD_ID.with(|id| id.set(0));
                true
            } else {
                false
            }
        });
        
        // Initialize or re-initialize if needed
        if rng_opt.is_none() || needs_reinit {
            let rng = if let Ok(global_seed) = GLOBAL_SEED.read() {
                if let Some(seed) = *global_seed {
                    // Use deterministic seed derived from global seed and thread ID
                    THREAD_ID.with(|id| {
                        let tid = id.get();
                        if tid == 0 {
                            // Generate a unique thread ID using atomic counter
                            let new_id = THREAD_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
                            id.set(new_id);
                            FastrandCompat::with_seed(seed.wrapping_add(new_id))
                        } else {
                            FastrandCompat::with_seed(seed.wrapping_add(tid))
                        }
                    })
                } else {
                    FastrandCompat::new()
                }
            } else {
                FastrandCompat::new()
            };
            *rng_opt = Some(rng);
        }
        
        // Use the RNG without cloning
        f(rng_opt.as_mut().unwrap())
    })
}

/// Create a deterministic secure RNG for testing.
///
/// This should only be used in test code, never in production.
#[cfg(test)]
pub fn test_rng(seed: u64) -> SecureRng {
    SecureRng::with_seed(seed)
}

/// Migration helper: Drop-in replacement for fastrand::Rng.
///
/// This provides API compatibility with fastrand while using secure RNG.
/// All methods properly avoid modulo bias.
#[derive(Clone)]
pub struct FastrandCompat {
    inner: SecureRng,
}

impl std::fmt::Debug for FastrandCompat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastrandCompat")
            .field("inner", &"SecureRng")
            .finish()
    }
}

impl FastrandCompat {
    /// Create new RNG with OS entropy.
    pub fn new() -> Self {
        Self {
            inner: SecureRng::new(),
        }
    }

    /// Create RNG with seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            inner: SecureRng::with_seed(seed),
        }
    }

    /// Generate f64 in [0, 1).
    pub fn f64(&mut self) -> f64 {
        self.inner.f64()
    }

    /// Generate usize in range.
    pub fn usize(&mut self, range: std::ops::Range<usize>) -> usize {
        self.inner.usize(range)
    }

    /// Generate u64 in range (FIXED: no modulo bias).
    pub fn u64(&mut self, range: std::ops::Range<u64>) -> u64 {
        self.inner.u64(range)
    }
}

impl Default for FastrandCompat {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-local secure RNG for high-performance scenarios.
///
/// This provides a thread-local RNG instance that avoids global lock contention.
/// Each thread gets its own independent RNG seeded from OS entropy.
///
/// Use this instead of the global RNG functions when:
/// - Running thousands of parallel simulations
/// - Performance is critical
/// - You don't need centralized audit logging
///
/// NOTE: Audit logs will be distributed across thread-local instances.
/// You'll need to aggregate them if centralized logging is required.
pub struct ThreadLocalRng {
    rng: std::cell::RefCell<SecureRng>,
}

impl ThreadLocalRng {
    /// Get the thread-local RNG instance.
    pub fn with<F, R>(f: F) -> R
    where
        F: FnOnce(&mut SecureRng) -> R,
    {
        // Use the unified thread-local RNG implementation
        with_thread_local_rng(|compat| {
            // Access the inner SecureRng from FastrandCompat
            f(&mut compat.inner)
        })
    }

    /// Generate a random f64 in [0, 1).
    pub fn f64() -> f64 {
        with_thread_local_rng(|rng| rng.f64())
    }

    /// Generate a random usize in range.
    pub fn usize(range: std::ops::Range<usize>) -> usize {
        with_thread_local_rng(|rng| rng.usize(range))
    }

    /// Generate a random u64 in range.
    pub fn u64(range: std::ops::Range<u64>) -> u64 {
        with_thread_local_rng(|rng| rng.u64(range))
    }

    /// Enable audit logging for the thread-local instance.
    pub fn enable_audit() {
        Self::with(|rng| rng.enable_audit())
    }

    /// Get audit log from the thread-local instance.
    pub fn get_audit_log() -> FractalResult<Vec<RngAuditEntry>> {
        Self::with(|rng| rng.get_audit_log())
    }
}

/// Documentation about audit log persistence.
///
/// IMPORTANT: Audit logs are stored entirely in memory with a maximum of 10,000 entries.
/// In case of application crash, all unpersisted logs will be lost.
///
/// For production financial systems, you MUST:
/// 1. Periodically call get_audit_log() or get_global_audit_log()
/// 2. Persist the logs to a database, file, or logging service
/// 3. Implement a recovery mechanism to handle partial log loss
///
/// Example persistence strategy:
/// ```no_run
/// // Run this in a background task every 60 seconds
/// let logs = secure_rng::get_global_audit_log()?;
/// for entry in logs {
///     database.insert_audit_log(entry)?;
/// }
/// ```
pub mod audit_persistence {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secure_rng_determinism() {
        let mut rng1 = SecureRng::with_seed(12345);
        let mut rng2 = SecureRng::with_seed(12345);

        // Should produce same sequence with same seed
        for _ in 0..100 {
            assert_eq!(rng1.f64(), rng2.f64());
        }
    }

    #[test]
    fn test_secure_rng_range() {
        let mut rng = SecureRng::new();

        // Test f64 is in [0, 1)
        for _ in 0..1000 {
            let val = rng.f64();
            assert!(val >= 0.0 && val < 1.0);
        }

        // Test usize range
        for _ in 0..1000 {
            let val = rng.usize(10..20);
            assert!(val >= 10 && val < 20);
        }

        // Test u64 range (no modulo bias)
        for _ in 0..1000 {
            let val = rng.u64(1000..2000);
            assert!(val >= 1000 && val < 2000);
        }
    }

    #[test]
    fn test_audit_logging() {
        let mut rng = SecureRng::with_seed(999);
        rng.enable_audit();

        // Generate some random values
        for _ in 0..10 {
            rng.f64();
        }

        let log = rng.get_audit_log().unwrap();
        assert_eq!(log.len(), 11); // 1 init + 10 operations
        assert_eq!(log[0].operation, "init_with_seed");
        assert!(log.iter().skip(1).all(|entry| entry.operation == "f64"));

        // Check that IDs are unique
        let mut ids: Vec<u64> = log.iter().map(|e| e.operation_id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 11);
    }

    #[test]
    fn test_global_rng() {
        // Test that global RNG works
        let val1 = secure_random_f64().unwrap();
        let val2 = secure_random_f64().unwrap();

        assert!(val1 >= 0.0 && val1 < 1.0);
        assert!(val2 >= 0.0 && val2 < 1.0);
        assert_ne!(val1, val2); // Should be different
    }

    #[test]
    fn test_fastrand_compat() {
        let mut rng = FastrandCompat::with_seed(42);

        // Test basic operations work
        let _ = rng.f64();
        let _ = rng.usize(0..100);
        let _ = rng.u64(1000..2000);
    }

    #[test]
    fn test_audit_log_rotation() {
        let mut rng = SecureRng::new();
        rng.enable_audit();

        // Generate more than 10000 operations
        for _ in 0..15000 {
            rng.f64();
        }

        let log = rng.get_audit_log().unwrap();
        assert_eq!(log.len(), 10000); // Should be capped at 10000
    }

    #[test]
    fn test_seed_recorded_in_audit() {
        let rng = SecureRng::with_seed(42);
        let log = rng.get_audit_log().unwrap();

        // Should have initialization entry
        assert!(!log.is_empty());
        assert_eq!(log[0].operation, "init_with_seed");
        assert_eq!(log[0].seed, Some(42));
    }

    #[test]
    fn test_nanosecond_timestamp_precision() {
        let mut rng = SecureRng::new();
        rng.enable_audit();

        rng.f64();
        std::thread::sleep(std::time::Duration::from_millis(1));
        rng.f64();

        let log = rng.get_audit_log().unwrap();
        assert!(log.len() >= 2);

        // Timestamps should be different at nanosecond precision
        assert_ne!(log[0].timestamp, log[1].timestamp);

        // Timestamp should be reasonable (after year 2020 in nanoseconds)
        let year_2020_nanos = 1_577_836_800_000_000_000u128;
        assert!(log[0].timestamp > year_2020_nanos);
    }

    #[test]
    fn test_thread_local_rng() {
        // Test basic operations
        let val = ThreadLocalRng::f64();
        assert!(val >= 0.0 && val < 1.0);

        let val = ThreadLocalRng::usize(10..20);
        assert!(val >= 10 && val < 20);

        let val = ThreadLocalRng::u64(1000..2000);
        assert!(val >= 1000 && val < 2000);
    }

    #[test]
    fn test_thread_local_independence() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::thread;

        let flag = Arc::new(AtomicBool::new(false));
        let flag_clone = flag.clone();

        // Generate values in parallel threads
        let handle = thread::spawn(move || {
            let mut values = Vec::new();
            for _ in 0..100 {
                values.push(ThreadLocalRng::f64());
            }
            flag_clone.store(true, Ordering::SeqCst);
            values
        });

        let mut main_values = Vec::new();
        for _ in 0..100 {
            main_values.push(ThreadLocalRng::f64());
        }

        let thread_values = handle.join().unwrap();

        // Values should be different (extremely unlikely to match)
        assert_ne!(main_values, thread_values);
    }
}
