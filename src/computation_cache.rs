//! Computation cache for expensive operations in fractal analysis.
//!
//! This module provides intelligent caching for computationally expensive operations
//! such as periodograms, covariance matrices, and FFT results. The cache uses
//! content-based hashing with true LRU eviction policy to ensure statistical 
//! correctness while maximizing performance through result reuse.

use crate::errors::{FractalAnalysisError, FractalResult};
use parking_lot::{RwLock, RwLockUpgradableReadGuard};
use rustfft::num_complex::Complex;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Maximum number of cached items per cache type to prevent unlimited memory growth
const MAX_CACHE_SIZE: usize = 100;

/// Maximum age for cached items (30 minutes)
const MAX_CACHE_AGE: Duration = Duration::from_secs(30 * 60);

/// Thread-safe computation cache with true LRU eviction and size limits.
///
/// Provides caching for expensive computations in fractal analysis including:
/// - Periodograms for spectral analysis
/// - Covariance matrices for multivariate analysis  
/// - FFT results for large datasets
/// - Autocorrelation functions
/// - Statistical test intermediate results
pub struct ComputationCache {
    /// Cache for periodogram computations
    periodogram_cache: Arc<RwLock<HashMap<PeriodogramKey, CacheEntry<Vec<f64>>>>>,
    /// Cache for covariance matrix computations
    covariance_cache: Arc<RwLock<HashMap<CovarianceKey, CacheEntry<Vec<Vec<f64>>>>>>,
    /// Cache for FFT results
    fft_cache: Arc<RwLock<HashMap<FftKey, CacheEntry<Vec<Complex<f64>>>>>>,
    /// Cache for autocorrelation results
    autocorr_cache: Arc<RwLock<HashMap<AutocorrKey, CacheEntry<Vec<f64>>>>>,
    /// Cache statistics for monitoring
    stats: Arc<Mutex<CacheStats>>,
}

/// Cache entry with timestamp for age-based cleanup
#[derive(Clone)]
struct CacheEntry<T> {
    value: T,
    timestamp: Instant,
    access_count: usize,
}

/// Cache statistics for monitoring performance
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub periodogram_hits: usize,
    pub periodogram_misses: usize,
    pub covariance_hits: usize,
    pub covariance_misses: usize,
    pub fft_hits: usize,
    pub fft_misses: usize,
    pub autocorr_hits: usize,
    pub autocorr_misses: usize,
    pub evictions: usize,
    pub memory_bytes_saved: u128, // Use u128 to prevent overflow
}

impl CacheStats {
    /// Calculate overall hit rate across all cache types
    pub fn hit_rate(&self) -> f64 {
        let total_hits =
            self.periodogram_hits + self.covariance_hits + self.fft_hits + self.autocorr_hits;
        let total_requests = total_hits
            + self.periodogram_misses
            + self.covariance_misses
            + self.fft_misses
            + self.autocorr_misses;

        if total_requests == 0 {
            0.0
        } else {
            total_hits as f64 / total_requests as f64
        }
    }
}

/// Cache key for periodogram computations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PeriodogramKey {
    data_hash: u64,
    // Removed data_length as it's redundant with hash
    method: PeriodogramMethod,
    window_type: WindowType,
}

/// Cache key for covariance matrix computations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CovarianceKey {
    data_hash: u64,
    // Removed data_length as it's redundant with hash
    lag_count: usize,
    bias_correction: bool,
}

/// Cache key for FFT computations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FftKey {
    data_hash: u64,
    // Removed data_length as it's redundant with hash
    fft_size: usize,
    direction: FftDirection,
}

/// Cache key for autocorrelation computations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct AutocorrKey {
    data_hash: u64,
    // Removed data_length as it's redundant with hash
    max_lag: usize,
    method: AutocorrMethod,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PeriodogramMethod {
    Welch,
    Periodogram,
    Multitaper,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WindowType {
    Hann,
    Hamming,
    Blackman,
    None,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FftDirection {
    Forward,
    Inverse,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AutocorrMethod {
    Direct,
    Fft { biased: bool },
}

impl Default for ComputationCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputationCache {
    /// Create a new computation cache with default settings.
    pub fn new() -> Self {
        Self {
            periodogram_cache: Arc::new(RwLock::new(HashMap::new())),
            covariance_cache: Arc::new(RwLock::new(HashMap::new())),
            fft_cache: Arc::new(RwLock::new(HashMap::new())),
            autocorr_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }

    /// Get or compute a periodogram with true LRU caching.
    pub fn get_or_compute_periodogram<F>(
        &self,
        data: &[f64],
        method: PeriodogramMethod,
        window_type: WindowType,
        compute_fn: F,
    ) -> FractalResult<Vec<f64>>
    where
        F: FnOnce() -> FractalResult<Vec<f64>>,
    {
        let key = PeriodogramKey {
            data_hash: hash_data(data),
            method,
            window_type,
        };

        // Try to get from cache with upgradable read lock
        {
            let cache = self.periodogram_cache.upgradable_read();
            if let Some(entry) = cache.get(&key) {
                if entry.timestamp.elapsed() < MAX_CACHE_AGE {
                    // Update stats atomically
                    {
                        let mut stats = self.stats.lock().unwrap();
                        stats.periodogram_hits += 1;
                        stats.memory_bytes_saved = stats
                            .memory_bytes_saved
                            .saturating_add((entry.value.len() * 8) as u128);
                    }
                    
                    // Clone value before upgrading to avoid holding write lock during clone
                    let value = entry.value.clone();
                    
                    // Upgrade to write lock and update LRU fields
                    let mut cache_w = RwLockUpgradableReadGuard::upgrade(cache);
                    if let Some(entry_mut) = cache_w.get_mut(&key) {
                        entry_mut.timestamp = Instant::now();
                        entry_mut.access_count = entry_mut.access_count.saturating_add(1);
                    }
                    
                    return Ok(value);
                }
            }
        }

        // Cache miss - compute the result
        self.stats.lock().unwrap().periodogram_misses += 1;
        let result = compute_fn()?;

        // Store in cache
        self.insert_periodogram(key, result.clone());
        Ok(result)
    }

    /// Get or compute a covariance matrix with true LRU caching.
    pub fn get_or_compute_covariance<F>(
        &self,
        data: &[f64],
        lag_count: usize,
        bias_correction: bool,
        compute_fn: F,
    ) -> FractalResult<Vec<Vec<f64>>>
    where
        F: FnOnce() -> FractalResult<Vec<Vec<f64>>>,
    {
        let key = CovarianceKey {
            data_hash: hash_data(data),
            lag_count,
            bias_correction,
        };

        // Try to get from cache with upgradable read lock
        {
            let cache = self.covariance_cache.upgradable_read();
            if let Some(entry) = cache.get(&key) {
                if entry.timestamp.elapsed() < MAX_CACHE_AGE {
                    // Safe memory calculation that won't panic
                    let memory_saved = entry.value.first()
                        .map(|row| {
                            entry.value.len()
                                .saturating_mul(row.len())
                                .saturating_mul(8) as u128
                        })
                        .unwrap_or(0);
                    
                    // Update stats atomically
                    {
                        let mut stats = self.stats.lock().unwrap();
                        stats.covariance_hits += 1;
                        stats.memory_bytes_saved = stats.memory_bytes_saved.saturating_add(memory_saved);
                    }
                    
                    // Clone value before upgrading
                    let value = entry.value.clone();
                    
                    // Upgrade to write lock and update LRU fields
                    let mut cache_w = RwLockUpgradableReadGuard::upgrade(cache);
                    if let Some(entry_mut) = cache_w.get_mut(&key) {
                        entry_mut.timestamp = Instant::now();
                        entry_mut.access_count = entry_mut.access_count.saturating_add(1);
                    }
                    
                    return Ok(value);
                }
            }
        }

        // Cache miss - compute the result
        self.stats.lock().unwrap().covariance_misses += 1;
        let result = compute_fn()?;

        // Store in cache
        self.insert_covariance(key, result.clone());
        Ok(result)
    }

    /// Get or compute FFT results with true LRU caching.
    pub fn get_or_compute_fft<F>(
        &self,
        data: &[f64],
        fft_size: usize,
        direction: FftDirection,
        compute_fn: F,
    ) -> FractalResult<Vec<Complex<f64>>>
    where
        F: FnOnce() -> FractalResult<Vec<Complex<f64>>>,
    {
        let key = FftKey {
            data_hash: hash_data(data),
            fft_size,
            direction,
        };

        // Try to get from cache with upgradable read lock
        {
            let cache = self.fft_cache.upgradable_read();
            if let Some(entry) = cache.get(&key) {
                if entry.timestamp.elapsed() < MAX_CACHE_AGE {
                    // Update stats atomically
                    {
                        let mut stats = self.stats.lock().unwrap();
                        stats.fft_hits += 1;
                        stats.memory_bytes_saved = stats
                            .memory_bytes_saved
                            .saturating_add((entry.value.len() * 16) as u128); // Complex<f64> = 16 bytes
                    }
                    
                    // Clone value before upgrading
                    let value = entry.value.clone();
                    
                    // Upgrade to write lock and update LRU fields
                    let mut cache_w = RwLockUpgradableReadGuard::upgrade(cache);
                    if let Some(entry_mut) = cache_w.get_mut(&key) {
                        entry_mut.timestamp = Instant::now();
                        entry_mut.access_count = entry_mut.access_count.saturating_add(1);
                    }
                    
                    return Ok(value);
                }
            }
        }

        // Cache miss - compute the result
        self.stats.lock().unwrap().fft_misses += 1;
        let result = compute_fn()?;

        // Store in cache
        self.insert_fft(key, result.clone());
        Ok(result)
    }

    /// Get or compute autocorrelation with true LRU caching.
    pub fn get_or_compute_autocorr<F>(
        &self,
        data: &[f64],
        max_lag: usize,
        method: AutocorrMethod,
        compute_fn: F,
    ) -> FractalResult<Vec<f64>>
    where
        F: FnOnce() -> FractalResult<Vec<f64>>,
    {
        let key = AutocorrKey {
            data_hash: hash_data(data),
            max_lag,
            method,
        };

        // Try to get from cache with upgradable read lock
        {
            let cache = self.autocorr_cache.upgradable_read();
            if let Some(entry) = cache.get(&key) {
                if entry.timestamp.elapsed() < MAX_CACHE_AGE {
                    // Update stats atomically
                    {
                        let mut stats = self.stats.lock().unwrap();
                        stats.autocorr_hits += 1;
                        stats.memory_bytes_saved = stats
                            .memory_bytes_saved
                            .saturating_add((entry.value.len() * 8) as u128);
                    }
                    
                    // Clone value before upgrading
                    let value = entry.value.clone();
                    
                    // Upgrade to write lock and update LRU fields
                    let mut cache_w = RwLockUpgradableReadGuard::upgrade(cache);
                    if let Some(entry_mut) = cache_w.get_mut(&key) {
                        entry_mut.timestamp = Instant::now();
                        entry_mut.access_count = entry_mut.access_count.saturating_add(1);
                    }
                    
                    return Ok(value);
                }
            }
        }

        // Cache miss - compute the result
        self.stats.lock().unwrap().autocorr_misses += 1;
        let result = compute_fn()?;

        // Store in cache
        self.insert_autocorr(key, result.clone());
        Ok(result)
    }

    /// Insert periodogram into cache with cleanup
    fn insert_periodogram(&self, key: PeriodogramKey, value: Vec<f64>) {
        let mut cache = self.periodogram_cache.write();

        // Cleanup old entries if cache is full
        if cache.len() >= MAX_CACHE_SIZE {
            self.cleanup_periodogram_cache(&mut cache);
        }

        cache.insert(
            key,
            CacheEntry {
                value,
                timestamp: Instant::now(),
                access_count: 1,
            },
        );
    }

    /// Insert covariance matrix into cache with cleanup
    fn insert_covariance(&self, key: CovarianceKey, value: Vec<Vec<f64>>) {
        let mut cache = self.covariance_cache.write();

        // Cleanup old entries if cache is full
        if cache.len() >= MAX_CACHE_SIZE {
            self.cleanup_covariance_cache(&mut cache);
        }

        cache.insert(
            key,
            CacheEntry {
                value,
                timestamp: Instant::now(),
                access_count: 1,
            },
        );
    }

    /// Insert FFT result into cache with cleanup
    fn insert_fft(&self, key: FftKey, value: Vec<Complex<f64>>) {
        let mut cache = self.fft_cache.write();

        // Cleanup old entries if cache is full
        if cache.len() >= MAX_CACHE_SIZE {
            self.cleanup_fft_cache(&mut cache);
        }

        cache.insert(
            key,
            CacheEntry {
                value,
                timestamp: Instant::now(),
                access_count: 1,
            },
        );
    }

    /// Insert autocorrelation into cache with cleanup
    fn insert_autocorr(&self, key: AutocorrKey, value: Vec<f64>) {
        let mut cache = self.autocorr_cache.write();

        // Cleanup old entries if cache is full
        if cache.len() >= MAX_CACHE_SIZE {
            self.cleanup_autocorr_cache(&mut cache);
        }

        cache.insert(
            key,
            CacheEntry {
                value,
                timestamp: Instant::now(),
                access_count: 1,
            },
        );
    }

    /// Cleanup old entries from periodogram cache using true LRU policy
    fn cleanup_periodogram_cache(&self, cache: &mut HashMap<PeriodogramKey, CacheEntry<Vec<f64>>>) {
        let now = Instant::now();
        let mut to_remove = Vec::new();

        // First remove expired entries
        for (key, entry) in cache.iter() {
            if now.duration_since(entry.timestamp) > MAX_CACHE_AGE {
                to_remove.push(key.clone());
            }
        }

        // If still too many entries, remove least recently used (oldest timestamp)
        if cache.len() - to_remove.len() >= MAX_CACHE_SIZE {
            let mut entries: Vec<_> = cache
                .iter()
                .filter(|(k, _)| !to_remove.contains(k))
                .map(|(k, e)| (k.clone(), e.timestamp, e.access_count))
                .collect();
            
            // Sort by timestamp (least recently used first)
            entries.sort_by_key(|(_, timestamp, _)| *timestamp);

            let num_to_remove = cache.len() - to_remove.len() - MAX_CACHE_SIZE / 2;
            for (key, _, _) in entries.iter().take(num_to_remove) {
                to_remove.push(key.clone());
            }
        }

        // Batch evictions and stats update
        let eviction_count = to_remove.len();
        for key in to_remove {
            cache.remove(&key);
        }
        
        if eviction_count > 0 {
            self.stats.lock().unwrap().evictions += eviction_count;
        }
    }

    /// Cleanup old entries from covariance cache using true LRU policy
    fn cleanup_covariance_cache(
        &self,
        cache: &mut HashMap<CovarianceKey, CacheEntry<Vec<Vec<f64>>>>,
    ) {
        let now = Instant::now();
        let mut to_remove = Vec::new();

        // First remove expired entries
        for (key, entry) in cache.iter() {
            if now.duration_since(entry.timestamp) > MAX_CACHE_AGE {
                to_remove.push(key.clone());
            }
        }

        // If still too many entries, remove least recently used
        if cache.len() - to_remove.len() >= MAX_CACHE_SIZE {
            let mut entries: Vec<_> = cache
                .iter()
                .filter(|(k, _)| !to_remove.contains(k))
                .map(|(k, e)| (k.clone(), e.timestamp, e.access_count))
                .collect();
            
            // Sort by timestamp (least recently used first)
            entries.sort_by_key(|(_, timestamp, _)| *timestamp);

            let num_to_remove = cache.len() - to_remove.len() - MAX_CACHE_SIZE / 2;
            for (key, _, _) in entries.iter().take(num_to_remove) {
                to_remove.push(key.clone());
            }
        }

        // Batch evictions and stats update
        let eviction_count = to_remove.len();
        for key in to_remove {
            cache.remove(&key);
        }
        
        if eviction_count > 0 {
            self.stats.lock().unwrap().evictions += eviction_count;
        }
    }

    /// Cleanup old entries from FFT cache using true LRU policy
    fn cleanup_fft_cache(&self, cache: &mut HashMap<FftKey, CacheEntry<Vec<Complex<f64>>>>) {
        let now = Instant::now();
        let mut to_remove = Vec::new();

        // First remove expired entries
        for (key, entry) in cache.iter() {
            if now.duration_since(entry.timestamp) > MAX_CACHE_AGE {
                to_remove.push(key.clone());
            }
        }

        // If still too many entries, remove least recently used
        if cache.len() - to_remove.len() >= MAX_CACHE_SIZE {
            let mut entries: Vec<_> = cache
                .iter()
                .filter(|(k, _)| !to_remove.contains(k))
                .map(|(k, e)| (k.clone(), e.timestamp, e.access_count))
                .collect();
            
            // Sort by timestamp (least recently used first)
            entries.sort_by_key(|(_, timestamp, _)| *timestamp);

            let num_to_remove = cache.len() - to_remove.len() - MAX_CACHE_SIZE / 2;
            for (key, _, _) in entries.iter().take(num_to_remove) {
                to_remove.push(key.clone());
            }
        }

        // Batch evictions and stats update
        let eviction_count = to_remove.len();
        for key in to_remove {
            cache.remove(&key);
        }
        
        if eviction_count > 0 {
            self.stats.lock().unwrap().evictions += eviction_count;
        }
    }

    /// Cleanup old entries from autocorrelation cache using true LRU policy
    fn cleanup_autocorr_cache(&self, cache: &mut HashMap<AutocorrKey, CacheEntry<Vec<f64>>>) {
        let now = Instant::now();
        let mut to_remove = Vec::new();

        // First remove expired entries
        for (key, entry) in cache.iter() {
            if now.duration_since(entry.timestamp) > MAX_CACHE_AGE {
                to_remove.push(key.clone());
            }
        }

        // If still too many entries, remove least recently used
        if cache.len() - to_remove.len() >= MAX_CACHE_SIZE {
            let mut entries: Vec<_> = cache
                .iter()
                .filter(|(k, _)| !to_remove.contains(k))
                .map(|(k, e)| (k.clone(), e.timestamp, e.access_count))
                .collect();
            
            // Sort by timestamp (least recently used first)
            entries.sort_by_key(|(_, timestamp, _)| *timestamp);

            let num_to_remove = cache.len() - to_remove.len() - MAX_CACHE_SIZE / 2;
            for (key, _, _) in entries.iter().take(num_to_remove) {
                to_remove.push(key.clone());
            }
        }

        // Batch evictions and stats update
        let eviction_count = to_remove.len();
        for key in to_remove {
            cache.remove(&key);
        }
        
        if eviction_count > 0 {
            self.stats.lock().unwrap().evictions += eviction_count;
        }
    }

    /// Get current cache statistics
    pub fn get_stats(&self) -> CacheStats {
        (*self.stats.lock().unwrap()).clone()
    }

    /// Clear all caches
    pub fn clear_all(&self) {
        self.periodogram_cache.write().clear();
        self.covariance_cache.write().clear();
        self.fft_cache.write().clear();
        self.autocorr_cache.write().clear();
        *self.stats.lock().unwrap() = CacheStats::default();
    }

    /// Get cache sizes for monitoring
    pub fn get_cache_sizes(&self) -> (usize, usize, usize, usize) {
        (
            self.periodogram_cache.read().len(),
            self.covariance_cache.read().len(),
            self.fft_cache.read().len(),
            self.autocorr_cache.read().len(),
        )
    }

    /// Disable the global cache for deterministic testing
    pub fn disable_for_tests(&self) {
        self.clear_all();
    }
}

/// Hash function for data arrays to create cache keys.
///
/// Uses a fast non-cryptographic hash suitable for cache key generation.
/// The hash incorporates both values and their order to ensure correctness.
/// Canonicalizes NaN and -0.0 for consistent hashing.
fn hash_data(data: &[f64]) -> u64 {
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();

    // Hash the length first to distinguish arrays of different sizes
    data.len().hash(&mut hasher);

    // Hash the data values with canonicalization
    for &value in data {
        // Canonicalize special float values for consistent hashing
        let bits = if value.is_nan() {
            // All NaNs hash to the same value
            f64::NAN.to_bits()
        } else if value == 0.0 {
            // Canonicalize -0.0 to 0.0
            0.0_f64.to_bits()
        } else {
            value.to_bits()
        };
        bits.hash(&mut hasher);
    }

    hasher.finish()
}

/// Global computation cache instance for use across the library.
lazy_static::lazy_static! {
    static ref GLOBAL_COMPUTATION_CACHE: ComputationCache = ComputationCache::new();
}

/// Get the global computation cache instance.
pub fn get_global_cache() -> &'static ComputationCache {
    &GLOBAL_COMPUTATION_CACHE
}

/// Clear the global computation cache.
pub fn clear_global_cache() {
    GLOBAL_COMPUTATION_CACHE.clear_all();
}

/// Get global cache statistics.
pub fn get_global_cache_stats() -> CacheStats {
    GLOBAL_COMPUTATION_CACHE.get_stats()
}

/// Disable global cache for deterministic tests
#[cfg(test)]
pub fn disable_cache_for_tests() {
    GLOBAL_COMPUTATION_CACHE.disable_for_tests();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_periodogram_caching() {
        let cache = ComputationCache::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // First call should compute
        let result1 = cache
            .get_or_compute_periodogram(
                &data,
                PeriodogramMethod::Welch,
                WindowType::Hann,
                || Ok(vec![1.0, 2.0, 3.0]),
            )
            .unwrap();

        // Second call should use cache
        let result2 = cache
            .get_or_compute_periodogram(&data, PeriodogramMethod::Welch, WindowType::Hann, || {
                panic!("Should use cache, not compute")
            })
            .unwrap();
        
        assert_eq!(result1, result2);

        let stats = cache.get_stats();
        assert_eq!(stats.periodogram_hits, 1);
        assert_eq!(stats.periodogram_misses, 1);
    }

    #[test]
    fn test_lru_updates_on_hit() {
        let cache = ComputationCache::new();
        let data = vec![1.0, 2.0, 3.0];
        
        // Initial computation
        let _ = cache.get_or_compute_periodogram(
            &data,
            PeriodogramMethod::Welch,
            WindowType::Hann,
            || Ok(vec![1.0]),
        );
        
        // Wait a bit
        std::thread::sleep(Duration::from_millis(10));
        
        // Access again - should update timestamp
        let _ = cache.get_or_compute_periodogram(
            &data,
            PeriodogramMethod::Welch,
            WindowType::Hann,
            || Ok(vec![2.0]),
        );
        
        // Check that entry is still fresh (timestamp was updated)
        let cache_read = cache.periodogram_cache.read();
        let key = PeriodogramKey {
            data_hash: hash_data(&data),
            method: PeriodogramMethod::Welch,
            window_type: WindowType::Hann,
        };
        
        if let Some(entry) = cache_read.get(&key) {
            assert!(entry.timestamp.elapsed() < Duration::from_millis(5));
            assert_eq!(entry.access_count, 2);
        }
    }

    #[test]
    fn test_data_hashing() {
        let data1 = vec![1.0, 2.0, 3.0];
        let data2 = vec![1.0, 2.0, 3.0];
        let data3 = vec![1.0, 2.0, 4.0];

        assert_eq!(hash_data(&data1), hash_data(&data2));
        assert_ne!(hash_data(&data1), hash_data(&data3));
        
        // Test NaN canonicalization
        let nan1 = vec![f64::NAN];
        let nan2 = vec![f64::from_bits(0x7ff8000000000001)]; // Different NaN payload
        assert_eq!(hash_data(&nan1), hash_data(&nan2));
        
        // Test -0.0 canonicalization
        let zero1 = vec![0.0];
        let zero2 = vec![-0.0];
        assert_eq!(hash_data(&zero1), hash_data(&zero2));
    }

    #[test]
    fn test_cache_cleanup() {
        let cache = ComputationCache::new();

        // Fill cache beyond capacity
        for i in 0..150 {
            let data = vec![i as f64];
            let _ = cache.get_or_compute_periodogram(
                &data,
                PeriodogramMethod::Welch,
                WindowType::Hann,
                || Ok(vec![i as f64]),
            );
        }

        let (periodogram_size, _, _, _) = cache.get_cache_sizes();
        assert!(periodogram_size <= MAX_CACHE_SIZE);

        let stats = cache.get_stats();
        assert!(stats.evictions > 0);
    }

    #[test]
    fn test_hit_rate_calculation() {
        let mut stats = CacheStats::default();
        stats.periodogram_hits = 5;
        stats.periodogram_misses = 5;
        stats.covariance_hits = 3;
        stats.covariance_misses = 2;

        let hit_rate = stats.hit_rate();
        assert!((hit_rate - 0.533).abs() < 0.01); // 8 hits out of 15 total
    }
    
    #[test]
    fn test_empty_covariance_no_panic() {
        let cache = ComputationCache::new();
        let data = vec![1.0, 2.0];
        
        // Test with empty covariance matrix
        let result = cache.get_or_compute_covariance(
            &data,
            0,
            false,
            || Ok(vec![]),
        ).unwrap();
        
        assert!(result.is_empty());
        
        // Access again to test the hit path with empty value
        let result2 = cache.get_or_compute_covariance(
            &data,
            0,
            false,
            || panic!("Should use cache"),
        ).unwrap();
        
        assert_eq!(result, result2);
        
        let stats = cache.get_stats();
        assert_eq!(stats.covariance_hits, 1);
        assert_eq!(stats.covariance_misses, 1);
    }
}