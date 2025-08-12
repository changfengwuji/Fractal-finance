//! Deterministic collection types for reproducible financial computations.
//!
//! This module provides collection types that guarantee deterministic behavior
//! across different platforms and execution contexts. This is critical for
//! financial systems where reproducibility is required for auditing and compliance.

use crate::errors::{FractalAnalysisError, FractalResult};
use std::collections::BTreeMap;

/// Type alias for deterministic map with ordered iteration.
///
/// BTreeMap provides O(log n) operations but guarantees consistent iteration order
/// based on key ordering, unlike HashMap which has platform-dependent iteration.
///
/// # Use Cases
/// - Financial calculations requiring reproducible results
/// - Audit trails that must be consistent across systems
/// - Any computation where iteration order affects the output
pub type DeterministicMap<K, V> = BTreeMap<K, V>;

/// Memory-bounded vector with automatic size limits.
///
/// This structure prevents unbounded memory growth, which is critical in
/// production systems to avoid OOM errors and ensure predictable resource usage.
///
/// # Mathematical Properties
/// - Maintains FIFO ordering for insertions
/// - Provides O(1) access and append operations (until limit)
/// - Memory usage is strictly bounded by max_size * sizeof(T)
pub struct BoundedVec<T> {
    data: Vec<T>,
    max_size: usize,
}

impl<T> BoundedVec<T> {
    /// Create a new bounded vector with specified maximum size.
    pub fn new(max_size: usize) -> Self {
        Self {
            data: Vec::with_capacity(max_size.min(1024)), // Pre-allocate sensibly
            max_size,
        }
    }
    
    /// Create with initial capacity hint for better performance.
    pub fn with_capacity(max_size: usize, capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity.min(max_size)),
            max_size,
        }
    }
    
    /// Push a value, failing if the size limit would be exceeded.
    pub fn push(&mut self, value: T) -> FractalResult<()> {
        if self.data.len() >= self.max_size {
            return Err(FractalAnalysisError::NumericalError {
                reason: format!("BoundedVec size limit {} exceeded", self.max_size),
                operation: Some("BoundedVec::push".to_string()),
            });
        }
        self.data.push(value);
        Ok(())
    }
    
    /// Try to extend from an iterator, stopping at the size limit.
    pub fn try_extend<I: IntoIterator<Item = T>>(&mut self, iter: I) -> FractalResult<usize> {
        let mut added = 0;
        for item in iter {
            if self.data.len() >= self.max_size {
                break;
            }
            self.data.push(item);
            added += 1;
        }
        Ok(added)
    }
    
    /// Get an element by index.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }
    
    /// Get a mutable reference to an element.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.get_mut(index)
    }
    
    /// Current number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Get as slice for read-only access.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
    
    /// Get as mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
    
    /// Maximum allowed size.
    pub fn max_size(&self) -> usize {
        self.max_size
    }
    
    /// Remaining capacity before hitting limit.
    pub fn remaining_capacity(&self) -> usize {
        self.max_size.saturating_sub(self.data.len())
    }
    
    /// Clear all elements.
    pub fn clear(&mut self) {
        self.data.clear();
    }
    
    /// Convert to unbounded Vec, consuming self.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T> From<BoundedVec<T>> for Vec<T> {
    fn from(bounded: BoundedVec<T>) -> Self {
        bounded.data
    }
}

/// Deterministic cache with size limits and ordered eviction.
///
/// This cache provides deterministic Least Recently Used (LRU) eviction
/// with consistent behavior across platforms. All operations maintain
/// the invariant that iteration order is deterministic.
///
/// # Complexity
/// - Insert: O(log n) for map + O(n) for access tracking
/// - Get: O(log n) for map + O(n) for access tracking  
/// - Memory: O(n) for n entries
///
/// # Determinism Guarantee
/// Given the same sequence of operations, this cache will always
/// produce the same state and eviction pattern.
pub struct DeterministicCache<K: Ord, V> {
    data: BTreeMap<K, V>,
    max_entries: usize,
    access_order: Vec<K>,
}

impl<K: Ord + Clone, V> DeterministicCache<K, V> {
    /// Create a new deterministic cache with specified capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            data: BTreeMap::new(),
            max_entries,
            access_order: Vec::with_capacity(max_entries),
        }
    }
    
    /// Insert a key-value pair, evicting LRU entry if at capacity.
    pub fn insert(&mut self, key: K, value: V) -> FractalResult<Option<V>> {
        // Check if we need to evict
        if self.data.len() >= self.max_entries && !self.data.contains_key(&key) {
            // Evict oldest entry (deterministic LRU)
            if let Some(oldest_key) = self.access_order.first().cloned() {
                self.data.remove(&oldest_key);
                self.access_order.remove(0);
            }
        }
        
        // Update access order
        if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
            self.access_order.remove(pos);
        }
        self.access_order.push(key.clone());
        
        Ok(self.data.insert(key, value))
    }
    
    /// Get a value, updating its access time.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.data.contains_key(key) {
            // Update access order
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                let key_clone = self.access_order[pos].clone();
                self.access_order.remove(pos);
                self.access_order.push(key_clone);
            }
        }
        self.data.get(key)
    }
    
    /// Get a value without updating access time (peek).
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.data.get(key)
    }
    
    /// Check if a key exists.
    pub fn contains_key(&self, key: &K) -> bool {
        self.data.contains_key(key)
    }
    
    /// Remove a specific key.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            self.access_order.remove(pos);
        }
        self.data.remove(key)
    }
    
    /// Current number of entries.
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Maximum capacity.
    pub fn capacity(&self) -> usize {
        self.max_entries
    }
    
    /// Clear all entries.
    pub fn clear(&mut self) {
        self.data.clear();
        self.access_order.clear();
    }
    
    /// Iterate over entries in key order (deterministic).
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.data.iter()
    }
    
    /// Get the least recently used key.
    pub fn lru_key(&self) -> Option<&K> {
        self.access_order.first()
    }
    
    /// Get the most recently used key.
    pub fn mru_key(&self) -> Option<&K> {
        self.access_order.last()
    }
}

/// Complexity guard to prevent algorithmic complexity attacks.
///
/// This guard tracks iteration count and fails if a computation exceeds
/// reasonable bounds, preventing both accidental infinite loops and
/// deliberate denial-of-service attacks.
///
/// # Mathematical Foundation
/// Many financial algorithms have known complexity bounds. This guard
/// ensures we stay within those bounds, failing fast if something goes wrong.
pub struct ComplexityGuard {
    max_iterations: usize,
    current_iterations: usize,
    operation: String,
}

impl ComplexityGuard {
    /// Create a new complexity guard for an operation.
    pub fn new(max_iterations: usize, operation: impl Into<String>) -> Self {
        Self {
            max_iterations,
            current_iterations: 0,
            operation: operation.into(),
        }
    }
    
    /// Check and increment iteration count.
    pub fn check_iteration(&mut self) -> FractalResult<()> {
        self.current_iterations += 1;
        if self.current_iterations > self.max_iterations {
            return Err(FractalAnalysisError::NumericalError {
                reason: format!(
                    "Operation '{}' exceeded maximum iterations: {} > {}",
                    self.operation, self.current_iterations, self.max_iterations
                ),
                operation: Some(self.operation.clone()),
            });
        }
        Ok(())
    }
    
    /// Batch check for multiple iterations at once.
    pub fn check_iterations(&mut self, count: usize) -> FractalResult<()> {
        self.current_iterations += count;
        if self.current_iterations > self.max_iterations {
            return Err(FractalAnalysisError::NumericalError {
                reason: format!(
                    "Operation '{}' exceeded maximum iterations: {} > {}",
                    self.operation, self.current_iterations, self.max_iterations
                ),
                operation: Some(self.operation.clone()),
            });
        }
        Ok(())
    }
    
    /// Get current iteration count.
    pub fn iterations(&self) -> usize {
        self.current_iterations
    }
    
    /// Get remaining iterations before limit.
    pub fn remaining(&self) -> usize {
        self.max_iterations.saturating_sub(self.current_iterations)
    }
    
    /// Reset the counter.
    pub fn reset(&mut self) {
        self.current_iterations = 0;
    }
    
    /// Check if we're close to the limit (within 10%).
    pub fn is_near_limit(&self) -> bool {
        self.current_iterations as f64 > 0.9 * self.max_iterations as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bounded_vec() {
        let mut vec = BoundedVec::new(3);
        
        // Should accept up to limit
        assert!(vec.push(1).is_ok());
        assert!(vec.push(2).is_ok());
        assert!(vec.push(3).is_ok());
        assert_eq!(vec.len(), 3);
        
        // Should reject beyond limit
        assert!(vec.push(4).is_err());
        assert_eq!(vec.len(), 3);
        
        // Test access
        assert_eq!(vec.get(0), Some(&1));
        assert_eq!(vec.get(3), None);
        
        // Test remaining capacity
        assert_eq!(vec.remaining_capacity(), 0);
    }
    
    #[test]
    fn test_deterministic_cache() {
        let mut cache = DeterministicCache::new(3);
        
        // Insert within capacity
        assert!(cache.insert(1, "a").unwrap().is_none());
        assert!(cache.insert(2, "b").unwrap().is_none());
        assert!(cache.insert(3, "c").unwrap().is_none());
        assert_eq!(cache.len(), 3);
        
        // Access middle element (changes LRU order)
        assert_eq!(cache.get(&2), Some(&"b"));
        
        // Insert beyond capacity should evict LRU (which is now 1)
        assert!(cache.insert(4, "d").unwrap().is_none());
        assert_eq!(cache.len(), 3);
        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert!(cache.contains_key(&4));
        
        // Test peek doesn't affect order
        // After accessing 2 and inserting 4, the LRU order is [3, 2, 4]
        // Peek at 3 should NOT change this order
        assert_eq!(cache.peek(&3), Some(&"c"));
        
        // Insert 5 should evict the LRU (which is still 3 since peek didn't update order)
        cache.insert(5, "e").unwrap();
        assert!(!cache.contains_key(&3)); // 3 should be evicted as it's still the LRU
        assert!(cache.contains_key(&2));  // 2 should still be there
        assert!(cache.contains_key(&4));  // 4 should still be there
        assert!(cache.contains_key(&5));  // 5 was just inserted
    }
    
    #[test]
    fn test_complexity_guard() {
        let mut guard = ComplexityGuard::new(100, "test_operation");
        
        // Should allow up to limit
        for _ in 0..100 {
            assert!(guard.check_iteration().is_ok());
        }
        assert_eq!(guard.iterations(), 100);
        assert_eq!(guard.remaining(), 0);
        assert!(guard.is_near_limit());
        
        // Should fail beyond limit
        assert!(guard.check_iteration().is_err());
        
        // Test batch check
        guard.reset();
        assert!(guard.check_iterations(50).is_ok());
        assert_eq!(guard.iterations(), 50);
        assert!(guard.check_iterations(51).is_err()); // Would exceed
    }
    
    #[test]
    fn test_deterministic_map_ordering() {
        let mut map: DeterministicMap<i32, &str> = DeterministicMap::new();
        
        // Insert in random order
        map.insert(3, "three");
        map.insert(1, "one");
        map.insert(4, "four");
        map.insert(2, "two");
        
        // Iteration should always be in key order
        let keys: Vec<i32> = map.keys().copied().collect();
        assert_eq!(keys, vec![1, 2, 3, 4]);
    }
}