//! Audit trail and transaction logging for financial computations.
//!
//! This module provides comprehensive auditing capabilities required for
//! financial systems, including transaction logging, operation tracking,
//! and compliance reporting. All operations are designed to be thread-safe
//! and maintain chronological ordering.

use crate::errors::{FractalAnalysisError, FractalResult};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Transaction log entry for audit trail.
///
/// Each log entry captures a complete snapshot of a computation,
/// including inputs, outputs, timing, and checksums for verification.
#[derive(Clone, Debug)]
pub struct TransactionLog {
    /// Unique identifier for this transaction
    pub id: u64,
    /// Unix timestamp in milliseconds
    pub timestamp: u64,
    /// Operation name or identifier
    pub operation: String,
    /// Checksum of input data for verification
    pub input_checksum: u64,
    /// Checksum of output data for verification
    pub output_checksum: u64,
    /// Execution duration in milliseconds
    pub duration_ms: u64,
    /// Optional metadata as key-value pairs
    pub metadata: Vec<(String, String)>,
}

impl TransactionLog {
    /// Create a new transaction log entry.
    pub fn new(
        id: u64,
        operation: String,
        input_checksum: u64,
        output_checksum: u64,
        duration_ms: u64,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        Self {
            id,
            timestamp,
            operation,
            input_checksum,
            output_checksum,
            duration_ms,
            metadata: Vec::new(),
        }
    }
    
    /// Add metadata to the log entry.
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.push((key, value));
        self
    }
    
    /// Format as a single-line audit log entry.
    pub fn to_audit_string(&self) -> String {
        let metadata_str = if self.metadata.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> = self.metadata
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            format!(" [{}]", pairs.join(", "))
        };
        
        format!(
            "TX#{:08x} {} op={} in={:016x} out={:016x} {}ms{}",
            self.id,
            self.timestamp,
            self.operation,
            self.input_checksum,
            self.output_checksum,
            self.duration_ms,
            metadata_str
        )
    }
}

/// Thread-safe transaction logger with automatic rotation.
///
/// This logger maintains an append-only audit trail with automatic
/// size management through log rotation. All operations are thread-safe
/// and maintain strict chronological ordering.
pub struct TransactionLogger {
    logs: Arc<RwLock<Vec<TransactionLog>>>,
    max_logs: usize,
    transaction_counter: Arc<RwLock<u64>>,
}

impl TransactionLogger {
    /// Create a new transaction logger with specified capacity.
    pub fn new(max_logs: usize) -> Self {
        Self {
            logs: Arc::new(RwLock::new(Vec::with_capacity(max_logs))),
            max_logs,
            transaction_counter: Arc::new(RwLock::new(0)),
        }
    }
    
    /// Log a transaction, returning its unique ID.
    pub fn log_transaction(
        &self,
        operation: String,
        input_checksum: u64,
        output_checksum: u64,
        duration_ms: u64,
    ) -> FractalResult<u64> {
        // Get unique transaction ID
        let id = self.next_transaction_id()?;
        
        // Create log entry
        let log = TransactionLog::new(
            id,
            operation,
            input_checksum,
            output_checksum,
            duration_ms,
        );
        
        // Append to log with rotation
        self.append_log(log)?;
        
        Ok(id)
    }
    
    /// Log a transaction with metadata.
    pub fn log_transaction_with_metadata(
        &self,
        operation: String,
        input_checksum: u64,
        output_checksum: u64,
        duration_ms: u64,
        metadata: Vec<(String, String)>,
    ) -> FractalResult<u64> {
        let id = self.next_transaction_id()?;
        
        let mut log = TransactionLog::new(
            id,
            operation,
            input_checksum,
            output_checksum,
            duration_ms,
        );
        log.metadata = metadata;
        
        self.append_log(log)?;
        
        Ok(id)
    }
    
    /// Get a copy of all current logs.
    pub fn get_logs(&self) -> FractalResult<Vec<TransactionLog>> {
        let logs = self.logs.read().map_err(|_| {
            FractalAnalysisError::ConcurrencyError {
                resource: "transaction_logs".to_string(),
            }
        })?;
        Ok(logs.clone())
    }
    
    /// Get logs for a specific operation.
    pub fn get_logs_for_operation(&self, operation: &str) -> FractalResult<Vec<TransactionLog>> {
        let logs = self.logs.read().map_err(|_| {
            FractalAnalysisError::ConcurrencyError {
                resource: "transaction_logs".to_string(),
            }
        })?;
        
        Ok(logs
            .iter()
            .filter(|log| log.operation == operation)
            .cloned()
            .collect())
    }
    
    /// Get logs within a time range.
    pub fn get_logs_in_range(&self, start_ms: u64, end_ms: u64) -> FractalResult<Vec<TransactionLog>> {
        let logs = self.logs.read().map_err(|_| {
            FractalAnalysisError::ConcurrencyError {
                resource: "transaction_logs".to_string(),
            }
        })?;
        
        Ok(logs
            .iter()
            .filter(|log| log.timestamp >= start_ms && log.timestamp <= end_ms)
            .cloned()
            .collect())
    }
    
    /// Clear all logs.
    pub fn clear(&self) -> FractalResult<()> {
        let mut logs = self.logs.write().map_err(|_| {
            FractalAnalysisError::ConcurrencyError {
                resource: "transaction_logs".to_string(),
            }
        })?;
        logs.clear();
        Ok(())
    }
    
    /// Get current number of logs.
    pub fn len(&self) -> FractalResult<usize> {
        let logs = self.logs.read().map_err(|_| {
            FractalAnalysisError::ConcurrencyError {
                resource: "transaction_logs".to_string(),
            }
        })?;
        Ok(logs.len())
    }
    
    /// Export logs as audit trail text.
    pub fn export_audit_trail(&self) -> FractalResult<String> {
        let logs = self.get_logs()?;
        Ok(logs
            .iter()
            .map(|log| log.to_audit_string())
            .collect::<Vec<_>>()
            .join("\n"))
    }
    
    // Private helper methods
    
    fn next_transaction_id(&self) -> FractalResult<u64> {
        let mut counter = self.transaction_counter.write().map_err(|_| {
            FractalAnalysisError::ConcurrencyError {
                resource: "transaction_counter".to_string(),
            }
        })?;
        *counter += 1;
        Ok(*counter)
    }
    
    fn append_log(&self, log: TransactionLog) -> FractalResult<()> {
        let mut logs = self.logs.write().map_err(|_| {
            FractalAnalysisError::ConcurrencyError {
                resource: "transaction_logs".to_string(),
            }
        })?;
        
        // Rotate if necessary (remove oldest)
        if logs.len() >= self.max_logs {
            logs.remove(0);
        }
        
        logs.push(log);
        Ok(())
    }
}

/// Compute a simple checksum for data verification.
///
/// This uses the FNV-1a hash algorithm, which is fast and sufficient
/// for integrity checking (not cryptographic security).
pub fn compute_checksum(data: &[f64]) -> u64 {
    const FNV_PRIME: u64 = 1099511628211;
    const FNV_OFFSET: u64 = 14695981039346656037;
    
    let mut hash = FNV_OFFSET;
    
    for &value in data {
        // Convert f64 to bytes for hashing
        let bytes = value.to_bits().to_ne_bytes();
        for &byte in &bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    
    hash
}

/// Operation timer for measuring execution duration.
///
/// This provides a simple RAII timer that automatically measures
/// the duration of an operation.
pub struct OperationTimer {
    start: std::time::Instant,
    operation: String,
}

impl OperationTimer {
    /// Start timing an operation.
    pub fn start(operation: impl Into<String>) -> Self {
        Self {
            start: std::time::Instant::now(),
            operation: operation.into(),
        }
    }
    
    /// Get elapsed time in milliseconds.
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
    
    /// Get the operation name.
    pub fn operation(&self) -> &str {
        &self.operation
    }
    
    /// Finish timing and return duration.
    pub fn finish(self) -> (String, u64) {
        let elapsed = self.elapsed_ms();
        (self.operation, elapsed)
    }
}

/// Audit context for wrapping operations with automatic logging.
pub struct AuditContext {
    logger: Arc<TransactionLogger>,
    operation: String,
    timer: OperationTimer,
    input_checksum: u64,
}

impl AuditContext {
    /// Create a new audit context for an operation.
    pub fn new(logger: Arc<TransactionLogger>, operation: impl Into<String>, input: &[f64]) -> Self {
        let operation = operation.into();
        Self {
            logger,
            operation: operation.clone(),
            timer: OperationTimer::start(operation),
            input_checksum: compute_checksum(input),
        }
    }
    
    /// Complete the audit with output data.
    pub fn complete(self, output: &[f64]) -> FractalResult<u64> {
        let output_checksum = compute_checksum(output);
        let duration_ms = self.timer.elapsed_ms();
        
        self.logger.log_transaction(
            self.operation,
            self.input_checksum,
            output_checksum,
            duration_ms,
        )
    }
    
    /// Complete with error (logs with zero output checksum).
    pub fn complete_with_error(self, error: &FractalAnalysisError) -> FractalResult<u64> {
        let duration_ms = self.timer.elapsed_ms();
        
        self.logger.log_transaction_with_metadata(
            self.operation,
            self.input_checksum,
            0, // Zero checksum indicates error
            duration_ms,
            vec![("error".to_string(), format!("{:?}", error))],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_transaction_logger() {
        let logger = TransactionLogger::new(3);
        
        // Log some transactions
        let id1 = logger.log_transaction(
            "operation1".to_string(),
            0x1234,
            0x5678,
            100,
        ).unwrap();
        assert_eq!(id1, 1);
        
        let id2 = logger.log_transaction(
            "operation2".to_string(),
            0x2345,
            0x6789,
            200,
        ).unwrap();
        assert_eq!(id2, 2);
        
        // Check logs
        let logs = logger.get_logs().unwrap();
        assert_eq!(logs.len(), 2);
        assert_eq!(logs[0].operation, "operation1");
        assert_eq!(logs[1].operation, "operation2");
        
        // Test rotation
        logger.log_transaction("op3".to_string(), 0, 0, 0).unwrap();
        logger.log_transaction("op4".to_string(), 0, 0, 0).unwrap();
        
        let logs = logger.get_logs().unwrap();
        assert_eq!(logs.len(), 3); // Should have rotated out the first
        assert_eq!(logs[0].operation, "operation2"); // First one should be gone
    }
    
    #[test]
    fn test_checksum() {
        let data1 = vec![1.0, 2.0, 3.0];
        let data2 = vec![1.0, 2.0, 3.0];
        let data3 = vec![1.0, 2.0, 3.1];
        
        let checksum1 = compute_checksum(&data1);
        let checksum2 = compute_checksum(&data2);
        let checksum3 = compute_checksum(&data3);
        
        assert_eq!(checksum1, checksum2); // Same data should give same checksum
        assert_ne!(checksum1, checksum3); // Different data should give different checksum
    }
    
    #[test]
    fn test_operation_timer() {
        let timer = OperationTimer::start("test_op");
        std::thread::sleep(std::time::Duration::from_millis(10));
        
        let elapsed = timer.elapsed_ms();
        assert!(elapsed >= 10);
        assert!(elapsed < 100); // Should be reasonably fast
        
        let (op, duration) = timer.finish();
        assert_eq!(op, "test_op");
        assert!(duration >= 10);
    }
    
    #[test]
    fn test_audit_context() {
        let logger = Arc::new(TransactionLogger::new(10));
        let input = vec![1.0, 2.0, 3.0];
        let output = vec![2.0, 4.0, 6.0];
        
        let context = AuditContext::new(logger.clone(), "double_values", &input);
        std::thread::sleep(std::time::Duration::from_millis(5));
        let id = context.complete(&output).unwrap();
        
        let logs = logger.get_logs().unwrap();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].id, id);
        assert_eq!(logs[0].operation, "double_values");
        assert!(logs[0].duration_ms >= 5);
    }
    
    #[test]
    fn test_log_metadata() {
        let logger = TransactionLogger::new(10);
        
        let id = logger.log_transaction_with_metadata(
            "test_op".to_string(),
            0x1234,
            0x5678,
            100,
            vec![
                ("user".to_string(), "alice".to_string()),
                ("version".to_string(), "1.0".to_string()),
            ],
        ).unwrap();
        
        let logs = logger.get_logs().unwrap();
        assert_eq!(logs[0].metadata.len(), 2);
        assert_eq!(logs[0].metadata[0], ("user".to_string(), "alice".to_string()));
        
        let audit_string = logs[0].to_audit_string();
        assert!(audit_string.contains("user=alice"));
        assert!(audit_string.contains("version=1.0"));
    }
}