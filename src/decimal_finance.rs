//! Production-grade financial calculations using exact decimal arithmetic
//!
//! This module provides mathematically exact financial calculations using
//! Decimal types instead of floating-point. This is REQUIRED for any system
//! handling real money to avoid rounding errors and ensure exact results.

use crate::errors::{FractalAnalysisError, FractalResult};
use rust_decimal::prelude::*;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Financial epsilon for exact comparisons
/// Much tighter than f64 epsilon - suitable for financial calculations
pub const FINANCIAL_DECIMAL_EPSILON: Decimal = dec!(0.0000000001); // 10^-10

/// Global audit counter using SeqCst for proper ordering
static AUDIT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Financial amount with exact decimal representation
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FinancialAmount {
    /// Exact decimal value - no rounding errors
    value: Decimal,
    /// Scale (decimal places) for display
    scale: u32,
}

impl FinancialAmount {
    /// Create a new financial amount from a string
    /// This is the ONLY safe way to create monetary values
    pub fn from_str(s: &str) -> FractalResult<Self> {
        let value = Decimal::from_str(s).map_err(|e| FractalAnalysisError::NumericalError {
            reason: format!("Invalid decimal value '{}': {}", s, e),
            operation: None,
        })?;

        // Financial amounts typically use 2-4 decimal places
        let scale = value.scale();
        if scale > 10 {
            return Err(FractalAnalysisError::NumericalError {
                reason: format!("Excessive precision: {} decimal places", scale),
                operation: None,
            });
        }

        Ok(Self { value, scale })
    }

    /// Create from an integer number of cents (or smallest currency unit)
    pub fn from_cents(cents: i64) -> Self {
        let value = Decimal::from(cents) / dec!(100);
        Self { value, scale: 2 }
    }

    /// Add two amounts with exact precision
    pub fn add(&self, other: &Self) -> FractalResult<Self> {
        let result = self.value.checked_add(other.value).ok_or_else(|| {
            FractalAnalysisError::NumericalError {
                reason: "Decimal overflow in addition".to_string(),
                operation: None,
            }
        })?;

        Ok(Self {
            value: result,
            scale: self.scale.max(other.scale),
        })
    }

    /// Subtract with exact precision
    pub fn subtract(&self, other: &Self) -> FractalResult<Self> {
        let result = self.value.checked_sub(other.value).ok_or_else(|| {
            FractalAnalysisError::NumericalError {
                reason: "Decimal overflow in subtraction".to_string(),
                operation: None,
            }
        })?;

        Ok(Self {
            value: result,
            scale: self.scale.max(other.scale),
        })
    }

    /// Multiply with controlled precision
    pub fn multiply(&self, factor: Decimal) -> FractalResult<Self> {
        let result =
            self.value
                .checked_mul(factor)
                .ok_or_else(|| FractalAnalysisError::NumericalError {
                    reason: "Decimal overflow in multiplication".to_string(),
                    operation: None,
                })?;

        // Round to original scale to avoid precision creep
        let rounded = result.round_dp(self.scale);

        Ok(Self {
            value: rounded,
            scale: self.scale,
        })
    }

    /// Divide with exact precision and proper rounding
    pub fn divide(&self, divisor: Decimal) -> FractalResult<Self> {
        if divisor.is_zero() {
            return Err(FractalAnalysisError::NumericalError {
                reason: "Division by zero".to_string(),
                operation: None,
            });
        }

        let result = self.value.checked_div(divisor).ok_or_else(|| {
            FractalAnalysisError::NumericalError {
                reason: "Decimal overflow in division".to_string(),
                operation: None,
            }
        })?;

        // Round to original scale
        let rounded = result.round_dp(self.scale);

        Ok(Self {
            value: rounded,
            scale: self.scale,
        })
    }

    /// Get the exact value as Decimal
    pub fn as_decimal(&self) -> Decimal {
        self.value
    }

    /// Convert to cents (smallest unit)
    pub fn to_cents(&self) -> FractalResult<i64> {
        let cents = (self.value * dec!(100)).round();
        cents
            .to_i64()
            .ok_or_else(|| FractalAnalysisError::NumericalError {
                reason: "Value too large to convert to cents".to_string(),
                operation: None,
            })
    }
}

/// Numerically stable variance calculation using Welford's algorithm
/// This is the CORRECT way to calculate variance for financial data
pub fn welford_variance_decimal(values: &[Decimal]) -> FractalResult<Decimal> {
    if values.len() < 2 {
        return Err(FractalAnalysisError::InsufficientData {
            required: 2,
            actual: values.len(),
        });
    }

    let mut mean = Decimal::ZERO;
    let mut m2 = Decimal::ZERO;
    let mut count = Decimal::ZERO;

    for &value in values {
        count += Decimal::ONE;
        let delta = value - mean;
        mean += delta / count;
        let delta2 = value - mean;
        m2 += delta * delta2;
    }

    // Bessel's correction: divide by (n-1) not n
    Ok(m2 / (count - Decimal::ONE))
}

/// Kahan summation for Decimal types - maintains precision.
/// For f64 values, use math_utils::kahan_sum() instead.
/// This is specifically for exact decimal arithmetic in financial calculations.
pub fn kahan_sum_decimal(values: &[Decimal]) -> Decimal {
    let mut sum = Decimal::ZERO;
    let mut c = Decimal::ZERO;

    for &value in values {
        let y = value - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    sum
}

/// Thread-safe audit log with proper memory ordering
pub struct FinancialAuditLog {
    entries: Arc<Mutex<Vec<AuditEntry>>>,
    counter: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub id: u64,
    pub timestamp: u64,
    pub operation: String,
    pub amount: Option<FinancialAmount>,
    pub checksum: u64,
}

impl FinancialAuditLog {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(Mutex::new(Vec::new())),
            counter: AtomicU64::new(0),
        }
    }

    /// Log an operation with SeqCst ordering for proper synchronization
    pub fn log_operation(
        &self,
        operation: String,
        amount: Option<FinancialAmount>,
    ) -> FractalResult<u64> {
        // Use SeqCst for audit log - CRITICAL for forensic analysis
        let id = self.counter.fetch_add(1, Ordering::SeqCst);
        let global_id = AUDIT_COUNTER.fetch_add(1, Ordering::SeqCst);

        let checksum = self.calculate_checksum(&operation, &amount);

        let entry = AuditEntry {
            id,
            timestamp: global_id, // Use monotonic counter, not wall clock
            operation,
            amount,
            checksum,
        };

        let mut entries =
            self.entries
                .lock()
                .map_err(|_| FractalAnalysisError::ConcurrencyError {
                    resource: "audit_log".to_string(),
                })?;

        entries.push(entry);

        Ok(id)
    }

    fn calculate_checksum(&self, operation: &str, amount: &Option<FinancialAmount>) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(operation, &mut hasher);
        if let Some(amt) = amount {
            std::hash::Hash::hash(&amt.value.to_string(), &mut hasher);
        }
        std::hash::Hasher::finish(&hasher)
    }

    pub fn get_entries(&self) -> FractalResult<Vec<AuditEntry>> {
        let entries = self
            .entries
            .lock()
            .map_err(|_| FractalAnalysisError::ConcurrencyError {
                resource: "audit_log".to_string(),
            })?;
        Ok(entries.clone())
    }
}

/// Portfolio of financial amounts with exact arithmetic
pub struct FinancialPortfolio {
    holdings: BTreeMap<String, FinancialAmount>,
    audit_log: FinancialAuditLog,
}

impl FinancialPortfolio {
    pub fn new() -> Self {
        Self {
            holdings: BTreeMap::new(),
            audit_log: FinancialAuditLog::new(),
        }
    }

    /// Add or update a holding with audit trail
    pub fn update_holding(&mut self, symbol: String, amount: FinancialAmount) -> FractalResult<()> {
        self.audit_log
            .log_operation(format!("UPDATE_HOLDING:{}", symbol), Some(amount.clone()))?;

        self.holdings.insert(symbol, amount);
        Ok(())
    }

    /// Calculate total portfolio value with EXACT precision
    pub fn total_value(&self) -> FractalResult<FinancialAmount> {
        let mut total = FinancialAmount::from_cents(0);

        for (_, amount) in &self.holdings {
            total = total.add(amount)?;
        }

        self.audit_log
            .log_operation("CALCULATE_TOTAL".to_string(), Some(total.clone()))?;

        Ok(total)
    }

    /// Get variance of holdings using Welford's algorithm
    pub fn holdings_variance(&self) -> FractalResult<Decimal> {
        if self.holdings.len() < 2 {
            return Err(FractalAnalysisError::InsufficientData {
                required: 2,
                actual: self.holdings.len(),
            });
        }

        let values: Vec<Decimal> = self.holdings.values().map(|amt| amt.as_decimal()).collect();

        welford_variance_decimal(&values)
    }
}

/// Convert legacy f64 data to Decimal (with warnings about precision loss)
pub fn migrate_f64_to_decimal(value: f64) -> FractalResult<Decimal> {
    if !value.is_finite() {
        return Err(FractalAnalysisError::NumericalError {
            reason: format!("Cannot convert non-finite f64: {}", value),
            operation: None,
        });
    }

    // Try to convert with a reasonable number of decimal places
    let s = format!("{:.10}", value);
    Decimal::from_str(&s).map_err(|e| FractalAnalysisError::NumericalError {
        reason: format!("Failed to convert f64 {} to Decimal: {}", value, e),
        operation: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_decimal_arithmetic() {
        // THIS is why we use Decimal - exact representation
        let one_tenth = FinancialAmount::from_str("0.1").unwrap();
        let two_tenths = FinancialAmount::from_str("0.2").unwrap();
        let three_tenths = FinancialAmount::from_str("0.3").unwrap();

        let sum = one_tenth.add(&two_tenths).unwrap();

        // EXACT equality - no epsilon needed!
        assert_eq!(sum, three_tenths);
        assert_eq!(sum.as_decimal(), dec!(0.3));
    }

    #[test]
    fn test_no_precision_loss_in_pennies() {
        // Add 1 cent a million times
        let mut total = FinancialAmount::from_cents(0);
        let penny = FinancialAmount::from_cents(1);

        for _ in 0..1_000_000 {
            total = total.add(&penny).unwrap();
        }

        // Exactly $10,000.00 - no rounding errors!
        assert_eq!(total.to_cents().unwrap(), 1_000_000);
        assert_eq!(total.as_decimal(), dec!(10000.00));
    }

    #[test]
    fn test_welford_variance_decimal() {
        let values = vec![
            dec!(100.00),
            dec!(200.00),
            dec!(300.00),
            dec!(400.00),
            dec!(500.00),
        ];

        let variance = welford_variance_decimal(&values).unwrap();
        let expected = dec!(25000.00); // Exact variance

        // Can use exact equality with Decimal!
        assert_eq!(variance, expected);
    }

    #[test]
    fn test_division_rounding() {
        let amount = FinancialAmount::from_str("100.00").unwrap();
        let result = amount.divide(dec!(3)).unwrap();

        // Should round to 33.33 (2 decimal places)
        assert_eq!(result.as_decimal(), dec!(33.33));
    }

    #[test]
    fn test_audit_log_ordering() {
        let log = FinancialAuditLog::new();

        // Log operations
        let id1 = log.log_operation("OP1".to_string(), None).unwrap();
        let id2 = log.log_operation("OP2".to_string(), None).unwrap();

        // IDs should be strictly ordered due to SeqCst
        assert!(id2 > id1);

        let entries = log.get_entries().unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries[1].timestamp > entries[0].timestamp);
    }

    #[test]
    fn test_portfolio_exact_total() {
        let mut portfolio = FinancialPortfolio::new();

        portfolio
            .update_holding(
                "AAPL".to_string(),
                FinancialAmount::from_str("10000.00").unwrap(),
            )
            .unwrap();

        portfolio
            .update_holding(
                "GOOGL".to_string(),
                FinancialAmount::from_str("20000.00").unwrap(),
            )
            .unwrap();

        portfolio
            .update_holding(
                "MSFT".to_string(),
                FinancialAmount::from_str("15000.00").unwrap(),
            )
            .unwrap();

        let total = portfolio.total_value().unwrap();

        // Exact total - no floating point errors!
        assert_eq!(total.as_decimal(), dec!(45000.00));
    }

    #[test]
    fn test_financial_precision() {
        // Test that we maintain precision for financial calculations
        let values = vec![dec!(1000000.01), dec!(1000000.02), dec!(1000000.03)];

        let sum = kahan_sum_decimal(&values);

        // Exact sum even with large numbers and small decimals
        assert_eq!(sum, dec!(3000000.06));
    }
}
