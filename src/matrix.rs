use serde::{Deserialize, Serialize};

use crate::{MoeError, Result};

/// Finite row-major F32 matrix used at the model contract boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Matrix {
    rows: usize,
    columns: usize,
    values: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, columns: usize, values: Vec<f32>) -> Result<Self> {
        if rows == 0 || columns == 0 {
            return Err(MoeError::InvalidTensor(
                "matrix dimensions must be non-zero".to_string(),
            ));
        }
        let expected = rows.checked_mul(columns).ok_or_else(|| {
            MoeError::InvalidTensor("matrix element count overflowed".to_string())
        })?;
        if values.len() != expected {
            return Err(MoeError::InvalidTensor(format!(
                "matrix shape [{rows}, {columns}] requires {expected} values, found {}",
                values.len()
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(MoeError::InvalidTensor(
                "matrix values must be finite".to_string(),
            ));
        }
        Ok(Self {
            rows,
            columns,
            values,
        })
    }

    pub fn zeros(rows: usize, columns: usize) -> Result<Self> {
        let elements = rows.checked_mul(columns).ok_or_else(|| {
            MoeError::InvalidTensor("matrix element count overflowed".to_string())
        })?;
        Self::new(rows, columns, vec![0.0; elements])
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn row(&self, row: usize) -> Result<&[f32]> {
        if row >= self.rows {
            return Err(MoeError::InvalidTensor(format!(
                "matrix row {row} is outside 0..{}",
                self.rows
            )));
        }
        let start = row * self.columns;
        Ok(&self.values[start..start + self.columns])
    }

    pub(crate) fn row_mut(&mut self, row: usize) -> Result<&mut [f32]> {
        if row >= self.rows {
            return Err(MoeError::InvalidTensor(format!(
                "matrix row {row} is outside 0..{}",
                self.rows
            )));
        }
        let start = row * self.columns;
        Ok(&mut self.values[start..start + self.columns])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_shape_mismatch_and_non_finite_values() {
        assert!(Matrix::new(2, 2, vec![1.0]).is_err());
        assert!(Matrix::new(1, 1, vec![f32::NAN]).is_err());
        assert!(Matrix::new(0, 1, Vec::new()).is_err());
    }

    #[test]
    fn exposes_exact_row_major_rows() {
        let matrix = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(matrix.row(0).unwrap(), [1.0, 2.0]);
        assert_eq!(matrix.row(1).unwrap(), [3.0, 4.0]);
        assert!(matrix.row(2).is_err());
    }
}
