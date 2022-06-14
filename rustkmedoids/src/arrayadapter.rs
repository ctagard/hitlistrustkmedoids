//! Adapter trait for accessing different types of arrays.
//!
//! Includes adapters for `ndarray::Array2` and a serialized lower triangular matrix in a `Vec`.

/// Adapter trait for accessing different types of arrays
#[allow(clippy::len_without_is_empty)]
pub trait ArrayAdapter<N> {
	/// Get the length of an array structure
	fn len(&self) -> usize;
	/// Verify that it is a square matrix
	fn is_square(&self) -> bool;
	/// Get the contents at cell x,y
	fn get(&self, x: usize, y: usize) -> N;
}

/// Adapter trait for using `ndarray::Array2` and similar
#[cfg(feature = "ndarray")]
impl<A, N> ArrayAdapter<N> for ndarray::ArrayBase<A, ndarray::Ix2>
where
	A: ndarray::Data<Elem = N>,
	N: Copy,
{
	#[inline]
	fn len(&self) -> usize {
		self.shape()[0]
	}
	#[inline]
	fn is_square(&self) -> bool {
		self.shape()[0] == self.shape()[1]
	}
	#[inline]
	fn get(&self, x: usize, y: usize) -> N {
		self[[x, y]]
	}
}

/// Lower triangular matrix in serial form (without diagonal)
///

#[derive(Debug, Clone)]
pub struct LowerTriangle<N> {
	/// Matrix size
	pub n: usize,
	// Matrix data, lower triangular form without diagonal
	pub data: Vec<N>,
}
/// Adapter implementation for LowerTriangle
impl<N: Copy + num_traits::Zero> ArrayAdapter<N> for LowerTriangle<N> {
	#[inline]
	fn len(&self) -> usize {
		self.n
	}
	#[inline]
	fn is_square(&self) -> bool {
		self.data.len() == (self.n * (self.n - 1)) >> 1
	}
	#[inline]
	fn get(&self, x: usize, y: usize) -> N {
		match x.cmp(&y) {
			std::cmp::Ordering::Less => self.data[((y * (y - 1)) >> 1) + x],
			std::cmp::Ordering::Greater => self.data[((x * (x - 1)) >> 1) + y],
			std::cmp::Ordering::Equal => N::zero(),
		}
	}
}