//! k-Medoids Clustering with the FasterPAM Algorithm
//!
//! For details on the implemented FasterPAM algorithm, please see:
//!
//! Erich Schubert, Peter J. Rousseeuw
//! **Fast and Eager k-Medoids Clustering:
//! O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms**
//! Information Systems (101), 2021, 101804
//! <https://doi.org/10.1016/j.is.2021.101804> (open access)
//!
//! Erich Schubert, Peter J. Rousseeuw:
//! **Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms**
//! In: 12th International Conference on Similarity Search and Applications (SISAP 2019), 171-187.
//! <https://doi.org/10.1007/978-3-030-32047-8_16>
//! Preprint: <https://arxiv.org/abs/1810.05691>
//!
//! This is a port of the original Java code from [ELKI](https://elki-project.github.io/) to Rust.
//! But it does not include all functionality in the original benchmarks.
//!
//! If you use this in scientific work, please consider citing above articles.
//!

mod alternating;
pub mod arrayadapter;
mod fasterpam;
mod fastpam1;
mod initialization;
mod pam;
#[cfg(feature = "parallel")]
mod par_fasterpam;
#[cfg(feature = "parallel")]
mod par_silhouette;
mod silhouette;
mod util;

pub use crate::alternating::*;
pub use crate::arrayadapter::ArrayAdapter;
pub use crate::fasterpam::*;
pub use crate::fastpam1::*;
pub use crate::initialization::*;
pub use crate::pam::*;
#[cfg(feature = "parallel")]
pub use crate::par_fasterpam::*;
#[cfg(feature = "parallel")]
pub use crate::par_silhouette::*;
pub use crate::silhouette::*;
