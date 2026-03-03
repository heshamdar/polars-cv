#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Interpolation method for affine transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum InterpolationType {
    Nearest,
    Bilinear,
}

/// Parameters for a 2D affine warp operation.
///
/// The matrix `[a, b, tx, c, d, ty]` is a **forward** mapping from source to
/// destination (same convention as OpenCV's `warpAffine`):
///
/// ```text
/// x_dst = a * x_src + b * y_src + tx
/// y_dst = c * x_src + d * y_src + ty
/// ```
///
/// The kernel inverts this matrix internally for interpolation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AffineParams {
    /// 2x3 affine matrix: `[a, b, tx, c, d, ty]`.
    pub matrix: [f64; 6],
    /// Output image height.
    pub output_height: u32,
    /// Output image width.
    pub output_width: u32,
    /// Interpolation method.
    pub interpolation: InterpolationType,
    /// Value used for out-of-bounds pixels.
    pub border_value: f64,
}

impl AffineParams {
    /// Identity transform (preserves input, requires explicit output size).
    pub fn identity(output_height: u32, output_width: u32) -> Self {
        Self {
            matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            output_height,
            output_width,
            interpolation: InterpolationType::Bilinear,
            border_value: 0.0,
        }
    }

    /// Compose two affine transforms by matrix multiplication.
    ///
    /// If `self` is the inner transform (applied first) and `other` is the
    /// outer transform (applied second), the result is `other * self`.
    /// The output dimensions and interpolation are taken from `other`.
    pub fn combine(&self, other: &Self) -> Self {
        let [a1, b1, tx1, c1, d1, ty1] = self.matrix;
        let [a2, b2, tx2, c2, d2, ty2] = other.matrix;

        // 3x3 matrix multiplication (homogeneous coordinates):
        // | a2 b2 tx2 |   | a1 b1 tx1 |
        // | c2 d2 ty2 | × | c1 d1 ty1 |
        // | 0  0  1   |   | 0  0  1   |
        Self {
            matrix: [
                a2 * a1 + b2 * c1,
                a2 * b1 + b2 * d1,
                a2 * tx1 + b2 * ty1 + tx2,
                c2 * a1 + d2 * c1,
                c2 * b1 + d2 * d1,
                c2 * tx1 + d2 * ty1 + ty2,
            ],
            output_height: other.output_height,
            output_width: other.output_width,
            interpolation: other.interpolation,
            border_value: other.border_value,
        }
    }

    /// Check whether the matrix is the identity matrix.
    pub fn is_identity(&self) -> bool {
        let [a, b, tx, c, d, ty] = self.matrix;
        (a - 1.0).abs() < 1e-12
            && b.abs() < 1e-12
            && tx.abs() < 1e-12
            && c.abs() < 1e-12
            && (d - 1.0).abs() < 1e-12
            && ty.abs() < 1e-12
    }

    /// Build an `AffineParams` that performs a rotation around the image
    /// center, optionally expanding the canvas to fit the full rotated image.
    pub fn from_rotation(
        angle_deg: f32,
        input_height: u32,
        input_width: u32,
        expand: bool,
        interpolation: InterpolationType,
        border_value: f64,
    ) -> Self {
        let rad = (angle_deg as f64) * std::f64::consts::PI / 180.0;
        let cos_a = rad.cos();
        let sin_a = rad.sin();

        let ih = input_height as f64;
        let iw = input_width as f64;
        let cx = iw / 2.0;
        let cy = ih / 2.0;

        let (oh, ow) = if expand {
            let abs_cos = cos_a.abs();
            let abs_sin = sin_a.abs();
            let new_w = (iw * abs_cos + ih * abs_sin).round() as u32;
            let new_h = (ih * abs_cos + iw * abs_sin).round() as u32;
            (new_h, new_w)
        } else {
            (input_height, input_width)
        };

        let new_cx = ow as f64 / 2.0;
        let new_cy = oh as f64 / 2.0;

        let tx = -cx * cos_a - cy * (-sin_a) + new_cx;
        let ty = -cx * sin_a - cy * cos_a + new_cy;

        Self {
            matrix: [cos_a, -sin_a, tx, sin_a, cos_a, ty],
            output_height: oh,
            output_width: ow,
            interpolation,
            border_value,
        }
    }
}
