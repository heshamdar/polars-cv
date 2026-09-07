#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Interpolation method for affine transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum InterpolationType {
    Nearest,
    Bilinear,
}

crate::naming::named_variants!(InterpolationType {
    "nearest" => Nearest,
    "bilinear" => Bilinear,
});

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

    /// The determinant of the matrix's linear part, `a*d - b*c`.
    ///
    /// Zero means the transform collapses the plane onto a line or a point, so
    /// it has no inverse — and warping is implemented by inverse mapping (for
    /// each output pixel, where did it come from?). See
    /// [`is_invertible`](Self::is_invertible).
    pub fn determinant(&self) -> f64 {
        let [a, b, _, c, d, _] = self.matrix;
        a * d - b * c
    }

    /// Whether the transform can be inverted, and so applied at all.
    ///
    /// The single authority for this question, so the check cannot be spelled
    /// one way at the boundary that rejects a bad matrix and another way in the
    /// runner that would have to cope with one.
    ///
    /// The threshold is deliberately an exact-zero neighbourhood rather than a
    /// conditioning test: a nearly-singular matrix is a legitimate (if extreme)
    /// transform and produces a real, if heavily stretched, image.
    pub fn is_invertible(&self) -> bool {
        self.determinant().abs() >= Self::SINGULAR_EPSILON
    }

    /// Below this, [`determinant`](Self::determinant) counts as zero.
    pub const SINGULAR_EPSILON: f64 = 1e-15;

    /// The 2x3 forward rotation+scale matrix about `(cx, cy)` — **the** rotation
    /// matrix authority (OpenCV's `getRotationMatrix2D(center, angle, scale)`
    /// convention: positive `angle_deg` = clockwise in image coordinates). It
    /// maps the pivot `(cx, cy)` to itself, so the output canvas is the same
    /// size as the input.
    ///
    /// [`from_rotation`](Self::from_rotation) builds on this for the
    /// image-center + optional-expand case, and the plugin's `rotation_matrix_2d`
    /// FFI exposes it so the Python planner reads this matrix instead of
    /// recomputing the trig for a literal `rotate_and_scale`.
    ///
    /// `angle_deg` is `f64` (not the `f32` `from_rotation` takes) so the FFI
    /// reproduces the planner's f64 arithmetic bit-for-bit.
    pub fn rotation_matrix_2d(angle_deg: f64, cx: f64, cy: f64, scale: f64) -> [f64; 6] {
        let rad = angle_deg * std::f64::consts::PI / 180.0;
        let cos_a = rad.cos() * scale;
        let sin_a = rad.sin() * scale;
        let tx = (1.0 - cos_a) * cx + sin_a * cy;
        let ty = -sin_a * cx + (1.0 - cos_a) * cy;
        [cos_a, -sin_a, tx, sin_a, cos_a, ty]
    }

    /// Build an `AffineParams` that performs a rotation around the image
    /// center, optionally expanding the canvas to fit the full rotated image.
    ///
    /// The rotation matrix itself comes from [`rotation_matrix_2d`](Self::rotation_matrix_2d)
    /// (scale 1, pivot = image center); this only adds the expand canvas and the
    /// recentering that moves the pivot to the enlarged canvas's centre.
    pub fn from_rotation(
        angle_deg: f32,
        input_height: u32,
        input_width: u32,
        expand: bool,
        interpolation: InterpolationType,
        border_value: f64,
    ) -> Self {
        let ih = input_height as f64;
        let iw = input_width as f64;
        let cx = iw / 2.0;
        let cy = ih / 2.0;

        let base = Self::rotation_matrix_2d(angle_deg as f64, cx, cy, 1.0);
        let cos_a = base[0];
        let sin_a = base[3];

        let (oh, ow) = if expand {
            let abs_cos = cos_a.abs();
            let abs_sin = sin_a.abs();
            let new_w = (iw * abs_cos + ih * abs_sin).round() as u32;
            let new_h = (ih * abs_cos + iw * abs_sin).round() as u32;
            (new_h, new_w)
        } else {
            (input_height, input_width)
        };

        // `base` maps the pivot to itself; expanding moves it to the enlarged
        // canvas's centre, a pure translation offset on tx/ty.
        let (tx, ty) = if expand {
            (
                base[2] + (ow as f64 / 2.0 - cx),
                base[5] + (oh as f64 / 2.0 - cy),
            )
        } else {
            (base[2], base[5])
        };

        Self {
            matrix: [cos_a, base[1], tx, sin_a, cos_a, ty],
            output_height: oh,
            output_width: ow,
            interpolation,
            border_value,
        }
    }
}
