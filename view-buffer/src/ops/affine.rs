#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AffineParams {
    // A simplified 2x3 matrix for 2D affine transforms: [a, b, tx, c, d, ty]
    // Maps (x, y) -> (ax + by + tx, cx + dy + ty)
    pub matrix: [f32; 6],
}

impl AffineParams {
    pub fn identity() -> Self {
        Self {
            matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        }
    }

    /// Combines two affine transforms.
    /// If we have T1 (inner) then T2 (outer), the result is T2 * T1.
    pub fn combine(&self, other: &Self) -> Self {
        let [a1, b1, tx1, c1, d1, ty1] = self.matrix;
        let [a2, b2, tx2, c2, d2, ty2] = other.matrix;

        // Matrix multiplication (3x3 treated as 2D affine)
        // | a2 b2 tx2 |   | a1 b1 tx1 |
        // | c2 d2 ty2 | x | c1 d1 ty1 |
        // | 0  0  1   |   | 0  0  1   |

        let new_a = a2 * a1 + b2 * c1;
        let new_b = a2 * b1 + b2 * d1;
        let new_tx = a2 * tx1 + b2 * ty1 + tx2;

        let new_c = c2 * a1 + d2 * c1;
        let new_d = c2 * b1 + d2 * d1;
        let new_ty = c2 * tx1 + d2 * ty1 + ty2;

        Self {
            matrix: [new_a, new_b, new_tx, new_c, new_d, new_ty],
        }
    }
}
