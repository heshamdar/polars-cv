//! Core contour types and basic operations.

use geo::BooleanOps;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A 2D point with f64 coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Point {
    /// X coordinate (horizontal position).
    pub x: f64,
    /// Y coordinate (vertical position).
    pub y: f64,
}

impl Point {
    /// Creates a new point.
    #[inline]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Returns the Euclidean distance to another point.
    #[inline]
    pub fn distance_to(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Returns the squared distance to another point (avoids sqrt).
    #[inline]
    pub fn distance_squared_to(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }
}

impl From<(f64, f64)> for Point {
    fn from((x, y): (f64, f64)) -> Self {
        Self { x, y }
    }
}

impl From<(i32, i32)> for Point {
    fn from((x, y): (i32, i32)) -> Self {
        Self {
            x: x as f64,
            y: y as f64,
        }
    }
}

/// Winding direction of a contour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Winding {
    /// Counter-clockwise (positive signed area in standard math coordinates).
    CounterClockwise,
    /// Clockwise (negative signed area in standard math coordinates).
    Clockwise,
}

impl Winding {
    /// Returns the opposite winding direction.
    pub fn flip(self) -> Self {
        match self {
            Winding::CounterClockwise => Winding::Clockwise,
            Winding::Clockwise => Winding::CounterClockwise,
        }
    }
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BoundingBox {
    /// X coordinate of top-left corner.
    pub x: f64,
    /// Y coordinate of top-left corner.
    pub y: f64,
    /// Width of the bounding box.
    pub width: f64,
    /// Height of the bounding box.
    pub height: f64,
}

impl BoundingBox {
    /// Creates a new bounding box.
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Computes the bounding box from a set of points.
    pub fn from_points(points: &[Point]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for p in points {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }

        Some(Self {
            x: min_x,
            y: min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        })
    }

    /// Returns the center point of the bounding box.
    pub fn center(&self) -> Point {
        Point {
            x: self.x + self.width / 2.0,
            y: self.y + self.height / 2.0,
        }
    }

    /// Returns the area of the bounding box.
    pub fn area(&self) -> f64 {
        self.width * self.height
    }

    /// Checks if this bounding box intersects with another.
    pub fn intersects(&self, other: &BoundingBox) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }

    /// Computes the intersection of two bounding boxes.
    pub fn intersection(&self, other: &BoundingBox) -> Option<BoundingBox> {
        if !self.intersects(other) {
            return None;
        }

        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let right = (self.x + self.width).min(other.x + other.width);
        let bottom = (self.y + self.height).min(other.y + other.height);

        Some(BoundingBox {
            x,
            y,
            width: right - x,
            height: bottom - y,
        })
    }

    /// Computes the union (bounding box that contains both).
    pub fn union(&self, other: &BoundingBox) -> BoundingBox {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let right = (self.x + self.width).max(other.x + other.width);
        let bottom = (self.y + self.height).max(other.y + other.height);

        BoundingBox {
            x,
            y,
            width: right - x,
            height: bottom - y,
        }
    }
}

/// A polygon contour with an exterior ring and optional interior holes.
///
/// The `holes` field is the **sole** carrier of hole-ness. Point order is never
/// interpreted as a hole signal: a ring listed in `holes` is a hole whichever way it
/// is wound, and every operation here is winding-independent. Winding is only
/// *reported* (by [`measures::contour_winding`]) and *set* (by
/// [`transforms::ensure_winding`]) on request.
///
/// Rings are implicitly closed — the first point is not repeated at the end.
///
/// [`measures::contour_winding`]: super::measures::contour_winding
/// [`transforms::ensure_winding`]: super::transforms::ensure_winding
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Contour {
    /// The exterior ring of the polygon.
    pub exterior: Vec<Point>,
    /// Interior holes (if any).
    pub holes: Vec<Vec<Point>>,
}

impl Contour {
    /// Creates a new contour with no holes.
    pub fn new(exterior: Vec<Point>) -> Self {
        Self {
            exterior,
            holes: Vec::new(),
        }
    }

    /// Creates a contour with holes.
    pub fn with_holes(exterior: Vec<Point>, holes: Vec<Vec<Point>>) -> Self {
        Self { exterior, holes }
    }

    /// Creates a contour from a slice of (x, y) tuples.
    pub fn from_tuples(points: &[(f64, f64)]) -> Self {
        let exterior = points.iter().map(|&(x, y)| Point::new(x, y)).collect();
        Self::new(exterior)
    }

    /// Creates a contour from integer coordinates.
    pub fn from_int_tuples(points: &[(i32, i32)]) -> Self {
        let exterior = points
            .iter()
            .map(|&(x, y)| Point::new(x as f64, y as f64))
            .collect();
        Self::new(exterior)
    }

    /// Returns the number of points in the exterior ring.
    pub fn len(&self) -> usize {
        self.exterior.len()
    }

    /// Returns true if the contour has no points.
    pub fn is_empty(&self) -> bool {
        self.exterior.is_empty()
    }

    /// Returns true if the contour has holes.
    pub fn has_holes(&self) -> bool {
        !self.holes.is_empty()
    }

    /// Adds a hole to the contour.
    pub fn add_hole(&mut self, hole: Vec<Point>) {
        self.holes.push(hole);
    }

    /// Returns an iterator over the exterior points.
    pub fn iter(&self) -> impl Iterator<Item = &Point> {
        self.exterior.iter()
    }

    /// Returns the exterior points as a slice.
    pub fn points(&self) -> &[Point] {
        &self.exterior
    }

    /// Returns the cached or computed bounding box.
    pub fn bounding_box(&self) -> Option<BoundingBox> {
        BoundingBox::from_points(&self.exterior)
    }

    /// Converts to a [`geo::Polygon`], the representation the exact geometry
    /// algorithms operate on.
    ///
    /// `geo::Polygon::new` closes each ring for us. Because every consumer treats
    /// hole-ness structurally, the rings are handed over with their point order intact.
    pub fn to_geo(&self) -> geo::Polygon<f64> {
        geo::Polygon::new(
            ring_to_geo(&self.exterior),
            self.holes.iter().map(|h| ring_to_geo(h)).collect(),
        )
    }

    /// The filled region this contour describes: the exterior minus the **union**
    /// of every hole ring.
    ///
    /// This is the representation area and overlap must use, and it is not the same
    /// as [`Self::to_geo`]. A `geo::Polygon` carries its holes as interior rings,
    /// where two independent rules disagree once rings overlap or nest:
    /// `geo::Area` subtracts each interior ring's absolute area (double-counting
    /// the shared part), while `BooleanOps` applies the even-odd rule (filling a
    /// ring nested inside another). Differencing the hole union out once leaves a
    /// region on which every operation agrees, and matches what `contains_point`
    /// and `rasterize` have always reported.
    ///
    /// Hole-free contours — the overwhelming majority — skip the boolean op.
    pub fn to_geo_region(&self) -> geo::MultiPolygon<f64> {
        let exterior = geo::Polygon::new(ring_to_geo(&self.exterior), Vec::new());

        if self.holes.is_empty() {
            return geo::MultiPolygon::new(vec![exterior]);
        }

        self.holes
            .iter()
            .fold(geo::MultiPolygon::new(vec![exterior]), |region, hole| {
                region.difference(&geo::Polygon::new(ring_to_geo(hole), Vec::new()))
            })
    }

    /// Converts back from a [`geo::Polygon`], dropping the repeated closing point so
    /// the result follows this crate's implicitly-closed ring convention.
    pub fn from_geo(polygon: &geo::Polygon<f64>) -> Self {
        Self::with_holes(
            ring_from_geo(polygon.exterior()),
            polygon.interiors().iter().map(ring_from_geo).collect(),
        )
    }
}

fn ring_to_geo(points: &[Point]) -> geo::LineString<f64> {
    geo::LineString::from(
        points
            .iter()
            .map(|p| geo::coord! { x: p.x, y: p.y })
            .collect::<Vec<_>>(),
    )
}

fn ring_from_geo(ring: &geo::LineString<f64>) -> Vec<Point> {
    let coords = ring.0.as_slice();
    // A closed `LineString` repeats its first coordinate; our rings do not.
    let coords = match coords {
        [first, .., last] if first == last => &coords[..coords.len() - 1],
        _ => coords,
    };
    coords.iter().map(|c| Point::new(c.x, c.y)).collect()
}

impl From<Vec<Point>> for Contour {
    fn from(exterior: Vec<Point>) -> Self {
        Self::new(exterior)
    }
}

impl From<Vec<(f64, f64)>> for Contour {
    fn from(points: Vec<(f64, f64)>) -> Self {
        Self::from_tuples(&points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_distance() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(3.0, 4.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_bounding_box_from_points() {
        let points = vec![
            Point::new(10.0, 20.0),
            Point::new(30.0, 40.0),
            Point::new(20.0, 10.0),
        ];
        let bbox = BoundingBox::from_points(&points).unwrap();
        assert_eq!(bbox.x, 10.0);
        assert_eq!(bbox.y, 10.0);
        assert_eq!(bbox.width, 20.0);
        assert_eq!(bbox.height, 30.0);
    }

    #[test]
    fn test_bounding_box_intersection() {
        let b1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let b2 = BoundingBox::new(5.0, 5.0, 10.0, 10.0);
        let inter = b1.intersection(&b2).unwrap();
        assert_eq!(inter.x, 5.0);
        assert_eq!(inter.y, 5.0);
        assert_eq!(inter.width, 5.0);
        assert_eq!(inter.height, 5.0);
    }

    #[test]
    fn test_contour_creation() {
        let contour = Contour::from_tuples(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]);
        assert_eq!(contour.len(), 4);
        assert!(!contour.has_holes());
    }

    #[test]
    fn test_contour_with_holes() {
        let exterior = vec![
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            Point::new(100.0, 100.0),
            Point::new(0.0, 100.0),
        ];
        let hole = vec![
            Point::new(25.0, 25.0),
            Point::new(75.0, 25.0),
            Point::new(75.0, 75.0),
            Point::new(25.0, 75.0),
        ];
        let contour = Contour::with_holes(exterior, vec![hole]);
        assert!(contour.has_holes());
        assert_eq!(contour.holes.len(), 1);
    }
}
