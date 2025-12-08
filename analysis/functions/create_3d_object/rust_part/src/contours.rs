use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray2};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

type Point3D = [f32; 3];
type Matrix3x3 = [[f32; 3]; 3];

fn rotate_point(point: &Point3D, matrix: &Matrix3x3) -> Point3D {
    let x = point[0] * matrix[0][0] + point[1] * matrix[0][1] + point[2] * matrix[0][2];
    let y = point[0] * matrix[1][0] + point[1] * matrix[1][1] + point[2] * matrix[1][2];
    let z = point[0] * matrix[2][0] + point[1] * matrix[2][1] + point[2] * matrix[2][2];
    [x, y, z]
}

fn point_in_polygon_2d(point: (f32, f32), polygon: &[(f32, f32)]) -> bool {
    let mut wn = 0i32;
    let n = polygon.len();

    for i in 0..n {
        let j = (i + 1) % n;
        let vi = polygon[i];
        let vj = polygon[j];

        if vi.1 <= point.1 {
            if vj.1 > point.1 && is_left(vi, vj, point) > 0.0 {
                wn += 1;
            }
        } else {
            if vj.1 <= point.1 && is_left(vi, vj, point) < 0.0 {
                wn -= 1;
            }
        }
    }

    wn != 0
}

#[inline(always)]
fn is_left(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32)) -> f32 {
    (p1.0 - p0.0) * (p2.1 - p0.1) - (p2.0 - p0.0) * (p1.1 - p0.1)
}

#[pyfunction]
pub fn process_contours_optimized<'py>(
    py: Python<'py>,
    parallelepiped: PyReadonlyArray2<'py, f32>,
    contours: Vec<(PyReadonlyArray2<'py, f32>, PyReadonlyArray2<'py, f32>)>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let parallelepiped_array = parallelepiped.as_array();
    let parallelepiped_vec: Vec<Point3D> = parallelepiped_array
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect();

    let mut contours_vec = Vec::new();
    
    for (contour_points_array, matrix_array) in contours.iter() {
        let points = contour_points_array.as_array();
        let contour_points: Vec<Point3D> = points
            .rows()
            .into_iter()
            .map(|row| [row[0], row[1], row[2]])
            .collect();

        let matrix_data = matrix_array.as_array();
        let matrix: Matrix3x3 = [
            [matrix_data[[0, 0]], matrix_data[[0, 1]], matrix_data[[0, 2]]],
            [matrix_data[[1, 0]], matrix_data[[1, 1]], matrix_data[[1, 2]]],
            [matrix_data[[2, 0]], matrix_data[[2, 1]], matrix_data[[2, 2]]],
        ];

        contours_vec.push((contour_points, matrix));
    }

    let result = py.allow_threads(|| {
        let n_points = parallelepiped_vec.len();

        let result_vec: Vec<AtomicU32> = (0..n_points)
            .map(|_| AtomicU32::new(0))
            .collect();

        contours_vec.par_iter().for_each(|(contour_points, rotation_matrix)| {
            let rotated_contour: Vec<Point3D> = contour_points
                .iter()
                .map(|p| rotate_point(p, rotation_matrix))
                .collect();

            let contour_2d: Vec<(f32, f32)> = rotated_contour
                .iter()
                .map(|p| (p[0], p[1]))
                .collect();

            for (i, point) in parallelepiped_vec.iter().enumerate() {
                let rotated_point = rotate_point(point, rotation_matrix);

                if !point_in_polygon_2d((rotated_point[0], rotated_point[1]), &contour_2d) {
                    result_vec[i].fetch_add(1, Ordering::Relaxed);
                }
            }
        });

        result_vec
            .into_iter()
            .map(|atomic| atomic.into_inner())
            .collect::<Vec<u32>>()
    });

    Ok(PyArray1::from_vec_bound(py, result))
}