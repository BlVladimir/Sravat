use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyList, PyTuple};
use std::f32;
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
fn process_contours_optimized(
    py: Python<'_>,
    parallelepiped: &PyList,
    contours: &PyList,
) -> PyResult<Py<PyList>> {
    let parallelepiped_vec: Vec<Point3D> = parallelepiped
        .iter()
        .map(|item| {
            let point: &PyList = item.downcast()?;
            Ok([
                point[0].extract::<f32>()?,
                point[1].extract::<f32>()?,
                point[2].extract::<f32>()?,
            ])
        })
        .collect::<PyResult<_>>()?;

    let mut contours_vec = Vec::new();

    for contour_item in contours.iter() {
        let contour_tuple: &PyTuple = contour_item.downcast()?;
        if contour_tuple.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Each contour must be a tuple (points, matrix)",
            ));
        }

        let contour_points_list: &PyList = contour_tuple[0].downcast()?;
        let contour_points: Vec<Point3D> = contour_points_list
            .iter()
            .map(|p| {
                let point: &PyList = p.downcast()?;
                Ok([
                    point[0].extract::<f32>()?,
                    point[1].extract::<f32>()?,
                    point[2].extract::<f32>()?,
                ])
            })
            .collect::<PyResult<_>>()?;

        let matrix_list: &PyList = contour_tuple[1].downcast()?;
        let mut matrix: Matrix3x3 = [[0.0; 3]; 3];

        for i in 0..3 {
            let row: &PyList = matrix_list[i].downcast()?;
            for j in 0..3 {
                matrix[i][j] = row[j].extract::<f32>()?;
            }
        }

        contours_vec.push((contour_points, matrix));
    }

    let result = py.allow_threads(|| {
        let n_points = parallelepiped_vec.len();

        let result_vec: Vec<AtomicU32> = (0..n_points)
            .map(|_| AtomicU32::new(0))
            .collect();

        contours_vec.par_iter().for_each(|(contour_points, rotation_matrix)| {
            let contour_2d: Vec<(f32, f32)> = contour_points
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

    let py_result = PyList::new(py, result.iter().map(|&x| x.into_py(py)));
    Ok(py_result.into())
}

#[pymodule]
fn scanning_optimized(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_contours_optimized, m)?)?;
    Ok(())
}