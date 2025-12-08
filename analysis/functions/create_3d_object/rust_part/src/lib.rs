use pyo3::prelude::*;

mod contours;
mod mesh;

#[pymodule]
fn scanning_optimized(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(contours::process_contours_optimized, m)?)?;

    m.add_function(wrap_pyfunction!(mesh::build_voxel_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(mesh::build_voxel_mesh_with_normals, m)?)?;

    Ok(())
}
