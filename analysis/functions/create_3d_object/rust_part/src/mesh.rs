use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2, PyArrayMethods};
use rayon::prelude::*;
use std::collections::HashSet;
use pyo3::types::PyTuple;

#[pyfunction]
pub fn build_voxel_mesh<'py>(
    py: Python<'py>,
    centers: PyReadonlyArray2<'py, f32>,
    cube_side: f32,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u32>>)> {
    // Конвертируем центры в вектор точек
    let centers_array = centers.as_array();
    let centers_vec: Vec<[f32; 3]> = centers_array
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect();

    // Создаем HashSet для быстрой проверки существования куба
    let centers_set: HashSet<[i32; 3]> = centers_vec
        .par_iter()
        .map(|&[x, y, z]| {
            [
                (x / cube_side).round() as i32,
                (y / cube_side).round() as i32,
                (z / cube_side).round() as i32,
            ]
        })
        .collect();

    // Предварительно вычисляем вершины одного куба
    let half_side = cube_side / 2.0;
    let cube_vertices = [
        [-half_side, -half_side, -half_side],
        [half_side, -half_side, -half_side],
        [half_side, half_side, -half_side],
        [-half_side, half_side, -half_side],
        [-half_side, -half_side, half_side],
        [half_side, -half_side, half_side],
        [half_side, half_side, half_side],
        [-half_side, half_side, half_side],
    ];

    // Индексы вершин для каждой грани куба
    let face_indices = [
        [0, 1, 2, 3], // задняя
        [4, 5, 6, 7], // передняя
        [0, 1, 5, 4], // нижняя
        [2, 3, 7, 6], // верхняя
        [0, 3, 7, 4], // левая
        [1, 2, 6, 5], // правая
    ];

    // Направления для проверки соседей
    let neighbor_offsets = [
        ([1, 0, 0], 5),   // право
        ([-1, 0, 0], 4),  // лево
        ([0, 1, 0], 3),   // верх
        ([0, -1, 0], 2),  // низ
        ([0, 0, 1], 1),   // перед
        ([0, 0, -1], 0),  // зад
    ];

    // Собираем вершины и индексы параллельно
    let (vertices_list, indices_list): (Vec<Vec<[f32; 3]>>, Vec<Vec<[u32; 3]>>) = centers_vec
        .par_iter()
        .map(|&[cx, cy, cz]| {
            let mut local_vertices = Vec::new();
            let mut local_indices = Vec::new();
            let mut vertex_offset_map = std::collections::HashMap::new();
            
            let discrete_coords = [
                (cx / cube_side).round() as i32,
                (cy / cube_side).round() as i32,
                (cz / cube_side).round() as i32,
            ];
            
            let mut next_vertex_idx = 0u32;
            
            for (face_idx, &face_verts) in face_indices.iter().enumerate() {
                let (offset, _) = neighbor_offsets.iter()
                    .find(|&&(_, of)| of == face_idx)
                    .unwrap();
                
                let neighbor_coords = [
                    discrete_coords[0] + offset[0],
                    discrete_coords[1] + offset[1],
                    discrete_coords[2] + offset[2],
                ];
                
                if centers_set.contains(&neighbor_coords) {
                    continue;
                }
                
                let mut face_vertex_indices = [0u32; 4];
                for (i, &vert_idx) in face_verts.iter().enumerate() {
                    let vertex = [
                        cx + cube_vertices[vert_idx][0],
                        cy + cube_vertices[vert_idx][1],
                        cz + cube_vertices[vert_idx][2],
                    ];
                    
                    let rounded_vertex = [
                        (vertex[0] * 1000.0).round() as i32,
                        (vertex[1] * 1000.0).round() as i32,
                        (vertex[2] * 1000.0).round() as i32,
                    ];
                    
                    let vertex_key = (rounded_vertex[0], rounded_vertex[1], rounded_vertex[2]);
                    
                    let global_idx = if let Some(&idx) = vertex_offset_map.get(&vertex_key) {
                        idx
                    } else {
                        let idx = next_vertex_idx;
                        vertex_offset_map.insert(vertex_key, idx);
                        local_vertices.push(vertex);
                        next_vertex_idx += 1;
                        idx
                    };
                    
                    face_vertex_indices[i] = global_idx;
                }
                
                local_indices.push([
                    face_vertex_indices[0],
                    face_vertex_indices[1],
                    face_vertex_indices[2],
                ]);
                local_indices.push([
                    face_vertex_indices[0],
                    face_vertex_indices[2],
                    face_vertex_indices[3],
                ]);
            }
            
            (local_vertices, local_indices)
        })
        .unzip();

    // Объединяем все локальные данные
    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();
    let mut global_vertex_offset = 0u32;
    
    for (vertices_chunk, indices_chunk) in vertices_list.into_iter().zip(indices_list.into_iter()) {
        all_vertices.extend(vertices_chunk.iter());
        
        for mut triangle in indices_chunk {
            for idx in triangle.iter_mut() {
                *idx += global_vertex_offset;
            }
            all_indices.push(triangle);
        }
        
        global_vertex_offset += vertices_chunk.len() as u32;
    }
    
    // Удаляем дубликаты вершин
    let (unique_vertices, vertex_map) = deduplicate_vertices(&all_vertices);
    
    // Перенумеровываем индексы
    let mut unique_indices = Vec::new();
    for triangle in all_indices {
        unique_indices.push([
            vertex_map[&create_vertex_key(&all_vertices[triangle[0] as usize])],
            vertex_map[&create_vertex_key(&all_vertices[triangle[1] as usize])],
            vertex_map[&create_vertex_key(&all_vertices[triangle[2] as usize])],
        ]);
    }
    
    // Конвертируем в формат для numpy
    let vertices_2d: Vec<Vec<f32>> = unique_vertices
        .iter()
        .map(|&[x, y, z]| vec![x, y, z])
        .collect();
    
    let indices_2d: Vec<Vec<u32>> = unique_indices
        .iter()
        .map(|&[a, b, c]| vec![a, b, c])
        .collect();
    
    let vertices_array = PyArray2::from_vec2_bound(py, &vertices_2d)?;
    let indices_array = PyArray2::from_vec2_bound(py, &indices_2d)?;
    
    Ok((vertices_array, indices_array))
}

fn create_vertex_key(vertex: &[f32; 3]) -> (i32, i32, i32) {
    (
        (vertex[0] * 10000.0).round() as i32,
        (vertex[1] * 10000.0).round() as i32,
        (vertex[2] * 10000.0).round() as i32,
    )
}

fn deduplicate_vertices(vertices: &[[f32; 3]]) -> (Vec<[f32; 3]>, std::collections::HashMap<(i32, i32, i32), u32>) {
    let mut unique_vertices = Vec::new();
    let mut vertex_map = std::collections::HashMap::new();
    let mut next_index = 0u32;
    
    for vertex in vertices {
        let key = create_vertex_key(vertex);
        if !vertex_map.contains_key(&key) {
            vertex_map.insert(key, next_index);
            unique_vertices.push(*vertex);
            next_index += 1;
        }
    }
    
    (unique_vertices, vertex_map)
}

#[pyfunction]
pub fn build_voxel_mesh_with_normals<'py>(
    py: Python<'py>,
    centers: PyReadonlyArray2<'py, f32>,
    cube_side: f32,
) -> PyResult<PyObject> {
    let (vertices_array, indices_array) = build_voxel_mesh(py, centers, cube_side)?;
    
    // Получаем данные из массивов
    let vertices = unsafe { vertices_array.as_array() };
    let indices = unsafe { indices_array.as_array() };
    
    let num_vertices = vertices.nrows();
    let mut normals = vec![[0.0f32; 3]; num_vertices];
    
    for tri_idx in 0..indices.nrows() {
        let idx0 = indices[[tri_idx, 0]] as usize;
        let idx1 = indices[[tri_idx, 1]] as usize;
        let idx2 = indices[[tri_idx, 2]] as usize;
        
        let v0 = [vertices[[idx0, 0]], vertices[[idx0, 1]], vertices[[idx0, 2]]];
        let v1 = [vertices[[idx1, 0]], vertices[[idx1, 1]], vertices[[idx1, 2]]];
        let v2 = [vertices[[idx2, 0]], vertices[[idx2, 1]], vertices[[idx2, 2]]];
        
        let edge1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let edge2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        
        let normal = [
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0],
        ];
        
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if length > 0.0 {
            let inv_length = 1.0 / length;
            let normalized_normal = [
                normal[0] * inv_length,
                normal[1] * inv_length,
                normal[2] * inv_length,
            ];
            
            for &idx in &[idx0, idx1, idx2] {
                normals[idx][0] += normalized_normal[0];
                normals[idx][1] += normalized_normal[1];
                normals[idx][2] += normalized_normal[2];
            }
        }
    }
    
    for normal in &mut normals {
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if length > 0.0 {
            let inv_length = 1.0 / length;
            normal[0] *= inv_length;
            normal[1] *= inv_length;
            normal[2] *= inv_length;
        }
    }
    
    let normals_2d: Vec<Vec<f32>> = normals
        .iter()
        .map(|&[x, y, z]| vec![x, y, z])
        .collect();
    let normals_array = PyArray2::from_vec2_bound(py, &normals_2d)?;
    
    let tuple = PyTuple::new_bound(
        py,
        &[
            vertices_array.into_py(py),
            indices_array.into_py(py),
            normals_array.into_py(py)
        ]
    );
    Ok(tuple.into())
}