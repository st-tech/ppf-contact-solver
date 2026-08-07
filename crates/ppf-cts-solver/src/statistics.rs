// File: crates/ppf-cts-solver/src/statistics.rs
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! Solver-side per-object statistics for exact emitted output poses.

use std::path::Path;

use ppf_cts_formats::statistics::{
    encode_statistics_frame, encode_statistics_manifest, ObjectStatistics, StatisticChannel,
    StatisticsFrame, StatisticsInput, StatisticsManifest, StatisticsObject,
};

use crate::data::DataSet;
use crate::mesh::MeshInfo;

pub struct TimelineStatistics {
    input: StatisticsInput,
    manifest: StatisticsManifest,
    rest_area: Vec<Option<f64>>,
    rest_volume: Vec<Option<f64>>,
    rest_length: Vec<Option<f64>>,
    previous_location: Vec<Option<[f64; 3]>>,
    previous_velocity: Vec<Option<[f64; 3]>>,
    previous_positions: Vec<Option<Vec<[f64; 3]>>>,
    output_dir: String,
    fps: f64,
}

impl TimelineStatistics {
    pub fn load(input_dir: &str, output_dir: &str, mesh: &MeshInfo, fps: f64) -> Option<Self> {
        let path = Path::new(input_dir).join(ppf_cts_formats::files::STATISTICS_INPUT);
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
            Err(error) => panic!("failed to read {}: {error}", path.display()),
        };
        let input = ppf_cts_formats::statistics::decode_statistics_input(&bytes)
            .unwrap_or_else(|error| panic!("failed to decode {}: {error}", path.display()));
        validate_input(&input, mesh);
        let manifest = StatisticsManifest {
            objects: input
                .objects
                .iter()
                .map(|object| StatisticsObject {
                    object_index: object.object_index,
                    object_uuid: object.object_uuid.clone(),
                    object_name: object.object_name.clone(),
                    dynamics_type: object.dynamics_type.clone(),
                    supported_channels: supported_channels(object),
                })
                .collect(),
        };
        manifest
            .validate()
            .expect("solver produced an invalid statistics manifest");
        let manifest_bytes =
            encode_statistics_manifest(&manifest).expect("failed to encode statistics manifest");
        let manifest_path = Path::new(output_dir).join(ppf_cts_formats::files::STATISTICS_MANIFEST);
        match std::fs::read(&manifest_path) {
            Ok(existing) => assert_eq!(
                existing, manifest_bytes,
                "statistics manifest changed across a resumed run"
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                write_atomic(&manifest_path, &manifest_bytes);
            }
            Err(error) => panic!("failed to read {}: {error}", manifest_path.display()),
        }
        let object_count = input.objects.len();
        Some(Self {
            input,
            manifest,
            rest_area: vec![None; object_count],
            rest_volume: vec![None; object_count],
            rest_length: vec![None; object_count],
            previous_location: vec![None; object_count],
            previous_velocity: vec![None; object_count],
            previous_positions: vec![None; object_count],
            output_dir: output_dir.to_string(),
            fps,
        })
    }

    pub fn configure_dataset(&self, dataset: &mut DataSet) {
        let mut dynamic = vec![u32::MAX; dataset.vertex.curr.size as usize];
        let mut static_mesh = vec![u32::MAX; dataset.constraint.mesh.vertex.size as usize];
        for object in &self.input.objects {
            let target = if object.static_object {
                &mut static_mesh
            } else {
                &mut dynamic
            };
            for &vertex in &object.vertex_indices {
                let slot = target.get_mut(vertex as usize).unwrap_or_else(|| {
                    panic!(
                        "statistics object {} vertex {} out of range",
                        object.object_index, vertex
                    )
                });
                assert_eq!(
                    *slot,
                    u32::MAX,
                    "statistics vertex {vertex} belongs to multiple objects"
                );
                *slot = object.object_index;
            }
        }
        dataset.statistics_object_index = crate::cvec::CVec::from(dynamic.as_slice());
        dataset.statistics_static_object_index = crate::cvec::CVec::from(static_mesh.as_slice());
        dataset.statistics_contact_count =
            crate::cvec::CVec::from(vec![0_u32; self.input.objects.len()].as_slice());
    }

    pub fn analyze_and_write(
        &mut self,
        solver_frame: u32,
        time_seconds: f64,
        positions: &[[f64; 3]],
        mesh: &MeshInfo,
        dataset: &DataSet,
        inverse_world_scale: f64,
    ) {
        let dynamic_masses = dataset.prop.vertex.as_slice();
        let static_masses = dataset.constraint.mesh.prop.vertex.as_slice();
        let static_positions: Vec<[f64; 3]> = dataset
            .constraint
            .mesh
            .vertex
            .as_slice()
            .iter()
            .map(|position| {
                [
                    f64::from(position[0]) * inverse_world_scale,
                    f64::from(position[1]) * inverse_world_scale,
                    f64::from(position[2]) * inverse_world_scale,
                ]
            })
            .collect();
        let static_faces = dataset.constraint.mesh.face.as_slice();
        let mut records = Vec::with_capacity(self.input.objects.len());
        for (object_position, object) in self.input.objects.iter().enumerate() {
            let mut record = ObjectStatistics {
                object_index: object.object_index,
                ..Default::default()
            };
            if !cfg!(feature = "emulated") {
                if let Some(&count) = dataset
                    .statistics_contact_count
                    .as_slice()
                    .get(object.object_index as usize)
                {
                    record.contact_count = count as u64;
                    record.valid_channels |= StatisticChannel::ContactCount.bit();
                }
            }
            let (object_positions, masses) = if object.static_object {
                (static_positions.as_slice(), static_masses)
            } else {
                (positions, dynamic_masses)
            };
            for &index in &object.vertex_indices {
                assert!(
                    (index as usize) < object_positions.len(),
                    "statistics object {} vertex {} out of range",
                    object.object_index,
                    index
                );
            }
            let location = center(object, object_positions, masses);
            record.location = cast3(location);
            record.valid_channels |= vector_bits(
                StatisticChannel::LocationX,
                StatisticChannel::LocationY,
                StatisticChannel::LocationZ,
            );

            let area = if object.static_object {
                static_surface_area(object, object_positions, static_faces)
            } else {
                surface_area(object, object_positions, mesh)
            };
            if let Some(area) = area {
                record.surface_area = area as f32;
                record.valid_channels |= StatisticChannel::SurfaceArea.bit();
                let rest = self.rest_area[object_position].get_or_insert(area);
                if *rest > 0.0 {
                    record.area_stretch = (area / *rest) as f32;
                    record.valid_channels |= StatisticChannel::AreaStretch.bit();
                }
            }

            let length = if object.static_object {
                None
            } else {
                rod_length(object, object_positions, mesh)
            };
            if let Some(length) = length {
                record.rod_length = length as f32;
                record.valid_channels |= StatisticChannel::RodLength.bit();
                let rest = self.rest_length[object_position].get_or_insert(length);
                if *rest > 0.0 {
                    record.length_stretch = (length / *rest) as f32;
                    record.valid_channels |= StatisticChannel::LengthStretch.bit();
                }
            }

            let object_volume = if object.static_object {
                static_volume(object, object_positions, static_faces)
            } else {
                volume(object, object_positions, mesh)
            };
            if let Some(volume) = object_volume {
                record.volume = volume as f32;
                record.valid_channels |= StatisticChannel::Volume.bit();
                let rest = self.rest_volume[object_position].get_or_insert(volume);
                if *rest > 0.0 {
                    record.volume_stretch = (volume / *rest) as f32;
                    record.valid_channels |= StatisticChannel::VolumeStretch.bit();
                }
            }

            if let Some(previous) = self.previous_location[object_position] {
                let velocity = scale(sub(location, previous), self.fps);
                record.velocity = cast3(velocity);
                record.speed = norm(velocity) as f32;
                record.valid_channels |= vector_bits(
                    StatisticChannel::VelocityX,
                    StatisticChannel::VelocityY,
                    StatisticChannel::VelocityZ,
                ) | StatisticChannel::Speed.bit();
                if let Some(previous_velocity) = self.previous_velocity[object_position] {
                    let acceleration = scale(sub(velocity, previous_velocity), self.fps);
                    record.acceleration = cast3(acceleration);
                    record.acceleration_magnitude = norm(acceleration) as f32;
                    record.valid_channels |= vector_bits(
                        StatisticChannel::AccelerationX,
                        StatisticChannel::AccelerationY,
                        StatisticChannel::AccelerationZ,
                    ) | StatisticChannel::AccelerationMagnitude.bit();
                }
                self.previous_velocity[object_position] = Some(velocity);
            }
            if let Some(previous_positions) = &self.previous_positions[object_position] {
                if let Some(angular_velocity) = angular_velocity(
                    object,
                    object_positions,
                    previous_positions,
                    location,
                    cast3_to_f64(record.velocity),
                    masses,
                    self.fps,
                ) {
                    record.angular_velocity = cast3(angular_velocity);
                    record.angular_speed = norm(angular_velocity) as f32;
                    record.valid_channels |= vector_bits(
                        StatisticChannel::AngularVelocityX,
                        StatisticChannel::AngularVelocityY,
                        StatisticChannel::AngularVelocityZ,
                    ) | StatisticChannel::AngularSpeed.bit();
                    if record.angular_speed > 0.0 {
                        record.angular_axis =
                            cast3(scale(angular_velocity, 1.0 / record.angular_speed as f64));
                        record.valid_channels |= vector_bits(
                            StatisticChannel::AngularAxisX,
                            StatisticChannel::AngularAxisY,
                            StatisticChannel::AngularAxisZ,
                        );
                    }
                }
            }
            self.previous_location[object_position] = Some(location);
            self.previous_positions[object_position] = Some(
                object
                    .vertex_indices
                    .iter()
                    .map(|&index| object_positions[index as usize])
                    .collect(),
            );
            records.push(record);
        }

        let frame = StatisticsFrame {
            solver_frame,
            time_seconds,
            objects: records,
        };
        frame
            .validate(&self.manifest)
            .expect("solver produced an invalid statistics frame");
        let bytes = encode_statistics_frame(&frame).expect("failed to encode statistics frame");
        let path = Path::new(&self.output_dir).join(ppf_cts_formats::files::statistics_filename(
            solver_frame as i32,
        ));
        write_atomic(&path, &bytes);
    }
}

fn validate_input(input: &StatisticsInput, mesh: &MeshInfo) {
    for (position, object) in input.objects.iter().enumerate() {
        assert_eq!(
            object.object_index, position as u32,
            "statistics input object order mismatch at {position}"
        );
        assert!(
            !object.object_uuid.is_empty(),
            "statistics input object {position} has an empty UUID"
        );
        if object.static_object {
            continue;
        }
        for &vertex in &object.vertex_indices {
            assert!(
                (vertex as usize) < mesh.vertex_count,
                "statistics object {position} vertex {vertex} out of range"
            );
        }
        validate_range("rod", position, object.rod_range, mesh.edge.ncols());
        validate_range("face", position, object.face_range, mesh.face.ncols());
        validate_range("tet", position, object.tet_range, mesh.tet.ncols());
        if let Some(radius) = object.grain_radius {
            assert!(
                radius.is_finite() && radius > 0.0,
                "statistics object {position} has invalid grain radius {radius}"
            );
        }
    }
}

fn validate_range(kind: &str, object: usize, range: [u32; 2], total: usize) {
    let start = range[0] as usize;
    let count = range[1] as usize;
    assert!(
        start <= total && count <= total - start,
        "statistics object {object} {kind} range [{start}, {count}] exceeds {total}"
    );
}

fn supported_channels(object: &ppf_cts_formats::statistics::StatisticsInputObject) -> u64 {
    let mut channels = vector_bits(
        StatisticChannel::LocationX,
        StatisticChannel::LocationY,
        StatisticChannel::LocationZ,
    ) | vector_bits(
        StatisticChannel::VelocityX,
        StatisticChannel::VelocityY,
        StatisticChannel::VelocityZ,
    ) | StatisticChannel::Speed.bit()
        | vector_bits(
            StatisticChannel::AccelerationX,
            StatisticChannel::AccelerationY,
            StatisticChannel::AccelerationZ,
        )
        | StatisticChannel::AccelerationMagnitude.bit();
    channels |= vector_bits(
        StatisticChannel::AngularVelocityX,
        StatisticChannel::AngularVelocityY,
        StatisticChannel::AngularVelocityZ,
    ) | StatisticChannel::AngularSpeed.bit()
        | vector_bits(
            StatisticChannel::AngularAxisX,
            StatisticChannel::AngularAxisY,
            StatisticChannel::AngularAxisZ,
        );
    if object.face_range[1] > 0 || object.dynamics_type == "SAND" {
        channels |= StatisticChannel::SurfaceArea.bit() | StatisticChannel::AreaStretch.bit();
    }
    if object.tet_range[1] > 0 || object.closed_surface || object.dynamics_type == "SAND" {
        channels |= StatisticChannel::Volume.bit() | StatisticChannel::VolumeStretch.bit();
    }
    if object.rod_range[1] > 0 {
        channels |= StatisticChannel::RodLength.bit() | StatisticChannel::LengthStretch.bit();
    }
    if !cfg!(feature = "emulated") {
        channels |= StatisticChannel::ContactCount.bit();
    }
    channels
}

fn vector_bits(x: StatisticChannel, y: StatisticChannel, z: StatisticChannel) -> u64 {
    x.bit() | y.bit() | z.bit()
}

fn center(
    object: &ppf_cts_formats::statistics::StatisticsInputObject,
    positions: &[[f64; 3]],
    masses: &[crate::data::VertexProp],
) -> [f64; 3] {
    assert!(
        !object.vertex_indices.is_empty(),
        "statistics object {} has no vertices",
        object.object_index
    );
    let mut weighted = [0.0; 3];
    let mut total_mass = 0.0;
    for &index in &object.vertex_indices {
        let index = index as usize;
        let mass = masses.get(index).map_or(0.0, |prop| prop.mass as f64);
        if mass > 0.0 {
            weighted = add(weighted, scale(positions[index], mass));
            total_mass += mass;
        }
    }
    if total_mass > 0.0 {
        return scale(weighted, 1.0 / total_mass);
    }
    let mut sum = [0.0; 3];
    for &index in &object.vertex_indices {
        sum = add(sum, positions[index as usize]);
    }
    scale(sum, 1.0 / object.vertex_indices.len() as f64)
}

fn surface_area(
    object: &ppf_cts_formats::statistics::StatisticsInputObject,
    positions: &[[f64; 3]],
    mesh: &MeshInfo,
) -> Option<f64> {
    if object.dynamics_type == "SAND" {
        let radius = object.grain_radius?;
        return Some(
            object.vertex_indices.len() as f64
                * 4.0
                * std::f64::consts::PI
                * radius as f64
                * radius as f64,
        );
    }
    let [start, count] = object.face_range;
    if count == 0 {
        return None;
    }
    let mut area = 0.0;
    for face_index in start as usize..(start + count) as usize {
        let face = mesh.face.column(face_index);
        let a = positions[face[0]];
        let b = positions[face[1]];
        let c = positions[face[2]];
        area += 0.5 * norm(cross(sub(b, a), sub(c, a)));
    }
    Some(area)
}

fn rod_length(
    object: &ppf_cts_formats::statistics::StatisticsInputObject,
    positions: &[[f64; 3]],
    mesh: &MeshInfo,
) -> Option<f64> {
    let [start, count] = object.rod_range;
    if count == 0 {
        return None;
    }
    let mut length = 0.0;
    for edge_index in start as usize..(start + count) as usize {
        let edge = mesh.edge.column(edge_index);
        length += norm(sub(positions[edge[1]], positions[edge[0]]));
    }
    Some(length)
}

fn volume(
    object: &ppf_cts_formats::statistics::StatisticsInputObject,
    positions: &[[f64; 3]],
    mesh: &MeshInfo,
) -> Option<f64> {
    if object.dynamics_type == "SAND" {
        let radius = object.grain_radius?;
        return Some(
            object.vertex_indices.len() as f64
                * (4.0 / 3.0)
                * std::f64::consts::PI
                * radius as f64
                * radius as f64
                * radius as f64,
        );
    }
    let [tet_start, tet_count] = object.tet_range;
    if tet_count > 0 {
        let mut total = 0.0;
        for tet_index in tet_start as usize..(tet_start + tet_count) as usize {
            let tet = mesh.tet.column(tet_index);
            let anchor = positions[tet[0]];
            total += dot(
                sub(positions[tet[1]], anchor),
                cross(
                    sub(positions[tet[2]], anchor),
                    sub(positions[tet[3]], anchor),
                ),
            ) / 6.0;
        }
        return Some(total.abs());
    }
    if !object.closed_surface || object.face_range[1] == 0 {
        return None;
    }
    let anchor = positions[*object.vertex_indices.first()? as usize];
    let [face_start, face_count] = object.face_range;
    let mut total = 0.0;
    for face_index in face_start as usize..(face_start + face_count) as usize {
        let face = mesh.face.column(face_index);
        total += dot(
            sub(positions[face[0]], anchor),
            cross(
                sub(positions[face[1]], anchor),
                sub(positions[face[2]], anchor),
            ),
        ) / 6.0;
    }
    Some(total.abs())
}

fn static_surface_area(
    object: &ppf_cts_formats::statistics::StatisticsInputObject,
    positions: &[[f64; 3]],
    faces: &[crate::data::Vec3u],
) -> Option<f64> {
    let [start, count] = object.face_range;
    if count == 0 {
        return None;
    }
    assert!(
        (start + count) as usize <= faces.len(),
        "statistics static face range out of bounds"
    );
    let mut area = 0.0;
    for face in &faces[start as usize..(start + count) as usize] {
        let a = positions[face[0] as usize];
        let b = positions[face[1] as usize];
        let c = positions[face[2] as usize];
        area += 0.5 * norm(cross(sub(b, a), sub(c, a)));
    }
    Some(area)
}

fn static_volume(
    object: &ppf_cts_formats::statistics::StatisticsInputObject,
    positions: &[[f64; 3]],
    faces: &[crate::data::Vec3u],
) -> Option<f64> {
    if !object.closed_surface || object.face_range[1] == 0 {
        return None;
    }
    let anchor = positions[*object.vertex_indices.first()? as usize];
    let [start, count] = object.face_range;
    assert!(
        (start + count) as usize <= faces.len(),
        "statistics static face range out of bounds"
    );
    let mut total = 0.0;
    for face in &faces[start as usize..(start + count) as usize] {
        total += dot(
            sub(positions[face[0] as usize], anchor),
            cross(
                sub(positions[face[1] as usize], anchor),
                sub(positions[face[2] as usize], anchor),
            ),
        ) / 6.0;
    }
    Some(total.abs())
}

fn angular_velocity(
    object: &ppf_cts_formats::statistics::StatisticsInputObject,
    positions: &[[f64; 3]],
    previous_positions: &[[f64; 3]],
    location: [f64; 3],
    linear_velocity: [f64; 3],
    masses: &[crate::data::VertexProp],
    fps: f64,
) -> Option<[f64; 3]> {
    assert_eq!(
        previous_positions.len(),
        object.vertex_indices.len(),
        "statistics angular history size mismatch"
    );
    let has_positive_mass = object.vertex_indices.iter().any(|&index| {
        masses
            .get(index as usize)
            .is_some_and(|property| property.mass > 0.0)
    });
    let mut matrix = [[0.0_f64; 3]; 3];
    let mut rhs = [0.0_f64; 3];
    for (local_index, &global_index) in object.vertex_indices.iter().enumerate() {
        let global_index = global_index as usize;
        let weight = if has_positive_mass {
            masses
                .get(global_index)
                .map_or(0.0, |property| property.mass as f64)
        } else {
            1.0
        };
        if weight == 0.0 {
            continue;
        }
        let current = positions[global_index];
        let radius = sub(current, location);
        let point_velocity = scale(sub(current, previous_positions[local_index]), fps);
        let relative_velocity = sub(point_velocity, linear_velocity);
        let radius_sq = dot(radius, radius);
        for row in 0..3 {
            for col in 0..3 {
                matrix[row][col] += weight
                    * if row == col {
                        radius_sq - radius[row] * radius[col]
                    } else {
                        -radius[row] * radius[col]
                    };
            }
        }
        rhs = add(rhs, scale(cross(radius, relative_velocity), weight));
    }
    solve_3x3(matrix, rhs)
}

fn solve_3x3(matrix: [[f64; 3]; 3], rhs: [f64; 3]) -> Option<[f64; 3]> {
    let matrix = na::Matrix3::from_row_slice(&[
        matrix[0][0],
        matrix[0][1],
        matrix[0][2],
        matrix[1][0],
        matrix[1][1],
        matrix[1][2],
        matrix[2][0],
        matrix[2][1],
        matrix[2][2],
    ]);
    let eigen = na::linalg::SymmetricEigen::new(matrix);
    let largest = eigen.eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    if largest == 0.0 {
        return None;
    }
    // A straight rod has one genuinely unobservable rotational axis. Use the
    // symmetric pseudoinverse so its two observable components remain
    // available, dropping only eigenvalues at the round-off scale of this
    // three-row decomposition.
    let noise = 64.0 * f64::EPSILON * largest;
    let rhs = na::Vector3::from_row_slice(&rhs);
    let mut solution = na::Vector3::zeros();
    let mut resolved = false;
    for column in 0..3 {
        let value = eigen.eigenvalues[column];
        assert!(
            value >= -noise,
            "statistics angular inertia matrix is not positive semidefinite"
        );
        if value <= noise {
            continue;
        }
        let axis = eigen.eigenvectors.column(column);
        solution += axis * (axis.dot(&rhs) / value);
        resolved = true;
    }
    (resolved && solution.iter().all(|value| value.is_finite())).then_some([
        solution[0],
        solution[1],
        solution[2],
    ])
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn cast3(value: [f64; 3]) -> [f32; 3] {
    [value[0] as f32, value[1] as f32, value[2] as f32]
}

fn cast3_to_f64(value: [f32; 3]) -> [f64; 3] {
    [value[0] as f64, value[1] as f64, value[2] as f64]
}

fn write_atomic(path: &Path, bytes: &[u8]) {
    let temporary = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or_default()
    ));
    {
        let mut file = std::fs::File::create(&temporary)
            .unwrap_or_else(|error| panic!("failed to create {}: {error}", temporary.display()));
        std::io::Write::write_all(&mut file, bytes)
            .unwrap_or_else(|error| panic!("failed to write {}: {error}", temporary.display()));
        file.sync_all()
            .unwrap_or_else(|error| panic!("failed to sync {}: {error}", temporary.display()));
    }
    std::fs::rename(&temporary, path).unwrap_or_else(|error| {
        panic!(
            "failed to rename {} to {}: {error}",
            temporary.display(),
            path.display()
        )
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angular_pseudoinverse_keeps_observable_rod_components() {
        let result = solve_3x3(
            [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            [0.0, 2.0, 4.0],
        )
        .unwrap();
        assert_eq!(result, [0.0, 1.0, 2.0]);
    }
}
