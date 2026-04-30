// File: builder.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use super::cvec::*;
use super::cvecvec::*;
use super::data::{self, *};
use super::{mesh::Mesh, MeshSet, SimArgs};
use more_asserts::*;
use na::{vector, Matrix2, Matrix2xX, Matrix3, Matrix3x2, Matrix3xX};
use rayon::prelude::*;
use std::collections::HashMap;

pub struct Props {
    pub edge: Vec<EdgeProp>,
    pub face: Vec<FaceProp>,
    pub tet: Vec<TetProp>,
    pub edge_params: Vec<EdgeParam>,
    pub face_params: Vec<FaceParam>,
    pub tet_params: Vec<TetParam>,
}

/// Signed dihedral angle between the two faces sharing the edge
/// `v0-v1` with opposite vertices `v2` and `v3`. Mirrors the device-side
/// `face_dihedral_angle` after `remap(hinge)` in `dihedral_angle.hpp`:
/// that kernel reorders the hinge to (v[2], v[1], v[0], v[3]) and computes
/// the angle, so here the hinge column `(h0, h1, h2, h3)` maps to
/// `v0 = h2, v1 = h1, v2 = h0, v3 = h3`. Callers pass the reordered verts.
fn signed_dihedral_angle(
    v0: &na::Vector3<f32>,
    v1: &na::Vector3<f32>,
    v2: &na::Vector3<f32>,
    v3: &na::Vector3<f32>,
) -> f32 {
    let n1 = (v1 - v0).cross(&(v2 - v0));
    let n2 = (v2 - v3).cross(&(v1 - v3));
    let n1_sq = n1.norm_squared();
    let n2_sq = n2.norm_squared();
    if n1_sq <= 0.0 || n2_sq <= 0.0 {
        return 0.0;
    }
    let dot = n1.dot(&n2) / (n1_sq * n2_sq).sqrt();
    let angle = dot.clamp(-1.0, 1.0).acos();
    if n2.cross(&n1).dot(&(v1 - v2)) < 0.0 {
        -angle
    } else {
        angle
    }
}

pub fn build(
    sim_args: &SimArgs,
    mesh: &MeshSet,
    velocity: &Matrix3xX<f32>,
    props: &mut Props,
    constraint: Constraint,
) -> data::DataSet {
    let dt = sim_args.dt.min(0.9999 / sim_args.fps as f32);
    let (vertex, uv) = (&mesh.vertex, &mesh.uv);
    let n_vert = vertex.ncols();
    let n_shells = mesh.mesh.mesh.shell_face_count;
    let rod_count = mesh.mesh.mesh.rod_count;
    let neighbor = Neighbor {
        vertex: VertexNeighbor {
            face: CVecVec::from(&mesh.mesh.neighbor.vertex.face.to_u32()[..]),
            hinge: CVecVec::from(&mesh.mesh.neighbor.vertex.hinge.to_u32()[..]),
            edge: CVecVec::from(&mesh.mesh.neighbor.vertex.edge.to_u32()[..]),
            rod: CVecVec::from(&mesh.mesh.neighbor.vertex.rod.to_u32()[..]),
        },
        hinge: HingeNeighbor {
            face: CVecVec::from(&mesh.mesh.neighbor.hinge.face.to_u32()[..]),
        },
        edge: EdgeNeighbor {
            face: CVecVec::from(&mesh.mesh.neighbor.edge.face.to_u32()[..]),
        },
    };

    // Update fixed flags based on vertex fix_index (parallelized)
    // Pre-collect fixed vertex indices into a HashSet for thread-safe access
    use std::collections::HashSet;
    let fixed_vertices: HashSet<usize> = constraint
        .fix
        .iter()
        .map(|pair| pair.index as usize)
        .collect();

    props.face.par_iter_mut().enumerate().for_each(|(i, prop)| {
        if mesh
            .mesh
            .mesh
            .face
            .column(i)
            .iter()
            .all(|&j| fixed_vertices.contains(&j))
        {
            prop.fixed = true;
        }
    });
    props.edge.par_iter_mut().enumerate().for_each(|(i, prop)| {
        if mesh
            .mesh
            .mesh
            .edge
            .column(i)
            .iter()
            .all(|&j| fixed_vertices.contains(&j))
        {
            prop.fixed = true;
        }
    });
    props.tet.par_iter_mut().enumerate().for_each(|(i, prop)| {
        if mesh
            .mesh
            .mesh
            .tet
            .column(i)
            .iter()
            .all(|&j| fixed_vertices.contains(&j))
        {
            prop.fixed = true;
        }
    });

    // Props now contains final props and params directly
    let edge_props = &props.edge;
    let face_props = &props.face;
    let tet_props = &props.tet;
    let edge_params = &props.edge_params;
    let face_params = &props.face_params;
    let tet_params = &props.tet_params;

    // Build vertex props and params
    let mut vertex_prop = vec![VertexProp::default(); n_vert];
    // rod-bend rest angle defaults to π (straight); mutated below for interior
    // rod vertices whose adjacent edge params request rest-from-geometry.
    for vp in vertex_prop.iter_mut() {
        vp.rest_bend_angle = std::f32::consts::PI;
    }
    let mut temp_vertex_params = Vec::new();
    for (i, pair) in constraint.fix.iter().enumerate() {
        vertex_prop[pair.index as usize].fix_index = (i + 1) as u32;
    }
    for (i, pair) in constraint.pull.iter().enumerate() {
        vertex_prop[pair.index as usize].pull_index = (i + 1) as u32;
    }

    // Populate rod-bend rest angle for interior rod vertices whose adjacent
    // rod-edges request rest-from-geometry. Matches the device-side gate at
    // src/cpp/energy/energy.cu:333: vertex has exactly 2 rod edges and 0
    // faces.
    for j in 0..n_vert {
        let adj_edges: Vec<usize> = mesh
            .mesh
            .neighbor
            .vertex
            .edge[j]
            .iter()
            .copied()
            .filter(|&ei| ei < rod_count)
            .collect();
        if adj_edges.len() != 2 {
            continue;
        }
        if !mesh.mesh.neighbor.vertex.face[j].is_empty() {
            continue;
        }
        let from_geometry = adj_edges.iter().any(|&ei| {
            let edge_prop = &edge_props[ei];
            edge_params[edge_prop.param_index as usize]
                .bend_rest_from_geometry
        });
        if !from_geometry {
            continue;
        }
        let edge_0 = mesh.mesh.mesh.edge.column(adj_edges[0]);
        let edge_1 = mesh.mesh.mesh.edge.column(adj_edges[1]);
        let other_0 = if edge_0[0] == j { edge_0[1] } else { edge_0[0] };
        let other_1 = if edge_1[0] == j { edge_1[1] } else { edge_1[0] };
        let x0 = vertex.column(other_0).into_owned();
        let x1 = vertex.column(j).into_owned();
        let x2 = vertex.column(other_1).into_owned();
        let e0 = x0 - x1;
        let e1 = x2 - x1;
        let n0 = e0.norm();
        let n1 = e1.norm();
        if n0 <= 0.0 || n1 <= 0.0 {
            continue;
        }
        let cos_theta = (e0.dot(&e1) / (n0 * n1)).clamp(-1.0, 1.0);
        vertex_prop[j].rest_bend_angle = cos_theta.acos();
    }

    // Aggregate vertex area from all faces (needed for wind/air forces on all surfaces)
    let area_contributions: Vec<(usize, f32)> = face_props
        .par_iter()
        .enumerate()
        .flat_map(|(i, face_prop)| {
            mesh.mesh
                .mesh
                .face
                .column(i)
                .iter()
                .map(|&j| (j, face_prop.area / 3.0))
                .collect::<Vec<_>>()
        })
        .collect();
    for (j, area) in area_contributions {
        vertex_prop[j].area += area;
    }

    // Aggregate vertex mass from faces (shell faces always, solid faces only if include_face_mass)
    let mass_contributions: Vec<(usize, f32)> = face_props
        .par_iter()
        .enumerate()
        .filter(|(i, _)| *i < n_shells || sim_args.include_face_mass)
        .flat_map(|(i, face_prop)| {
            mesh.mesh
                .mesh
                .face
                .column(i)
                .iter()
                .map(|&j| (j, face_prop.mass / 3.0))
                .collect::<Vec<_>>()
        })
        .collect();
    for (j, mass) in mass_contributions {
        vertex_prop[j].mass += mass;
    }

    // Aggregate vertex mass from edges (parallelized collection, sequential merge)
    let edge_contributions: Vec<(usize, f32)> = edge_props
        .par_iter()
        .enumerate()
        .filter(|(i, _)| *i < rod_count)
        .flat_map(|(i, edge_prop)| {
            mesh.mesh
                .mesh
                .edge
                .column(i)
                .iter()
                .map(|&j| (j, edge_prop.mass / 2.0))
                .collect::<Vec<_>>()
        })
        .collect();
    for (j, mass) in edge_contributions {
        vertex_prop[j].mass += mass;
    }

    // Aggregate vertex mass/volume from tets (parallelized collection, sequential merge)
    let tet_contributions: Vec<(usize, f32, f32)> = tet_props
        .par_iter()
        .enumerate()
        .flat_map(|(i, tet_prop)| {
            mesh.mesh
                .mesh
                .tet
                .column(i)
                .iter()
                .map(|&j| (j, tet_prop.mass / 4.0, tet_prop.volume / 4.0))
                .collect::<Vec<_>>()
        })
        .collect();
    for (j, mass, volume) in tet_contributions {
        vertex_prop[j].mass += mass;
        vertex_prop[j].volume += volume;
    }

    // Build vertex params by aggregating from face/edge params (parallelized)
    // Phase 1: Parallel computation of vertex params
    let vertex_params_data: Vec<Option<VertexParam>> = (0..n_vert)
        .into_par_iter()
        .map(|j| {
            let mut ghat = 0.0f32;
            let mut offset = 0.0f32;
            let mut friction = 0.0f32;
            let mut found = false;

            // Aggregate from faces
            for &fi in mesh.mesh.neighbor.vertex.face[j].iter() {
                if fi < face_props.len() {
                    let face_prop = &face_props[fi];
                    let face_param = &face_params[face_prop.param_index as usize];
                    ghat = face_param.ghat;
                    offset = face_param.offset;
                    friction = friction.max(face_param.friction);
                    found = true;
                }
            }

            // Aggregate from edges (for rods)
            for &ei in mesh.mesh.neighbor.vertex.edge[j].iter() {
                if ei < rod_count && ei < edge_props.len() {
                    let edge_prop = &edge_props[ei];
                    let edge_param = &edge_params[edge_prop.param_index as usize];
                    ghat = edge_param.ghat;
                    offset = edge_param.offset;
                    friction = friction.max(edge_param.friction);
                    found = true;
                }
            }

            if found {
                Some(VertexParam { ghat, offset, friction })
            } else {
                None
            }
        })
        .collect();

    // Phase 2: Sequential deduplication
    let mut vertex_param_map: HashMap<VertexParam, u32> = HashMap::new();
    for (j, vparam_opt) in vertex_params_data.into_iter().enumerate() {
        if let Some(vparam) = vparam_opt {
            let param_idx = *vertex_param_map.entry(vparam).or_insert_with(|| {
                let new_idx = temp_vertex_params.len() as u32;
                temp_vertex_params.push(vparam);
                new_idx
            });
            vertex_prop[j].param_index = param_idx;
        }
    }

    // Build hinge props and params (parallelized)
    // Phase 1: Parallel computation of hinge data
    let hinge_columns: Vec<_> = mesh.mesh.mesh.hinge.column_iter().collect();
    let hinge_data: Vec<(f32, f32, bool, HingeParam)> = hinge_columns
        .into_par_iter()
        .enumerate()
        .map(|(i, hinge)| {
            let x = vertex.column(hinge[0]) - vertex.column(hinge[1]);
            let length = x.norm();
            let mut offset_sum = 0.0;
            let mut ghat_sum = 0.0;
            let mut bend_sum = 0.0;
            let mut plasticity_sum = 0.0;
            let mut plasticity_threshold_sum = 0.0;
            let mut area_sum = 0.0;
            let mut all_fixed = true;
            let mut from_geometry = false;
            for &j in mesh.mesh.neighbor.hinge.face[i].iter() {
                let face_prop = &face_props[j];
                let face_param = &face_params[face_prop.param_index as usize];
                all_fixed = all_fixed && face_prop.fixed;
                offset_sum += face_prop.area * face_param.offset;
                ghat_sum += face_prop.area * face_param.ghat;
                bend_sum += face_prop.area * face_param.bend;
                plasticity_sum += face_prop.area * face_param.bend_plasticity;
                plasticity_threshold_sum +=
                    face_prop.area * face_param.bend_plasticity_threshold;
                area_sum += face_prop.area;
                if face_param.bend_rest_from_geometry {
                    from_geometry = true;
                }
            }
            assert_gt!(area_sum, 0.0);
            let hparam = HingeParam {
                bend: if all_fixed { 0.0 } else { bend_sum / area_sum },
                ghat: ghat_sum / area_sum,
                offset: offset_sum / area_sum,
                plasticity: plasticity_sum / area_sum,
                plasticity_threshold: plasticity_threshold_sum / area_sum,
            };
            let rest_angle = if from_geometry {
                // Mirror the device-side `remap` in dihedral_angle.hpp:14
                // before calling face_dihedral_angle: (h2, h1, h0, h3).
                let v0 = vertex.column(hinge[2]).into_owned();
                let v1 = vertex.column(hinge[1]).into_owned();
                let v2 = vertex.column(hinge[0]).into_owned();
                let v3 = vertex.column(hinge[3]).into_owned();
                signed_dihedral_angle(&v0, &v1, &v2, &v3)
            } else {
                0.0
            };
            (length, rest_angle, all_fixed, hparam)
        })
        .collect();

    // Phase 2: Sequential deduplication
    let mut temp_hinge_props = Vec::with_capacity(hinge_data.len());
    let mut temp_hinge_params = Vec::new();
    let mut hinge_param_map: HashMap<HingeParam, u32> = HashMap::new();
    for (length, rest_angle, all_fixed, hparam) in hinge_data {
        let param_idx = *hinge_param_map.entry(hparam).or_insert_with(|| {
            let new_idx = temp_hinge_params.len() as u32;
            temp_hinge_params.push(hparam);
            new_idx
        });
        temp_hinge_props.push(HingeProp {
            fixed: all_fixed,
            length,
            rest_angle,
            param_index: param_idx,
        });
    }

    let prop_set = PropSet {
        vertex: CVec::from(vertex_prop.as_ref()),
        edge: CVec::from(edge_props.as_ref()),
        face: CVec::from(face_props.as_ref()),
        hinge: CVec::from(temp_hinge_props.as_ref()),
        tet: CVec::from(tet_props.as_ref()),
    };

    let param_arrays = ParamArrays {
        vertex: CVec::from(temp_vertex_params.as_slice()),
        edge: CVec::from(edge_params.as_slice()),
        face: CVec::from(face_params.as_slice()),
        hinge: CVec::from(temp_hinge_params.as_slice()),
        tet: CVec::from(tet_params.as_slice()),
    };

    // Parallel computation of inv_rest2x2 matrices
    let rest_v = mesh.rest_vertex.as_ref().unwrap_or(&mesh.vertex);
    let inv_rest2x2: Vec<Matrix2<f32>> = (0..mesh.mesh.mesh.shell_face_count)
        .into_par_iter()
        .map(|i| {
            let f = mesh.mesh.mesh.face.column(i);
            // Compute rest-pose from 3D geometry projected into tangent plane
            let (x0, x1, x2) = (
                rest_v.column(f[0]),
                rest_v.column(f[1]),
                rest_v.column(f[2]),
            );
            let dx = Matrix3x2::<f32>::from_columns(&[
                x1 - x0,
                x2 - x0,
            ]);
            let n = dx.column(0).cross(&dx.column(1));
            let e2 = n.cross(&dx.column(0)).normalize();
            let proj_mat =
                Matrix3x2::<f32>::from_columns(&[dx.column(0).normalize(), e2]).transpose();
            let d_mat = proj_mat * dx;
            // When UV data exists, rotate d_mat to align with the UV first-edge
            // direction and apply shrink. This preserves UV orientation for
            // Baraff-Witkin anisotropy while keeping F ≈ I at rest.
            let d_mat = if let Some(uv) = uv.as_ref() {
                let uv_e0 = uv[i].column(1) - uv[i].column(0);
                let lu0 = uv_e0.norm();
                if lu0 > 0.0 {
                    let uv_dir = uv_e0 / lu0;
                    // Rotation from (1,0) to uv_dir
                    let rot = Matrix2::<f32>::new(
                        uv_dir[0], -uv_dir[1],
                        uv_dir[1],  uv_dir[0],
                    );
                    let face_param = &face_params[face_props[i].param_index as usize];
                    debug_assert!(
                        face_param.shrink_x > 0.0 && face_param.shrink_y > 0.0,
                        "FaceParam::shrink_x/shrink_y uninitialized"
                    );
                    let shrink = Matrix2::<f32>::new(
                        face_param.shrink_x, 0.0,
                        0.0, face_param.shrink_y,
                    );
                    shrink * rot * d_mat
                } else {
                    d_mat
                }
            } else {
                d_mat
            };
            d_mat.try_inverse().unwrap()
        })
        .collect();

    // Parallel computation of inv_rest3x3 matrices
    let tet_columns: Vec<_> = mesh.mesh.mesh.tet.column_iter().collect();
    let inv_rest3x3: Vec<Matrix3<f32>> = tet_columns
        .into_par_iter()
        .enumerate()
        .map(|(i, tet)| {
            let (x0, x1, x2, x3) = (
                rest_v.column(tet[0]),
                rest_v.column(tet[1]),
                rest_v.column(tet[2]),
                rest_v.column(tet[3]),
            );
            let tet_param = &tet_params[tet_props[i].param_index as usize];
            let s = tet_param.shrink;
            debug_assert!(s > 0.0, "TetParam::shrink uninitialized");
            let mat = s * Matrix3::<f32>::from_columns(&[
                x1 - x0,
                x2 - x0,
                x3 - x0,
            ]);
            if let Some(inv_mat) = mat.try_inverse() {
                inv_mat
            } else {
                println!("{}", mat);
                panic!("Degenerate tetrahedron");
            }
        })
        .collect();

    let inv_rest2x2 = CVec::from(&inv_rest2x2[..]);
    let inv_rest3x3 = CVec::from(&inv_rest3x3[..]);
    let vertex_count = mesh.mesh.mesh.vertex_count as u32;
    let surface_vert_count = mesh.mesh.mesh.surface_vert_count as u32;

    let mut fixed_index_table = vec![Vec::new(); vertex_count as usize];
    let mut insert = |i: usize, j: usize| {
        if i <= j {
            let mut index = 0;
            while index < fixed_index_table[i].len() && fixed_index_table[i][index] < j as u32 {
                index += 1;
            }
            if index == fixed_index_table[i].len() {
                fixed_index_table[i].push(j as u32);
            } else if fixed_index_table[i][index] != j as u32 {
                fixed_index_table[i].insert(index, j as u32);
            }
        }
    };
    for i in 0..vertex_count {
        insert(i as usize, i as usize);
    }
    for f in mesh.mesh.mesh.edge.column_iter() {
        for k1 in 0..2 {
            for k2 in 0..2 {
                insert(f[k1], f[k2]);
            }
        }
    }
    for f in mesh.mesh.mesh.face.column_iter() {
        for k1 in 0..3 {
            for k2 in 0..3 {
                insert(f[k1], f[k2]);
            }
        }
    }
    for hinge in mesh.mesh.mesh.hinge.column_iter() {
        for k1 in 0..4 {
            for k2 in 0..4 {
                insert(hinge[k1], hinge[k2]);
            }
        }
    }
    for tet in mesh.mesh.mesh.tet.column_iter() {
        for k1 in 0..4 {
            for k2 in 0..4 {
                insert(tet[k1], tet[k2]);
            }
        }
    }
    for seam in constraint.stitch.iter() {
        for &i in seam.index.iter() {
            for &j in seam.index.iter() {
                insert(i as usize, j as usize);
            }
        }
    }
    let mut transpose_table = vec![Vec::new(); vertex_count as usize];
    let mut index_sum = 0;
    for (i, row) in fixed_index_table.iter().enumerate() {
        for (k, &j) in row.iter().enumerate() {
            if i as u32 != j {
                transpose_table[j as usize].push(vector![i as u32, (index_sum + k) as u32]);
            }
        }
        index_sum += row.len();
    }

    let num_face = mesh.mesh.mesh.face.ncols();
    let shell_face_count = mesh.mesh.mesh.shell_face_count;
    let rod_count = mesh.mesh.mesh.rod_count;
    let mut face_type = vec![0_u8; num_face];
    let mut vertex_type = vec![0_u8; vertex_count as usize];
    let mut hinge_type = vec![0_u8; mesh.mesh.mesh.hinge.ncols()];
    for (i, x) in face_type.iter_mut().enumerate() {
        if i >= shell_face_count {
            *x |= 1;
        }
    }
    for &i in mesh.mesh.mesh.tet.iter() {
        vertex_type[i] |= 1;
    }
    for (i, face_neighbors) in mesh.mesh.neighbor.hinge.face.iter().enumerate() {
        for &f in face_neighbors {
            if face_type[f] & 1 == 1 {
                hinge_type[i] |= 1;
                break;
            }
        }
    }
    let ttype = data::Type {
        face: CVec::from(&face_type[..]),
        vertex: CVec::from(&vertex_type[..]),
        hinge: CVec::from(&hinge_type[..]),
    };

    let mesh = data::Mesh {
        face: CVec::from(
            mesh.mesh
                .mesh
                .face
                .map(|x| x as u32)
                .column_iter()
                .map(|x| x.into())
                .collect::<Vec<_>>()
                .as_slice(),
        ),
        edge: CVec::from(
            mesh.mesh
                .mesh
                .edge
                .map(|x| x as u32)
                .column_iter()
                .map(|x| x.into())
                .collect::<Vec<_>>()
                .as_slice(),
        ),
        hinge: CVec::from(
            mesh.mesh
                .mesh
                .hinge
                .map(|x| x as u32)
                .column_iter()
                .map(|x| x.into())
                .collect::<Vec<_>>()
                .as_slice(),
        ),
        tet: CVec::from(
            mesh.mesh
                .mesh
                .tet
                .map(|x| x as u32)
                .column_iter()
                .map(|x| x.into())
                .collect::<Vec<_>>()
                .as_slice(),
        ),
    };
    let mesh_info = data::MeshInfo {
        mesh,
        neighbor,
        ttype,
    };
    let vertex = VertexSet {
        prev: CVec::from(
            vertex
                .column_iter()
                .zip(velocity.column_iter())
                .map(|(x, y)| x - dt * y)
                .collect::<Vec<_>>()
                .as_slice(),
        ),
        curr: CVec::from(
            vertex
                .column_iter()
                .map(|x| x.into_owned())
                .collect::<Vec<_>>()
                .as_slice(),
        ),
    };

    data::DataSet {
        vertex,
        mesh: mesh_info,
        prop: prop_set,
        param_arrays,
        inv_rest2x2,
        inv_rest3x3,
        constraint,
        fixed_index_table: CVecVec::from(&fixed_index_table[..]),
        transpose_table: CVecVec::from(&transpose_table[..]),
        rod_count: rod_count as u32,
        shell_face_count: shell_face_count as u32,
        surface_vert_count,
    }
}

pub fn make_param(args: &SimArgs) -> data::ParamSet {
    let dt = args.dt.min(0.9999 / args.fps as f32);
    let wind = Vec3f::new(args.wind[0], args.wind[1], args.wind[2]);
    data::ParamSet {
        time: 0.0,
        disable_contact: args.disable_contact,
        inactive_momentum: args.inactive_momentum,
        air_friction: args.air_friction,
        air_density: args.air_density,
        constraint_tol: args.constraint_tol,
        prev_dt: dt,
        dt,
        playback: args.playback,
        min_newton_steps: args.min_newton_steps,
        target_toi: args.target_toi,
        stitch_stiffness: args.stitch_stiffness,
        cg_max_iter: args.cg_max_iter,
        cg_tol: args.cg_tol,
        line_search_max_t: args.line_search_max_t,
        ccd_eps: args.ccd_eps,
        ccd_reduction: args.ccd_reduction,
        ccd_max_iter: args.ccd_max_iter,
        max_dx: args.max_dx,
        eiganalysis_eps: args.eiganalysis_eps,
        friction_eps: args.friction_eps,
        isotropic_air_friction: args.isotropic_air_friction,
        gravity: Vec3f::new(args.gravity[0], args.gravity[1], args.gravity[2]),
        wind,
        barrier: match args.barrier.as_str() {
            "cubic" => data::Barrier::Cubic,
            "quad" => data::Barrier::Quad,
            "log" => data::Barrier::Log,
            _ => panic!("Invalid barrier: {}", args.barrier),
        },
        friction_mode: match args.friction_mode.as_str() {
            "min" => data::FrictionMode::Min,
            "max" => data::FrictionMode::Max,
            "mean" => data::FrictionMode::Mean,
            _ => panic!("Invalid friction-mode: {}", args.friction_mode),
        },
        csrmat_max_nnz: args.csrmat_max_nnz,
        fix_xz: args.fix_xz,
    }
}

pub fn copy_to_dataset(
    curr_vertex: &Matrix3xX<f32>,
    prev_vertex: &Matrix3xX<f32>,
    dataset: &mut data::DataSet,
) {
    let vertex = VertexSet {
        prev: CVec::from(
            prev_vertex
                .column_iter()
                .map(|x| x.into())
                .collect::<Vec<_>>()
                .as_slice(),
        ),
        curr: CVec::from(
            curr_vertex
                .column_iter()
                .map(|x| x.into())
                .collect::<Vec<_>>()
                .as_slice(),
        ),
    };
    dataset.vertex = vertex;
}

trait ConvertToU32 {
    fn to_u32(&self) -> Vec<Vec<u32>>;
}

impl ConvertToU32 for Vec<Vec<usize>> {
    fn to_u32(&self) -> Vec<Vec<u32>> {
        self.iter()
            .map(|inner_vec| inner_vec.iter().map(|&x| x as u32).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    }
}

pub fn convert_prop(young_mod: f32, poiss_rat: f32) -> (f32, f32) {
    let mu = young_mod / (2.0 * (1.0 + poiss_rat));
    let lambda = young_mod * poiss_rat / ((1.0 + poiss_rat) * (1.0 - 2.0 * poiss_rat));
    (mu, lambda)
}

pub fn make_collision_mesh(
    vertex: &Matrix3xX<f32>,
    face: &Matrix3xX<usize>,
    face_props: &[FaceProp],
    face_params: &[FaceParam],
) -> CollisionMesh {
    let mesh = Mesh::new(
        Matrix2xX::<usize>::zeros(0),
        face.clone(),
        na::Matrix4xX::zeros(0),
        face.ncols(),
    );
    let neighbor = Neighbor {
        vertex: VertexNeighbor {
            face: CVecVec::from(&mesh.neighbor.vertex.face.to_u32()[..]),
            hinge: CVecVec::from(&mesh.neighbor.vertex.hinge.to_u32()[..]),
            edge: CVecVec::from(&mesh.neighbor.vertex.edge.to_u32()[..]),
            rod: CVecVec::from(&mesh.neighbor.vertex.rod.to_u32()[..]),
        },
        hinge: HingeNeighbor {
            face: CVecVec::from(&mesh.neighbor.hinge.face.to_u32()[..]),
        },
        edge: EdgeNeighbor {
            face: CVecVec::from(&mesh.neighbor.edge.face.to_u32()[..]),
        },
    };
    let n_vert = vertex.ncols();
    let n_edge = mesh.mesh.edge.ncols();
    let n_face = face.ncols();
    assert_eq!(n_face, face_props.len());

    // Build edge props and params
    let mut edge_param_map: HashMap<EdgeParam, u32> = HashMap::new();
    let mut unique_edge_params = Vec::new();
    let mut collision_edge_props = Vec::new();

    for i in 0..n_edge {
        let mut ghat_sum = 0.0;
        let mut offset_sum = 0.0;
        let mut friction_sum = 0.0;
        let mut area_sum = 0.0;
        for &j in mesh.neighbor.edge.face[i].iter() {
            let face_prop = &face_props[j];
            let face_param = &face_params[face_prop.param_index as usize];
            ghat_sum += face_param.ghat * face_prop.area;
            offset_sum += face_param.offset * face_prop.area;
            friction_sum += face_param.friction * face_prop.area;
            area_sum += face_prop.area;
        }
        assert_gt!(area_sum, 0.0);
        let param = EdgeParam {
            stiffness: 0.0,
            bend: 0.0,
            ghat: ghat_sum / area_sum,
            offset: offset_sum / area_sum,
            friction: friction_sum / area_sum,
            strainlimit: 0.0,
            plasticity: 0.0,
            plasticity_threshold: 0.0,
            bend_rest_from_geometry: false,
        };
        let param_idx = *edge_param_map.entry(param).or_insert_with(|| {
            let new_idx = unique_edge_params.len() as u32;
            unique_edge_params.push(param);
            new_idx
        });
        collision_edge_props.push(EdgeProp {
            length: 0.0,
            initial_length: 0.0,
            mass: 0.0,
            fixed: false,
            param_index: param_idx,
        });
    }

    // Build vertex props and params
    let mut vertex_param_map: HashMap<VertexParam, u32> = HashMap::new();
    let mut unique_vertex_params = Vec::new();
    let mut collision_vertex_props = Vec::new();

    for i in 0..n_vert {
        let mut ghat_sum = 0.0;
        let mut offset_sum = 0.0;
        let mut friction_sum = 0.0;
        let mut area_sum = 0.0;
        for &j in mesh.neighbor.vertex.face[i].iter() {
            let face_prop = &face_props[j];
            let face_param = &face_params[face_prop.param_index as usize];
            ghat_sum += face_param.ghat * face_prop.area;
            offset_sum += face_param.offset * face_prop.area;
            friction_sum += face_param.friction * face_prop.area;
            area_sum += face_prop.area;
        }
        assert_gt!(area_sum, 0.0);
        let param = VertexParam {
            ghat: ghat_sum / area_sum,
            offset: offset_sum / area_sum,
            friction: friction_sum / area_sum,
        };
        let param_idx = *vertex_param_map.entry(param).or_insert_with(|| {
            let new_idx = unique_vertex_params.len() as u32;
            unique_vertex_params.push(param);
            new_idx
        });
        collision_vertex_props.push(VertexProp {
            area: 0.0,
            volume: 0.0,
            mass: 0.0,
            rest_bend_angle: std::f32::consts::PI,
            fix_index: 0,
            pull_index: 0,
            param_index: param_idx,
        });
    }

    CollisionMesh {
        vertex: CVec::from(
            vertex
                .column_iter()
                .map(|x| x.into_owned())
                .collect::<Vec<_>>()
                .as_slice(),
        ),
        face: CVec::from(
            mesh.mesh
                .face
                .column_iter()
                .map(|x| x.map(|x| x as u32))
                .collect::<Vec<_>>()
                .as_slice(),
        ),
        edge: CVec::from(
            mesh.mesh
                .edge
                .column_iter()
                .map(|x| x.map(|x| x as u32))
                .collect::<Vec<_>>()
                .as_slice(),
        ),
        prop: CollisionMeshPropSet {
            vertex: CVec::from(collision_vertex_props.as_slice()),
            edge: CVec::from(collision_edge_props.as_slice()),
            face: CVec::from(face_props),
        },
        param_arrays: CollisionMeshParamArrays {
            vertex: CVec::from(unique_vertex_params.as_slice()),
            face: CVec::from(face_params),
            edge: CVec::from(unique_edge_params.as_slice()),
        },
        neighbor,
    }
}
