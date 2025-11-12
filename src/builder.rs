// File: builder.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use super::bvh::{self, Aabb};
use super::cvec::*;
use super::cvecvec::*;
use super::data::{self, *};

use super::{mesh::Mesh, MeshSet, SimArgs};
use more_asserts::*;
use na::{vector, Matrix2, Matrix2xX, Matrix3, Matrix3x2, Matrix3xX};
use std::collections::HashMap;

pub struct Props {
    pub edge: Vec<EdgeProp>,
    pub face: Vec<FaceProp>,
    pub tet: Vec<TetProp>,
    pub edge_params: Vec<EdgeParam>,
    pub face_params: Vec<FaceParam>,
    pub tet_params: Vec<TetParam>,
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

    // Update fixed flags based on vertex fix_index
    for (i, prop) in props.face.iter_mut().enumerate() {
        if mesh.mesh.mesh.face.column(i).iter().all(|&j| {
            let mut vp = VertexProp::default();
            for (fi, pair) in constraint.fix.iter().enumerate() {
                if pair.index as usize == j {
                    vp.fix_index = (fi + 1) as u32;
                }
            }
            vp.fix_index > 0
        }) {
            prop.fixed = true;
        }
    }
    for (i, prop) in props.edge.iter_mut().enumerate() {
        if mesh
            .mesh
            .mesh
            .edge
            .column(i)
            .iter()
            .all(|&j| constraint.fix.iter().any(|pair| pair.index as usize == j))
        {
            prop.fixed = true;
        }
    }
    for (i, prop) in props.tet.iter_mut().enumerate() {
        if mesh
            .mesh
            .mesh
            .tet
            .column(i)
            .iter()
            .all(|&j| constraint.fix.iter().any(|pair| pair.index as usize == j))
        {
            prop.fixed = true;
        }
    }

    // Props now contains final props and params directly
    let edge_props = &props.edge;
    let face_props = &props.face;
    let tet_props = &props.tet;
    let edge_params = &props.edge_params;
    let face_params = &props.face_params;
    let tet_params = &props.tet_params;

    // Build vertex props and params
    let mut vertex_prop = vec![VertexProp::default(); n_vert];
    let mut temp_vertex_params = Vec::new();
    for (i, pair) in constraint.fix.iter().enumerate() {
        vertex_prop[pair.index as usize].fix_index = (i + 1) as u32;
    }
    for (i, pair) in constraint.pull.iter().enumerate() {
        vertex_prop[pair.index as usize].pull_index = (i + 1) as u32;
    }

    // Aggregate vertex params from faces and edges
    for (i, face_prop) in face_props.iter().enumerate() {
        for &j in mesh.mesh.mesh.face.column(i).iter() {
            if i < n_shells || sim_args.include_face_mass {
                vertex_prop[j].mass += face_prop.mass / 3.0;
                vertex_prop[j].area += face_prop.area / 3.0;
            }
        }
    }
    for (i, edge_prop) in edge_props.iter().enumerate() {
        if i < mesh.mesh.mesh.rod_count {
            for &j in mesh.mesh.mesh.edge.column(i).iter() {
                vertex_prop[j].mass += edge_prop.mass / 2.0;
            }
        }
    }
    for (i, tet_prop) in tet_props.iter().enumerate() {
        for &j in mesh.mesh.mesh.tet.column(i).iter() {
            vertex_prop[j].mass += tet_prop.mass / 4.0;
            vertex_prop[j].volume += tet_prop.volume / 4.0;
        }
    }

    // Build vertex params by aggregating from face/edge params
    let mut vertex_param_map: HashMap<VertexParam, u32> = HashMap::new();
    for (j, vp) in vertex_prop.iter_mut().enumerate() {
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
            if ei < mesh.mesh.mesh.rod_count && ei < edge_props.len() {
                let edge_prop = &edge_props[ei];
                let edge_param = &edge_params[edge_prop.param_index as usize];
                ghat = edge_param.ghat;
                offset = edge_param.offset;
                friction = friction.max(edge_param.friction);
                found = true;
            }
        }

        if found {
            let vparam = VertexParam {
                ghat,
                offset,
                friction,
            };
            let param_idx = *vertex_param_map.entry(vparam).or_insert_with(|| {
                let new_idx = temp_vertex_params.len() as u32;
                temp_vertex_params.push(vparam);
                new_idx
            });
            vp.param_index = param_idx;
        }
    }

    // Build hinge props and params
    let mut temp_hinge_props = Vec::new();
    let mut temp_hinge_params = Vec::new();
    let mut hinge_param_map: HashMap<HingeParam, u32> = HashMap::new();

    for (i, hinge) in mesh.mesh.mesh.hinge.column_iter().enumerate() {
        let x = vertex.column(hinge[0]) - vertex.column(hinge[1]);
        let length = x.norm();
        let mut offset_sum = 0.0;
        let mut ghat_sum = 0.0;
        let mut bend_sum = 0.0;
        let mut area_sum = 0.0;
        let mut all_fixed = true;
        for &j in mesh.mesh.neighbor.hinge.face[i].iter() {
            let face_prop = &face_props[j];
            let face_param = &face_params[face_prop.param_index as usize];
            all_fixed = all_fixed && face_prop.fixed;
            offset_sum += face_prop.area * face_param.offset;
            ghat_sum += face_prop.area * face_param.ghat;
            bend_sum += face_prop.area * face_param.bend;
            area_sum += face_prop.area;
        }
        assert_gt!(area_sum, 0.0);
        let hparam = HingeParam {
            bend: if all_fixed { 0.0 } else { bend_sum / area_sum },
            ghat: ghat_sum / area_sum,
            offset: offset_sum / area_sum,
        };
        let param_idx = *hinge_param_map.entry(hparam).or_insert_with(|| {
            let new_idx = temp_hinge_params.len() as u32;
            temp_hinge_params.push(hparam);
            new_idx
        });
        temp_hinge_props.push(HingeProp {
            fixed: all_fixed,
            length,
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

    let n_surface_vert = mesh.mesh.mesh.surface_vert_count;
    let surface_vert: Matrix3xX<f32> = vertex.columns(0, n_surface_vert).into_owned();
    let bvh_set = BvhSet {
        face: build_bvh(vertex, Some(&mesh.mesh.mesh.face)),
        edge: build_bvh(vertex, Some(&mesh.mesh.mesh.edge)),
        vertex: build_bvh::<1>(&surface_vert, None),
    };
    let mut inv_rest2x2 = Vec::new();
    for i in 0..mesh.mesh.mesh.shell_face_count {
        let f = mesh.mesh.mesh.face.column(i);
        let mut d_mat_inv = None;
        if let Some(uv) = uv.as_ref() {
            let (x0, x1, x2) = (uv[i].column(0), uv[i].column(1), uv[i].column(2));
            let (y0, y1, y2) = (
                vertex.column(f[0]),
                vertex.column(f[1]),
                vertex.column(f[2]),
            );
            let (lx0, lx1) = ((x1 - x0).norm(), (x2 - x0).norm());
            let (ly0, ly1) = ((y1 - y0).norm(), (y2 - y0).norm());
            let face_param = &face_params[face_props[i].param_index as usize];
            let shrink_factor = face_param.shrink;
            let d_mat = Matrix2::<f32>::from_columns(&[
                shrink_factor * ly0 * (x1 - x0) / lx0,
                shrink_factor * ly1 * (x2 - x0) / lx1,
            ]);
            if lx0 > 0.0 && lx1 > 0.0 && d_mat.norm_squared() > 0.0 {
                d_mat_inv = d_mat.try_inverse();
            }
        }
        if let Some(d_mat_inv) = d_mat_inv {
            inv_rest2x2.push(d_mat_inv);
        } else {
            let (x0, x1, x2) = (
                vertex.column(f[0]),
                vertex.column(f[1]),
                vertex.column(f[2]),
            );
            let d_mat = Matrix3x2::<f32>::from_columns(&[(x1 - x0), (x2 - x0)]);
            let e1 = d_mat.column(0).cross(&d_mat.column(1));
            let e2 = e1.cross(&d_mat.column(0)).normalize();
            let proj_mat =
                Matrix3x2::<f32>::from_columns(&[d_mat.column(0).normalize(), e2]).transpose();
            let d_mat = proj_mat * d_mat;
            inv_rest2x2.push(d_mat.try_inverse().unwrap());
        }
    }

    let mut inv_rest3x3 = Vec::new();
    for tet in mesh.mesh.mesh.tet.column_iter() {
        let (x0, x1, x2, x3) = (
            vertex.column(tet[0]),
            vertex.column(tet[1]),
            vertex.column(tet[2]),
            vertex.column(tet[3]),
        );
        let mat = Matrix3::<f32>::from_columns(&[(x1 - x0), (x2 - x0), (x3 - x0)]);
        if let Some(inv_mat) = mat.try_inverse() {
            inv_rest3x3.push(inv_mat);
        } else {
            println!("{}", mat);
            panic!("Degenerate tetrahedron");
        }
    }

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
                .map(|(x, y)| x - (dt * y))
                .collect::<Vec<_>>()
                .as_slice(),
        ),
        curr: CVec::from(
            vertex
                .column_iter()
                .map(|x| x.into())
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
        bvh: bvh_set,
        fixed_index_table: CVecVec::from(&fixed_index_table[..]),
        transpose_table: CVecVec::from(&transpose_table[..]),
        rod_count: rod_count as u32,
        shell_face_count: shell_face_count as u32,
        surface_vert_count,
    }
}

pub fn make_param(args: &SimArgs) -> data::ParamSet {
    let dt = args.dt.min(0.9999 / args.fps as f32);
    let wind = match args.wind_dim {
        0 => Vec3f::new(args.wind, 0.0, 0.0),
        1 => Vec3f::new(0.0, args.wind, 0.0),
        2 => Vec3f::new(0.0, 0.0, args.wind),
        _ => panic!("Invalid wind dimension: {}", args.wind_dim),
    };
    data::ParamSet {
        time: 0.0,
        disable_contact: args.disable_contact,
        fitting: args.fitting,
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
        gravity: Vec3f::new(0.0, args.gravity, 0.0),
        wind,
        barrier: match args.barrier.as_str() {
            "cubic" => data::Barrier::Cubic,
            "quad" => data::Barrier::Quad,
            "log" => data::Barrier::Log,
            _ => panic!("Invalid barrier: {}", args.barrier),
        },
        csrmat_max_nnz: args.csrmat_max_nnz,
        bvh_alloc_factor: args.bvh_alloc_factor,
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

pub type ArbitrayElement<const N: usize> =
    na::Matrix<usize, na::Const<N>, na::Dyn, na::VecStorage<usize, na::Const<N>, na::Dyn>>;

pub fn build_bvh<const N: usize>(
    vertex: &Matrix3xX<f32>,
    elements: Option<&ArbitrayElement<N>>,
) -> Bvh {
    let aabb = if let Some(elements) = elements {
        bvh::generate_aabb(vertex, elements)
    } else {
        vertex
            .column_iter()
            .enumerate()
            .map(|(i, x)| {
                (
                    Aabb {
                        min: x.map(f64::from),
                        max: x.map(f64::from),
                    },
                    i,
                )
            })
            .collect::<Vec<_>>()
    };
    if !aabb.is_empty() {
        let tree = bvh::Tree::build_tree(aabb);
        let node = tree
            .node
            .iter()
            .map(|&x| match x {
                bvh::Node::Parent(left, right) => {
                    data::Vec2u::new(left as u32 + 1, right as u32 + 1)
                }
                bvh::Node::Leaf(i) => data::Vec2u::new(i as u32 + 1, 0),
            })
            .collect::<Vec<_>>();
        data::Bvh {
            node: CVec::from(&node[..]),
            level: CVecVec::from(&tree.level.to_u32()[..]),
        }
    } else {
        data::Bvh {
            node: CVec::new(),
            level: CVecVec::new(),
        }
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
        };
        let param_idx = *edge_param_map.entry(param).or_insert_with(|| {
            let new_idx = unique_edge_params.len() as u32;
            unique_edge_params.push(param);
            new_idx
        });
        collision_edge_props.push(EdgeProp {
            length: 0.0,
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
            fix_index: 0,
            pull_index: 0,
            param_index: param_idx,
        });
    }

    CollisionMesh {
        vertex: CVec::from(
            vertex
                .column_iter()
                .map(|x| x.into())
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
        face_bvh: build_bvh(vertex, Some(&mesh.mesh.face)),
        edge_bvh: build_bvh(vertex, Some(&mesh.mesh.edge)),
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
