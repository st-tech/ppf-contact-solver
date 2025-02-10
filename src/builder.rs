// File: builder.rs
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use super::bvh::{self};
use super::cvec::*;
use super::cvecvec::*;
use super::data::{self, *};
use super::{mesh::Mesh, Args, MeshSet};
use na::{vector, Matrix2, Matrix2xX, Matrix3, Matrix3x2, Matrix3xX};

pub struct Props {
    pub rod: Vec<RodProp>,
    pub face: Vec<FaceProp>,
    pub tet: Vec<TetProp>,
}

pub fn build(
    args: &Args,
    mesh: &MeshSet,
    velocity: &Matrix3xX<f32>,
    props: &Props,
    constraint: Constraint,
) -> data::DataSet {
    let dt = args.dt.min(0.9999 / args.fps as f32);
    let (vertex, uv) = (&mesh.vertex, &mesh.uv);
    let n_vert = vertex.ncols();
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

    let mut vertex_prop = vec![
        VertexProp {
            area: 0.0,
            volume: 0.0,
            mass: 0.0
        };
        n_vert
    ];
    for (i, prop) in props.rod.iter().enumerate() {
        for &j in mesh.mesh.mesh.edge.column(i).iter() {
            let area = prop.length * prop.radius;
            vertex_prop[j].mass += prop.mass / 2.0;
            vertex_prop[j].area += area / 2.0;
        }
    }
    for (i, face) in mesh.mesh.mesh.face.column_iter().enumerate() {
        for &j in face.iter() {
            vertex_prop[j].mass += props.face[i].mass / 3.0;
            vertex_prop[j].area += props.face[i].area / 3.0;
        }
    }
    for (i, tet) in mesh.mesh.mesh.tet.column_iter().enumerate() {
        for &j in tet.iter() {
            vertex_prop[j].mass += props.tet[i].mass / 4.0;
            vertex_prop[j].volume += props.tet[i].volume / 4.0;
        }
    }

    let mut hinge_prop = Vec::new();
    for hinge in mesh.mesh.mesh.hinge.column_iter() {
        let x = vertex.column(hinge[0]) - vertex.column(hinge[1]);
        let length = x.norm();
        hinge_prop.push(HingeProp { length });
    }
    let prop_set = PropSet {
        vertex: CVec::from(vertex_prop.as_ref()),
        rod: CVec::from(props.rod.as_ref()),
        face: CVec::from(props.face.as_ref()),
        hinge: CVec::from(hinge_prop.as_ref()),
        tet: CVec::from(props.tet.as_ref()),
    };

    let bvh_set = BvhSet {
        face: build_bvh(vertex, &mesh.mesh.mesh.face),
        edge: build_bvh(vertex, &mesh.mesh.mesh.edge),
        vertex: build_bvh(vertex, &mesh.mesh.mesh.vertex),
    };
    let mut inv_rest2x2 = Vec::new();
    for i in 0..mesh.mesh.mesh.shell_face_count {
        let f = mesh.mesh.mesh.face.column(i);
        let mut d_mat_inv = None;
        if let Some(uv) = uv.as_ref() {
            let (x0, x1, x2) = (uv.column(f[0]), uv.column(f[1]), uv.column(f[2]));
            let d_mat = Matrix2::<f32>::from_columns(&[x1 - x0, x2 - x0]);
            if d_mat.norm_squared() > 0.0 {
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
            let d_mat = Matrix3x2::<f32>::from_columns(&[x1 - x0, x2 - x0]);
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
        let mat = Matrix3::<f32>::from_columns(&[x1 - x0, x2 - x0, x3 - x0]);
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
                .map(|(x, y)| x - dt * y)
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
        inv_rest2x2,
        inv_rest3x3,
        constraint,
        prop: prop_set,
        bvh: bvh_set,
        fixed_index_table: CVecVec::from(&fixed_index_table[..]),
        transpose_table: CVecVec::from(&transpose_table[..]),
        rod_count: rod_count as u32,
        shell_face_count: shell_face_count as u32,
        surface_vert_count,
    }
}

pub fn make_param(args: &Args) -> data::ParamSet {
    let dt = args.dt.min(0.9999 / args.fps as f32);
    let strain_limit_eps = if args.disable_strain_limit {
        0.0
    } else {
        args.strain_limit_eps
    };
    let wind = match args.wind_dim {
        0 => Vec3f::new(args.wind, 0.0, 0.0),
        1 => Vec3f::new(0.0, args.wind, 0.0),
        2 => Vec3f::new(0.0, 0.0, args.wind),
        _ => panic!("Invalid wind dimension: {}", args.wind_dim),
    };
    data::ParamSet {
        time: 0.0,
        fitting: false,
        air_friction: args.air_friction,
        air_density: args.air_density,
        strain_limit_eps,
        contact_ghat: args.contact_ghat,
        contact_offset: args.contact_offset,
        rod_offset: args.rod_offset,
        constraint_ghat: args.constraint_ghat,
        constraint_tol: args.constraint_tol,
        prev_dt: dt,
        dt,
        playback: args.playback,
        min_newton_steps: args.min_newton_steps,
        target_toi: args.target_toi,
        bend: args.bend,
        rod_bend: args.rod_bend,
        stitch_stiffness: args.stitch_stiffness,
        cg_max_iter: args.cg_max_iter,
        cg_tol: args.cg_tol,
        line_search_max_t: args.line_search_max_t,
        ccd_tol: args.contact_ghat,
        ccd_reduction: args.ccd_reduction,
        eiganalysis_eps: args.eiganalysis_eps,
        friction: args.friction,
        friction_eps: args.friction_eps,
        isotropic_air_friction: args.isotropic_air_friction,
        gravity: Vec3f::new(0.0, args.gravity, 0.0),
        wind,
        model_shell: match args.model_shell.as_str() {
            "arap" => data::Model::Arap,
            "stvk" => data::Model::StVK,
            "snhk" => data::Model::SNHk,
            "baraffwitkin" => data::Model::BaraffWitkin,
            _ => panic!("Invalid model: {}", args.model_shell),
        },
        model_tet: match args.model_tet.as_str() {
            "arap" => data::Model::Arap,
            "stvk" => data::Model::StVK,
            "snhk" => data::Model::SNHk,
            _ => panic!("Invalid model: {}", args.model_tet),
        },
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

pub fn build_bvh<const N: usize>(vertex: &Matrix3xX<f32>, elements: &ArbitrayElement<N>) -> Bvh {
    let aabb = bvh::generate_aabb(vertex, elements);
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

pub fn make_collision_mesh(vertex: &Matrix3xX<f32>, face: &Matrix3xX<usize>) -> CollisionMesh {
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
    CollisionMesh {
        active: true,
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
        face_bvh: build_bvh(vertex, &mesh.mesh.face),
        edge_bvh: build_bvh(vertex, &mesh.mesh.edge),
        neighbor,
    }
}
