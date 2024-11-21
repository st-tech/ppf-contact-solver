// File: bvh.rs
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use more_asserts::*;
use na::{vector, Matrix3xX, Vector3};
use rayon::prelude::*;

#[derive(Default)]
pub struct Tree {
    pub node: Vec<Node>,
    pub level: Vec<Vec<usize>>,
}

#[derive(Copy, Clone, Debug)]
pub struct Aabb {
    pub min: Vector3<f32>,
    pub max: Vector3<f32>,
}

#[derive(Copy, Clone)]
pub enum Node {
    Parent(usize, usize),
    Leaf(usize),
}

impl Tree {
    pub fn build_tree(aggregate: Vec<(Aabb, usize)>) -> Tree {
        assert_ge!(aggregate.len(), 1);
        let mut node = Vec::new();
        let root = subdivide(aggregate, &mut node);
        node.push(root);
        let level = gather_level(&node);
        Tree { node, level }
    }
}

fn gather_level(node: &[Node]) -> Vec<Vec<usize>> {
    let mut level = Vec::new();
    let mut queue = vec![node.len() - 1];
    while !queue.is_empty() {
        let mut next_queue = Vec::new();
        for &i in &queue {
            match node[i] {
                Node::Parent(left, right) => {
                    next_queue.push(left);
                    next_queue.push(right);
                }
                Node::Leaf(_) => {}
            }
        }
        level.push(queue);
        queue = next_queue;
    }
    level.reverse();
    level
}

pub fn generate_aabb<const N: usize>(
    vertex: &Matrix3xX<f32>,
    element: &na::Matrix<
        usize,
        na::Const<N>,
        na::Dyn,
        na::VecStorage<usize, na::Const<N>, na::Dyn>,
    >,
) -> Vec<(Aabb, usize)> {
    element
        .as_slice()
        .par_chunks(N)
        .enumerate()
        .map(|(i, elm)| {
            let x = Matrix3xX::from_fn(N, |i, j| vertex.column(elm[j])[i]);
            (
                Aabb {
                    min: Vector3::<f32>::from_fn(|i, _| x.row(i).min()),
                    max: Vector3::<f32>::from_fn(|i, _| x.row(i).max()),
                },
                i,
            )
        })
        .collect()
}

fn subdivide(mut obj: Vec<(Aabb, usize)>, node: &mut Vec<Node>) -> Node {
    if obj.len() > 1 {
        let mean = compute_mean(obj.iter().map(|(aabb, _)| aabb));
        let var = compute_var(obj.iter().map(|(aabb, _)| aabb), mean);
        let (dim, _) = var.argmax();
        obj.sort_by(|a, b| {
            let a = a.0.min[dim] + a.0.max[dim];
            let b = b.0.min[dim] + b.0.max[dim];
            a.partial_cmp(&b).unwrap()
        });
        let obj_len = obj.len();
        let left_obj = Vec::from_iter(obj.drain(..obj_len / 2));
        let right_obj = obj;
        let (left_node, right_node) = (subdivide(left_obj, node), subdivide(right_obj, node));
        let left_index = node.len();
        node.push(left_node);
        let right_index = node.len();
        node.push(right_node);
        Node::Parent(left_index, right_index)
    } else {
        assert!(!obj.is_empty());
        let entry = obj.pop().unwrap();
        Node::Leaf(entry.1)
    }
}

pub fn compute_mean<'a>(aabb_iter: impl Iterator<Item = &'a Aabb>) -> Vector3<f32> {
    let (count, sum) = aabb_iter.fold((0, Vector3::<f32>::zeros()), |acc, x| {
        let mean = (x.min + x.max) / 2.0;
        (acc.0 + 1, acc.1 + mean)
    });
    sum / count as f32
}

pub fn compute_var<'a>(
    aabb_iter: impl Iterator<Item = &'a Aabb>,
    mean: Vector3<f32>,
) -> Vector3<f32> {
    aabb_iter.fold(Vector3::<f32>::zeros(), |acc, aabb| {
        let m = (aabb.min + aabb.max) / 2.0;
        let x = (mean.x - m.x) * (mean.x - m.x);
        let y = (mean.y - m.y) * (mean.y - m.y);
        let z = (mean.z - m.z) * (mean.z - m.z);
        vector![x, y, z] + acc
    })
}
