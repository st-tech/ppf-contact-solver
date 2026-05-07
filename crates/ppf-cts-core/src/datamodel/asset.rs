// File: crates/ppf-cts-core/src/datamodel/asset.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Asset registry data model:
//
//   * AssetRegistry maps `name -> Asset` and rejects duplicates.
//   * Asset is a tagged enum (`Tri`, `Tet`, `Rod`, `Stitch`).
//   * Numeric arrays are `ndarray::Array2` so PyO3 can return them
//     to Python as zero-copy `numpy.ndarray` views via the `numpy`
//     crate.
//
// Validation: every index in F (or T or E) must be < V.shape[0].
// Shape errors are reported with stable human-readable messages so
// existing callers' error-string assertions keep working.

use std::collections::BTreeMap;

use ndarray::Array2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetKind {
    Tri,
    Tet,
    Rod,
    Stitch,
}

impl AssetKind {
    pub fn as_str(self) -> &'static str {
        match self {
            AssetKind::Tri => "tri",
            AssetKind::Tet => "tet",
            AssetKind::Rod => "rod",
            AssetKind::Stitch => "stitch",
        }
    }
}

impl std::fmt::Display for AssetKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One mesh asset. UV is optional on Tri: 5-column V emits separate
/// xyz/uv arrays; 3-column V leaves uv as None.
#[derive(Debug, Clone)]
pub enum Asset {
    Tri {
        verts: Array2<f32>,
        faces: Array2<u32>,
        uv: Option<Array2<f32>>,
    },
    Tet {
        verts: Array2<f32>,
        faces: Array2<u32>,
        tets: Array2<u32>,
    },
    Rod {
        verts: Array2<f32>,
        edges: Array2<u32>,
    },
    Stitch {
        ind: Array2<i32>,
        w: Array2<f32>,
    },
}

impl Asset {
    pub fn kind(&self) -> AssetKind {
        match self {
            Asset::Tri { .. } => AssetKind::Tri,
            Asset::Tet { .. } => AssetKind::Tet,
            Asset::Rod { .. } => AssetKind::Rod,
            Asset::Stitch { .. } => AssetKind::Stitch,
        }
    }
}

/// Bound-check that every index in `indices` is `< n_verts`.
fn check_bounds(indices: &Array2<u32>, n_verts: usize) -> Result<(), AssetError> {
    if let Some(&max) = indices.iter().max() {
        if (max as usize) >= n_verts {
            return Err(AssetError::OutOfBounds {
                index: max,
                n: n_verts,
            });
        }
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    #[error("name '{0}' already exists")]
    Duplicate(String),
    #[error("Asset {0} does not exist")]
    NotFound(String),
    #[error("Asset {name} is not a {expected}")]
    KindMismatch {
        name: String,
        expected: AssetKind,
    },
    #[error("E contains index {index} out of bounds ({n})")]
    OutOfBounds { index: u32, n: usize },
    #[error("{0} must have {1} columns")]
    InvalidShape(&'static str, usize),
}

/// Registry of named assets. The chained-API uploader / fetcher are
/// convenience wrappers around `add_*` / `get_*` here; the PyO3 layer
/// reproduces the chained shape for Python callers.
#[derive(Debug, Clone, Default)]
pub struct AssetRegistry {
    assets: BTreeMap<String, Asset>,
}

impl AssetRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// All registered names. Sorted (BTreeMap) so order is stable
    /// across runs.
    pub fn list(&self) -> Vec<&str> {
        self.assets.keys().map(String::as_str).collect()
    }

    /// True when an asset with this name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.assets.contains_key(name)
    }

    /// Drop an asset. Returns `true` if one existed.
    pub fn remove(&mut self, name: &str) -> bool {
        self.assets.remove(name).is_some()
    }

    /// Wipe the registry.
    pub fn clear(&mut self) {
        self.assets.clear();
    }

    pub fn get(&self, name: &str) -> Option<&Asset> {
        self.assets.get(name)
    }

    pub fn len(&self) -> usize {
        self.assets.len()
    }
    pub fn is_empty(&self) -> bool {
        self.assets.is_empty()
    }

    // ---- Typed adders (mirror AssetUploader.tri/tet/rod/stitch) ----

    pub fn add_tri(
        &mut self,
        name: &str,
        verts: Array2<f32>,
        faces: Array2<u32>,
        uv: Option<Array2<f32>>,
    ) -> Result<(), AssetError> {
        if verts.ncols() != 3 {
            return Err(AssetError::InvalidShape("V", 3));
        }
        if faces.ncols() != 3 {
            return Err(AssetError::InvalidShape("F", 3));
        }
        if let Some(ref uv_arr) = uv {
            if uv_arr.ncols() != 2 {
                return Err(AssetError::InvalidShape("UV", 2));
            }
        }
        if self.assets.contains_key(name) {
            return Err(AssetError::Duplicate(name.to_string()));
        }
        check_bounds(&faces, verts.nrows())?;
        self.assets.insert(
            name.to_string(),
            Asset::Tri {
                verts,
                faces,
                uv,
            },
        );
        Ok(())
    }

    pub fn add_tet(
        &mut self,
        name: &str,
        verts: Array2<f32>,
        faces: Array2<u32>,
        tets: Array2<u32>,
    ) -> Result<(), AssetError> {
        if verts.ncols() != 3 {
            return Err(AssetError::InvalidShape("V", 3));
        }
        if faces.ncols() != 3 {
            return Err(AssetError::InvalidShape("F", 3));
        }
        if tets.ncols() != 4 {
            return Err(AssetError::InvalidShape("T", 4));
        }
        if self.assets.contains_key(name) {
            return Err(AssetError::Duplicate(name.to_string()));
        }
        check_bounds(&faces, verts.nrows())?;
        check_bounds(&tets, verts.nrows())?;
        self.assets.insert(
            name.to_string(),
            Asset::Tet {
                verts,
                faces,
                tets,
            },
        );
        Ok(())
    }

    pub fn add_rod(
        &mut self,
        name: &str,
        verts: Array2<f32>,
        edges: Array2<u32>,
    ) -> Result<(), AssetError> {
        if verts.ncols() != 3 {
            return Err(AssetError::InvalidShape("V", 3));
        }
        if edges.ncols() != 2 {
            return Err(AssetError::InvalidShape("E", 2));
        }
        if self.assets.contains_key(name) {
            return Err(AssetError::Duplicate(name.to_string()));
        }
        check_bounds(&edges, verts.nrows())?;
        self.assets.insert(
            name.to_string(),
            Asset::Rod { verts, edges },
        );
        Ok(())
    }

    pub fn add_stitch(
        &mut self,
        name: &str,
        ind: Array2<i32>,
        w: Array2<f32>,
    ) -> Result<(), AssetError> {
        if ind.ncols() != 4 {
            return Err(AssetError::InvalidShape("Ind", 4));
        }
        if w.ncols() != 4 {
            return Err(AssetError::InvalidShape("W", 4));
        }
        if self.assets.contains_key(name) {
            return Err(AssetError::Duplicate(name.to_string()));
        }
        self.assets.insert(name.to_string(), Asset::Stitch { ind, w });
        Ok(())
    }

    // ---- Typed fetchers (mirror AssetFetcher.tri/tet/rod/stitch) ----

    pub fn get_tri(
        &self,
        name: &str,
    ) -> Result<(&Array2<f32>, &Array2<u32>, Option<&Array2<f32>>), AssetError> {
        match self.assets.get(name) {
            None => Err(AssetError::NotFound(name.to_string())),
            Some(Asset::Tri { verts, faces, uv }) => Ok((verts, faces, uv.as_ref())),
            Some(_) => Err(AssetError::KindMismatch {
                name: name.to_string(),
                expected: AssetKind::Tri,
            }),
        }
    }

    pub fn get_tet(
        &self,
        name: &str,
    ) -> Result<(&Array2<f32>, &Array2<u32>, &Array2<u32>), AssetError> {
        match self.assets.get(name) {
            None => Err(AssetError::NotFound(name.to_string())),
            Some(Asset::Tet { verts, faces, tets }) => Ok((verts, faces, tets)),
            Some(_) => Err(AssetError::KindMismatch {
                name: name.to_string(),
                expected: AssetKind::Tet,
            }),
        }
    }

    pub fn get_rod(&self, name: &str) -> Result<(&Array2<f32>, &Array2<u32>), AssetError> {
        match self.assets.get(name) {
            None => Err(AssetError::NotFound(name.to_string())),
            Some(Asset::Rod { verts, edges }) => Ok((verts, edges)),
            Some(_) => Err(AssetError::KindMismatch {
                name: name.to_string(),
                expected: AssetKind::Rod,
            }),
        }
    }

    pub fn get_stitch(&self, name: &str) -> Result<(&Array2<i32>, &Array2<f32>), AssetError> {
        match self.assets.get(name) {
            None => Err(AssetError::NotFound(name.to_string())),
            Some(Asset::Stitch { ind, w }) => Ok((ind, w)),
            Some(_) => Err(AssetError::KindMismatch {
                name: name.to_string(),
                expected: AssetKind::Stitch,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn unit_tri() -> (Array2<f32>, Array2<u32>) {
        let v = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let f = array![[0u32, 1, 2]];
        (v, f)
    }

    #[test]
    fn add_tri_then_fetch() {
        let mut r = AssetRegistry::new();
        let (v, f) = unit_tri();
        r.add_tri("sheet", v.clone(), f.clone(), None).unwrap();
        assert!(r.contains("sheet"));
        assert_eq!(r.list(), vec!["sheet"]);
        let (vv, ff, uv) = r.get_tri("sheet").unwrap();
        assert_eq!(vv, &v);
        assert_eq!(ff, &f);
        assert!(uv.is_none());
    }

    #[test]
    fn add_tri_with_uv() {
        let mut r = AssetRegistry::new();
        let (v, f) = unit_tri();
        let uv = array![[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
        r.add_tri("sheet", v, f, Some(uv.clone())).unwrap();
        let (_, _, got_uv) = r.get_tri("sheet").unwrap();
        assert_eq!(got_uv.unwrap(), &uv);
    }

    #[test]
    fn duplicate_name_rejected() {
        let mut r = AssetRegistry::new();
        let (v, f) = unit_tri();
        r.add_tri("x", v.clone(), f.clone(), None).unwrap();
        let err = r.add_tri("x", v, f, None).unwrap_err();
        assert!(matches!(err, AssetError::Duplicate(_)));
    }

    #[test]
    fn out_of_bounds_index_rejected() {
        let mut r = AssetRegistry::new();
        let v = array![[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let f = array![[0u32, 1, 5]]; // 5 >= 2
        let err = r.add_tri("bad", v, f, None).unwrap_err();
        match err {
            AssetError::OutOfBounds { index, n } => {
                assert_eq!(index, 5);
                assert_eq!(n, 2);
            }
            other => panic!("expected OutOfBounds, got {other:?}"),
        }
    }

    #[test]
    fn invalid_shape_rejected() {
        let mut r = AssetRegistry::new();
        let v = array![[0.0f32, 0.0, 0.0, 0.0]]; // 4 cols
        let f = array![[0u32, 0, 0]];
        let err = r.add_tri("bad", v, f, None).unwrap_err();
        assert!(matches!(err, AssetError::InvalidShape("V", 3)));
    }

    #[test]
    fn fetch_wrong_kind_errors() {
        let mut r = AssetRegistry::new();
        let (v, f) = unit_tri();
        r.add_tri("sheet", v, f, None).unwrap();
        let err = r.get_tet("sheet").unwrap_err();
        assert!(matches!(err, AssetError::KindMismatch { .. }));
    }

    #[test]
    fn add_tet_validates_all_index_arrays() {
        let mut r = AssetRegistry::new();
        let v = array![
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let f = array![[0u32, 1, 2]];
        let t = array![[0u32, 1, 2, 3]];
        r.add_tet("ball", v, f, t).unwrap();
        assert_eq!(r.get("ball").unwrap().kind(), AssetKind::Tet);
    }

    #[test]
    fn add_rod_and_stitch() {
        let mut r = AssetRegistry::new();
        let v = array![[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let e = array![[0u32, 1]];
        r.add_rod("strand", v, e).unwrap();

        let ind = array![[0i32, 1, 2, 3]];
        let w = array![[0.25f32, 0.25, 0.25, 0.25]];
        r.add_stitch("glue", ind, w).unwrap();

        assert_eq!(r.list(), vec!["glue", "strand"]);
    }

    #[test]
    fn remove_and_clear() {
        let mut r = AssetRegistry::new();
        let (v, f) = unit_tri();
        r.add_tri("a", v.clone(), f.clone(), None).unwrap();
        r.add_tri("b", v, f, None).unwrap();
        assert!(r.remove("a"));
        assert!(!r.remove("missing"));
        assert_eq!(r.list(), vec!["b"]);
        r.clear();
        assert!(r.is_empty());
    }
}
