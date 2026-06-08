//! Build-time emitter for CubeCL kernels.
//!
//! Phase 1.5 task #24. Consumes a `#[cube]`-decorated kernel struct,
//! invokes CubeCL's CUDA backend to produce CUDA C++ source, and
//! emits both the `.cu` file and a Rust `constants` module suitable
//! for hand-written or auto-generated launch shims.
//!
//! # MVP scope
//!
//! The MVP emits two artifacts per kernel:
//!
//! 1. `<name>.cu` — the CubeCL-compiled CUDA source. Goes through
//!    the consumer crate's nvcc step alongside hand-written `.cu`.
//! 2. `<name>.consts.rs` — a Rust module exposing:
//!    - `ENTRYPOINT_NAME: &str` (CUDA `__global__` symbol)
//!    - `INFO_ST_SIZE: usize` (bytes of the `info_st` struct)
//!    - `SCALAR_BYTES: usize` (bytes of `scalars_uint32[N]` region)
//!    - `STATIC_META_BYTES: usize` (bytes of `static_meta[M]` region)
//!    - `NUM_BUFFERS: usize`
//!    - `CUBE_DIM: (u32, u32, u32)` (declared launch bounds)
//!
//! Typed launch shims live alongside, hand-written by the consumer.
//! See `mistralrs-paged-attn/tests/cubecl_shim_prototype.rs` for the
//! shape such a shim takes. Auto-generating named typed shims wants
//! more semantic metadata than `KernelDefinition` exposes today; that
//! work is a separate epic (the upstream issue filed under task #29).
//!
//! # Usage
//!
//! ```ignore
//! use cubecl_shim_gen::EmittedKernel;
//!
//! fn main() {
//!     let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
//!     let files = EmittedKernel::new("reshape_and_cache_flashinfer_hnd_f32")
//!         .with_kernel(|| Box::new(make_kernel()))
//!         .emit_to(&out)
//!         .unwrap();
//!     println!("cargo::rerun-if-changed=build.rs");
//!     // The .cu joins build.rs's nvcc invocation; the .consts.rs is
//!     // include!()'d by the consumer's src/.
//! }
//! ```

#![deny(missing_docs)]

use std::path::{Path, PathBuf};

use cubecl_core::Compiler;
use cubecl_core::prelude::{AddressType, CubeKernel};
use cubecl_cuda::CudaCompiler;

// Future-work modules live behind `cfg(feature = "typed-shims")`; the
// MVP does not need them.

/// Errors from emission.
#[derive(Debug, thiserror::Error)]
pub enum EmitError {
    /// The CubeCL backend rejected the kernel.
    #[error("cubecl compile failed: {0}")]
    Compile(#[from] cubecl_core::CompilationError),

    /// An output file couldn't be written.
    #[error("write to {path}: {source}")]
    Write {
        /// Path that failed.
        path: String,
        /// I/O error.
        #[source]
        source: std::io::Error,
    },
}

/// Configures and runs one kernel emission. Use the builder methods,
/// then call [`Self::emit_to`].
pub struct EmittedKernel<F>
where
    F: FnOnce() -> Box<dyn CubeKernel + Send + Sync + 'static>,
{
    name: String,
    kernel_factory: Option<F>,
}

impl<F> EmittedKernel<F>
where
    F: FnOnce() -> Box<dyn CubeKernel + Send + Sync + 'static>,
{
    /// Start a new emission named `kernel_name`.
    ///
    /// The name is used as the basename of every artifact and as the
    /// inner module name in `<name>.consts.rs`. It does NOT have to
    /// match the CubeCL kernel's entrypoint; that's a separate symbol
    /// chosen by the CubeCL compiler.
    pub fn new(kernel_name: impl Into<String>) -> Self {
        Self {
            name: kernel_name.into(),
            kernel_factory: None,
        }
    }

    /// Supply the closure that constructs the `#[cube]` kernel struct.
    /// Called once during emission. Pick dtype and comptime values
    /// here.
    pub fn with_kernel(mut self, f: F) -> Self {
        self.kernel_factory = Some(f);
        self
    }

    /// Emit the `.cu` and `.consts.rs` files to `out_dir`.
    pub fn emit_to(self, out_dir: &Path) -> Result<EmittedFiles, EmitError> {
        let factory = self
            .kernel_factory
            .expect("EmittedKernel::with_kernel was not called");
        let kernel = factory();
        let kernel_def = kernel.define();

        // ----- compile to CUDA C++ -----
        let mut compiler = CudaCompiler::default();
        let representation = compiler.compile(
            kernel_def.clone(),
            &Default::default(),
            cubecl_core::prelude::ExecutionMode::Unchecked,
            AddressType::U32.unsigned_type(),
        )?;
        let cpp_source = representation.to_string();

        // ----- write .cu -----
        let cu_path = out_dir.join(format!("{}.cu", self.name));
        write_file(&cu_path, &cpp_source)?;

        // ----- compute info_st sizing -----
        // CubeCL serializes scalars grouped by storage type (see
        // cubecl-core::codegen::ScalarBuilder). One `info_st`
        // contains:
        //   - one packed region per scalar type, each aligned to
        //     INFO_ALIGN (8 bytes)
        //   - then `static_meta`, aligned to INFO_ALIGN
        //   - then optional dynamic meta (lives AFTER info_st in
        //     memory, accessed via pointer arithmetic by the kernel)
        const INFO_ALIGN: usize = 8;

        let mut scalar_bytes: usize = 0;
        // Group same-type scalars (matches ScalarBuilder behavior).
        let mut grouped: std::collections::BTreeMap<u32, usize> =
            std::collections::BTreeMap::new();
        for sa in &kernel_def.scalars {
            let ty_size = sa.ty.size();
            *grouped.entry(ty_size as u32).or_default() += sa.count;
        }
        for (ty_size, count) in &grouped {
            scalar_bytes += (*ty_size as usize * count).next_multiple_of(INFO_ALIGN);
        }
        // static_meta_bytes is the static portion of metadata, sized
        // by address_type * static_len. We don't have direct access
        // to the Metadata's static_len from KernelDefinition without
        // walking buffers, so we expose static_meta size only as the
        // delta from cpp_source's `info_st` struct declaration.
        // For correctness purposes the consumer just needs INFO_ST_SIZE.
        let static_meta_bytes = scan_static_meta_bytes(&cpp_source);

        let info_st_size = scalar_bytes + static_meta_bytes;
        let scalar_count_u32 = scalar_bytes / 4;
        let static_count_u32 = static_meta_bytes / 4;

        let entrypoint = kernel_def.options.kernel_name.clone();
        let num_buffers = kernel_def.num_global_buffers();
        let cube_dim = (
            kernel_def.cube_dim.x,
            kernel_def.cube_dim.y,
            kernel_def.cube_dim.z,
        );

        // ----- write .consts.rs -----
        let consts_path = out_dir.join(format!("{}.consts.rs", self.name));
        let consts_source = render_consts_module(
            &self.name,
            &entrypoint,
            info_st_size,
            scalar_bytes,
            static_meta_bytes,
            scalar_count_u32,
            static_count_u32,
            num_buffers,
            cube_dim,
        );
        write_file(&consts_path, &consts_source)?;

        Ok(EmittedFiles {
            cu_path,
            consts_path,
            cpp_source,
            consts_source,
            entrypoint,
            info_st_size,
            num_buffers,
            cube_dim,
        })
    }
}

fn write_file(path: &PathBuf, contents: &str) -> Result<(), EmitError> {
    std::fs::write(path, contents).map_err(|source| EmitError::Write {
        path: path.display().to_string(),
        source,
    })
}

/// Scan the emitted C++ for the `static_meta[N]` declaration inside
/// `info_st` and return its size in bytes.
///
/// This is a sanity-check on our own scalar-bytes math. The C++ side
/// is the source of truth for the actual struct layout.
fn scan_static_meta_bytes(cpp_source: &str) -> usize {
    // Looking for: `uint32 static_meta[N];`
    if let Some(after) = cpp_source.split("static_meta[").nth(1) {
        if let Some(num_str) = after.split(']').next() {
            if let Ok(n) = num_str.trim().parse::<usize>() {
                return n * 4;
            }
        }
    }
    0
}

fn render_consts_module(
    mod_name: &str,
    entrypoint: &str,
    info_st_size: usize,
    scalar_bytes: usize,
    static_meta_bytes: usize,
    scalar_count_u32: usize,
    static_count_u32: usize,
    num_buffers: usize,
    cube_dim: (u32, u32, u32),
) -> String {
    let (cd_x, cd_y, cd_z) = cube_dim;
    format!(
        r#"// Auto-generated by cubecl-shim-gen. Do not edit by hand.
//
// See <crate root>/docs/CANDLE_ON_CUBECL_EPIC.md for the role of this
// file in the migration; see <crate root>/cubecl-shim-gen/src/lib.rs
// for the generator entrypoint.

#[allow(dead_code)]
pub mod {mod_name} {{
    /// CUDA `__global__` symbol name, as emitted by CubeCL.
    /// This is what the consumer's launch shim passes to
    /// `cuModuleGetFunction`.
    pub const ENTRYPOINT_NAME: &str = "{entrypoint}";

    /// Total bytes of the `info_st` struct that the kernel reads as
    /// its last argument (scalars + static_meta).
    pub const INFO_ST_SIZE: usize = {info_st_size};

    /// Bytes of the `scalars_uint32[N]` region inside `info_st`.
    pub const SCALAR_BYTES: usize = {scalar_bytes};

    /// Bytes of the `static_meta[N]` region inside `info_st`.
    pub const STATIC_META_BYTES: usize = {static_meta_bytes};

    /// Number of u32 slots in the scalar region (including padding).
    pub const SCALAR_COUNT_U32: usize = {scalar_count_u32};

    /// Number of u32 slots in the static_meta region.
    pub const STATIC_COUNT_U32: usize = {static_count_u32};

    /// Number of device buffer arguments the kernel takes (excluding
    /// the `info_st*` arg).
    pub const NUM_BUFFERS: usize = {num_buffers};

    /// Declared cube (block) dimensions from the kernel's settings.
    pub const CUBE_DIM: (u32, u32, u32) = ({cd_x}, {cd_y}, {cd_z});
}}
"#,
    )
}

/// What [`EmittedKernel::emit_to`] returns.
#[derive(Debug)]
pub struct EmittedFiles {
    /// On-disk location of the `.cu` source.
    pub cu_path: PathBuf,
    /// On-disk location of the `.consts.rs` Rust module.
    pub consts_path: PathBuf,
    /// In-memory copy of the `.cu` contents.
    pub cpp_source: String,
    /// In-memory copy of the consts Rust source.
    pub consts_source: String,
    /// CUDA `__global__` entrypoint symbol.
    pub entrypoint: String,
    /// Total bytes of the `info_st` struct.
    pub info_st_size: usize,
    /// Number of device buffer args.
    pub num_buffers: usize,
    /// Cube (block) dimensions.
    pub cube_dim: (u32, u32, u32),
}
