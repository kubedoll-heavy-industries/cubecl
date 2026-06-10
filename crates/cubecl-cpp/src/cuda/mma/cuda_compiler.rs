use super::{WMMA_MINIMUM_VERSION, WMMA_NAMESPACE};
use crate::{
    cuda::{
        CudaDialect,
        arch::CudaArchitecture,
        mma::{
            compile_manual_mma, compile_scaled_mma, supported_mma_combinations,
            supported_scaled_mma_combinations,
        },
    },
    shared::{
        Architecture, DialectWmmaCompiler, Elem, Flags, Fragment, FragmentIdent, FragmentLayout,
        ManualMma, SupportedMmaCombinations, SupportedScaledMmaCombinations, Variable,
        WmmaInstruction, wmma_api_base,
    },
};
use cubecl_core::ir::{self as gpu, features::MmaConfig};
use itertools::Itertools;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CudaWmmaCompiler {}

/// `nvcc` does not emit `ldmatrix` for WMMA loads of `__nv_bfloat16` fragments from
/// shared memory — it falls back to scalar `LDS`, costing ~30% on bf16 GEMMs.
/// `ldmatrix` is dtype-agnostic (it moves raw 16-bit lanes), so for bf16
/// `matrix_a`/`matrix_b` fragments we:
///   1. declare the fragment as `__half`,
///   2. load it through a pointer cast to `const __half*`,
///   3. `reinterpret_cast` the fragment back to its true `__nv_bfloat16` fragment type
///      at `mma_sync`, so overload resolution still selects the `.BF16` HMMA path.
/// This is bit-exact: no value conversion happens at any point.
/// Accumulators are unaffected (f32, and bf16 accumulators were already declared as
/// `__half` by `wmma_api_base::compile_fragment`).
fn is_bf16_input_fragment(frag: &Fragment<CudaDialect<CudaWmmaCompiler>>) -> bool {
    matches!(frag.ident, FragmentIdent::A | FragmentIdent::B) && matches!(frag.elem, Elem::BF16)
}

fn try_variable_to_frag(
    var: &Variable<CudaDialect<CudaWmmaCompiler>>,
) -> Option<Fragment<CudaDialect<CudaWmmaCompiler>>> {
    match var {
        Variable::WmmaFragment { frag, .. } => Some(*frag),
        _ => None,
    }
}

/// Displays a WMMA fragment operand, reinterpreting `__half`-declared bf16 input
/// fragments back to their true `__nv_bfloat16` fragment type.
struct Bf16MmaOperand<'a>(&'a Variable<CudaDialect<CudaWmmaCompiler>>);

impl std::fmt::Display for Bf16MmaOperand<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match try_variable_to_frag(self.0) {
            Some(frag) if is_bf16_input_fragment(&frag) => {
                write!(f, "(*reinterpret_cast<")?;
                // `wmma_api_base::compile_fragment` prints the true (bf16) element type
                // for `matrix_a`/`matrix_b` fragments.
                wmma_api_base::compile_fragment(f, WMMA_NAMESPACE, &frag)?;
                write!(f, "*>(&{}))", self.0)
            }
            _ => write!(f, "{}", self.0),
        }
    }
}

impl DialectWmmaCompiler<CudaDialect<Self>> for CudaWmmaCompiler {
    fn compile_wmma_includes(
        f: &mut std::fmt::Formatter<'_>,
        _flags: &Flags<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        f.write_str("#include <mma.h>\n")
    }

    fn compile_wmma_fragment_declaration(
        f: &mut std::fmt::Formatter<'_>,
        var: &crate::shared::Variable<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment_declaration(f, var)
    }

    fn compile_wwma_fragment_ident(
        f: &mut std::fmt::Formatter<'_>,
        ident: &FragmentIdent<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment_ident(f, WMMA_NAMESPACE, ident)
    }

    fn compile_wmma_fragment_layout(
        f: &mut std::fmt::Formatter<'_>,
        layout: &FragmentLayout<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment_layout(f, WMMA_NAMESPACE, layout)
    }

    fn compile_wmma_fragment(
        f: &mut std::fmt::Formatter<'_>,
        fragment: &Fragment<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        if is_bf16_input_fragment(fragment) {
            // Declared as `__half` so nvcc emits `ldmatrix` for shared memory loads.
            // See `is_bf16_input_fragment` for the full rationale.
            let half_fragment = Fragment::<CudaDialect<Self>> {
                elem: Elem::F16,
                ..*fragment
            };
            wmma_api_base::compile_fragment(f, WMMA_NAMESPACE, &half_fragment)
        } else {
            wmma_api_base::compile_fragment(f, WMMA_NAMESPACE, fragment)
        }
    }

    fn compile_wmma_instruction(
        f: &mut std::fmt::Formatter<'_>,
        instruction: &WmmaInstruction<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        match instruction {
            WmmaInstruction::Fill { frag, value }
                if try_variable_to_frag(frag).is_some_and(|f| is_bf16_input_fragment(&f)) =>
            {
                // The fragment is declared as `__half`; fill it through its true
                // bf16 fragment type so the value isn't converted through f16.
                writeln!(
                    f,
                    "{WMMA_NAMESPACE}::fill_fragment({}, {value});",
                    Bf16MmaOperand(frag)
                )
            }
            WmmaInstruction::Load {
                frag,
                ptr,
                stride,
                layout,
            } if try_variable_to_frag(frag).is_some_and(|f| is_bf16_input_fragment(&f)) => {
                // Load the `__half`-declared fragment through a `__half` pointer:
                // `ldmatrix` moves raw 16-bit lanes, so the bits stay bf16.
                let elem = Elem::<CudaDialect<Self>>::F16;
                match layout {
                    Some(layout) => {
                        let layout = match layout {
                            FragmentLayout::ColMajor => format!("{WMMA_NAMESPACE}::mem_col_major"),
                            FragmentLayout::RowMajor => format!("{WMMA_NAMESPACE}::mem_row_major"),
                            FragmentLayout::_Dialect(_) => String::new(),
                        };
                        writeln!(
                            f,
                            "{WMMA_NAMESPACE}::load_matrix_sync({frag}, reinterpret_cast<const {elem}*>({ptr}), {stride}, {layout});"
                        )
                    }
                    None => writeln!(
                        f,
                        "{WMMA_NAMESPACE}::load_matrix_sync({frag}, reinterpret_cast<const {elem}*>({ptr}), {stride});"
                    ),
                }
            }
            WmmaInstruction::Execute {
                frag_a,
                frag_b,
                frag_c,
                frag_d,
                ..
            } if [frag_a, frag_b].iter().any(|frag| {
                try_variable_to_frag(frag).is_some_and(|f| is_bf16_input_fragment(&f))
            }) =>
            {
                // Reinterpret the `__half`-declared input fragments back to their true
                // bf16 fragment types so overload resolution selects the `.BF16` HMMA.
                writeln!(
                    f,
                    "{WMMA_NAMESPACE}::mma_sync({frag_d}, {}, {}, {frag_c});",
                    Bf16MmaOperand(frag_a),
                    Bf16MmaOperand(frag_b)
                )
            }
            _ => wmma_api_base::compile_instruction(f, WMMA_NAMESPACE, instruction),
        }
    }

    fn compile_manual_mma(
        f: &mut std::fmt::Formatter<'_>,
        mma: ManualMma<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        compile_manual_mma(f, mma)
    }

    fn compile_scaled_mma(
        f: &mut std::fmt::Formatter<'_>,
        mma: ManualMma<CudaDialect<Self>>,
        scales_a: Variable<CudaDialect<Self>>,
        scales_b: Variable<CudaDialect<Self>>,
        scales_factor: u32,
    ) -> std::fmt::Result {
        compile_scaled_mma(f, mma, scales_a, scales_b, scales_factor)
    }

    fn supported_wmma_combinations(arch: &CudaArchitecture) -> SupportedMmaCombinations {
        let mut result: SupportedMmaCombinations = vec![];
        if arch.get_version() >= WMMA_MINIMUM_VERSION {
            let tdims = vec![(16, 16, 16), (32, 8, 16), (8, 32, 16)];
            // Types fully supported.
            let types = vec![
                (
                    gpu::ElemType::Float(gpu::FloatKind::F16), // m
                    gpu::ElemType::Float(gpu::FloatKind::F16), // n
                    gpu::ElemType::Float(gpu::FloatKind::F16), // k
                ),
                (
                    gpu::ElemType::Float(gpu::FloatKind::F16),
                    gpu::ElemType::Float(gpu::FloatKind::F16),
                    gpu::ElemType::Float(gpu::FloatKind::F32),
                ),
                (
                    gpu::ElemType::Float(gpu::FloatKind::BF16),
                    gpu::ElemType::Float(gpu::FloatKind::BF16),
                    gpu::ElemType::Float(gpu::FloatKind::F32),
                ),
                (
                    gpu::ElemType::Int(gpu::IntKind::I8),
                    gpu::ElemType::Int(gpu::IntKind::I8),
                    gpu::ElemType::Int(gpu::IntKind::I32),
                ),
                (
                    gpu::ElemType::UInt(gpu::UIntKind::U8),
                    gpu::ElemType::UInt(gpu::UIntKind::U8),
                    gpu::ElemType::Int(gpu::IntKind::I32),
                ),
            ];
            let combinations: SupportedMmaCombinations = types
                .into_iter()
                .cartesian_product(tdims)
                .map(|((a, b, c), (m, n, k))| MmaConfig {
                    a_type: a.into(),
                    b_type: b.into(),
                    cd_type: c.into(),
                    m,
                    n,
                    k,
                })
                .collect();
            result.extend(combinations);
            if arch.get_version() >= 80 {
                result.push(MmaConfig {
                    a_type: gpu::ElemType::Float(gpu::FloatKind::TF32).into(),
                    b_type: gpu::ElemType::Float(gpu::FloatKind::TF32).into(),
                    cd_type: gpu::ElemType::Float(gpu::FloatKind::F32).into(),
                    m: 16,
                    n: 16,
                    k: 8,
                });
            }
        }
        result
    }

    fn supported_mma_combinations(arch: &CudaArchitecture) -> SupportedMmaCombinations {
        supported_mma_combinations(arch)
    }

    fn supported_scaled_mma_combinations(
        arch: &CudaArchitecture,
    ) -> SupportedScaledMmaCombinations {
        supported_scaled_mma_combinations(arch)
    }
}
