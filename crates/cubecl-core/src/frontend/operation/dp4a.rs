use crate::{prelude::*, unexpanded};

/// 4-element signed int8 dot product with int32 accumulator.
///
/// `a` and `b` are containers (`i32` or `u32`) packing four `i8` values each in
/// little-endian byte order. The result is
/// `c + Σᵢ₌₀..₄ (a.byte(i) as i32 * b.byte(i) as i32)`.
///
/// On CUDA devices with SM_61+ (Pascal P40 / GTX 1080 onwards) this lowers
/// to the hardware `dp4a.s32.s32` instruction. The CUDA runtime emulates the
/// same semantics on older devices using four packed multiply-adds, but at a
/// significant performance penalty.
///
/// This is the inner-loop primitive for INT8 quantized GEMV kernels (Q4_K,
/// Q5_K, Q8_0, etc. against Q8_1 activations).
#[allow(unused_variables)]
pub fn dp4a(a: i32, b: i32, c: i32) -> i32 {
    unexpanded!()
}

/// Expand method of [`dp4a()`].
pub mod dp4a {
    use super::*;
    use cubecl_ir::{Arithmetic, Dp4aOperands, Instruction, Scope};

    pub fn expand(
        scope: &Scope,
        a: NativeExpand<i32>,
        b: NativeExpand<i32>,
        c: NativeExpand<i32>,
    ) -> NativeExpand<i32> {
        let output = scope.create_local(a.expand.value_type());
        let a = a.expand;
        let b = b.expand;
        let c = c.expand;

        scope.register(Instruction::new(
            Arithmetic::Dp4a(Dp4aOperands { a, b, c }),
            output,
        ));

        output.into()
    }
}
