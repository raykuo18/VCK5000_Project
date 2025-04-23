#pragma once
#include <adf.h>

void kernel_matmul_8x16x8_int8(
    input_window_int8* __restrict in,
    output_window_int8* __restrict out);
