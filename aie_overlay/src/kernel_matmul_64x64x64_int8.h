#pragma once
#include <adf.h>

void kernel_matmul_64x64x64_int8(
    input_window_int8* __restrict in,
    output_window_int8* __restrict out);
