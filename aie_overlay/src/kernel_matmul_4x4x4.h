#pragma once

#include <adf.h>

void kernel_matmul_4x4x4(
    input_window_int16* __restrict in,
    output_window_int16* __restrict out);