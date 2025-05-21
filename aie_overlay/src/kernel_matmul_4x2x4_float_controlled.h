#pragma once

#include <adf.h>

void kernel_matmul_4x2x4_float_controlled(
    input_window<int32>* __restrict ctl,
    input_window<float>* __restrict in,
    output_window<float>* __restrict out);
