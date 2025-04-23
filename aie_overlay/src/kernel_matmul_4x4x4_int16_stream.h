#pragma once

#include <adf.h>

void kernel_matmul_4x4x4_int16_stream(
    input_stream<int16>* __restrict in_stream,
    output_stream<int16>* __restrict out_stream);