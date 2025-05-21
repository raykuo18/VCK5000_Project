#include <aie_api/aie.hpp>

__attribute__((noinline)) void kernel_matmul_4x2x4_float_controlled(
    input_window<int32>* __restrict ctl_win,
    input_window<float>* __restrict in_win,
    output_window<float>* __restrict out_win) {

    static aie::vector<float, 8> b_matrix;      // 4x2 = 8 elements
    static aie::vector<float, 16> acc_buffer;    // 4x2 = 8 elements
    static bool is_initialized = false;

    if (!is_initialized) {
        b_matrix = aie::zeros<float, 8>();
        acc_buffer = aie::zeros<float, 16>();
        is_initialized = true;
    }

    int32 ctl = window_readincr(ctl_win);

    if (ctl == 0) {
        for (int i = 0; i < 8; ++i)
            b_matrix[i] = window_readincr(in_win);
    }
    else if (ctl == 1) {
        aie::vector<float, 8> a_matrix;
        for (int i = 0; i < 8; ++i)
            a_matrix[i] = window_readincr(in_win);

        aie::mmul<4, 2, 4, float, float> m;  // A: 4x2, B: 2x4
        m.mul(a_matrix, b_matrix);
        acc_buffer = aie::add(acc_buffer, m.to_vector<float>(0));
    }
    else if (ctl == 2) {
        for (int i = 0; i < 16; ++i)
            window_writeincr(out_win, acc_buffer[i]);
    }
    else if (ctl == 3) {
        acc_buffer = aie::zeros<float, 16>();
    }
}
