#include <aie_api/aie.hpp>

__attribute__((noinline)) void kernel_matmul_4x4x4_controlled(
    input_window_int32* __restrict ctl_win,
    input_window_int16* __restrict in_win,
    output_window_int16* __restrict out_win) {

    // Buffer to store a matrix
    static aie::vector<int16, 16> b_matrix;
    static aie::vector<int16, 16> acc_buffer;
    static bool is_initialized = false;

    if (!is_initialized) {
        b_matrix = aie::zeros<int16, 16>();
        acc_buffer = aie::zeros<int16, 16>();
        is_initialized = true;
    }

    // Read control signal
    int32 ctl = window_readincr(ctl_win);

    if (ctl == 0) {
        // Read b_matrix
        for (int i = 0; i < 16; ++i)
            b_matrix[i] = window_readincr(in_win);
    } else if (ctl == 1) {
        // Matmul and accumulate
        aie::vector<int16, 16> a_matrix;
        for (int i = 0; i < 16; ++i)
            a_matrix[i] = window_readincr(in_win);
        aie::mmul<4, 4, 4, int16, int16> m;
        m.mul(a_matrix, b_matrix);
        acc_buffer = aie::add(acc_buffer, m.to_vector<int16>(0));
    } else if (ctl == 2) {
        // Write output
        for (int i = 0; i < 16; ++i) {
            window_writeincr(out_win, acc_buffer[i]);
        }
    } else if (ctl == 3) {
        // Clear buffer
        acc_buffer = aie::zeros<int16, 16>();
    }
}
