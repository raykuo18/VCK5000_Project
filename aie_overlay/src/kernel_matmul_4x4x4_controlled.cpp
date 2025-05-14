#include <aie_api/aie.hpp>

__attribute__((noinline)) void kernel_matmul_4x4x4_controlled(
    input_window_int32* __restrict ctl_win,
    input_window_int16* __restrict in_win,
    output_window_int16* __restrict out_win) {

  // Buffer to store a matrix
  static aie::vector<int16, 16> buffer_matrix;
  static bool is_initialized = false;

  if (!is_initialized) {
    buffer_matrix = aie::zeros<int16, 16>();
    is_initialized = true;
  }

  aie::vector<int16, 16> a_vec;
  aie::vector<int16, 16> b_vec;

  // Read control signal
  int32 ctl = window_readincr(ctl_win);  // Read control signal

  // Read A and B
  for (int i = 0; i < 16; ++i)
    a_vec[i] = window_readincr(in_win);
  for (int i = 0; i < 16; ++i)
    b_vec[i] = window_readincr(in_win);
  
  if (ctl == 0)
    buffer_matrix = b_vec;

  // Matmul
  aie::mmul<4, 4, 4, int16, int16> m;
  m.mul(a_vec, b_vec);
  aie::vector<int16, 16> c_vec = m.to_vector<int16>(0);

  // Write output
  for (int i = 0; i < 16; ++i) {
    if (ctl == 1)
        window_writeincr(out_win, c_vec[i] + buffer_matrix[i]);
    else
        window_writeincr(out_win, c_vec[i]);
  }
}
