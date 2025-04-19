#include <aie_api/aie.hpp>

__attribute__((noinline)) void kernel_matmul(
    input_window_int16* __restrict in_win,
    output_window_int16* __restrict out_win) {

  aie::vector<int16, 16> a_vec;
  aie::vector<int16, 16> b_vec;

  // Read A and B
  for (int i = 0; i < 16; ++i)
    a_vec[i] = window_readincr(in_win);
  for (int i = 0; i < 16; ++i)
    b_vec[i] = window_readincr(in_win);

  // Matmul
  aie::mmul<4, 4, 4, int16, int16> m;
  m.mul(a_vec, b_vec);
  aie::vector<int16, 16> c_vec = m.to_vector<int16>(0);

  // Write output
  for (int i = 0; i < 16; ++i)
    window_writeincr(out_win, c_vec[i]);
}
