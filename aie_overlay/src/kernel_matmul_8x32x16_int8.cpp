#include <aie_api/aie.hpp>
// 4 x 16 x 8
__attribute__((noinline))
void kernel_matmul_8x32x16_int8(
    input_window<int8>* __restrict in_win,
    output_window<int8>* __restrict out_win) {

    aie::mmul<4, 16, 8, int8, int8> m;
    
    aie::vector<int8, 64> a_tile_1;
    aie::vector<int8, 64> a_tile_2;

    aie::vector<int8, 128> b_tile_1;
    aie::vector<int8, 128> b_tile_2;
    aie::vector<int8, 128> b_tile_3;
    aie::vector<int8, 128> b_tile_4;

    aie::vector<int8, 32> c_tile_1;
    aie::vector<int8, 32> c_tile_2;

    // first row tiles in A
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 16; ++j) {
            a_tile_1[i * 16 + j] = window_readincr(in_win);
        }
        for (int j = 0; j < 16; ++j) {
            a_tile_2[i * 16 + j] = window_readincr(in_win);
        }
    }
    window_incr(in_win, 128);
    // all tiles in B
    for (int i = 0; i < 16; ++ i) {
        for (int j = 0; j < 8; ++j) {
            b_tile_1[i * 8 + j] = window_readincr(in_win);
        }
        for (int j = 0; j < 8; ++j) {
            b_tile_2[i * 16 + j] = window_readincr(in_win);
        }
    }
    for (int i = 0; i < 16; ++ i) {
        for (int j = 0; j < 8; ++j) {
            b_tile_3[i * 8 + j] = window_readincr(in_win);
        }
        for (int j = 0; j < 8; ++j) {
            if (i == 15 && j == 7) b_tile_4[i * 16 + j] = window_readincr(in_win);
            else b_tile_4[i * 16 + j] = window_read(in_win);
        }
    }
    
    m.mul(a_tile_1, b_tile_1);
    m.mac(a_tile_2, b_tile_3);
    c_tile_1 = m.to_vector<int8>(0);
    m.mul(a_tile_1, b_tile_2);
    m.mac(a_tile_2, b_tile_4);
    c_tile_2 = m.to_vector<int8>(0);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++ j) {
            window_writeincr(out_win, c_tile_1[i * 8 + j]);
        }
        for (int j = 0; j < 8; ++ j) {
            window_writeincr(out_win, c_tile_2[i * 8 + j]);
        }
    }

    /////////////////////////////////////////////////////////
    window_decr(in_win, 767);
    window_incr(in_win, 128);
    // second row tiles in A
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 16; ++j) {
            a_tile_1[i * 16 + j] = window_readincr(in_win);
        }
        for (int j = 0; j < 16; ++j) {
            a_tile_2[i * 16 + j] = window_readincr(in_win);
        }
    }

    m.mul(a_tile_1, b_tile_1);
    m.mac(a_tile_2, b_tile_3);
    c_tile_1 = m.to_vector<int8>(0);
    m.mul(a_tile_1, b_tile_2);
    m.mac(a_tile_2, b_tile_4);
    c_tile_2 = m.to_vector<int8>(0);
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++ j) {
            window_writeincr(out_win, c_tile_1[i * 8 + j]);
        }
        for (int j = 0; j < 8; ++ j) {
            window_writeincr(out_win, c_tile_2[i * 8 + j]);
        }
    }
}