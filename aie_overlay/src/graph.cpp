//
// Copyright 2021 Xilinx, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "graph.h"

using namespace adf;

// graph_overlay my_graph;
// graph_matmul_4x4x4 my_graph_matmul_4x4x4;
// graph_matmul_4x4x4_int16_stream my_graph_matmul_4x4x4_int16_stream;
// graph_matmul_64x64x64_int8 my_graph_matmul_64x64x64_int8;
// graph_matmul_4x16x8_int8 my_graph_matmul_4x16x8_int8;
// graph_matmul_8x16x8_int8 my_graph_matmul_8x16x8_int8;
// graph_matmul_8x32x16_int8 my_graph_matmul_8x32x16_int8;
// graph_matmul_4x4x4_controlled my_graph_matmul_4x4x4_controlled;
graph_matmul_4x2x4_float_controlled my_graph_matmul_4x2x4_float_controlled;

#define MY_GRAPH my_graph_matmul_4x2x4_float_controlled

int main(int argc, char ** argv)
{
	MY_GRAPH.init();
	MY_GRAPH.run(8);
	MY_GRAPH.end();
	return 0;
}
