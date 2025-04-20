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
graph_matmul_64_64_64_int8 my_graph_matmul_64_64_64_int8;

#define MY_GRAPH my_graph_matmul_64_64_64_int8

int main(int argc, char ** argv)
{
	MY_GRAPH.init();
	MY_GRAPH.run(1);
	MY_GRAPH.end();
	return 0;
}
