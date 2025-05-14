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

#pragma once
#include <adf.h>
#include "kernel_overlay.h"
#include "kernel_cvt.h"
#include "kernel_matmul_4x4x4.h"
#include "kernel_matmul_4x4x4_int16_stream.h"
#include "kernel_matmul_64x64x64_int8.h"
#include "kernel_matmul_4x16x8_int8.h"
#include "kernel_matmul_8x16x8_int8.h"
#include "kernel_matmul_4x4x4_controlled.h"

using namespace adf;

class graph_overlay: public graph
{

private:
	kernel k_cvt;
	kernel k_overlay;

public:
	input_plio p_s0;
	input_plio p_s1;
	output_plio p_s2;

	graph_overlay()
	{
		// create kernel
		k_cvt =  kernel::create(kernel_cvt);
		k_overlay = kernel::create(kernel_overlay);

		// create port
		p_s0 = input_plio::create("s0", plio_128_bits, "data/s0.txt");
		p_s1 = input_plio::create("s1", plio_128_bits, "data/s1.txt");
		p_s2 = output_plio::create("s2", plio_32_bits, "data/s2_act.txt");

		// connect port and kernel
		connect<window<64>>(p_s0.out[0], async(k_overlay.in[0]));
		connect<window<64>>(p_s0.out[0], async(k_cvt.in[0]));
		connect<window<192>>(p_s1.out[0], async(k_cvt.in[1]));
		connect<window<192>>(async(k_cvt.out[0]), async(k_overlay.in[1]));
		connect<window<256>>(async(k_overlay.out[0]), p_s2.in[0]);

		// set kernel source codes
		source(k_cvt)     	= "src/kernel_cvt.cpp";
		source(k_overlay) 	= "src/kernel_overlay.cpp";
		headers(k_cvt) 	  	= {"src/kernel_cvt.h","../common/common.h"};
		headers(k_overlay)	= {"src/kernel_overlay.h","../common/common.h"};

		// set ratio
		runtime<ratio>(k_cvt)     =0.9;
		runtime<ratio>(k_overlay) =0.9;

	};

};

class graph_matmul_4x4x4: public graph
{

private:
    kernel k_matmul;

public:
    input_plio in;
    output_plio out;

	graph_matmul_4x4x4()
	{
		// create kernel
        k_matmul = kernel::create(kernel_matmul_4x4x4);

		// create port
        in = input_plio::create("in", plio_32_bits, "data/input_4x4x4.txt");
        out = output_plio::create("out", plio_32_bits, "data/output_4x4x4.txt");

		// connect port and kernel
        connect<window<64>>(in.out[0], k_matmul.in[0]);
        connect<window<32>>(k_matmul.out[0], out.in[0]);

		// set kernel source codes
		source(k_matmul)     	= "src/kernel_matmul_4x4x4.cpp";
		headers(k_matmul) 	  	= {"src/kernel_matmul_4x4x4.h"};

		// set ratio
		runtime<ratio>(k_matmul)    =0.9;

	};

};

class graph_matmul_4x4x4_controlled: public graph
{

private:
    kernel k_matmul;

public:
    input_plio in;
    output_plio out;

	graph_matmul_4x4x4_controlled()
	{
		// create kernel
        k_matmul = kernel::create(kernel_matmul_4x4x4_controlled);

		// create port
        in = input_plio::create("in", plio_32_bits, "data/input_4x4x4.txt");
        out = output_plio::create("out", plio_32_bits, "data/output_4x4x4_controlled.txt");

		// connect port and kernel
        connect<window<64>>(in.out[0], k_matmul.in[0]);
        connect<window<32>>(k_matmul.out[0], out.in[0]);

		// set kernel source codes
		source(k_matmul)     	= "src/kernel_matmul_4x4x4_controlled.cpp";
		headers(k_matmul) 	  	= {"src/kernel_matmul_4x4x4_controlled.h"};

		// set ratio
		runtime<ratio>(k_matmul)    =0.9;

	};

};

class graph_matmul_4x4x4_int16_stream : public graph {
private:
    kernel k_matmul;

public:
    input_plio in;
    output_plio out;

    graph_matmul_4x4x4_int16_stream() {
        // Create kernel
        k_matmul = kernel::create(kernel_matmul_4x4x4_int16_stream);

        // Create PLIO ports (must assign to class members!)
        in = input_plio::create("in0", plio_32_bits, "data/input_4x4x4_stream.txt");
        out = output_plio::create("out0", plio_32_bits, "data/output_4x4x4_stream.txt");

        // Connect PLIOs to the kernel
        connect<>(in.out[0], k_matmul.in[0]);
        connect<>(k_matmul.out[0], out.in[0]);

        // Set kernel files
        source(k_matmul)  = "src/kernel_matmul_4x4x4_int16_stream.cpp";
        headers(k_matmul) = { "src/kernel_matmul_4x4x4_int16_stream.h" };

        // Set execution ratio
        runtime<ratio>(k_matmul) = 0.9;
    }
};

class graph_matmul_64x64x64_int8 : public graph {
private:
    kernel k_matmul;

public:
    input_plio in;
    output_plio out;

    graph_matmul_64x64x64_int8() {
        // Create kernel
        k_matmul = kernel::create(kernel_matmul_64x64x64_int8);

        // Create PLIO ports (128 bits = 16 bytes = 16 int8s per line)
        in = input_plio::create("in", plio_128_bits, "data/input_identity_64x64x64.txt");
        out = output_plio::create("out", plio_128_bits, "data/output_identity_64x64x64.txt");

        // Connect PLIOs to kernel
        connect<window<8192>>(in.out[0], k_matmul.in[0]);    // 64x64x2 int8 = 8192 bytes
        connect<window<4096>>(k_matmul.out[0], out.in[0]);   // 64x64 int8 = 4096 bytes

        // Register source and header files
        source(k_matmul) = "src/kernel_matmul_64x64x64_int8.cpp";
        headers(k_matmul) = {"src/kernel_matmul_64x64x64_int8.h"};

        // Performance estimation
        runtime<ratio>(k_matmul) = 0.9;
    }
};

class graph_matmul_4x16x8_int8 : public graph {
private:
    kernel k_matmul;

public:
    input_plio in;
    output_plio out;

    graph_matmul_4x16x8_int8() {
        // Create kernel
        k_matmul = kernel::create(kernel_matmul_4x16x8_int8);

        // Create PLIO ports (128 bits = 16 bytes = 16 int8s per line)
        in = input_plio::create("in", plio_128_bits, "data/input_random_4x16x8.txt");
        out = output_plio::create("out", plio_128_bits, "data/output_random_4x16x8.txt");

        // Connect PLIOs to kernel
        connect<window<192>>(in.out[0], k_matmul.in[0]);    // 64x64x2 int8 = 8192 bytes
        connect<window<32>>(k_matmul.out[0], out.in[0]);   // 64x64 int8 = 4096 bytes

        // Register source and header files
        source(k_matmul) = "src/kernel_matmul_4x16x8_int8.cpp";
        headers(k_matmul) = {"src/kernel_matmul_4x16x8_int8.h"};

        // Performance estimation
        runtime<ratio>(k_matmul) = 0.9;
    }
};

class graph_matmul_8x16x8_int8 : public graph {
private:
    kernel k_matmul;

public:
    input_plio in;
    output_plio out;

    graph_matmul_8x16x8_int8() {
        // Create kernel
        k_matmul = kernel::create(kernel_matmul_8x16x8_int8);

        // Create PLIO ports (128 bits = 16 bytes = 16 int8s per line)
        in = input_plio::create("in", plio_128_bits, "data/input_random_8x16x8.txt");
        out = output_plio::create("out", plio_128_bits, "data/output_random_8x16x8.txt");

        // Connect PLIOs to kernel
        connect<window<256>>(in.out[0], k_matmul.in[0]);    // 64x64x2 int8 = 8192 bytes
        connect<window<64>>(k_matmul.out[0], out.in[0]);   // 64x64 int8 = 4096 bytes

        // Register source and header files
        source(k_matmul) = "src/kernel_matmul_8x16x8_int8.cpp";
        headers(k_matmul) = {"src/kernel_matmul_8x16x8_int8.h"};

        // Performance estimation
        runtime<ratio>(k_matmul) = 0.9;
    }
};

