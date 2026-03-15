`timescale 1ns / 1ps

module inst_mem(
	input [9:0] addr,
	input clk,
	input host_we,
	input [9:0] host_waddr,
	input [31:0] host_wdata,
	output reg [31:0] instr
);
	reg [31:0] memory [0:1023];
	initial begin : init_mem
        integer i;
        for(i=0; i<200; i=i+1) memory[i] = 32'h00000000;

        // ========================================================
        // Super Kernel: 一次執行 5 種運算
        // ========================================================
    /*    memory[0]  = 32'hE1600000; // GETTID r11
        memory[1]  = 32'h00000000; // NOP
        memory[2]  = 32'h00000000; // NOP
        memory[3]  = 32'h00000000; // NOP
        memory[4]  = 32'hC169003F; // BEQ r11, r9, Target: 63
        memory[5]  = 32'h00000000; // NOP
        memory[6]  = 32'h00000000; // NOP
        memory[7]  = 32'h00000000; // NOP

        // --- 1. Vector Add --- (使用 r1, r2, r3)
        memory[8]  = 32'h04810000; // VLD r4, [r1]
        memory[9]  = 32'h04A20000; // VLD r5, [r2]
        memory[10] = 32'h00000000; // NOP
        memory[11] = 32'h00000000; // NOP
        memory[12] = 32'h00000000; // NOP
        memory[13] = 32'h20C42800; // VADD.I16 r6, r4, r5
        memory[14] = 32'h00000000; // NOP
        memory[15] = 32'h00000000; // NOP
        memory[16] = 32'h00000000; // NOP
        memory[17] = 32'h08033000; // VST [r3], r6

        // --- 2. Vector Sub --- (使用 r12, r13, r14)
        memory[18] = 32'h048C0000; // VLD r4, [r12]
        memory[19] = 32'h04AD0000; // VLD r5, [r13]
        memory[20] = 32'h00000000; // NOP
        memory[21] = 32'h00000000; // NOP
        memory[22] = 32'h00000000; // NOP
        memory[23] = 32'h24C42800; // VSUB.I16 r6, r4, r5
        memory[24] = 32'h00000000; // NOP
        memory[25] = 32'h00000000; // NOP
        memory[26] = 32'h00000000; // NOP
        memory[27] = 32'h080E3000; // VST [r14], r6

        // --- 3. Vector Mul (BF16) --- (使用 r15, r16, r17)
        memory[28] = 32'h048F0000; // VLD r4, [r15]
        memory[29] = 32'h04B00000; // VLD r5, [r16]
        memory[30] = 32'h00000000; // NOP
        memory[31] = 32'h00000000; // NOP
        memory[32] = 32'h00000000; // NOP
        memory[33] = 32'h40C42800; // VMUL.BF16 r6, r4, r5
        memory[34] = 32'h00000000; // NOP
        memory[35] = 32'h00000000; // NOP
        memory[36] = 32'h00000000; // NOP
        memory[37] = 32'h08113000; // VST [r17], r6

        // --- 4. BFloat16 FMA --- (使用 r18, r19, r20, r21)
        memory[38] = 32'h04B20000; // VLD r5, [r18]
        memory[39] = 32'h04D30000; // VLD r6, [r19]
        memory[40] = 32'h04F40000; // VLD r7, [r20]
        memory[41] = 32'h00000000; // NOP
        memory[42] = 32'h00000000; // NOP
        memory[43] = 32'h00000000; // NOP
        memory[44] = 32'h450531C0; // VFMA.BF16 r8, r5, r6, r7
        memory[45] = 32'h00000000; // NOP
        memory[46] = 32'h00000000; // NOP
        memory[47] = 32'h00000000; // NOP
        memory[48] = 32'h08154000; // VST [r21], r8

        // --- 5. ReLU --- (使用 r22, r23)
        memory[49] = 32'h04960000; // VLD r4, [r22]
        memory[50] = 32'h00000000; // NOP
        memory[51] = 32'h00000000; // NOP
        memory[52] = 32'h00000000; // NOP
        memory[53] = 32'h60A40000; // VRELU r5, r4
        memory[54] = 32'h00000000; // NOP
        memory[55] = 32'h00000000; // NOP
        memory[56] = 32'h00000000; // NOP
        memory[57] = 32'h08172800; // VST [r23], r5

        // --- 迴圈推進與結束 ---
        memory[58] = 32'hFC000000; // ADVANCETID
        memory[59] = 32'h00000000; // NOP
        memory[60] = 32'h00000000; // NOP
        memory[61] = 32'h00000000; // NOP
        memory[62] = 32'hC4000000; // JMP 0 (Loop back)

        // --- HALT ---
        memory[63] = 32'hC400003F; // JMP 63 (Halt 停機) */
        
    end
	always@(posedge clk)begin
		if (host_we)
			memory[host_waddr] <= host_wdata;
		instr <= memory[addr];
	end
endmodule