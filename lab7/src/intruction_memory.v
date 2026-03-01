`timescale 1ns / 1ps

module inst_mem(
	input [9:0] addr,
	input clk,
	output reg [31:0] instr
);
	reg [31:0] memory [0:1023];
	initial begin : init_mem
		integer i;
        for(i=0; i<100; i=i+1) memory[i] = 32'h00000000;
		memory[0] = 32'hE1600000;
		memory[1] = 32'h00000000;
		memory[2] = 32'h00000000;
		memory[3] = 32'h00000000;
		memory[4] = 32'hC169003F;
		memory[5] = 32'h00000000;
		memory[6] = 32'h00000000;
		memory[7] = 32'h00000000;
		memory[8] = 32'h04810000;
		memory[9] = 32'h04A20000;
		memory[10] = 32'h00000000;
		memory[11] = 32'h00000000;
		memory[12] = 32'h00000000;
		memory[13] = 32'h20C42800;
		memory[14] = 32'h00000000;
		memory[15] = 32'h00000000;
		memory[16] = 32'h00000000;
		memory[17] = 32'h08033000;
		memory[18] = 32'h048C0000;
		memory[19] = 32'h04AD0000;
		memory[20] = 32'h00000000;
		memory[21] = 32'h00000000;
		memory[22] = 32'h00000000;
		memory[23] = 32'h24C42800;
		memory[24] = 32'h00000000;
		memory[25] = 32'h00000000;
		memory[26] = 32'h00000000;
		memory[27] = 32'h080E3000;
		memory[28] = 32'h048F0000;
		memory[29] = 32'h04B00000;
		memory[30] = 32'h00000000;
		memory[31] = 32'h00000000;
		memory[32] = 32'h00000000;
		memory[33] = 32'h40C42800;
		memory[34] = 32'h00000000;
		memory[35] = 32'h00000000;
		memory[36] = 32'h00000000;
		memory[37] = 32'h08113000;
		memory[38] = 32'h04B20000;
		memory[39] = 32'h04D30000;
		memory[40] = 32'h04F40000;
		memory[41] = 32'h00000000;
		memory[42] = 32'h00000000;
		memory[43] = 32'h00000000;
		memory[44] = 32'h450531C0;
		memory[45] = 32'h00000000;
		memory[46] = 32'h00000000;
		memory[47] = 32'h00000000;
		memory[48] = 32'h08154000;
		memory[49] = 32'h04960000;
		memory[50] = 32'h00000000;
		memory[51] = 32'h00000000;
		memory[52] = 32'h00000000;
		memory[53] = 32'h60A40000;
		memory[54] = 32'h00000000;
		memory[55] = 32'h00000000;
		memory[56] = 32'h00000000;
		memory[57] = 32'h08172800;
		memory[58] = 32'hFC000000;
		memory[59] = 32'h00000000;
		memory[60] = 32'h00000000;
		memory[61] = 32'h00000000;
		memory[62] = 32'hC4000000;
		memory[63] = 32'hC400003F;
		
		//$readmemh("/root/netfpga/projects/lab7/src/inst_init.mem",memory);
    end
	always@(posedge clk)begin
		instr <= memory[addr];
	end
endmodule