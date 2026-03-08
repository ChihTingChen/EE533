`timescale 1ns / 1ps

module inst_mem(
    input clk,
    input [9:0] addr,
    output reg [31:0] instr
);
    reg [31:0] memory [0:1023];
    integer i;

    initial begin
		for (i = 0;   i < 256;  i = i + 1) memory[i] = 32'hE1A00000;
        for (i = 256; i < 512;  i = i + 1) memory[i] = 32'hE1A00000;
        for (i = 512; i < 768;  i = i + 1) memory[i] = 32'hE1A00000;
        for (i = 768; i < 1024; i = i + 1) memory[i] = 32'hE1A00000;
        memory[0] = 32'hE3A01001;
        memory[1] = 32'hE3A02002;
        memory[2] = 32'hE3A03003;
        memory[3] = 32'hE3A04004;

        memory[4] = 32'hEC110050; 
        memory[5] = 32'hEC120060; 
        memory[6] = 32'hEC130070; 

        memory[7] = 32'hE1A00000; 
        memory[8] = 32'hE1A00000; 
        memory[9] = 32'hE1A00000; 

        memory[10] = 32'hED156780;

        memory[11] = 32'hE1A00000;
        memory[12] = 32'hE1A00000;
        memory[13] = 32'hE1A00000;

        memory[14] = 32'hEC248000;

        memory[15] = 32'hE3A050FF;

        memory[16] = 32'hE1A00000;
        memory[17] = 32'hE1A00000;
        memory[18] = 32'hE1A00000;

        memory[19] = 32'hE5855000;

        memory[20] = 32'hEAFFFFFE;
    end

	always@(posedge clk)begin
		instr <= memory[addr];
	end
	
endmodule