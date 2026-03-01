`timescale 1ns / 1ps

module regfile_two_input(
    input clk, reset, we,
    input [63:0] din,
    input [4:0] addra, addrb, waddr,
    output reg [63:0] outa, outb
);
    integer i;
    reg [63:0] memory [0:31];
	
	initial begin : init_regfile
		for (i = 0; i < 32; i = i + 1) begin
            memory[i] = 64'b0;
        end

        memory[1]  = 64'd0;
        memory[2]  = 64'd1;
        memory[3]  = 64'd2;

        memory[12] = 64'd10;
        memory[13] = 64'd11;
        memory[14] = 64'd12;

        memory[15] = 64'd20;
        memory[16] = 64'd21;
        memory[17] = 64'd22;

        memory[18] = 64'd30;
        memory[19] = 64'd31;
        memory[20] = 64'd32;
        memory[21] = 64'd33;

        memory[22] = 64'd40;
        memory[23] = 64'd41;

        memory[9]  = 64'd1;  
        memory[10] = 64'd1; 
        memory[11] = 64'd0;  
    end
   
    always @(posedge clk) begin
		if(we)begin
			memory[waddr] <= din;
		end
        outa <= memory[addra];
        outb <= memory[addrb];
    end
endmodule