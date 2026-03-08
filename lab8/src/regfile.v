`timescale 1ns / 1ps
// ARM 16x32-bit
module regfile(
    input clk, reset, wena,
    input [3:0] waddr, r0addr, r1addr,
    input [31:0] wdata,
    output [31:0] r0data, r1data
);
    reg [31:0] register [0:15]; 
    integer i;
    initial begin
        for (i = 0; i < 16; i = i + 1) register[i] = 32'd0;
    end
    always @(posedge clk) begin
        if (wena) register[waddr] <= wdata;
    end
    assign r0data = register[r0addr];
    assign r1data = register[r1addr];
endmodule