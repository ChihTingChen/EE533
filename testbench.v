`timescale 1ns/1ps

module ALU32_tb;

    reg  clk;
    reg  [2:0]  sel;
    reg  [31:0] A;
    reg  [31:0] B;
    reg  Cin;
    wire [31:0] Sum;
    wire  Cout;

    ALU32 dut (
        .clk (clk),
        .sel (sel),
        .A   (A),
        .B   (B),
        .Cin (Cin),
        .Sum (Sum),
        .Cout(Cout)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        sel = 3'b000;
        A   = 32'd0;
        B   = 32'd0;
        Cin = 1'b0;

        #10;

        sel = 3'b000;
        A   = 32'd1;
        B   = 32'd1;
        Cin = 1'b0;
        #10;

        sel = 3'b001;
        A   = 32'd2;
        B   = 32'd1;
        Cin = 1'b0;
        #10;

        sel = 3'b010;
        A   = 32'd1;
        B   = 32'd0;
        #10;
        sel = 3'b011;
        A   = 32'd1;
        B   = 32'd2;
        #10;

        sel = 3'b011;
        A   = 32'hFFFFFFFF; 
        B   = 32'd1;
        #10;

        sel = 3'b100;
        A   = 32'd5;
        B   = 32'd5;
        #10;

        sel = 3'b100;
        A   = 32'd5;
        B   = 32'd3;
        #10;

        sel = 3'b000;
        A   = 32'hFFFFFFFF;
        B   = 32'd1;
        Cin = 1'b0;
        #10;
        $stop;
    end

endmodule

