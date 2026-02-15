`timescale 1ns / 1ps

module alu_tb;

    reg [63:0] A;
    reg [63:0] B;
    reg [2:0] aluctrl;
    wire [63:0] Z;
    wire overflow;

    reg [15*8:1] op_name; 

    parameter Add   = 3'd0;
    parameter Sub   = 3'd1;
    parameter BwAND = 3'd2;
    parameter BwOR  = 3'd3;
    parameter BwXOR = 3'd4;
    parameter Equal = 3'd5;
    parameter Shift = 3'd6;

    alu uut (
        .A(A), 
        .B(B), 
        .aluctrl(aluctrl), 
        .Z(Z), 
        .overflow(overflow)
    );

    initial begin
        $monitor("Time=%0t | Op=%-10s | Ctrl=%d | A=%h | B=%h | Z=%h | Ovf=%b", 
                 $time, op_name, aluctrl, A, B, Z, overflow);

        // --- Case 1: Add ---
        op_name = "Add";  
        A = 64'd5; B = 64'd10; aluctrl = Add;
        #10; 

        // --- Case 2: Add Overflow Test ---
        op_name = "Add_Ovf";
        A = 64'h7FFFFFFFFFFFFFFF; B = 64'd1; aluctrl = Add;
        #10;

        // --- Case 3: Sub ---
        op_name = "Sub";
        A = 64'd50; B = 64'd20; aluctrl = Sub;
        #10;

        // --- Case 4: Sub Overflow Test ---
        op_name = "Sub_Ovf";
        A = 64'h8000000000000000; B = 64'd1; aluctrl = Sub;
        #10;

        // --- Case 5: Bitwise AND ---
        op_name = "AND";
        A = 64'hF0F0F0F0F0F0F0F0; B = 64'h0F0F0F0F0F0F0F0F; aluctrl = BwAND;
        #10;

        // --- Case 6: Bitwise OR ---
        op_name = "OR";
        A = 64'hF0F0F0F0F0F0F0F0; B = 64'h0F0F0F0F0F0F0F0F; aluctrl = BwOR;
        #10;

        // --- Case 7: Bitwise XOR ---
        op_name = "XOR";
        A = 64'hAAAAAAAAAAAAAAAA; B = 64'h5555555555555555; aluctrl = BwXOR;
        #10;

        // --- Case 8: Equal ---
        op_name = "Equal(True)";
        A = 64'd5; B = 64'd5; aluctrl = Equal;
        #10;
        
        op_name = "Equal(False)";
        A = 64'd5; B = 64'd3; aluctrl = Equal;
        #10;

        // --- Case 9: Shift Right ---
        op_name = "Shift";
        A = 64'd8; B = 64'd1; aluctrl = Shift;
        #10;
        
        A = 64'hF000000000000000; B = 64'd4; aluctrl = Shift;
        #10;

        $finish;
    end

endmodule