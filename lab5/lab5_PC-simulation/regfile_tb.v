`timescale 1ns / 1ps

module regfile_tb;

    reg clk;
    reg wena;
    reg [2:0] waddr;
    reg [63:0] wdata;
    reg [2:0] r0addr;
    reg [2:0] r1addr;

    wire [63:0] r0data;
    wire [63:0] r1data;

    reg [15*8:1] op_name; 

    regfile uut (
        .clk(clk), 
        .wena(wena), 
        .waddr(waddr), 
        .wdata(wdata), 
        .r0addr(r0addr), 
        .r1addr(r1addr), 
        .r0data(r0data), 
        .r1data(r1data)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        $monitor("Time=%0t | Op=%-15s | WEn=%b | WAddr=%d | WData=%h | R0Addr=%d -> R0Data=%h | R1Addr=%d -> R1Data=%h", 
                 $time, op_name, wena, waddr, wdata, r0addr, r0data, r1addr, r1data);

        op_name = "Init";
        wena = 0; waddr = 0; wdata = 0; r0addr = 0; r1addr = 0;
        #10;

        // --- Case 1: Write Reg 1 ---
        op_name = "Write Reg 1";
        @(negedge clk); 
        wena = 1; waddr = 3'd1; wdata = 64'hAAAAAAAAAAAAAAAA;
        #10; 

        // --- Case 2: Write Reg 2 ---
        op_name = "Write Reg 2";
        @(negedge clk);
        wena = 1; waddr = 3'd2; wdata = 64'hBBBBBBBBBBBBBBBB;
        #10;

        //check Reg1&2 ---
        op_name = "Read R1 & R2";
        @(negedge clk);
        wena = 0; 
        waddr = 0; wdata = 0; 
        r0addr = 3'd1; 
        r1addr = 3'd2; 
        #10;

        // --- Case 4: Try Write when wena=0 ---
        op_name = "Try Write when wena=0";
        @(negedge clk);
        wena = 0; 
        waddr = 3'd3; wdata = 64'hFFFFFFFFFFFFFFFF;
        #10;

        // check reg3
        op_name = "Check Reg 3";
        r0addr = 3'd3; 
		r1addr = 3'd4;
        #10;


        $finish;
    end

endmodule