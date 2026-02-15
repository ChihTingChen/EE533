////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 1995-2008 Xilinx, Inc.  All rights reserved.
////////////////////////////////////////////////////////////////////////////////
//   ____  ____ 
//  /   /\/   / 
// /___/  \  /    Vendor: Xilinx 
// \   \   \/     Version : 10.1
//  \   \         Application : sch2verilog
//  /   /         Filename : detect7B.vf
// /___/   /\     Timestamp : 01/30/2026 00:51:50
// \   \  /  \ 
//  \___\/\___\ 
//
//Command: C:\Xilinx\10.1\ISE\bin\nt\unwrapped\sch2verilog.exe -intstyle ise -family virtex2p -w "C:/Documents and Settings/student/lab3/detect7B.sch" detect7B.vf
//Design Name: detect7B
//Device: virtex2p
//Purpose:
//    This verilog netlist is translated from an ECS schematic.It can be 
//    synthesized and simulated, but it should not be modified. 
//
`timescale 1ns / 1ps

module detect7B(ce, 
                clk, 
                hwregA, 
                match_en, 
                mrst, 
                pipe1, 
                match);

    input ce;
    input clk;
    input [63:0] hwregA;
    input match_en;
    input mrst;
    input [71:0] pipe1;
   output match;
   
   wire [71:0] pipe0;
   wire [111:0] XLXN_20;
   wire XLXN_26;
   wire XLXN_29;
   wire XLXN_34;
   wire match_DUMMY;
   
   assign match = match_DUMMY;
   AND3B1 XLXI_6 (.I0(match_DUMMY), 
                  .I1(match_en), 
                  .I2(XLXN_26), 
                  .O(XLXN_29));
   FDCE XLXI_7 (.C(clk), 
                .CE(XLXN_29), 
                .CLR(XLXN_34), 
                .D(XLXN_29), 
                .Q(match_DUMMY));
   defparam XLXI_7.INIT = 1'b0;
   FD XLXI_8 (.C(clk), 
              .D(mrst), 
              .Q(XLXN_34));
   defparam XLXI_8.INIT = 1'b0;
   reg9B XLXI_9 (.ce(ce), 
                 .clk(clk), 
                 .clr(XLXN_34), 
                 .d(pipe1[71:0]), 
                 .q(pipe0[71:0]));
   busmerge XLXI_10 (.da(pipe0[47:0]), 
                     .db(pipe1[63:0]), 
                     .q(XLXN_20[111:0]));
   wordmatch XLXI_11 (.datacomp(hwregA[55:0]), 
                      .datain(XLXN_20[111:0]), 
                      .wildcard(hwregA[62:56]), 
                      .match(XLXN_26));
endmodule
