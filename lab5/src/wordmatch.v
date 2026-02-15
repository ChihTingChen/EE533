////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 1995-2008 Xilinx, Inc.  All rights reserved.
////////////////////////////////////////////////////////////////////////////////
//   ____  ____ 
//  /   /\/   / 
// /___/  \  /    Vendor: Xilinx 
// \   \   \/     Version : 10.1
//  \   \         Application : sch2verilog
//  /   /         Filename : wordmatch.vf
// /___/   /\     Timestamp : 02/03/2026 20:53:05
// \   \  /  \ 
//  \___\/\___\ 
//
//Command: C:\Xilinx\10.1\ISE\bin\nt\unwrapped\sch2verilog.exe -intstyle ise -family virtex2p -w C:/lab3_test/wordmatch.sch wordmatch.vf
//Design Name: wordmatch
//Device: virtex2p
//Purpose:
//    This verilog netlist is translated from an ECS schematic.It can be 
//    synthesized and simulated, but it should not be modified. 
//
`timescale 1ns / 1ps

module OR8_MXILINX_wordmatch(I0, 
                             I1, 
                             I2, 
                             I3, 
                             I4, 
                             I5, 
                             I6, 
                             I7, 
                             O);

    input I0;
    input I1;
    input I2;
    input I3;
    input I4;
    input I5;
    input I6;
    input I7;
   output O;
   
   wire dummy;
   wire S0;
   wire S1;
   wire O_DUMMY;
   
   assign O = O_DUMMY;
   FMAP I_36_91 (.I1(S0), 
                 .I2(S1), 
                 .I3(dummy), 
                 .I4(dummy), 
                 .O(O_DUMMY));
   // synthesis attribute RLOC of I_36_91 is "X0Y1"
   OR2 I_36_94 (.I0(S0), 
                .I1(S1), 
                .O(O_DUMMY));
   OR4 I_36_95 (.I0(I4), 
                .I1(I5), 
                .I2(I6), 
                .I3(I7), 
                .O(S1));
   OR4 I_36_112 (.I0(I0), 
                 .I1(I1), 
                 .I2(I2), 
                 .I3(I3), 
                 .O(S0));
   FMAP I_36_116 (.I1(I0), 
                  .I2(I1), 
                  .I3(I2), 
                  .I4(I3), 
                  .O(S0));
   // synthesis attribute RLOC of I_36_116 is "X0Y0"
   FMAP I_36_117 (.I1(I4), 
                  .I2(I5), 
                  .I3(I6), 
                  .I4(I7), 
                  .O(S1));
   // synthesis attribute RLOC of I_36_117 is "X0Y0"
endmodule
`timescale 1ns / 1ps

module wordmatch(datacomp, 
                 datain, 
                 wildcard, 
                 match);

    input [55:0] datacomp;
    input [111:0] datain;
    input [6:0] wildcard;
   output match;
   
   wire XLXN_26;
   wire XLXN_27;
   wire XLXN_28;
   wire XLXN_29;
   wire XLXN_30;
   wire XLXN_35;
   wire XLXN_36;
   wire XLXN_37;
   
   OR8_MXILINX_wordmatch XLXI_13 (.I0(XLXN_37), 
                                  .I1(XLXN_36), 
                                  .I2(XLXN_35), 
                                  .I3(XLXN_30), 
                                  .I4(XLXN_29), 
                                  .I5(XLXN_28), 
                                  .I6(XLXN_27), 
                                  .I7(XLXN_26), 
                                  .O(match));
   // synthesis attribute HU_SET of XLXI_13 is "XLXI_13_0"
   comparator XLXI_14 (.a(datacomp[55:0]), 
                       .amask(wildcard[6:0]), 
                       .b(datain[55:0]), 
                       .match(XLXN_26));
   comparator XLXI_15 (.a(datacomp[55:0]), 
                       .amask(wildcard[6:0]), 
                       .b(datain[63:8]), 
                       .match(XLXN_27));
   comparator XLXI_16 (.a(datacomp[55:0]), 
                       .amask(wildcard[6:0]), 
                       .b(datain[71:16]), 
                       .match(XLXN_28));
   comparator XLXI_17 (.a(datacomp[55:0]), 
                       .amask(wildcard[6:0]), 
                       .b(datain[79:24]), 
                       .match(XLXN_29));
   comparator XLXI_18 (.a(datacomp[55:0]), 
                       .amask(wildcard[6:0]), 
                       .b(datain[87:32]), 
                       .match(XLXN_30));
   comparator XLXI_19 (.a(datacomp[55:0]), 
                       .amask(wildcard[6:0]), 
                       .b(datain[95:40]), 
                       .match(XLXN_35));
   comparator XLXI_20 (.a(datacomp[55:0]), 
                       .amask(wildcard[6:0]), 
                       .b(datain[103:48]), 
                       .match(XLXN_36));
   comparator XLXI_21 (.a(datacomp[55:0]), 
                       .amask(wildcard[6:0]), 
                       .b(datain[111:56]), 
                       .match(XLXN_37));
endmodule
