////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 1995-2008 Xilinx, Inc.  All rights reserved.
////////////////////////////////////////////////////////////////////////////////
//   ____  ____ 
//  /   /\/   / 
// /___/  \  /    Vendor: Xilinx 
// \   \   \/     Version : 10.1
//  \   \         Application : sch2verilog
//  /   /         Filename : comparator.vf
// /___/   /\     Timestamp : 01/30/2026 00:09:21
// \   \  /  \ 
//  \___\/\___\ 
//
//Command: C:\Xilinx\10.1\ISE\bin\nt\unwrapped\sch2verilog.exe -intstyle ise -family virtex2p -w "C:/Documents and Settings/student/lab3/comparator.sch" comparator.vf
//Design Name: comparator
//Device: virtex2p
//Purpose:
//    This verilog netlist is translated from an ECS schematic.It can be 
//    synthesized and simulated, but it should not be modified. 
//
`timescale 1ns / 1ps

module comp8_MUSER_comparator(A, 
                              B, 
                              EQ);

    input [7:0] A;
    input [7:0] B;
   output EQ;
   
   wire AB0;
   wire AB1;
   wire AB2;
   wire AB3;
   wire AB4;
   wire AB5;
   wire AB6;
   wire AB7;
   wire AB03;
   wire AB47;
   
   AND4 I_36_32 (.I0(AB7), 
                 .I1(AB6), 
                 .I2(AB5), 
                 .I3(AB4), 
                 .O(AB47));
   XNOR2 I_36_33 (.I0(B[6]), 
                  .I1(A[6]), 
                  .O(AB6));
   XNOR2 I_36_34 (.I0(B[7]), 
                  .I1(A[7]), 
                  .O(AB7));
   XNOR2 I_36_35 (.I0(B[5]), 
                  .I1(A[5]), 
                  .O(AB5));
   XNOR2 I_36_36 (.I0(B[4]), 
                  .I1(A[4]), 
                  .O(AB4));
   AND4 I_36_41 (.I0(AB3), 
                 .I1(AB2), 
                 .I2(AB1), 
                 .I3(AB0), 
                 .O(AB03));
   XNOR2 I_36_42 (.I0(B[2]), 
                  .I1(A[2]), 
                  .O(AB2));
   XNOR2 I_36_43 (.I0(B[3]), 
                  .I1(A[3]), 
                  .O(AB3));
   XNOR2 I_36_44 (.I0(B[1]), 
                  .I1(A[1]), 
                  .O(AB1));
   XNOR2 I_36_45 (.I0(B[0]), 
                  .I1(A[0]), 
                  .O(AB0));
   AND2 I_36_50 (.I0(AB47), 
                 .I1(AB03), 
                 .O(EQ));
endmodule
`timescale 1ns / 1ps

module AND7_MXILINX_comparator(I0, 
                               I1, 
                               I2, 
                               I3, 
                               I4, 
                               I5, 
                               I6, 
                               O);

    input I0;
    input I1;
    input I2;
    input I3;
    input I4;
    input I5;
    input I6;
   output O;
   
   wire I36;
   wire O_DUMMY;
   
   assign O = O_DUMMY;
   AND4 I_36_69 (.I0(I3), 
                 .I1(I4), 
                 .I2(I5), 
                 .I3(I6), 
                 .O(I36));
   AND4 I_36_85 (.I0(I0), 
                 .I1(I1), 
                 .I2(I2), 
                 .I3(I36), 
                 .O(O_DUMMY));
   FMAP I_36_98 (.I1(I0), 
                 .I2(I1), 
                 .I3(I2), 
                 .I4(I36), 
                 .O(O_DUMMY));
   // synthesis attribute RLOC of I_36_98 is "X0Y0"
   FMAP I_36_110 (.I1(I3), 
                  .I2(I4), 
                  .I3(I5), 
                  .I4(I6), 
                  .O(I36));
   // synthesis attribute RLOC of I_36_110 is "X0Y0"
endmodule
`timescale 1ns / 1ps

module comparator(a, 
                  amask, 
                  b, 
                  match);

    input [55:0] a;
    input [6:0] amask;
    input [55:0] b;
   output match;
   
   wire XLXN_48;
   wire XLXN_56;
   wire XLXN_57;
   wire XLXN_58;
   wire XLXN_59;
   wire XLXN_60;
   wire XLXN_61;
   wire XLXN_86;
   wire XLXN_87;
   wire XLXN_88;
   wire XLXN_89;
   wire XLXN_90;
   wire XLXN_91;
   wire XLXN_92;
   
   OR2B1 XLXI_28 (.I0(amask[6]), 
                  .I1(XLXN_48), 
                  .O(XLXN_86));
   OR2B1 XLXI_29 (.I0(amask[5]), 
                  .I1(XLXN_58), 
                  .O(XLXN_87));
   OR2B1 XLXI_30 (.I0(amask[4]), 
                  .I1(XLXN_56), 
                  .O(XLXN_88));
   OR2B1 XLXI_31 (.I0(amask[3]), 
                  .I1(XLXN_57), 
                  .O(XLXN_89));
   OR2B1 XLXI_32 (.I0(amask[2]), 
                  .I1(XLXN_59), 
                  .O(XLXN_90));
   OR2B1 XLXI_33 (.I0(amask[1]), 
                  .I1(XLXN_60), 
                  .O(XLXN_91));
   OR2B1 XLXI_34 (.I0(amask[0]), 
                  .I1(XLXN_61), 
                  .O(XLXN_92));
   AND7_MXILINX_comparator XLXI_42 (.I0(XLXN_92), 
                                    .I1(XLXN_91), 
                                    .I2(XLXN_90), 
                                    .I3(XLXN_89), 
                                    .I4(XLXN_88), 
                                    .I5(XLXN_87), 
                                    .I6(XLXN_86), 
                                    .O(match));
   // synthesis attribute HU_SET of XLXI_42 is "XLXI_42_0"
   comp8_MUSER_comparator XLXI_65 (.A(a[55:48]), 
                                   .B(b[55:48]), 
                                   .EQ(XLXN_48));
   comp8_MUSER_comparator XLXI_66 (.A(a[47:40]), 
                                   .B(b[47:40]), 
                                   .EQ(XLXN_58));
   comp8_MUSER_comparator XLXI_67 (.A(a[39:32]), 
                                   .B(b[39:32]), 
                                   .EQ(XLXN_56));
   comp8_MUSER_comparator XLXI_68 (.A(a[31:24]), 
                                   .B(b[31:24]), 
                                   .EQ(XLXN_57));
   comp8_MUSER_comparator XLXI_69 (.A(a[23:16]), 
                                   .B(b[23:16]), 
                                   .EQ(XLXN_59));
   comp8_MUSER_comparator XLXI_70 (.A(a[15:8]), 
                                   .B(b[15:8]), 
                                   .EQ(XLXN_60));
   comp8_MUSER_comparator XLXI_71 (.A(a[7:0]), 
                                   .B(b[7:0]), 
                                   .EQ(XLXN_61));
endmodule
