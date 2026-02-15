`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    17:25:14 02/11/2026 
// Design Name: 
// Module Name:    top 
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////
module top(
input clk,
input reset
    );

reg [8:0] PC;
reg [31:0] inst_out;
wire [31:0] instruction;
reg [31:0]IF_ID_reg;


reg [67:0] MEM_WB_reg;

wire WB_WRE   = MEM_WB_reg[67];
wire [2:0] WB_Waddr = MEM_WB_reg[66:64];
wire [63:0] WB_Wdata = MEM_WB_reg[63:0];


//PC
always@(posedge clk or posedge reset)begin
	if(reset) PC <= 0;
	else PC <= PC+1;
end


//inst_mem
inst_mem instmem(
	.clka(clk),
	.addra(PC),
	.douta(instruction)
);


//IF_ID_reg
always@(posedge clk or posedge reset)begin
	if(reset) IF_ID_reg <= 0;
	else IF_ID_reg <= instruction;
end



wire WME = IF_ID_reg[31];
wire WRE = IF_ID_reg[30];
wire [2:0] REG1  = IF_ID_reg[29:27];
wire [2:0] REG2  = IF_ID_reg[26:24];
wire [2:0] WREG1 = IF_ID_reg[23:21];
wire [63:0] r0data, r1data;



//regfile
regfile regg(
	.clk(clk),
	.reset(reset),
	.wena(WB_WRE),
	.waddr(WB_Waddr),
	.wdata(WB_Wdata),
	.r0addr(REG1),
	.r1addr(REG2),
	.r0data(r0data),
	.r1data(r1data)
);


//ID_EX_reg
//WME(1) + WRE(1) + WReg1(3) + R1out(64) + R2out(64)= 133bits
reg [132:0] ID_EX_reg;

always@(posedge clk or posedge reset)begin
	if(reset) ID_EX_reg <= 0;
	else begin 
		ID_EX_reg[132] <= WME;
		ID_EX_reg[131] <= WRE;
		ID_EX_reg[130:128] <= WREG1;
		ID_EX_reg[127:64] <= r0data;
		ID_EX_reg[63:0] <= r1data;
	end
end

wire ID_EX_WME  = ID_EX_reg[132];
wire ID_EX_WRE  = ID_EX_reg[131];
wire [2:0] ID_EX_WREG1  = ID_EX_reg[130:128];
wire [63:0] ID_EX_r0data  = ID_EX_reg[127:64];
wire [63:0] ID_EX_r1data  = ID_EX_reg[63:0];


//EX_MEM_reg
reg [132:0] EX_MEM_reg;

always@(posedge clk or posedge reset)begin
	if(reset) EX_MEM_reg <= 0;
	else begin 
		EX_MEM_reg[132] <= ID_EX_WME;
		EX_MEM_reg[131] <= ID_EX_WRE;
		EX_MEM_reg[130:128] <= ID_EX_WREG1;
		EX_MEM_reg[127:64] <= ID_EX_r0data;
		EX_MEM_reg[63:0] <= ID_EX_r1data;
	end
end


wire EX_MEM_WME = EX_MEM_reg[132];
wire EX_MEM_WRE = EX_MEM_reg[131];
wire [2:0]  EX_MEM_WREG1 = EX_MEM_reg[130:128];
wire [63:0] EX_MEM_addr = EX_MEM_reg[127:64];
wire [63:0] EX_MEM_data = EX_MEM_reg[63:0];

wire [63:0] mem_dout;

data_mem datamem(
	.clka(clk),
	.dina(EX_MEM_data),
	.addra(EX_MEM_addr[7:0]),
	.wea(EX_MEM_WME),
	.douta(mem_dout)
);



//MEM_WB

always@(posedge clk or posedge reset)begin
	if(reset) MEM_WB_reg <= 0;
	else begin
		MEM_WB_reg[67] <= EX_MEM_WRE;
        MEM_WB_reg[66:64] <= EX_MEM_WREG1;
        MEM_WB_reg[63:0] <= mem_dout;
	end
end



endmodule
