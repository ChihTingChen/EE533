`timescale 1ns / 1ps
/*/////////////////////////////////////////////////////////////////////////////////
	
      
       __   __      _____      __                                    
      /\ \  \ \    /\  __`\   /\ \         __                        
      \ \ \  \ \   \ \ \ \_\  \ \ \___    /\_\      ___        ___   
       \ \ \  \ \   \ \ \  __  \ \  _ `\  \/\ \    / __`\     / __`\ 
        \ \ \__\ \   \ \ \ \ \  \ \ \ \ \  \ \ \  /\ \_\ \_  /\ \_\ \
         \ \______\   \ \____/   \ \_\ \_\  \ \_\ \ \__/ \_\ \ \____/
          \/______/    \/___/     \/_/\/_/   \/_/  \/__/\/_/  \/___/ 
      


/////////////////////////////////////////////////////////////////////////////////*/

module cpu_top(
	input clk,
	input reset,
	input reg_req_in,     
	input reg_ack_in,     
	input reg_rd_wr_L_in, 
	input [`UDP_REG_ADDR_WIDTH-1:0] reg_addr_in,    
	input [`CPCI_NF2_DATA_WIDTH-1:0] reg_data_in,    
	input [1:0] reg_src_in,     
	
	output reg_req_out,    
	output reg_ack_out,    
	output reg_rd_wr_L_out,
	output [`UDP_REG_ADDR_WIDTH-1:0] reg_addr_out,   
	output [`CPCI_NF2_DATA_WIDTH-1:0] reg_data_out,   
	output [1:0] reg_src_out    
);	

wire [31:0] cpu_power; //1=cpu_off(write init index only when cpu_off)  0=cpu_on   
wire [31:0] host_addr;  
wire [31:0] host_wdata; 
wire [31:0] host_we;    
wire [31:0] host_rdata; 


reg [8:0] PC;
reg [31:0] inst_out;
wire [31:0] instruction;
reg [31:0]IF_ID_reg;
reg [67:0] MEM_WB_reg;


wire WB_WRE = MEM_WB_reg[67];
wire [2:0] WB_Waddr = MEM_WB_reg[66:64];
wire [63:0] WB_Wdata = MEM_WB_reg[63:0];


//for netfpga write in
wire rst = reset | cpu_power[0];
wire write_to_inst_mem = (cpu_power[0]) & (host_we[0]) & (host_addr < 32'd512);
wire write_to_data_mem = (cpu_power[0]) & (host_we[0]) & (host_addr > 32'd511 && host_addr < 32'd768);
wire [31:0] inst_data;
wire [63:0] data_data;
assign host_rdata = (host_addr < 32'd512)? inst_data : ((host_addr > 32'd511 && host_addr < 32'd768)? data_data[31:0] : 32'hBADDADDD);

//PC
always@(posedge clk or posedge rst)begin
	if(rst) PC <= 0;
	else PC <= PC+1;
end


//inst_mem
inst_mem instmem(
	.clka(clk),
	.addra(PC),
	.douta(instruction),
	
	.addrb(host_addr),
	.dinb(host_wdata),
	.web(write_to_inst_mem),
	.doutb(inst_data)
);


//IF_ID_reg
always@(posedge clk or posedge rst)begin
	if(rst) IF_ID_reg <= 0;
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
	.reset(rst),
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

always@(posedge clk or posedge rst)begin
	if(rst) ID_EX_reg <= 0;
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

always@(posedge clk or posedge rst)begin
	if(rst) EX_MEM_reg <= 0;
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


wire [31:0] data_addr = host_addr - 32'd512;


data_mem datamem(
	.clka(clk),
	.dina(EX_MEM_data),
	.addra(EX_MEM_addr[7:0]),
	.wea(EX_MEM_WME),
	.douta(mem_dout),
	
	.addrb(data_addr),
	.dinb(host_wdata),
	.web(write_to_data_mem),
	.doutb(data_data)
);



//MEM_WB

always@(posedge clk or posedge rst)begin
	if(rst) MEM_WB_reg <= 0;
	else begin
		MEM_WB_reg[67] <= EX_MEM_WRE;
        MEM_WB_reg[66:64] <= EX_MEM_WREG1;
        MEM_WB_reg[63:0] <= mem_dout;
	end
end



//=============generic reg
	generic_regs
   #( 
      .UDP_REG_SRC_WIDTH   (2),
      .TAG                 (`IDS_BLOCK_ADDR),          // Tag -- eg. MODULE_TAG
      .REG_ADDR_WIDTH      (`IDS_REG_ADDR_WIDTH),     // Width of block addresses -- eg. MODULE_REG_ADDR_WIDTH
      .NUM_COUNTERS        (0),                 // Number of counters
      .NUM_SOFTWARE_REGS   (4),                 // Number of sw regs
      .NUM_HARDWARE_REGS   (1)                  // Number of hw regs
   ) module_regs (
      .reg_req_in       (reg_req_in),
      .reg_ack_in       (reg_ack_in),
      .reg_rd_wr_L_in   (reg_rd_wr_L_in),
      .reg_addr_in      (reg_addr_in),
      .reg_data_in      (reg_data_in),
      .reg_src_in       (reg_src_in),

      .reg_req_out      (reg_req_out),
      .reg_ack_out      (reg_ack_out),
      .reg_rd_wr_L_out  (reg_rd_wr_L_out),
      .reg_addr_out     (reg_addr_out),
      .reg_data_out     (reg_data_out),
      .reg_src_out      (reg_src_out),

      // --- counters interface
      .counter_updates  (),
      .counter_decrement(),

      // --- SW regs interface
      .software_regs    ({host_we,host_wdata,host_addr,cpu_power}),
		
      // --- HW regs interface
      .hardware_regs    (host_rdata),

      .clk              (clk),
      .reset            (reset)
    );





endmodule
