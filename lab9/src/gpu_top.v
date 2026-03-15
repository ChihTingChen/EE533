`timescale 1ns / 1ps
module gpu_top(
	input clk, reset,
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

//====================================================================
//                             Generic Regs
//====================================================================
	wire [31:0] cpu_power;
	wire [31:0] host_addr;
	wire [31:0] host_wr_addr;
	wire [31:0] host_wr_data;
	wire [31:0] host_wr_ctrl;
	wire run_enable = cpu_power[0];
	
	wire [31:0] host_rdata_low;
	wire [31:0] host_rdata_high;
	
	generic_regs
   #( 
      .UDP_REG_SRC_WIDTH   (2),
      .TAG                 (`IDS_BLOCK_ADDR),
      .REG_ADDR_WIDTH      (`IDS_REG_ADDR_WIDTH),
      .NUM_COUNTERS        (0),
      .NUM_SOFTWARE_REGS   (5),
      .NUM_HARDWARE_REGS   (2)
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

      .counter_updates  (),
      .counter_decrement(),

      .software_regs    ({host_wr_ctrl, host_wr_data, host_wr_addr, host_addr, cpu_power}),
		
      .hardware_regs    ({host_rdata_high, host_rdata_low}),

      .clk              (clk),
      .reset            (reset)
    );

//====================================================================
//                     Host Write Trigger
//====================================================================
reg [31:0] wr_ctrl_prev;
always @(posedge clk) begin
	if (reset)
		wr_ctrl_prev <= 32'd0;
	else
		wr_ctrl_prev <= host_wr_ctrl;
end

wire wr_trigger = (host_wr_ctrl != wr_ctrl_prev) && (host_wr_ctrl != 32'd0) && (~run_enable);

// Latch low-half buffers
reg [31:0] dmem_low_buf;
reg [31:0] rf_low_buf;
always @(posedge clk) begin
	if (reset) begin
		dmem_low_buf <= 32'd0;
		rf_low_buf   <= 32'd0;
	end
	else if (wr_trigger) begin
		if (host_wr_ctrl[2:0] == 3'd2)
			dmem_low_buf <= host_wr_data;
		if (host_wr_ctrl[2:0] == 3'd4)
			rf_low_buf <= host_wr_data;
	end
end

// One-shot write enables
wire host_imem_we = wr_trigger && (host_wr_ctrl[2:0] == 3'd1);
wire host_dmem_we = wr_trigger && (host_wr_ctrl[2:0] == 3'd3);
wire host_rf_we   = wr_trigger && (host_wr_ctrl[2:0] == 3'd5);

// 64-bit write data: current wr_data = high, latched buf = low
wire [63:0] host_dmem_wdata_64 = {host_wr_data, dmem_low_buf};
wire [63:0] host_rf_wdata_64   = {host_wr_data, rf_low_buf}; 



//====================================================================
//                             IF Stage
//====================================================================	

	reg  [9:0] PC;
	wire [9:0] JTA;
	wire [9:0] BTA;
	
	wire is_jump, branch_taken;
	
	reg [9:0] slow_PC;
	
	wire [31:0] instr_in;
	wire [31:0] instr_out;
	
	reg [31:0] IF_ID_Instr;
	reg [9:0] IF_ID_PC;
	
	reg EX_MEM_branch_taken;
	reg [9:0] EX_MEM_BTA, EX_MEM_JTA;
	reg EX_MEM_JUMP;
	assign instr_in = (EX_MEM_branch_taken) ? EX_MEM_BTA : (EX_MEM_JUMP) ? EX_MEM_JTA : PC;
	
	always @(posedge clk) begin
		if (reset) PC <= 10'b0;
		else if (run_enable) begin
			if (EX_MEM_branch_taken) PC <= EX_MEM_BTA + 1;
			else if (EX_MEM_JUMP) PC <= EX_MEM_JTA + 1;
			else PC <= PC + 1;
		end
	end
	
	inst_mem im(
		.addr(instr_in),
		.clk(clk),
		.host_we(host_imem_we),
		.host_waddr(host_wr_addr[9:0]),
		.host_wdata(host_wr_data),
		.instr(instr_out)
	);
	
	always @(posedge clk) begin
		if (reset) IF_ID_Instr <= 32'b0;
		else if (run_enable) begin 
			slow_PC <= instr_in;
			IF_ID_Instr <= instr_out;
			IF_ID_PC <= slow_PC;
		end
	end

//====================================================================
//                             ID Stage
//====================================================================	
	
	reg ID_EX_use_tensor, ID_EX_ALU_Src, ID_EX_Mem_Read, ID_EX_Mem_Write, ID_EX_Reg_Write, ID_EX_Mem_to_Reg;
	always @(posedge clk) begin
		if (reset) begin
			ID_EX_use_tensor         <= 1'b0;
			ID_EX_ALU_Src            <= 1'b0;
			ID_EX_Mem_Read           <= 1'b0;
			ID_EX_Mem_Write          <= 1'b0;
			ID_EX_Reg_Write          <= 1'b0;
			ID_EX_Mem_to_Reg         <= 1'b0;
		end
		else if (run_enable) begin
			ID_EX_use_tensor <= (IF_ID_Instr[31:30] == 2'b01) ? 1'b1 : 1'b0;
			ID_EX_ALU_Src <= ((IF_ID_Instr[31:26] == 6'b000001)||(IF_ID_Instr[31:26] == 6'b000010)||(IF_ID_Instr[31:26] == 6'b110000)||(IF_ID_Instr[31:26] == 6'b110001)) ? 1'b1 : 1'b0;
			ID_EX_Mem_Read <= (IF_ID_Instr[31:26] == 6'b000001) ? 1'b1 : 1'b0;
			ID_EX_Mem_Write <= (IF_ID_Instr[31:26] == 6'b000010) ? 1'b1 : 1'b0;
			ID_EX_Reg_Write <= ((IF_ID_Instr[31:26]==6'b000001)||(IF_ID_Instr[31:26]==6'b001000)||(IF_ID_Instr[31:26]==6'b001001)||(IF_ID_Instr[31:26]==6'b010000)||(IF_ID_Instr[31:26]==6'b010001)||(IF_ID_Instr[31:26]==6'b011000)||(IF_ID_Instr[31:26]==6'b111000)) ? 1'b1 : 1'b0;
			ID_EX_Mem_to_Reg <= (IF_ID_Instr[31:26]==6'b000001) ? 1'b1 : 1'b0;
		end
	end
	
	reg [5:0] ID_EX_Op;
	always @(posedge clk) begin
		if (reset) ID_EX_Op <= 6'b0;
		else if (run_enable) ID_EX_Op <= IF_ID_Instr[31:26];
	end
	
	wire trigger_next_group = (IF_ID_Instr[31:26] == 6'b111111) ? run_enable : 1'b0;
	
	reg [9:0] ID_EX_PC;
	always @(posedge clk) begin
		if (reset) ID_EX_PC <= 10'b0;
		else if (run_enable) ID_EX_PC <= IF_ID_PC;
	end
	
	
	wire WB_Reg_Write;
	wire [63:0] WB_Data;
	wire [4:0] WB_Addr;
	
	// Regfile write mux: GPU writeback vs host load
	wire        rf_we    = run_enable ? WB_Reg_Write : host_rf_we; 
	wire [63:0] rf_din   = run_enable ? WB_Data      : host_rf_wdata_64; 
	wire [4:0]  rf_waddr = run_enable ? WB_Addr      : host_wr_addr[4:0]; 
	
	wire [63:0] rd1, rd2;
	regfile_two_input rf2p(
		.clk(clk),
		.reset(reset),
		.we(rf_we),
		.din(rf_din),
		.addra(IF_ID_Instr[20:16]),
		.addrb(IF_ID_Instr[15:11]),
		.waddr(rf_waddr),
		.outa(rd1),
		.outb(rd2)
	);
	
	wire [63:0] rd3;
	regfile_one_input rf1p(
		.clk(clk),
		.reset(reset),
		.we(rf_we),
		.din(rf_din),
		.addra(IF_ID_Instr[10:6]),
		.waddr(rf_waddr),
		.outa(rd3)
	);

	wire [63:0] current_tid_vector;
	thread_id_controller tic(
		.clk(clk),
		.reset(reset),
		.next_thread_group(trigger_next_group),
		.tid_vector(current_tid_vector)
	);
	
	wire [15:0] imm = IF_ID_Instr[15:0];
	reg [63:0] sign_ext_imm;
	always @(posedge clk) begin
		if (reset) sign_ext_imm <= 64'b0;
		else if (run_enable) sign_ext_imm <= {{48{imm[15]}}, imm};
	end
	
	reg [63:0] ID_EX_TID;
	reg [4:0] ID_EX_Rd;
	reg ID_EX_is_GETTID;
	always @(posedge clk) begin
		if (reset) begin
			ID_EX_is_GETTID <= 1'b0;
			ID_EX_TID <= 64'b0;
			ID_EX_Rd <= 5'b0;
		end
		else if (run_enable) begin
			ID_EX_is_GETTID <= (IF_ID_Instr[31:26] == 6'b111000);
			ID_EX_TID <= current_tid_vector;
			ID_EX_Rd <= IF_ID_Instr[25:21];
		end
	end
	
	wire [63:0] ID_EX_data_1 = ID_EX_is_GETTID ? ID_EX_TID : rd1;
	wire [63:0] ID_EX_data_2 = ID_EX_ALU_Src ? sign_ext_imm : rd2;
	wire [63:0] ID_EX_data_3 = rd3;
	wire [63:0] ID_EX_rd2 = rd2;

//====================================================================
//                             EX Stage
//====================================================================
	wire [63:0] alu_result;
	wire EX_branch_taken;
	ALU alu(
		.rs1_data(ID_EX_data_1),
		.rs2_data(ID_EX_data_2),
		.alu_op(ID_EX_Op),
		.alu_result(alu_result),
		.branch_taken(EX_branch_taken)
	);
	
	wire [63:0] tu_result;
	tensor_unit tu(
		.clk(clk),
		.rs1_data(ID_EX_data_1),
		.rs2_data(ID_EX_data_2),
		.rs3_data(ID_EX_data_3),
		.tu_op(ID_EX_Op),
		.tu_result(tu_result)
	);
	
	reg [63:0] EX_MEM_result;
	always @(posedge clk) begin
		if (reset) EX_MEM_result <= 64'b0;
		else if (run_enable) EX_MEM_result <= alu_result;
	end
	
	wire [9:0] EX_BTA = ID_EX_PC + ID_EX_data_2[9:0];
	wire EX_JUMP = (ID_EX_Op == 6'b110001) ? 1'b1 : 1'b0;
	wire [9:0] EX_JTA = ID_EX_PC + ID_EX_data_2[9:0];
	
	reg EX_MEM_use_tensor; 
	reg EX_MEM_Mem_Read, EX_MEM_Mem_Write, EX_MEM_Reg_Write, EX_MEM_Mem_to_Reg;
	reg [4:0] EX_MEM_Rd;
	reg [63:0] EX_MEM_rd2;
	
	always @(posedge clk) begin
		if (reset) begin
			EX_MEM_use_tensor <= 1'b0;
			EX_MEM_Mem_Read   <= 1'b0;
			EX_MEM_Mem_Write  <= 1'b0;
			EX_MEM_Reg_Write  <= 1'b0;
			EX_MEM_Mem_to_Reg <= 1'b0;
			EX_MEM_Rd         <= 1'b0;
			EX_MEM_branch_taken <= 1'b0;
			EX_MEM_BTA        <= 10'b0;
			EX_MEM_JUMP       <= 10'b0;
			EX_MEM_JTA        <= 10'b0;
			EX_MEM_rd2        <= 64'b0;
		end
		else if (run_enable) begin
			EX_MEM_use_tensor <= ID_EX_use_tensor;
			EX_MEM_Mem_Read   <= ID_EX_Mem_Read;
			EX_MEM_Mem_Write  <= ID_EX_Mem_Write;
			EX_MEM_Reg_Write  <= ID_EX_Reg_Write;
			EX_MEM_Mem_to_Reg <= ID_EX_Mem_to_Reg;
			EX_MEM_Rd         <= ID_EX_Rd;
			EX_MEM_branch_taken <= EX_branch_taken;
			EX_MEM_BTA        <= EX_BTA;
			EX_MEM_JUMP       <= EX_JUMP;
			EX_MEM_JTA        <= EX_JTA;
			EX_MEM_rd2        <= ID_EX_rd2;
		end
	end
	

//====================================================================
//                             MEM Stage
//====================================================================	
	wire [63:0] MEM_WB_read_data;
	// Mux between GPU pipeline and host
	wire [63:0] dmem_addr  = host_dmem_we ? {32'b0, host_wr_addr} : (run_enable ? EX_MEM_result : {32'b0, host_addr});
	wire        dmem_we    = run_enable ? EX_MEM_Mem_Write : host_dmem_we;
	wire [63:0] dmem_wdata = run_enable ? EX_MEM_rd2       : host_dmem_wdata_64; 
	
	wire [11:0] data_addr_out;
	data_mem dmem(
		.clk(clk),
		.mem_read_in(run_enable ? EX_MEM_Mem_Read : 1'b1),      
		.mem_write_in(dmem_we),    
		.alu_result_in(dmem_addr),      
		.rs2_data_in(dmem_wdata),           
		.addr_out(data_addr_out),
		.mem_read_data(MEM_WB_read_data)    
	);
	
	assign host_rdata_low  = MEM_WB_read_data[31:0];
	assign host_rdata_high = MEM_WB_read_data[63:32];
	
	reg MEM_WB_use_tensor;
	reg MEM_WB_Reg_Write;
	reg MEM_WB_Mem_to_Reg;
	reg [4:0]  MEM_WB_Rd;
	reg [63:0] MEM_WB_alu_result;
	
	always @(posedge clk) begin
		if (reset) begin
			MEM_WB_use_tensor <= 1'b0;
			MEM_WB_Reg_Write  <= 1'b0;
			MEM_WB_Mem_to_Reg <= 1'b0;
			MEM_WB_Rd         <= 5'b0;
			MEM_WB_alu_result <= 64'b0;
		end
		else if (run_enable) begin
			MEM_WB_use_tensor <= EX_MEM_use_tensor;
			MEM_WB_Reg_Write  <= EX_MEM_Reg_Write;
			MEM_WB_Mem_to_Reg <= EX_MEM_Mem_to_Reg;
			MEM_WB_Rd         <= EX_MEM_Rd;
			MEM_WB_alu_result <= EX_MEM_result;
		end
	end

//====================================================================
//                             WB Stage
//====================================================================
	
	assign WB_Data = MEM_WB_use_tensor ? tu_result :
	                 (MEM_WB_Mem_to_Reg ? MEM_WB_read_data : MEM_WB_alu_result);
					 
	assign WB_Reg_Write = MEM_WB_Reg_Write & run_enable;
	assign WB_Addr      = MEM_WB_Rd;

endmodule