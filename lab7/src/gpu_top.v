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
	wire [31:0] host_rdata_low;
	wire [31:0] host_rdata_high;
	wire run_enable = cpu_power[0];
	wire [31:0] host_reset;
	wire [31:0] host_d0, host_d1, host_d2, host_d3;
	
	
	generic_regs
   #( 
      .UDP_REG_SRC_WIDTH   (2),
      .TAG                 (`IDS_BLOCK_ADDR),          // Tag -- eg. MODULE_TAG
      .REG_ADDR_WIDTH      (`IDS_REG_ADDR_WIDTH),     // Width of block addresses -- eg. MODULE_REG_ADDR_WIDTH
      .NUM_COUNTERS        (0),                 // Number of counters
      .NUM_SOFTWARE_REGS   (2),                 // Number of sw regs
      .NUM_HARDWARE_REGS   (7)                  // Number of hw regs
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
      .software_regs    ({host_addr,cpu_power}),
		
      // --- HW regs interface
      .hardware_regs    ({host_rdata_high, host_rdata_low, host_reset, host_d0, host_d1, host_d2, host_d3}),

      .clk              (clk),
      .reset            (reset)
    );
	
//---------------
reg resettest;
always@(posedge clk or posedge reset)begin
	if(reset) resettest <= 1;
	else resettest <= (resettest)? 1:0;
end
assign host_reset = {31'b0,resettest};


//====================================================================
//                             IF Stage
//====================================================================	

	//宣告要被mux進instruction memory的訊號
	reg  [9:0] PC;//normal program counter
	wire [9:0] JTA;//Jump Target Address
	wire [9:0] BTA;//Branch Target Address
	
	//宣告mux的選擇訊號
	wire is_jump, branch_taken;
	
	//宣告要慢一拍的PC
	reg [9:0] slow_PC;
	
	//instruction memory input 和 output
	wire [31:0] instr_in;
	wire [31:0] instr_out;
	
	//宣告IF_ID的暫存器
	reg [31:0] IF_ID_Instr;
	reg [9:0] IF_ID_PC;
	
	//選擇要進instruction memory的訊號
	reg EX_MEM_branch_taken;
	reg [9:0] EX_MEM_BTA, EX_MEM_JTA;
	reg EX_MEM_JUMP;
	assign instr_in = (EX_MEM_branch_taken) ? EX_MEM_BTA : (EX_MEM_JUMP) ? EX_MEM_JTA : PC;
	
	//program counter +1 邏輯
	always@(posedge clk)begin
		if(reset) PC <= 10'b0;
		else if (run_enable) begin
			if(EX_MEM_branch_taken) PC <= EX_MEM_BTA + 1;
			else if (EX_MEM_JUMP) PC <= EX_MEM_JTA + 1;
			else PC <= PC + 1;
		end
	end
	
	//instruction memory 在這裡
	inst_mem im(
		.addr(instr_in),
		.clk(clk),
		.instr(instr_out)
	);
	
	//IF_ID暫存器邏輯
	always@(posedge clk)begin
		if(reset) IF_ID_Instr <= 32'b0;
		else if (run_enable) begin 
			slow_PC <= instr_in;//讓PC慢一拍進ID和instruction時序切一樣
			IF_ID_Instr <= instr_out;
			IF_ID_PC <= slow_PC;
		end
	end

//====================================================================
//                             ID Stage
//====================================================================	
	
	//control unit
	reg ID_EX_use_tensor, ID_EX_ALU_Src, ID_EX_Mem_Read, ID_EX_Mem_Write, ID_EX_Reg_Write, ID_EX_Mem_to_Reg;
	always@(posedge clk)begin
		if(reset)begin
			ID_EX_use_tensor         <= 1'b0;
			ID_EX_ALU_Src            <= 1'b0;
			ID_EX_Mem_Read           <= 1'b0;
			ID_EX_Mem_Write          <= 1'b0;
			ID_EX_Reg_Write          <= 1'b0;
			ID_EX_Mem_to_Reg         <= 1'b0;
		end
		else if (run_enable)begin
			//用tensor unit 或是 ALU
			ID_EX_use_tensor <= (IF_ID_Instr[31:30]== 2'b01) ? 1'b1 : 1'b0;
			//ALU第二個data來自哪(ALU_Src = 1代表來自imm,ALU_Src =0代表來自rs2)
			ID_EX_ALU_Src <= ((IF_ID_Instr[31:26]== 6'b000001)||(IF_ID_Instr[31:26]== 6'b000010)||(IF_ID_Instr[31:26]== 6'b110000)||(IF_ID_Instr[31:26]== 6'b110001)) ? 1'b1 : 1'b0;
			//記憶體讀取信號
			ID_EX_Mem_Read <= (IF_ID_Instr[31:26]== 6'b000001) ? 1'b1 : 1'b0;
			//記憶體寫入信號
			ID_EX_Mem_Write <= (IF_ID_Instr[31:26]== 6'b000010) ? 1'b1 : 1'b0;
			//regfile是否寫回
			ID_EX_Reg_Write <= ((IF_ID_Instr[31:26]==6'b000001)||(IF_ID_Instr[31:26]==6'b001000)||(IF_ID_Instr[31:26]==6'b001001)||(IF_ID_Instr[31:26]==6'b010000)||(IF_ID_Instr[31:26]==6'b010001)||(IF_ID_Instr[31:26]==6'b011000)||(IF_ID_Instr[31:26]==6'b111000)) ? 1'b1 : 1'b0;
			//寫回regfile來自ALU(0)或memory(1)
			ID_EX_Mem_to_Reg <= (IF_ID_Instr[31:26]==6'b000001) ? 1'b1 : 1'b0; 

		end
	end
	
	//OPcode傳進EX Stage
	reg [5:0] ID_EX_Op;
	always@(posedge clk)begin
		if(reset) ID_EX_Op <= 6'b0;
		else if (run_enable) ID_EX_Op <= IF_ID_Instr[31:26];
	end
	
	//ADVANCE_TID推進指令(先假設給他opcode111111)
	wire trigger_next_group = (IF_ID_Instr[31:26] == 6'b111111) ? run_enable : 1'b0;
	
	//PC
	reg [9:0] ID_EX_PC;
	always@(posedge clk)begin
		if(reset) ID_EX_PC <= 10'b0;
		else if (run_enable) ID_EX_PC <= IF_ID_PC;
	end
	
	//第一個regfile在這裡
	wire [63:0] rd1, rd2;
	wire WB_Reg_Write;
	wire [63:0] WB_Data;
	wire [4:0] WB_Addr;
	regfile_two_input rf2p(
		.clk(clk),
		.reset(reset),
		.we(WB_Reg_Write),
		.din(WB_Data),
		.addra(IF_ID_Instr[20:16]),
		.addrb(IF_ID_Instr[15:11]),
		.waddr(WB_Addr),
		.outa(rd1),
		.outb(rd2)
	);
	//第二個regfile在這裡
	wire [63:0] rd3;
	regfile_one_input rf1p(
		.clk(clk),
		.reset(reset),
		.we(WB_Reg_Write),
		.din(WB_Data),
		.addra(IF_ID_Instr[10:6]),
		.waddr(WB_Addr),
		.outa(rd3)
	);

	//Thread ID Controller在這裡
	wire [63:0] current_tid_vector;
	thread_id_controller tic(
		.clk(clk),
		.reset(reset),
		.next_thread_group(trigger_next_group),
		.tid_vector(current_tid_vector)
	);
	
	//sign-extend immediate
	wire [15:0] imm = IF_ID_Instr[15:0];
	reg [63:0] sign_ext_imm;
	always@(posedge clk)begin
		if(reset) sign_ext_imm <= 64'b0;
		else if (run_enable) sign_ext_imm <= {{48{imm[15]}},imm};
	end
	
	
	reg [63:0] ID_EX_TID;
	reg [4:0] ID_EX_Rd;
	reg ID_EX_is_GETTID;
	always@(posedge clk)begin
		if(reset)begin
			ID_EX_is_GETTID <= 1'b0;
			ID_EX_TID <= 64'b0;
			ID_EX_Rd <= 5'b0;
		end
		else if (run_enable) begin
			ID_EX_is_GETTID <= (IF_ID_Instr[31:26]==6'b111000);
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
	//ALU在這裡
	wire [63:0] alu_result;
	wire EX_branch_taken;
	ALU alu(
		.rs1_data(ID_EX_data_1),
		.rs2_data(ID_EX_data_2),
		.alu_op(ID_EX_Op),
		.alu_result(alu_result),
		.branch_taken(EX_branch_taken)
	);
	//tensor unit在這裡
	wire [63:0] tu_result;
	tensor_unit tu(
		.rs1_data(ID_EX_data_1),
		.rs2_data(ID_EX_data_2),
		.rs3_data(ID_EX_data_3),
		.tu_op(ID_EX_Op),
		.tu_result(tu_result)
	);
	//計算結果選擇
	reg [63:0] EX_MEM_result;
	always@(posedge clk)begin
		if(reset) EX_MEM_result <= 64'b0;
		else if (run_enable) EX_MEM_result <= (ID_EX_use_tensor) ? tu_result : alu_result;
	end
	
	//BTA計算
	wire [9:0] EX_BTA = ID_EX_PC + ID_EX_data_2[9:0];
	//JTA計算
	wire EX_JUMP = (ID_EX_Op == 6'b110001) ? 1'b1 : 1'b0;
	wire [9:0] EX_JTA = ID_EX_PC + ID_EX_data_2[9:0];
	
	
	//帶進下個stage的信號
	reg EX_MEM_Mem_Read, EX_MEM_Mem_Write, EX_MEM_Reg_Write, EX_MEM_Mem_to_Reg;
	reg [4:0] EX_MEM_Rd;
	reg [63:0] EX_MEM_rd2;
	always@(posedge clk)begin
		if(reset)begin
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
	wire [63:0] actual_dmem_addr = (run_enable) ? EX_MEM_result : {32'b0,host_addr};
	wire [11:0] data_addr_out;
	data_mem dmem(
		.clk(clk),
		.mem_read_in(run_enable ? EX_MEM_Mem_Read : 1'b1),      
		.mem_write_in(EX_MEM_Mem_Write & run_enable),    
		.alu_result_in(actual_dmem_addr),      
		.rs2_data_in(EX_MEM_rd2),           
		.addr_out(data_addr_out),
		.mem_read_data(MEM_WB_read_data)    
	);
	
	reg[31:0] data_0, data_1, data_2, data_3;
	always@(posedge clk)begin
		if(reset)begin
			data_0 <= 0;
			data_1 <= 0;
			data_2 <= 0;
			data_3 <= 0;
		end
		else begin
			data_0 <= (data_addr_out == 0)? MEM_WB_read_data[31:0] : data_0;
			data_1 <= (data_addr_out == 1)? MEM_WB_read_data[31:0] : data_1;
			data_2 <= (data_addr_out == 2)? MEM_WB_read_data[31:0] : data_2;
			data_3 <= (data_addr_out == 3)? MEM_WB_read_data[31:0] : data_3;
		end
	end
	
	assign host_d0 = data_0;
	assign host_d1 = data_1;
	assign host_d2 = data_2;
	assign host_d3 = data_3;
	
	
	assign host_rdata_low  = MEM_WB_read_data[31:0];
	assign host_rdata_high = MEM_WB_read_data[63:32];
	
	//要進WB的訊號
	reg MEM_WB_Reg_Write;
	reg MEM_WB_Mem_to_Reg;
	reg [4:0]  MEM_WB_Rd;
	reg [63:0] MEM_WB_alu_result;

	always@(posedge clk)begin
		if(reset)begin
			MEM_WB_Reg_Write  <= 1'b0;
			MEM_WB_Mem_to_Reg <= 1'b0;
			MEM_WB_Rd         <= 5'b0;
			MEM_WB_alu_result <= 64'b0;
		end
		else if (run_enable) begin
			MEM_WB_Reg_Write  <= EX_MEM_Reg_Write;
			MEM_WB_Mem_to_Reg <= EX_MEM_Mem_to_Reg;
			MEM_WB_Rd         <= EX_MEM_Rd;
			MEM_WB_alu_result <= EX_MEM_result;
		end
	end

//====================================================================
//                             WB Stage
//====================================================================
	//MEM_WB_Mem_to_Reg = 1  取store資料 ; MEM_WB_Mem_to_Reg = 0 取計算資料
	assign WB_Data = (MEM_WB_Mem_to_Reg) ? MEM_WB_read_data : MEM_WB_alu_result;
	assign WB_Reg_Write = MEM_WB_Reg_Write & run_enable;
	assign WB_Addr      = MEM_WB_Rd;
	

	
endmodule