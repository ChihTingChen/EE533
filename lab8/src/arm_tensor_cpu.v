`timescale 1ns / 1ps

module arm_tensor_cpu (
    input clk,
    input rst,
    input  [63:0] net_in_data,
    input  [7:0]  net_in_ctrl,
    input         net_in_wr,
    output        net_in_rdy,
    output [63:0] net_out_data,
    output [7:0]  net_out_ctrl,
    output        net_out_wr,
    input         net_out_rdy,
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
	generic_regs
   #( 
      .UDP_REG_SRC_WIDTH   (2),
      .TAG                 (`IDS_BLOCK_ADDR),        
      .REG_ADDR_WIDTH      (`IDS_REG_ADDR_WIDTH),    
      .NUM_COUNTERS        (0),                
      .NUM_SOFTWARE_REGS   (1),             
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

      .software_regs    ({host_addr}),
		
      .hardware_regs    ({host_rdata_high, host_rdata_low}),

      .clk              (clk),
      .reset            (reset)
    );
    wire FIFO_FULL, CPU_GO;
//====================================================================
//                            IF Stage
//====================================================================
    reg  [9:0] PC;
    wire [31:0] IF_Instr;
    
    wire EX_BranchTaken; 
    wire [9:0] EX_BranchTarget; 

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            PC <= 10'h0;
        end else if (CPU_GO) begin 
            if (EX_BranchTaken) PC <= EX_BranchTarget;
            else                PC <= PC + 1;
        end
    end

    inst_mem im (
        .clk(clk),
        .addr(PC),
        .instr(IF_Instr)
    );
    reg [9:0]  IF_ID_PC;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            IF_ID_PC    <= 10'b0;
        end 
		else if (CPU_GO) begin
            IF_ID_PC    <= PC;
        end
    end

//====================================================================
//                            ID Stage
//====================================================================
    
	wire [31:0] instr = EX_BranchTaken ? 32'hE1A00000 : IF_Instr;
    wire is_tensor = (instr[27:26] == 2'b11); 
    wire [5:0] tensor_op = is_tensor ? instr[25:20] : 6'b000000;
    
    localparam OP_VLD  = 6'b000001;
    localparam OP_VST  = 6'b000010; 
    localparam OP_FMA  = 6'b010001; 
    
    wire [31:0] arm_r0data, arm_r1data;
    wire arm_RegWrite; 
    wire [3:0] arm_waddr;
    wire [31:0] arm_wdata;
    
    wire [3:0] arm_r1_addr = (instr[27:26] == 2'b01) ? instr[15:12] : instr[3:0];
    regfile arm_rf(
        .clk(clk),
        .reset(rst),
        .wena(arm_RegWrite),
        .waddr(arm_waddr),
        .wdata(arm_wdata),
        .r0addr(instr[19:16]),
        .r1addr(arm_r1_addr), 
        .r0data(arm_r0data),
        .r1data(arm_r1data)
    );
    reg [31:0] arm_alu_operand_b;
    always @(*) begin
        if (instr[27:26] == 2'b00 && instr[25] == 1'b1) 
            arm_alu_operand_b = {24'b0, instr[7:0]};
        else if (instr[27:26] == 2'b01 && instr[25] == 1'b0)
            arm_alu_operand_b = {20'b0, instr[11:0]};
        else
            arm_alu_operand_b = arm_r1data;
    end

    wire [63:0] ten_rd1, ten_rd2, ten_rd3;
    wire ten_RegWrite;
    wire [4:0] ten_waddr;
    wire [63:0] ten_wdata;
    
    wire [4:0] ten_raddr1 = (is_tensor && tensor_op == OP_VST) ? {1'b0, instr[15:12]} : {1'b0, instr[19:16]};

    regfile_64b tensor_rf(
        .clk(clk),
        .reset(rst),
        .we(ten_RegWrite),
        .waddr(ten_waddr),
        .wdata(ten_wdata),
        .raddr1(ten_raddr1),             
        .raddr2({1'b0, instr[15:12]}),   
        .raddr3({1'b0, instr[11:8]}),    
        .rdata1(ten_rd1),
        .rdata2(ten_rd2),
        .rdata3(ten_rd3)
    );
    reg ID_EX_is_tensor;
    reg [5:0]  ID_EX_tensor_op;
    reg [31:0] ID_EX_arm_r0, ID_EX_arm_r1;
    reg [31:0] ID_EX_arm_alu_b; 
    reg [63:0] ID_EX_ten_rd1, ID_EX_ten_rd2, ID_EX_ten_rd3;
    reg [31:0] ID_EX_Instr;
    reg [9:0]  ID_EX_PC;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            ID_EX_is_tensor <= 1'b0;
            ID_EX_tensor_op <= 6'b0;
            ID_EX_arm_r0    <= 32'b0;
            ID_EX_arm_r1    <= 32'b0;
            ID_EX_arm_alu_b <= 32'b0;
            ID_EX_ten_rd1   <= 64'b0;
            ID_EX_ten_rd2   <= 64'b0;
            ID_EX_ten_rd3   <= 64'b0;
            ID_EX_Instr     <= 32'hE1A00000;
            ID_EX_PC        <= 10'b0;
        end else if (CPU_GO) begin
            if (EX_BranchTaken) begin
                ID_EX_is_tensor <= 1'b0;
                ID_EX_Instr     <= 32'hE1A00000;
            end else begin
                ID_EX_is_tensor <= is_tensor;
                ID_EX_tensor_op <= tensor_op;
                ID_EX_arm_r0    <= arm_r0data;
                ID_EX_arm_r1    <= arm_r1data;
                ID_EX_arm_alu_b <= arm_alu_operand_b;
                ID_EX_ten_rd1   <= ten_rd1;
                ID_EX_ten_rd2   <= ten_rd2;
                ID_EX_ten_rd3   <= ten_rd3;
                ID_EX_PC        <= IF_ID_PC;
                if (instr[27:26] == 2'b00 && instr[25] == 1'b1)
                    ID_EX_Instr <= {instr[31:12], 12'b0};
                else
                    ID_EX_Instr <= instr;
            end
        end
    end

//====================================================================
//                            EX Stage
//====================================================================
    wire [3:0] NZCV;
    wire [31:0] arm_alu_out;
    ALU arm_alu(
        .SRC_A(ID_EX_arm_r0), 
        .SRC_B(ID_EX_arm_alu_b), 
        .Shift_amt(ID_EX_Instr[11:7]), 
        .Shift_type(ID_EX_Instr[6:5]),
        .Opcode(ID_EX_Instr[24:21]), 
        .NZCV(NZCV),
        .ALU_out(arm_alu_out)
    );
    wire [63:0] ten_tu_out;
    tensor_unit tu(
        .rs1_data(ID_EX_ten_rd1),
        .rs2_data(ID_EX_ten_rd2),
        .rs3_data(ID_EX_ten_rd3),
        .tu_op(ID_EX_tensor_op),
        .tu_result(ten_tu_out)
    );
    reg [7:0]  EX_MEM_Addr;       
    reg [63:0] EX_MEM_WriteData;  
    reg EX_MEM_MemWrite;
    reg EX_MEM_MemRead;
    reg EX_MEM_is_tensor;
    reg [5:0]  EX_MEM_tensor_op;
    reg [4:0]  EX_MEM_TenRd;       
    reg [3:0]  EX_MEM_ArmRd;       
    reg [63:0] EX_MEM_TenResult;
    reg [31:0] EX_MEM_ArmResult;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            EX_MEM_MemWrite  <= 1'b0;
            EX_MEM_MemRead   <= 1'b0;
            EX_MEM_is_tensor <= 1'b0;
        end else if (CPU_GO) begin 
            EX_MEM_is_tensor <= ID_EX_is_tensor;
            EX_MEM_tensor_op <= ID_EX_tensor_op;
            EX_MEM_TenResult <= ten_tu_out;
            EX_MEM_ArmResult <= arm_alu_out;
            
            if (ID_EX_is_tensor) begin
                EX_MEM_Addr      <= ID_EX_arm_r0[7:0];
                EX_MEM_WriteData <= ID_EX_ten_rd1;
                EX_MEM_MemWrite  <= (ID_EX_tensor_op == OP_VST);
                EX_MEM_MemRead   <= (ID_EX_tensor_op == OP_VLD);
                EX_MEM_TenRd     <= {1'b0, ID_EX_Instr[7:4]};
            end else begin
                EX_MEM_Addr      <= arm_alu_out[7:0];
                EX_MEM_WriteData <= {32'b0, ID_EX_arm_r1}; 
                EX_MEM_MemWrite  <= (ID_EX_Instr[27:26] == 2'b01) && !ID_EX_Instr[20];
                EX_MEM_MemRead   <= (ID_EX_Instr[27:26] == 2'b01) &&  ID_EX_Instr[20]; 
                EX_MEM_ArmRd     <= ID_EX_Instr[15:12];
            end
        end
    end

    assign EX_BranchTaken = (!ID_EX_is_tensor) && (ID_EX_Instr[27:24] == 4'b1010);
    assign EX_BranchTarget = ID_EX_PC + {{2{ID_EX_Instr[23]}}, ID_EX_Instr[7:0]}; 

//====================================================================
//                            MEM Stage
//====================================================================
    wire [71:0] mem_read_data_72b;
    Convertible_FIFO_Mem mem_stage(
        .clk(clk),
        .reset(rst),
        .in_data(net_in_data),   .in_ctrl(net_in_ctrl),   .in_wr(net_in_wr),     .in_rdy(net_in_rdy),
        .out_data(net_out_data), .out_ctrl(net_out_ctrl), .out_wr(net_out_wr),   .out_rdy(net_out_rdy),
        
        .cpu_we(EX_MEM_MemWrite),
        .cpu_addr(EX_MEM_Addr),
        .cpu_wdata(EX_MEM_WriteData), 
        .mem_data_out(mem_read_data_72b),
        .FIFO_FULL(FIFO_FULL),
        .CPU_GO(CPU_GO),
		.test_addr(host_addr),
		.test_data_up(host_rdata_high),
		.test_data_down(host_rdata_low)
    );

    reg MEM_WB_is_tensor;
    reg [5:0]  MEM_WB_tensor_op;
    reg [63:0] MEM_WB_TenResult;
    reg [31:0] MEM_WB_ArmResult;
    reg [4:0]  MEM_WB_TenRd;
    reg [3:0]  MEM_WB_ArmRd;
    reg MEM_WB_MemRead;

    always @(posedge clk or posedge rst) begin
        if(rst) begin
            MEM_WB_is_tensor <= 0;
            MEM_WB_MemRead   <= 0;
        end else if (CPU_GO) begin
            MEM_WB_is_tensor <= EX_MEM_is_tensor;
            MEM_WB_tensor_op <= EX_MEM_tensor_op;
            MEM_WB_TenResult <= EX_MEM_TenResult;
            MEM_WB_ArmResult <= EX_MEM_ArmResult;
            MEM_WB_TenRd     <= EX_MEM_TenRd;
            MEM_WB_ArmRd     <= EX_MEM_ArmRd;
            MEM_WB_MemRead   <= EX_MEM_MemRead;
        end
    end

//====================================================================
//                            WB Stage
//====================================================================
    assign arm_wdata = MEM_WB_MemRead ? mem_read_data_72b[31:0] : MEM_WB_ArmResult;
    assign arm_waddr = MEM_WB_ArmRd;
    assign arm_RegWrite = (!MEM_WB_is_tensor) && (MEM_WB_ArmRd != 4'd0);

    assign ten_wdata = (MEM_WB_tensor_op == OP_VLD) ? mem_read_data_72b[63:0] : MEM_WB_TenResult;
    assign ten_waddr = MEM_WB_TenRd;
    assign ten_RegWrite = (MEM_WB_is_tensor) && (MEM_WB_tensor_op == OP_VLD || MEM_WB_tensor_op == OP_FMA);

endmodule