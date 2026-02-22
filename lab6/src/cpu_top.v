`timescale 1ns / 1ps
/*/////////////////////////////////////////////////////////////////////////////////
    
      
     ▒▓████████▓▒░▒▓████████▓▒░░▒▓██████▓▒░░▒▓██████████████▓▒░ ░▒▓██████▓▒░  
       ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
       ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
       ░▒▓█▓▒░   ░▒▓██████▓▒░ ░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░  
       ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
       ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
       ░▒▓█▓▒░   ░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░  
      

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

//====================================================================
//                          NetFPGA Setup
//====================================================================
wire [31:0] cpu_power;//0=cpu_off(write init index only when cpu_off)  1=cpu_on
wire [31:0] host_addr;
wire [31:0] host_wdata;
wire [31:0] host_we;
wire [31:0] host_rdata;
wire [31:0] host_thread;


//for netfpga write in
wire rst = reset | ~cpu_power[0];
wire write_to_data_mem = (cpu_power[0]) & (host_we[0]) & (host_addr < 32'd64);
wire [31:0] inst_addr = host_addr - 32'd64;
wire [31:0] inst_data;
wire [31:0] data_data;
assign host_rdata = (host_addr < 32'd64)? data_data : ((host_addr > 32'd63 && host_addr < 32'd192)? inst_data : 32'hBADDADDD);
//====================================================================
//                         For WriteBack
//====================================================================
wire [8:0] EX_Target_PC;
wire EX_branch_taken;
wire [1:0] EX_thread;


wire [31:0] WB_Datamem_rdata;
wire [31:0] WB_ALU_Result;
wire [8:0]  WB_Return_PC;

wire [3:0] WB_Rd;
wire [1:0] WB_thread;

wire WB_RegWrite;
wire WB_MemToReg;
wire WB_IsLINK;

wire [31:0] WB_Wdata;


//====================================================================
//                             IF Stage
//====================================================================
reg [8:0] IF_PC [0:3];
reg [1:0] IF_thread;
wire [31:0] ID_inst;
wire [8:0] PC_for_instmem;

//IF_PC
always@(posedge clk or posedge rst)begin
    if(rst)begin
        IF_PC[0] <= 0;
        IF_PC[1] <= 0;
        IF_PC[2] <= 0;
        IF_PC[3] <= 0;
    end 
    else begin
        IF_PC[IF_thread] <= IF_PC[IF_thread]+1;
        if(EX_branch_taken) IF_PC[EX_thread] <= EX_Target_PC;
    end
end

//IF_thread
always@(posedge clk or posedge rst)begin
    if(rst) IF_thread <= 0;
    else IF_thread <= IF_thread+1;
end

assign PC_for_instmem = IF_PC[IF_thread];


//inst_mem
inst_mem instmem(
    .clka(clk),
    .addra(PC_for_instmem),
    .threada(IF_thread),
    .douta(ID_inst),
    
    .addrb(inst_addr),
    .threadb(host_thread),
    .doutb(inst_data)
);




reg [42:32] IF_ID_meta_reg; 
always@(posedge clk or posedge rst)begin
    if(rst) IF_ID_meta_reg <= 0;
    else begin
        IF_ID_meta_reg [42:41] <= IF_thread;
        IF_ID_meta_reg [40:32] <= IF_PC[IF_thread];
    end
end



//====================================================================
//                             ID Stage
//====================================================================
wire [1:0] ID_r_thread = IF_ID_meta_reg[42:41];

wire [3:0] ID_Rn  = ID_inst[19:16];
wire [3:0] ID_Rd = ID_inst[15:12];
wire [3:0] ID_Rm = (ID_inst[27:26] == 2'b01 && ID_inst[20] == 1'b0) ? ID_inst[15:12] : ID_inst[3:0];

wire [31:0] ID_r0data, ID_r1data;



//regfile
regfile regg(
    .clk(clk),
    .reset(rst),
    
    .wena(WB_RegWrite),
    .waddr(WB_Rd),
    .wdata(WB_Wdata),
    .w_thread(WB_thread),
    
    .r0addr(ID_Rn),
    .r1addr(ID_Rm),
    .r0data(ID_r0data),
    .r1data(ID_r1data),
    .r_thread(ID_r_thread)
);

reg [31:0] ID_imm_ext;
wire [1:0] ID_op_type = ID_inst[27:26];

always@(*)begin
    case(ID_op_type)
        2'b00: begin 
            if (ID_inst[25] == 1'b1) ID_imm_ext = { 24'd0, ID_inst[7:0] };
            else ID_imm_ext = 32'd0;
        end
        
        2'b01: begin 
            ID_imm_ext = { 20'd0, ID_inst[11:0] };
        end
        
        2'b10: begin 
            ID_imm_ext = { {8{ID_inst[23]}}, ID_inst[23:0] };
        end
        
        2'b11: begin
            ID_imm_ext = 32'd0;
        end
    endcase
end


reg ID_RegWrite;
reg ID_MemWrite;
reg ID_MemToReg;
reg ID_LinkWrite;
wire ID_is_BX = (ID_inst[27:4] == 24'h12FFF1);
wire ID_is_Branch = (ID_op_type == 2'b10) || ID_is_BX;
wire ID_is_BL = ID_is_Branch && (ID_inst[24] == 1'b1);


always @(*) begin
    ID_RegWrite = 1'b0;
    ID_MemWrite = 1'b0;
    ID_MemToReg = 1'b0;
    ID_LinkWrite = 1'b0;
    
    case(ID_op_type)
        2'b00: begin 
            if(ID_inst[24:21] == 4'b1010) ID_RegWrite = 1'b0;
            else if(ID_is_BX) ID_RegWrite = 1'b0;
            else ID_RegWrite = 1'b1;
        end

        2'b01: begin 
            if(ID_inst[20] == 1'b1) begin 
                ID_RegWrite = 1'b1;
                ID_MemToReg = 1'b1;
            end 
            else begin 
                ID_MemWrite = 1'b1;
            end
        end

        2'b10: begin 
            if(ID_is_BL)begin
                ID_RegWrite  = 1'b1;
                ID_LinkWrite = 1'b1;
            end
        end
        
        default: begin  end
    endcase
end


//ID_EX_reg
reg [134:0] ID_EX_reg;

always@(posedge clk or posedge rst)begin
    if(rst) ID_EX_reg <= 0;
    else begin 
        ID_EX_reg[31:0] <= ID_r0data;
        ID_EX_reg[63:32] <= ID_r1data;
        ID_EX_reg[95:64] <= ID_imm_ext;
        
        ID_EX_reg[104:96] <= IF_ID_meta_reg[40:32];
        ID_EX_reg[108:105] <= (ID_is_BL) ? 4'd14 : ID_Rd;
        ID_EX_reg[110:109] <= IF_ID_meta_reg[42:41];
        
        ID_EX_reg[118:111] <= ID_inst[11:4];
        ID_EX_reg[122:119] <= ID_inst[24:21];
        ID_EX_reg[126:123] <= ID_inst[31:28];
        ID_EX_reg[127] <= ID_inst[25];
        ID_EX_reg[128] <= (ID_op_type == 2'b00) ? ID_inst[20] : 1'b0;
        
        ID_EX_reg[129] <= ID_RegWrite;
        ID_EX_reg[130] <= ID_MemWrite;
        ID_EX_reg[131] <= ID_MemToReg;
        ID_EX_reg[132] <= ID_LinkWrite;
        ID_EX_reg[133] <= ID_is_Branch;
        ID_EX_reg[134] <= ID_is_BX;
    end
end

//====================================================================
//                             EXE Stage
//====================================================================
wire [31:0] EX_r0data = ID_EX_reg[31:0];
wire [31:0] EX_r1data = ID_EX_reg[63:32];
wire [31:0] EX_imm_ext = ID_EX_reg[95:64];

wire [8:0] EX_PC = ID_EX_reg[104:96];
wire [3:0] EX_Rd = ID_EX_reg[108:105];
assign EX_thread = ID_EX_reg[110:109];

wire [7:0] EX_shift_amt = ID_EX_reg[118:111];
wire [3:0] EX_Opcode = ID_EX_reg[122:119];
wire [3:0] EX_Cond = ID_EX_reg[126:123];
wire  EX_I = ID_EX_reg[127];
wire  EX_S = ID_EX_reg[128];

wire EX_RegWrite = ID_EX_reg[129];
wire EX_MEMWrite = ID_EX_reg[130];
wire EX_MemToReg = ID_EX_reg[131];
wire EX_IsLINK = ID_EX_reg[132];
wire EX_is_Branch = ID_EX_reg[133];
wire EX_is_BX = ID_EX_reg[134];


wire [31:0] EX_ALU_Result;
wire [3:0]  EX_ALU_NZCV;

wire EX_is_MEM = (EX_MemToReg || EX_MEMWrite);
wire EX_use_imm = EX_is_MEM | EX_I;
wire [31:0] EX_ALU_SRC_B = (EX_use_imm)? EX_imm_ext : EX_r1data;
wire [3:0] EX_ALU_Opcode = (EX_is_MEM)? 4'b0100 : EX_Opcode;

wire [1:0] EX_shift_type = EX_shift_amt[2:1];
wire [4:0] EX_shift_val = EX_shift_amt[7:3];

wire [7:0] EX_shift_amt_in_ALU = (EX_I) ? 8'd0 : {3'd0, EX_shift_val};

ALU ex_alu_inst (
    .SRC_A(EX_r0data),
    .SRC_B(EX_ALU_SRC_B),
    .Shift_amt(EX_shift_amt_in_ALU),
    .Shift_type(EX_shift_type),
    .Opcode(EX_ALU_Opcode),
    
    .NZCV(EX_ALU_NZCV),
    .ALU_out(EX_ALU_Result)
);

reg [3:0] EX_NZCV [0:3];

//EX_NZCV
always @(posedge clk or posedge rst) begin
    if (rst) begin
        EX_NZCV[0] <= 4'b0000;
        EX_NZCV[1] <= 4'b0000;
        EX_NZCV[2] <= 4'b0000;
        EX_NZCV[3] <= 4'b0000;
    end 
    else begin
        if (EX_S == 1'b1) begin
            EX_NZCV[EX_thread] <= EX_ALU_NZCV;
        end
    end
end

wire EX_NZCV_N = EX_NZCV[EX_thread][3];
wire EX_NZCV_Z = EX_NZCV[EX_thread][2];
wire EX_NZCV_C = EX_NZCV[EX_thread][1];
wire EX_NZCV_V = EX_NZCV[EX_thread][0];


assign EX_Target_PC = (EX_is_BX)? EX_r1data[8:0] : ($signed(EX_PC) + $signed(EX_imm_ext[8:0]));


reg EX_cond_met;
always@(*)begin
    case (EX_Cond)
        4'b0000: EX_cond_met = EX_NZCV_Z;
        4'b0001: EX_cond_met = ~EX_NZCV_Z;
        4'b1010: EX_cond_met = (EX_NZCV_N == EX_NZCV_V);
        4'b1011: EX_cond_met = (EX_NZCV_N != EX_NZCV_V);
        4'b1100: EX_cond_met = (~EX_NZCV_Z) & (EX_NZCV_N == EX_NZCV_V);
        4'b1101: EX_cond_met = EX_NZCV_Z | (EX_NZCV_N != EX_NZCV_V);
        4'b1110: EX_cond_met = 1'b1;
        default: EX_cond_met = 1'b0;
    endcase
end

assign EX_branch_taken = EX_is_Branch & EX_cond_met;
wire EX_final_MEMWrite = EX_MEMWrite & EX_cond_met;
wire EX_final_RegWrite = EX_RegWrite & EX_cond_met;
wire [8:0] EX_Return_PC = EX_PC + 1'b1;


//EX_MEM_reg
reg [82:0] EX_MEM_reg;
always @(posedge clk or posedge rst) begin
    if (rst) begin
        EX_MEM_reg <= 0;
    end 
    else begin
        EX_MEM_reg[31:0]  <= EX_ALU_Result;
        EX_MEM_reg[63:32] <= EX_r1data;
        EX_MEM_reg[72:64] <= EX_Return_PC;

        EX_MEM_reg[76:73] <= EX_Rd;
        EX_MEM_reg[78:77] <= EX_thread;

        EX_MEM_reg[79]    <= EX_final_MEMWrite; 
        EX_MEM_reg[80]    <= EX_final_RegWrite;
        EX_MEM_reg[81]    <= EX_MemToReg;
        EX_MEM_reg[82]    <= EX_IsLINK;
    end
end

//====================================================================
//                             MEM Stage
//====================================================================
wire [31:0] MEM_ALU_Result = EX_MEM_reg[31:0];
wire [31:0] MEM_r1data = EX_MEM_reg[63:32];
wire [8:0] MEM_Return_PC = EX_MEM_reg[72:64];

wire [3:0] MEM_Rd = EX_MEM_reg[76:73];
wire [1:0] MEM_thread = EX_MEM_reg[78:77];

wire MEM_MEMWrite = EX_MEM_reg[79];
wire MEM_RegWrite = EX_MEM_reg[80];
wire MEM_MemToReg = EX_MEM_reg[81];
wire MEM_IsLINK = EX_MEM_reg[82];

wire [7:0] MEM_Datamem_addr = MEM_ALU_Result[9:2];
wire [31:0] WB_Datamem_rdata_bram;

data_mem datamem(
    .clka(clk),
    .dina(MEM_r1data),
    .addra(MEM_Datamem_addr),
    .wea(MEM_MEMWrite),
    .threada(MEM_thread),
    .douta(WB_Datamem_rdata_bram),
    
    
    .dinb(host_wdata),
    .addrb(host_addr),
    .web(write_to_data_mem),
    .threadb(host_thread),
    .doutb(data_data)
);

reg [81:32] MEM_WB_meta_reg;
always @(posedge clk or posedge rst) begin
    if (rst) begin
        MEM_WB_meta_reg <= 0;
    end else begin
        MEM_WB_meta_reg[63:32] <= MEM_ALU_Result;
        MEM_WB_meta_reg[72:64] <= MEM_Return_PC;
        
        MEM_WB_meta_reg[76:73] <= MEM_Rd;
        MEM_WB_meta_reg[78:77] <= MEM_thread;
        
        MEM_WB_meta_reg[79]    <= MEM_RegWrite;
        MEM_WB_meta_reg[80]    <= MEM_MemToReg;
        MEM_WB_meta_reg[81]    <= MEM_IsLINK;
    end
end



//====================================================================
//                             WB Stage
//====================================================================
assign WB_Datamem_rdata = WB_Datamem_rdata_bram;
assign WB_ALU_Result = MEM_WB_meta_reg[63:32];
assign WB_Return_PC = MEM_WB_meta_reg[72:64];

assign WB_Rd = MEM_WB_meta_reg[76:73];
assign WB_thread = MEM_WB_meta_reg[78:77];

assign WB_RegWrite = MEM_WB_meta_reg[79];
assign WB_MemToReg = MEM_WB_meta_reg[80];
assign WB_IsLINK = MEM_WB_meta_reg[81];

assign WB_Wdata = (WB_IsLINK) ? {23'd0, WB_Return_PC} : (WB_MemToReg) ? WB_Datamem_rdata : WB_ALU_Result;

//====================================================================
//                          Generic Reg
//====================================================================
   generic_regs
   #( 
      .UDP_REG_SRC_WIDTH   (2),
      .TAG                 (`IDS_BLOCK_ADDR),
      .REG_ADDR_WIDTH      (`IDS_REG_ADDR_WIDTH),
      .NUM_COUNTERS        (0),
      .NUM_SOFTWARE_REGS   (5),
      .NUM_HARDWARE_REGS   (1)
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
      .software_regs    ({host_we,host_wdata,host_addr,host_thread,cpu_power}),
        
      // --- HW regs interface
      .hardware_regs    (host_rdata),

      .clk              (clk),
      .reset            (reset)
    );

endmodule