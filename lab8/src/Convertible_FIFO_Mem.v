`timescale 1ns / 1ps

module Convertible_FIFO_Mem(
    input clk, reset,
    
    input  [63:0] in_data,
    input  [7:0]  in_ctrl,
    input         in_wr,
    output        in_rdy,
    output [63:0] out_data,
    output [7:0]  out_ctrl,
    output        out_wr,
    input         out_rdy,
   
    input         cpu_we,
    input  [7:0]  cpu_addr,
    input  [63:0] cpu_wdata,
    output [71:0] mem_data_out,
    
    output reg FIFO_FULL, 
    output CPU_GO,
	input [4:0] test_addr,
	output [31:0] test_data_up, test_data_down
);
    localparam STATE_IDLE    = 2'd0;
    localparam STATE_HEADER  = 2'd1;
    localparam STATE_PAYLOAD = 2'd2;
    reg [1:0] controller_state, controller_nx_state;
    reg cpu_or_fifo;

    always @(posedge clk) begin
        if(reset) controller_state <= STATE_IDLE;
        else      controller_state <= controller_nx_state;
    end
    
    always @(*) begin
        controller_nx_state = controller_state;
        case(controller_state)
            STATE_IDLE:    controller_nx_state = (in_ctrl == 8'hFF) ? STATE_HEADER : STATE_IDLE;
            STATE_HEADER:  controller_nx_state = (in_ctrl == 8'h00) ? STATE_PAYLOAD : STATE_HEADER;
            STATE_PAYLOAD: controller_nx_state = (in_ctrl != 8'h00 && in_ctrl != 8'hFF) ? STATE_IDLE : ((in_ctrl == 8'hFF) ? STATE_HEADER : STATE_PAYLOAD);
            default:       controller_nx_state = STATE_IDLE;
        endcase
    end
    
    always @(posedge clk) begin
        if(reset) cpu_or_fifo <= 1'b0;
        else      cpu_or_fifo <= FIFO_FULL;
    end
    
	always @(posedge clk) begin
        if(reset) FIFO_FULL <= 1'b0;
        else if (controller_state == STATE_PAYLOAD && controller_nx_state != STATE_PAYLOAD)
            FIFO_FULL <= 1'b1; 
        else if (cpu_or_fifo && cpu_we && cpu_addr == 8'hFF)
            FIFO_FULL <= 1'b0;
    end
    
    assign CPU_GO = FIFO_FULL;
    assign in_rdy = ~FIFO_FULL;
    
    reg [7:0] tail_addr, head_addr;
    always @(posedge clk) begin
        if(reset) tail_addr <= 8'd0;
        else if (in_wr && in_rdy && !cpu_or_fifo) tail_addr <= tail_addr + 1;
    end
    
    wire fifo_empty = (head_addr == tail_addr);
    assign out_wr = (!fifo_empty) && (!cpu_or_fifo);
    always @(posedge clk) begin
        if(reset) head_addr <= 8'd0;
        else if(out_wr && out_rdy) head_addr <= head_addr + 1;
    end
    

    wire [7:0] next_head_addr = (out_wr && out_rdy) ? (head_addr + 1) : head_addr;
    

    wire wen = cpu_or_fifo ? cpu_we : (in_wr & in_rdy);
    wire [7:0]  addrA = cpu_or_fifo ? cpu_addr : tail_addr;

    wire [7:0]  addrB = cpu_or_fifo ? cpu_addr : next_head_addr;
    wire [71:0] dataA = cpu_or_fifo ? {8'b0, cpu_wdata} : {in_ctrl, in_data};
    
    Dual_Port_SRAM sram(
        .clk(clk), .reset(reset), .wen(wen),
        .addrA(addrA), .addrB(addrB), .dataA(dataA), .data_outB(mem_data_out)
    );
    assign out_data = mem_data_out[63:0];
    assign out_ctrl = mem_data_out[71:64];
	
	
	reg [4:0] test_cnt;
    always @(posedge clk) begin
        if(reset) 
            test_cnt <= 0;
        else if(in_wr && in_rdy && !cpu_or_fifo) 
            test_cnt <= test_cnt + 1;
    end
    

    reg [63:0] test_mem [0:31];
    
    reg [63:0] test_mem_read_data; 

    always @(posedge clk) begin
        if(in_wr && in_rdy && !cpu_or_fifo) begin
            test_mem[test_cnt] <= mem_data_out[63:0];
        end
        test_mem_read_data <= test_mem[test_addr];
    end

    assign test_data_up   = test_mem_read_data[63:32];
    assign test_data_down = test_mem_read_data[31:0]; 

endmodule

module Dual_Port_SRAM(
    input clk, reset, wen,
    input [7:0] addrA, addrB,
    input [71:0] dataA,
    output reg [71:0]
);
    reg [71:0] memory [0:255];
    integer i;
    initial begin
        for(i=0; i<256; i=i+1) memory[i] = 72'd0;
        /*memory[10] = {8'h00, 64'h4000_4000_4000_4000}; // A = 2.0
        memory[11] = {8'h00, 64'h4040_4040_4040_4040}; // B = 3.0
        memory[12] = {8'h00, 64'h3F80_3F80_3F80_3F80}; // C = 1.0
		*/
    end
    always @(posedge clk) begin
        if(wen) begin
            memory[addrA] <= dataA;
        end
        if (wen && (addrA == addrB)) begin
            data_outB <= dataA;
        end else begin
            data_outB <= memory[addrB];
        end
    end
endmodule