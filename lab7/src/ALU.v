`timescale 1ns / 1ps

module ALU (
    input [63:0] rs1_data,    
    input [63:0] rs2_data,    
    input [5:0] alu_op,      
    
    output reg [63:0] alu_result,  
    output branch_taken 
);

    localparam ALU_NOP = 6'b000000; 
    localparam ALU_ADD = 6'b001000; 
    localparam ALU_SUB = 6'b001001; 
    localparam ALU_PASS = 6'b000011; 
	localparam BEQ = 6'b110000;
	localparam VLD = 6'b000001;
    localparam VST = 6'b000010;
	
    always @(*) begin
		alu_result = 64'd0;
        case (alu_op)
			VLD, VST: begin
                alu_result = rs1_data; 
            end
            ALU_ADD: begin
                alu_result[15:0]  = rs1_data[15:0]  + rs2_data[15:0]; // Lane 0
                alu_result[31:16] = rs1_data[31:16] + rs2_data[31:16]; // Lane 1
                alu_result[47:32] = rs1_data[47:32] + rs2_data[47:32]; // Lane 2
                alu_result[63:48] = rs1_data[63:48] + rs2_data[63:48]; // Lane 3
            end
            
            ALU_SUB: begin
                alu_result[15:0]  = rs1_data[15:0]  - rs2_data[15:0]; // Lane 0
                alu_result[31:16] = rs1_data[31:16] - rs2_data[31:16]; // Lane 1
                alu_result[47:32] = rs1_data[47:32] - rs2_data[47:32]; // Lane 2
                alu_result[63:48] = rs1_data[63:48] - rs2_data[63:48]; // Lane 3
            end

            ALU_PASS: begin
                alu_result = rs1_data;
            end
            
            default: begin
                alu_result = 64'd0; 
            end
        endcase
    end

    
    assign branch_taken = (alu_op == BEQ)? ((rs1_data == rs2_data) ? 1'b1 : 1'b0) : 1'b0;

endmodule