`timescale 1ns/1ps

module ALU(
    input [31:0] SRC_A, SRC_B,
    input [4:0]  Shift_amt, 
    input [1:0]  Shift_type,
    input [3:0]  Opcode,
    output [3:0] NZCV,
    output reg [31:0] ALU_out
);
    localparam ADD = 4'b0100;
    localparam SUB = 4'b0010;
    localparam MOV = 4'b1101;
    
    reg [32:0] temp_ALU_out;
    
    always @(*) begin
        ALU_out = 32'b0;
        temp_ALU_out = 33'b0;
        case(Opcode)
            ADD: begin
                temp_ALU_out = {1'b0, SRC_A} + {1'b0, SRC_B};
                ALU_out = temp_ALU_out[31:0];
            end
            SUB: begin
                temp_ALU_out = {1'b0, SRC_A} + {1'b0, ~SRC_B} + 1;
                ALU_out = temp_ALU_out[31:0];
            end
            MOV: ALU_out = SRC_B;   
            default: ALU_out = SRC_A + SRC_B; 
        endcase
    end
    
    assign NZCV[3] = ALU_out[31]; 
    assign NZCV[2] = (ALU_out == 32'b0); 
    assign NZCV[1] = temp_ALU_out[32]; 
    assign NZCV[0] = 1'b0; 
endmodule