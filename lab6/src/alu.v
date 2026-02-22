`timescale 1ns/1ps
module ALU(
	input [31:0] SRC_A, SRC_B,
	input [7:0] Shift_amt,
	input [1:0] Shift_type,
	input [3:0] Opcode,
	output [3:0] NZCV, 
	output reg [31:0] ALU_out
);
	parameter AND = 4'b0000;
	parameter ADD = 4'b0100;
	parameter SUB = 4'b0010;
	parameter CMP = 4'b1010;
	parameter ORR = 4'b1100;
	parameter MOV = 4'b1101;
	wire add_ov,sub_ov;
	reg [32:0] temp_ALU_out;
	reg [31:0] SRC_B_shift;
	
	always @(*) begin
		case(Shift_type) 
			2'b00: SRC_B_shift = SRC_B << Shift_amt;  // LSL
			2'b01: SRC_B_shift = SRC_B >> Shift_amt;  // LSR
			/*2'b10: SRC_B_shift = $signed(SRC_B) >>> Shift_amt; // ASR
			2'b11: SRC_B_shift = (Shift_amt == 0)? (SRC_B) : (SRC_B >> Shift_amt) | (SRC_B << (32-Shift_amt));*/ // ROR
			default: SRC_B_shift = SRC_B;
		endcase
	end
	
	always@(*)begin
		ALU_out = 32'b0;
		temp_ALU_out = 33'b0;
		case(Opcode)
			ADD:begin
				temp_ALU_out = {1'b0,SRC_A} + {1'b0,SRC_B_shift};
				ALU_out = temp_ALU_out[31:0];
			end
			
			SUB, CMP:begin
				temp_ALU_out = {1'b0,SRC_A} + {1'b0,~SRC_B_shift} + 1;
				ALU_out = temp_ALU_out[31:0];
			end
			
			MOV:begin
				temp_ALU_out = {1'b0,SRC_B_shift};
				ALU_out = SRC_B_shift;				
			end
			
			AND:begin
				ALU_out = SRC_A & SRC_B_shift;
				temp_ALU_out = {1'b0, ALU_out};
			end
			
			ORR:begin
				ALU_out = SRC_A | SRC_B_shift;
				temp_ALU_out = {1'b0, ALU_out};
			end
			
			default:begin
				ALU_out = 32'b0;
				temp_ALU_out = 33'b0;
			end
		endcase	
		
	end
		assign NZCV[3] = ALU_out[31];
		assign NZCV[2] = (ALU_out == 32'b0);
		assign NZCV[1] = (Opcode == MOV || Opcode == AND || Opcode == ORR) ? 0 : temp_ALU_out[32];
		assign add_ov = (SRC_A[31]==SRC_B_shift[31])&&(temp_ALU_out[31]!=SRC_A[31]);
		assign sub_ov = (SRC_A[31]!=SRC_B_shift[31])&&(temp_ALU_out[31]!=SRC_A[31]);
		assign NZCV[0] = (Opcode == ADD) ? add_ov : (Opcode == SUB||Opcode == CMP) ? sub_ov : 1'b0;
endmodule
