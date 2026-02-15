module alu(
input [63:0] A, B,
input [2:0] aluctrl,
output reg [63:0] Z,
output reg overflow
);

parameter Add = 'd0, Sub = 'd1, BwAND = 'd2, BwOR = 'd3, BwXOR = 'd4, Equal = 'd5, Shift = 'd6; 

//Z
always@(*)begin
	case(aluctrl)
		Add: Z = A+B;
		
		Sub: Z = A-B;
		
		BwAND: Z = A&B;
		
		BwOR: Z = A|B;
		
		BwXOR: Z = A^B;
		 
		Equal: Z = (A==B)? 64'd1 : 64'd0;
		 
		Shift: Z = A>>B;
		 
		default: Z = 64'd0;		
		 
	endcase
end

//overflow
always@(*)begin
	case(aluctrl)
		Add: overflow = (A[63] == B[63]) && (Z[63] != A[63]);
		
		Sub: overflow = (A[63] != B[63]) && (Z[63] != A[63]);
		
		default: overflow = 1'b0;
		
	endcase
end


endmodule