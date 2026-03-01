`timescale 1ns / 1ps

module tensor_unit (
    input [63:0] rs1_data,
    input [63:0] rs2_data,
    input [63:0] rs3_data,
    input [5:0]  tu_op,   
    
    output reg  [63:0] tu_result 
);

    
    localparam TU_MUL  = 6'b010000;
    localparam TU_FMA  = 6'b010001;
    localparam TU_RELU = 6'b011000;

    wire [15:0] fma_out_0, fma_out_1, fma_out_2, fma_out_3;

    wire [63:0] fma_c_input = (tu_op == TU_FMA) ? rs3_data : 64'h0000_0000_0000_0000;

    bf16_fma_lane lane0 (.a(rs1_data[15:0]),  .b(rs2_data[15:0]),  .c(fma_c_input[15:0]),  .out(fma_out_0));
    bf16_fma_lane lane1 (.a(rs1_data[31:16]), .b(rs2_data[31:16]), .c(fma_c_input[31:16]), .out(fma_out_1));
    bf16_fma_lane lane2 (.a(rs1_data[47:32]), .b(rs2_data[47:32]), .c(fma_c_input[47:32]), .out(fma_out_2));
    bf16_fma_lane lane3 (.a(rs1_data[63:48]), .b(rs2_data[63:48]), .c(fma_c_input[63:48]), .out(fma_out_3));

    always @(*) begin
        case (tu_op)
            TU_MUL, TU_FMA: begin
                tu_result[15:0]  = fma_out_0;
                tu_result[31:16] = fma_out_1;
                tu_result[47:32] = fma_out_2;
                tu_result[63:48] = fma_out_3;
            end
            
            TU_RELU: begin
                tu_result[15:0]  = (rs1_data[15] == 1'b1)  ? 16'd0 : rs1_data[15:0];
                tu_result[31:16] = (rs1_data[31] == 1'b1)  ? 16'd0 : rs1_data[31:16];
                tu_result[47:32] = (rs1_data[47] == 1'b1)  ? 16'd0 : rs1_data[47:32];
                tu_result[63:48] = (rs1_data[63] == 1'b1)  ? 16'd0 : rs1_data[63:48];
            end
            
            default: tu_result = 64'd0;
        endcase
    end
endmodule


`timescale 1ns / 1ps

module bf16_fma_lane (
    input  wire [15:0] a,
    input  wire [15:0] b,
    input  wire [15:0] c,
    output reg  [15:0] out
);


    wire sign_a = a[15];
    wire sign_b = b[15];
    wire sign_c = c[15];
    
    wire [7:0] exp_a = a[14:7];
    wire [7:0] exp_b = b[14:7];
    wire [7:0] exp_c = c[14:7];
    
    
    wire [7:0] mant_a = (exp_a == 0) ? 8'd0 : {1'b1, a[6:0]};
    wire [7:0] mant_b = (exp_b == 0) ? 8'd0 : {1'b1, b[6:0]};
    wire [14:0] mant_c = (exp_c == 0) ? 15'd0 : {1'b1, c[6:0], 7'd0}; 

    reg        sign_mul;
    reg [8:0]  exp_mul_temp;
    reg [7:0]  exp_mul_norm;
    reg [15:0] mant_mul_full;
    reg [14:0] mant_mul_norm;
    
    reg [7:0]  target_exp;
    reg [14:0] shifted_mul;
    reg [14:0] shifted_c;
    reg [7:0]  exp_diff;
    
    reg        final_sign;
    reg [15:0] mant_sum; 
    
    reg [7:0]  final_exp;
    reg [6:0]  final_mant;
    
    integer i;
    reg [4:0] shift_amt;

    always @(*) begin

        sign_mul = sign_a ^ sign_b;
        
        exp_mul_temp = exp_a + exp_b - 8'd127;
        
        mant_mul_full = mant_a * mant_b; 
        
        if (mant_mul_full[15] == 1'b1) begin
            mant_mul_norm = mant_mul_full[15:1];
            exp_mul_norm  = exp_mul_temp[7:0] + 1'b1;
        end else begin
            mant_mul_norm = mant_mul_full[14:0];
            exp_mul_norm  = exp_mul_temp[7:0];
        end
        

        if (exp_mul_norm > exp_c) begin
            exp_diff   = exp_mul_norm - exp_c;
            shifted_c  = (exp_diff > 15) ? 15'd0 : (mant_c >> exp_diff);
            shifted_mul = mant_mul_norm;
            target_exp = exp_mul_norm;
        end 
		else begin
            exp_diff   = exp_c - exp_mul_norm;
            shifted_mul = (exp_diff > 15) ? 15'd0 : (mant_mul_norm >> exp_diff);
            shifted_c  = mant_c;
            target_exp = exp_c;
        end


        if (sign_mul == sign_c) begin
            mant_sum   = shifted_mul + shifted_c;
            final_sign = sign_mul;
        end 
		else begin
            if (shifted_mul >= shifted_c) begin
                mant_sum   = shifted_mul - shifted_c;
                final_sign = sign_mul;
            end 
			else begin
                mant_sum   = shifted_c - shifted_mul;
                final_sign = sign_c;
            end
        end


        if (mant_sum == 16'd0) begin
            final_exp  = 8'd0;
            final_mant = 7'd0;
            final_sign = 1'b0;
        end 
		else if (mant_sum[15] == 1'b1) begin
            final_exp  = target_exp + 1'b1;
            final_mant = mant_sum[14:8];
        end 
		else begin
            shift_amt = 0;
            if (mant_sum[14]) shift_amt = 0;
            else if (mant_sum[13]) shift_amt = 1;
            else if (mant_sum[12]) shift_amt = 2;
            else if (mant_sum[11]) shift_amt = 3;
            else if (mant_sum[10]) shift_amt = 4;
            else if (mant_sum[9])  shift_amt = 5;
            else if (mant_sum[8])  shift_amt = 6;
            else if (mant_sum[7])  shift_amt = 7;
            else if (mant_sum[6])  shift_amt = 8;
            else if (mant_sum[5])  shift_amt = 9;
            else if (mant_sum[4])  shift_amt = 10;
            else if (mant_sum[3])  shift_amt = 11;
            else if (mant_sum[2])  shift_amt = 12;
            else if (mant_sum[1])  shift_amt = 13;
            else if (mant_sum[0])  shift_amt = 14;

            final_exp = target_exp - shift_amt;
            
            case (shift_amt)
                0:  final_mant = mant_sum[13:7];
                1:  final_mant = mant_sum[12:6];
                2:  final_mant = mant_sum[11:5];
                3:  final_mant = mant_sum[10:4];
                4:  final_mant = mant_sum[9:3];
                5:  final_mant = mant_sum[8:2];
                6:  final_mant = mant_sum[7:1];
                7:  final_mant = mant_sum[6:0];
                8:  final_mant = {mant_sum[5:0], 1'b0};
                9:  final_mant = {mant_sum[4:0], 2'b0};
                10: final_mant = {mant_sum[3:0], 3'b0};
                11: final_mant = {mant_sum[2:0], 4'b0};
                12: final_mant = {mant_sum[1:0], 5'b0};
                13: final_mant = {mant_sum[0],   6'b0};
                default: final_mant = 7'd0;
            endcase
        end

        out = {final_sign, final_exp, final_mant};
    end

endmodule
