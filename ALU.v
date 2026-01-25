module ALU32 (
    input clk,
    input rst,
    input [2:0] sel,
    input [31:0] A,
    input [31:0] B,
    input Cin,
    output reg [31:0] Sum,
    output reg Cout
);

    reg [32:0] tmp;
    reg [31:0] sum_c;
    reg cout_c;

    always @(*) begin
        tmp  = 33'd0;
        sum_c = 32'd0;
        cout_c= 1'b0;

        case (sel)
            3'b000: begin
                tmp  = {1'b0,A} + {1'b0,B} + Cin;
                sum_c = tmp[31:0];
                cout_c= tmp[32];
            end
            3'b001: begin
                tmp  = {1'b0, A} + {1'b0, (~B)} + 1'b1;
                sum_c = tmp[31:0];
                cout_c= tmp[32];
            end
            3'b010: begin
                sum_c = A << 1;
                cout_c= A[31];
            end
            3'b011: begin
                tmp = {1'b0,A} + {1'b0,(~B)} + 1'b1;
                sum_c = 32'd0;
                cout_c= ~tmp[32]; 
            end
            3'b100: begin
                sum_c = 32'd0;
                cout_c= (A == B);
            end

            default: begin
                sum_c = 32'd0;
                cout_c= 1'b0;
            end
        endcase
    end
    always @(posedge clk) begin
        if (rst) begin
            Sum  <= 32'd0;
            Cout <= 1'b0;
        end else begin
            Sum  <= sum_c;
            Cout <= cout_c;
        end
    end

endmodule
