module regfile(
	input clk,
	input reset,
	input wena,
	input [3:0] waddr,
	input [31:0] wdata,
	input [3:0] r0addr,
	input [3:0] r1addr,
	input [1:0] r_thread,
	input [1:0] w_thread,
	output [31:0] r0data,
	output [31:0] r1data
);

reg [31:0] register [0:63];

integer i;
initial begin
    for (i = 0; i < 64; i = i + 1) begin
        register[i] = 32'd0;
    end
end

always@(posedge clk) begin
	if(reset)begin end
    else if(wena && waddr != 4'd0) register[{w_thread, waddr}] <= wdata;
end



assign r0data = (r0addr == 4'd0) ? 32'd0 : register[{r_thread, r0addr}];
assign r1data = (r1addr == 4'd0) ? 32'd0 : register[{r_thread, r1addr}];

endmodule

