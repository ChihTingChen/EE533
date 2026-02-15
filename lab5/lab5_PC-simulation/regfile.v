module regfile(
	input clk,
	input reset,
	input wena,
	input [2:0] waddr,
	input [63:0] wdata,
	input [2:0] r0addr,
	input [2:0] r1addr,
	output [63:0] r0data,
	output [63:0] r1data
);

reg [63:0] register [0:7];

//register
always@(posedge clk or posedge reset)begin
	if(reset)begin
		register[0] <= 0;
		register[1] <= 0;
		register[2] <= 0;
		register[3] <= 0;
		register[4] <= 0;
		register[5] <= 0;
		register[6] <= 0;
		register[7] <= 0;
	end
	else if(wena) register[waddr] <= wdata;
end

//r0data
assign r0data = register[r0addr];

//r1data
assign r1data = register[r1addr];

endmodule

