module inst_mem(
	input clka,
	input [8:0] addra,
	output reg [31:0] douta,
	
	input [8:0] addrb,
	input [31:0] dinb,
	input web,
	output reg [31:0] doutb
);

reg [31:0] memory [0:511];

initial begin
    memory[0] = 32'hFFFAAFFF;  
end

always@(posedge clka)begin
	douta <= memory[addra];
end

always@(posedge clka)begin
	if(web) memory[addrb] <= dinb;
	doutb <= memory[addrb];
end


endmodule