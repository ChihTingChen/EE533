module data_mem(
    input clka,
	input [63:0] dina,
	input [7:0] addra,
	input wea,
	output reg [63:0] douta,
	
	input [63:0] dinb,
	input [7:0] addrb,
	input web,
	output reg [63:0] doutb
);

reg [63:0] memory [0:255];

initial begin
    memory[0] = {32'h00000000,32'hAAAFFAAA};  
end

//a
always@(posedge clka)begin
	douta <= memory[addra];
	if(wea) memory[addra] <= dina;
end


//b
always@(posedge clka)begin
	doutb <= memory[addrb];
	if(web) memory[addrb] <= dinb;
end

endmodule