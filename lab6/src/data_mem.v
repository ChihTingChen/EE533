module data_mem(
    input clka,
	input [31:0] dina,
	input [7:0] addra,
	input wea,
	input [1:0] threada,
	output reg [31:0] douta,
	
	input [31:0] dinb,
	input [7:0] addrb,
	input web,
	input [1:0] threadb,
	output reg [31:0] doutb
);

wire [5:0] addra_256 = addra[5:0];
wire [5:0] addrb_256 = addrb[5:0];


(* ram_style = "block" *)reg [31:0] memory [0:255];

integer i;
initial begin
    for (i = 0; i < 205; i = i + 1) begin
        memory[i] = 32'd0;
    end
    for (i = 205; i < 256; i = i + 1) begin
        memory[i] = 32'd0;
    end
    
	memory[0] = 32'h00000143;
	memory[1] = 32'h0000007b;
	memory[2] = 32'hfffffe39;
	memory[3] = 32'h00000002;
	memory[4] = 32'h00000062;
	memory[5] = 32'h0000007d;
	memory[6] = 32'h0000000a;
	memory[7] = 32'h00000041;
	memory[8] = 32'hffffffc8;
	memory[9] = 32'h00000000;
	
end

always @(posedge clka) begin
    if(wea) memory[{threada, addra_256}] <= dina;
    douta <= memory[{threada, addra_256}];
end
always @(posedge clka) begin
    if(web) memory[{threadb, addrb_256}] <= dinb;
    doutb <= memory[{threadb, addrb_256}];
end


endmodule
