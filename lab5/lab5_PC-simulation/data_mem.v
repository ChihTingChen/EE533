module data_mem(
    input clka,
	input [63:0] dina,
	input [7:0] addra,
	input wea,
	output reg [63:0] douta
);

reg [63:0] memory [0:255];

        integer i;


initial begin
        for (i = 0; i < 256; i = i + 1) begin
            memory[i] = 64'h0000000000000000;
        end
        $readmemh("data.txt", memory);
    end

always@(posedge clka)begin
	douta <= memory[addra];
	if(wea) memory[addra] <= dina;
end

endmodule