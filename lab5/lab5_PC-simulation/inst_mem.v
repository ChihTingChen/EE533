module inst_mem(
	input clka,
	input [8:0] addra,
	output reg [31:0] douta
);

reg [31:0] memory [0:511];

integer i;

initial begin
        
        for (i = 0; i < 512; i = i + 1) begin
            memory[i] = 32'h00000000;
        end

        $readmemh("inst.txt", memory);
    end

always@(posedge clka)begin
	douta <= memory[addra];
end


endmodule