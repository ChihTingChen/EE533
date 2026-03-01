`timescale 1ns / 1ps

module data_mem(
    input               clk, 
    input               mem_read_in,
    input               mem_write_in,
    input       [63:0]  alu_result_in,
    input       [63:0]  rs2_data_in,
    output reg  [63:0]  mem_read_data,
	output reg  [11:0]  addr_out
);
    wire [11:0] bram_addr = alu_result_in[11:0]; 
    reg [63:0] memory [0:4095];
    wire bram_en = mem_read_in | mem_write_in;
    wire bram_we = mem_write_in;
		initial begin: init_data_mem
        integer i;
        for(i = 0; i < 200; i = i + 1) begin
            memory[i] = 64'd0;
        end


        memory[0] = 64'h0004_0003_0002_0001; 
        memory[1] = 64'h0028_001E_0014_000A; 

        memory[10] = 64'h0028_001E_0014_000A;
        // B = {4, 3, 2, 1}
        memory[11] = 64'h0004_0003_0002_0001;

        memory[20] = 64'h3F80_C000_4040_4000;
        memory[21] = 64'h4000_4000_4000_4040;

        memory[30] = 64'h4000_4000_4000_4000;
        memory[31] = 64'h4040_4040_4040_4040;
        memory[32] = 64'h3F80_3F80_3F80_3F80;


        memory[40] = 64'h0008_FFFD_000A_FFFB;
 
    end
    always @(posedge clk) begin
        if (bram_en) begin
            if (bram_we) begin
                memory[bram_addr] <= rs2_data_in;
            end
            mem_read_data <= memory[bram_addr];
        end
    end
	
	always@(posedge clk)begin
		if (bram_en) begin
            addr_out <= bram_addr;
        end
	end
endmodule