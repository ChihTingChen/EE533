module regfile_64b(
    input clk, reset, we,
    input [4:0] waddr, raddr1, raddr2, raddr3,
    input [63:0] wdata,
    output [63:0] rdata1, rdata2, rdata3
);
    reg [63:0] memory [0:31]; 
    integer i;
    initial begin
        for(i=0; i<32; i=i+1) memory[i] = 64'd0;
    end
    always @(posedge clk) begin
        if(we) memory[waddr] <= wdata;
    end
    assign rdata1 = memory[raddr1];
    assign rdata2 = memory[raddr2];
    assign rdata3 = memory[raddr3];
endmodule