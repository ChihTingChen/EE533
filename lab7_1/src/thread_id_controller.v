`timescale 1ns / 1ps

module thread_id_controller (
    input clk,
    input reset,
    input next_thread_group, 
    
    output [63:0] tid_vector
);

    reg [15:0] base_id;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            base_id <= 16'd0; 
        end 
        else if (next_thread_group) begin
            base_id <= base_id + 16'd4; 
        end
    end

    assign tid_vector[15:0]  = base_id;         
    assign tid_vector[31:16] = base_id + 16'd1; 
    assign tid_vector[47:32] = base_id + 16'd2; 
    assign tid_vector[63:48] = base_id + 16'd3; 

endmodule