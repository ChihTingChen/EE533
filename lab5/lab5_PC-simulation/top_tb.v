`timescale 1ns / 1ps

module top_tb;

    
    reg clk;
    reg reset;

    
    
    top uut (
        .clk(clk),
        .reset(reset)
    );

    
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    
    initial begin
        
        $timeformat(-9, 0, " ns", 6); 

        
        $display("----------------------------------------------------------------------");
        $display("Time      | PC  | Inst     | WB_En | WB_Dest | WB_Data          | Note");
        $display("----------------------------------------------------------------------");

        
        reset = 1;
        #20; 
        reset = 0;
        $display("%t | Reset Released... CPU Start!", $time);

        
        
        
        #200;

        
        $display("----------------------------------------------------------------------");
        $display("Simulation Finished.");
        $finish;
    end

    
    
    
    always @(negedge clk) begin
        if (!reset && uut.WB_WRE) begin
            $display("%t | %3d | %h |   1   |   R%0d   | %16h | Write Back Occurred!", 
                     $time, 
                     uut.PC,          
                     uut.instruction, 
                     uut.WB_Waddr,    
                     uut.WB_Wdata     
            );
        end
    end

    always @(negedge clk) begin
        if (!reset && uut.EX_MEM_WME) begin 
             $display("%t | %3d | %h |  MEM  | Mem[%0d] | %16h | Store to Memory!", 
                     $time, 
                     uut.PC,
                     uut.instruction,
                     uut.EX_MEM_addr[7:0], 
                     uut.EX_MEM_data       
            );
        end
    end

endmodule