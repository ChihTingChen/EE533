`timescale 1ns / 1ps

module tb_arm_tensor();

    // --- Test Signal Declarations ---
    reg clk;
    reg rst;
    
    // NetFPGA Input (Network Interface Receiver)
    reg [63:0] net_in_data;
    reg [7:0]  net_in_ctrl;
    reg        net_in_wr;
    wire       net_in_rdy;
    
    // NetFPGA Output (Network Interface Transmitter)
    wire [63:0] net_out_data;
    wire [7:0]  net_out_ctrl;
    wire        net_out_wr;
    reg         net_out_rdy;

    // --- Instantiate Device Under Test (DUT) ---
    arm_tensor_cpu dut (
        .clk(clk),
        .rst(rst),
        .net_in_data(net_in_data),
        .net_in_ctrl(net_in_ctrl),
        .net_in_wr(net_in_wr),
        .net_in_rdy(net_in_rdy),
        .net_out_data(net_out_data),
        .net_out_ctrl(net_out_ctrl),
        .net_out_wr(net_out_wr),
        .net_out_rdy(net_out_rdy)
    );

    // --- Clock Generation (100MHz = 10ns period) ---
    always #5 clk = ~clk;

    // --- Main Simulation Flow ---
    initial begin
        // 1. System Initialization
        clk = 0; rst = 1;
        net_in_data = 0; net_in_ctrl = 0; net_in_wr = 0; net_out_rdy = 0;
        
        $display("==================================================");
        $display("[Time: %0t] System Reset...", $time);
        #20 rst = 0;
        #10;

        // 2. Simulate network packet flowing into FIFO
        $display("[Time: %0t] >>> Start injecting network packet (Push to FIFO)...", $time);
        
        // Cycle 1: Write Header (Ctrl = 0xFF)
        @(posedge clk);
        net_in_wr = 1; net_in_ctrl = 8'hFF; net_in_data = 64'hAAAA_BBBB_CCCC_DDDD;
        
        // Cycle 2: Write Payload A = 2.0 (Ctrl = 0x00)
        @(posedge clk);
        net_in_ctrl = 8'h00; net_in_data = 64'h4000_4000_4000_4000; 
        
        // Cycle 3: Write Payload B = 3.0 (Ctrl = 0x00)
        @(posedge clk);
        net_in_ctrl = 8'h00; net_in_data = 64'h4040_4040_4040_4040; 
        
        // Cycle 4: Write Payload C = 1.0 (Ctrl = 0x00)
        @(posedge clk);
        net_in_ctrl = 8'h00; net_in_data = 64'h3F80_3F80_3F80_3F80; 
        
        // Cycle 5: Reserve an empty slot for GPU to write back result D
        @(posedge clk);
        net_in_ctrl = 8'h00; net_in_data = 64'h0000_0000_0000_0000; 
        
        // Cycle 6: End of packet (Send non-00/FF Ctrl signal to trigger IDLE)
        @(posedge clk);
        net_in_wr = 0; net_in_ctrl = 8'h01; 
        $display("[Time: %0t] <<< Packet injection complete, FIFO locked, CPU started!", $time);

        // 3. Wait for CPU and GPU (Tensor Core) pipeline execution
        // Takes about 15~20 cycles for the pipeline to fetch, decode, compute FMA, and write back to memory
        #250; 
        
        $display("[Time: %0t] CPU should have finished computing and released FIFO control", $time);

        // 4. Simulate network side receiving the processed packet
        $display("[Time: %0t] >>> Start reading the processed network packet from FIFO...", $time);
        net_out_rdy = 1; // Tell FIFO the network side is ready to receive
        
        // Monitor the output packet data
        repeat(6) begin
            @(posedge clk);
            if (net_out_wr) begin
                if (net_out_data == 64'h40E0_40E0_40E0_40E0)
                    $display("    [SUCCESS] Received Tensor accelerated result: %h (Value: 7.0)", net_out_data);
                else if (net_out_data == 64'hAAAA_BBBB_CCCC_DDDD)
                    $display("    [RECEIVED] Packet Header: %h", net_out_data);
                else
                    $display("    [RECEIVED] Packet Payload: %h", net_out_data);
            end
        end

        net_out_rdy = 0;
        $display("==================================================");
        $display("[Time: %0t] Simulation Test Ended!", $time);
        $finish;
    end

endmodule