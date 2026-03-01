`timescale 1ns / 1ps

module thread_id_controller (
    input clk,
    input reset,
    // 來自 Control Unit 的訊號：告訴控制器「這一批 4 個算完了，給我下一批號碼」
    input next_thread_group, 
    
    // 輸出一個 64-bit 的向量，裡面包含了 4 個 16-bit 的連續 ID
    output [63:0] tid_vector
);

    // 儲存目前的「基礎號碼 (Base ID)」
    reg [15:0] base_id;

    // 時序邏輯：負責更新號碼牌
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            base_id <= 16'd0; // 系統重置時，從 0 開始
        end 
        else if (next_thread_group) begin
            // 因為我們一次處理 4 個元素，所以下一批的基礎號碼要 +4
            base_id <= base_id + 16'd4; 
        end
    end

    // 組合邏輯：把 1 個基礎號碼，展開成 4 個平行的號碼牌
    // 這樣 64-bit 暫存器裡就會是 [base+3, base+2, base+1, base]
    assign tid_vector[15:0]  = base_id;             // Lane 0 拿到的 ID
    assign tid_vector[31:16] = base_id + 16'd1;     // Lane 1 拿到的 ID
    assign tid_vector[47:32] = base_id + 16'd2;     // Lane 2 拿到的 ID
    assign tid_vector[63:48] = base_id + 16'd3;     // Lane 3 拿到的 ID

endmodule