module inst_mem(
	input clka,
	input [8:0] addra,
	input [1:0] threada,
	output reg [31:0] douta,
	
	input [8:0] addrb,
	input [1:0] threadb,
	output reg [31:0] doutb 
);

(* ram_style = "block" *)reg [31:0] memory [0:255];
integer i;

wire [5:0] adra = addra[5:0];
wire [5:0] adrb = addrb[5:0];

initial begin
    for (i = 0;    i < 205;  i = i + 1) memory[i] = 32'hE1A00000;
    for (i = 205;  i < 256;  i = i + 1) memory[i] = 32'hE1A00000;
    memory[0] = 32'hE3A01009; // MOV R1, #9
    memory[1] = 32'hE3A02000; // MOV R2, #0
    memory[2] = 32'hE3A03000; // MOV R3, #0
    memory[3] = 32'hE5924000; // LDR R4, [R2]
    memory[4] = 32'hE2825004; // ADD R5, R2, #4
    memory[5] = 32'hE5956000; // LDR R6, [R5]
    memory[6] = 32'hE1540006; // CMP R4, R6
    memory[7] = 32'hDA000003; // BLE +3
    memory[8] = 32'hE5826000; // STR R6, [R2]
    memory[9] = 32'hE5854000; // STR R4, [R5]
    memory[11]= 32'hE2822004; // j += 4
    memory[12]= 32'hE2833001; // count++
    memory[13]= 32'hE1530001; // CMP count, R1
    memory[14]= 32'hBBFFFFF5; // BLT -11
    memory[15]= 32'hE2511001; // SUBS R1, R1, #1
    memory[17]= 32'hCAFFFFF0; // BGT -16
    memory[18]= 32'hEAFFFFFE; // HALT

    // --- Thread 1: Shifter & Logic Test ---
    memory[64] = 32'hE3A01055; // MOV R1, #0x55
    memory[65] = 32'hE1A02101; // MOV R2, R1, LSL #2 
    memory[66] = 32'hE0013002; // AND R3, R1, R2   
    memory[67] = 32'hE5803000; // STR R3, [R0]       
    memory[68] = 32'hEAFFFFFE; // HALT

    // --- Thread 2: Function Call (BL/BX) Test ---
    memory[128] = 32'hE3A01000; // MOV R1, #0       
	memory[129] = 32'hE3A02005; // MOV R2, #5 
	memory[130] = 32'hE2811001; // ADD R1, R1, #1    
	memory[131] = 32'hE1510002; // CMP R1, R2       
	memory[132] = 32'h1AFFFFFE; // BNE -2          
	memory[133] = 32'hE5801000; // STR R1, [R0]    
	memory[134] = 32'hEAFFFFFE;

    // --- Thread 3: Full Conditional Execution Test ---
    memory[192] = 32'hE3A010AA; // MOV R1, #0xAA
    memory[193] = 32'hE3A020AA; // MOV R2, #0xAA
    memory[194] = 32'hE1510002; // CMP R1, R2
    memory[195] = 32'h03A03001; // MOVEQ R3, #1 
    memory[196] = 32'h13A04001; // MOVNE R4, #1
    memory[197] = 32'hE5803008; // STR R3, [R0, #8]
    memory[198] = 32'hEAFFFFFE; // HALT

end
always@(posedge clka)begin
    douta <= memory[{threada, adra}];
    doutb <= memory[{threadb, adrb}];
end





endmodule
