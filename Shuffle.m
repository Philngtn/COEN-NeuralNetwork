% Original matrix
C1 = [C1_Target;C1_Out];
C2 = [C2_Target;C2_Out];

C_Temp = [C1 C2];
C_Temp =  C_Temp(:,randperm(end));

C_in = C_Temp(1:2,:);
C_out = C_Temp(3:4,:);
