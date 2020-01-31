close all; clc; clear; 

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%C1
% Data
n_C1 = 1000;
n1_C1 = 300;

radius_C1 = 2;
xc1 = 0;
yc1 = -1;
R_C1 = [0.25 1];

% Engine
theta_C1 = rand(1,n_C1)*(pi/2);
r_C1 = sqrt(rand(1,n_C1)*range(R_C1)+min(R_C1))*radius_C1;
x_C1 = xc1 + r_C1.*cos(theta_C1);
y_C1 = yc1 + r_C1.*sin(theta_C1);

x1_C1 = xc1 + r_C1.*cos(theta_C1);
y1_C1 = yc1 - r_C1.*sin(theta_C1);

%Small circle C1
theta_s_C1 = rand(1,n1_C1)*(pi/2);
r_s_C1 = sqrt(rand(1,n1_C1));

x2_C1 = -r_s_C1.*cos(theta_s_C1);
y2_C1 = -r_s_C1.*sin(theta_s_C1);

x3_C1 = -r_s_C1.*cos(theta_s_C1);
y3_C1 =  r_s_C1.*sin(theta_s_C1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot C1
X1 = [x_C1 x1_C1];
Y1 = [y_C1 y1_C1];
X2 = [x2_C1 x3_C1];
Y2 = [y2_C1 y3_C1];
X_Data_C1 = [X1 X2];
Y_Data_C1 = [Y1 Y2];


C1_Target = [X_Data_C1;Y_Data_C1];

C1_Out_1 = ones(1,2600);
C1_Out_2 = zeros(1,2600);

C1_Out = C1_Out_1;

plot(X_Data_C1,Y_Data_C1,'.')

axis equal
hold on
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%C2
% Data
n_C2 = 1000;
n1_C2 = 300;

radius_C2 = 2;
xc2 = 0;
yc2 = 0;
R_C2 = [0.25 1];

% Engine
theta_C2 = rand(1,n_C2)*(pi/2);
r_C2 = sqrt(rand(1,n_C2)*range(R_C2)+min(R_C2))*radius_C2;

x_C2 = xc2 - r_C2.*cos(theta_C2);
y_C2 = yc2 + r_C2.*sin(theta_C2);

x1_C2 = xc2 - r_C2.*cos(theta_C2);
y1_C2 = yc2 - r_C2.*sin(theta_C2);

%Small circle
x_s_c2 = 0;
y_s_c2 = -1;
theta_s_C2 = rand(1,n1_C2)*(pi/2);
r_s_C2 = sqrt(rand(1,n1_C2));

x2_C2 = x_s_c2 + r_s_C2.*cos(theta_s_C2);
y2_C2 = y_s_c2 - r_s_C2.*sin(theta_s_C2);

x3_C2 = x_s_c2 + r_s_C2.*cos(theta_s_C2);
y3_C2 = y_s_c2 + r_s_C2.*sin(theta_s_C2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X1_C2 = [x_C2 x1_C2];
Y1_C2 = [y_C2 y1_C2];
X2_C2 = [x2_C2 x3_C2];
Y2_C2 = [y2_C2 y3_C2];

X_Data_C2 = [X1_C2 X2_C2];
Y_Data_C2 = [Y1_C2 Y2_C2];

C2_Target = [X_Data_C2;Y_Data_C2];

C2_Out_1 = ones(1,2600);
C2_Out_2 = zeros(1,2600);

C2_Out = C2_Out_2;


plot(X_Data_C2,Y_Data_C2,'.')

axis equal
% % Check
% plot(X,Y,'.')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C1 = [C1_Target;C1_Out];
C2 = [C2_Target;C2_Out];

C_Temp = [C1 C2];
C_Temp =  C_Temp(:,randperm(end));

C_input = C_Temp(1:2,:);
C_output = C_Temp(3,:);
