%% Artificial Intelligence Homework#1 Demo - 2020/04/28
clear
clc
%% 資料整理
load('iris.txt')
featureSel = [3 4];
cP = iris(iris(:,5)==2,[3,4]);
cN = iris(iris(:,5)==3,[3,4]);
trnP = cP( 1:25,:);
trnN = cN( 1:25,:);
tstP = cP(26:50,:);
tstN = cN(26:50,:);
trnSet = [trnP;trnN];
tstSet = [tstP;tstN];
trnY   = [ones(1,25),ones(1,25)*-1]';
tstY   = trnY;
%% 參數設定
ker = 'rbf';
C = 10;
S = 0.1;
%% Quadratic Programming
L = length(trnY);
for a = 1:L
    u = trnSet(a,:);
    for b = 1:a
        v = trnSet(b,:);
        H(a,b) = trnY(a)*trnY(b)*exp(-(u-v)*(u-v)'/(2*S^2));
        H(b,a) = H(a,b);
    end
end
f   =  -ones(L,1) ;
Aeq =    trnY'    ;
beq =      0      ;
lb  =  zeros(L,1) ;
ub  = C*ones(L,1) ;
alpha = quadprog(H,f,[],[],Aeq,beq,lb,ub);
alpha(alpha<   sqrt(eps) ) = 0;
alpha(alpha>(C-sqrt(eps))) = C;
%% Support Vector Machine
%=========================================================================%
%   接續二次規劃所求得之alpha計算bias，即可得到SVM之模型
%   此階段程式碼請自行實現
%=========================================================================%
%% Scatter plot and Hyperplane
[xx,yy] = meshgrid(linspace(min([cP(:,1);cN(:,1)]),max([cP(:,1);cN(:,1)]),201),...
    linspace(min([cP(:,2);cN(:,2)]),max([cP(:,2);cN(:,2)]),201));
xy = [xx(:),yy(:)];
%=========================================================================%
% 讀取demo用的SVM結果，此結果為上述xy帶入SVM模型後判別的decision results
% 該結果採用rbf-based SVM，C = 10，sigma = 0.1
% 此階段程式碼請自行實現
load('demoDecisionResult_rbf_C10_S1E-1')
%=========================================================================%
colorClass = D*-0.5+1.5;
Hyperplane = reshape(colorClass,size(xx));
figure(1)
clf
image(xx(1,:),yy(:,1),Hyperplane)
colormap([1,.4,.4;.4,.4,1]);
set(gca,'YDir','normal');
title(['Panelty weight = ',num2str(C),', kernal parameter (sigma/p) = ',num2str(S)])
hold on
plot(trnP(:,1),trnP(:,2),'ks','markerface','r','LineWidth',1,'MarkerSize',10)
plot(trnN(:,1),trnN(:,2),'ks','markerface','b','LineWidth',1,'MarkerSize',10)
plot(tstP(:,1),tstP(:,2),'yo','markerface','r','LineWidth',1,'MarkerSize',5)
plot(tstN(:,1),tstN(:,2),'yo','markerface','b','LineWidth',1,'MarkerSize',5)
legend('Class2 - Training set','Class3 - Training set',...
    'Class2 - Test set','Class3 - Test set','Location','southeast')








