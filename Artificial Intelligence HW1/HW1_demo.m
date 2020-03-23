%% Artificial Intelligence Homework#1 Demo - 2020/03/17

%%
clc;       % 清除command window
clear      % 清除workspace
close all  % 關閉所有figure

%% 讀取.txt資料
dataSet = load('iris.txt');
rawData = dataSet(:,1:4);    % 原始資料，75筆資料 x 4個特徵
label   = dataSet(:,5);      % 75筆資料所對應的標籤

%% 範例一、Scatter Plot
figure; % 開啟新的繪圖空間

plot(rawData(  1: 50,1),rawData(  1: 50,2),'ro',...
     rawData( 51:100,1),rawData( 51:100,2),'go',...
     rawData(101:150,1),rawData(101:150,2),'bo');   
     % 以plot繪圖指令分別畫出class1~3之第一與第二特徵。

title('Scatter Plot');                              % 圖名稱
legend('class1', 'class2', 'class3');               % 類別標號說明
xlabel('Feature1');                                 % 特徵標號註解
ylabel('Feature2');                                 % 特徵標號註解

%% 範例二、計算距離
trnSet = [rawData(  1: 25,1:4);...
          rawData( 51: 75,1:4);...
          rawData(101:125,1:4);]; 
          % 選取每類別前半，合併為training set

tstSet = [rawData( 26: 50,1:4);...
          rawData( 76:100,1:4);...
          rawData(126:150,1:4)]; 
          % 選取每類別後半，合併為test set

distance = norm(tstSet(1,:)-trnSet(1,:)); 
% norm(u-v): 求出向量u與向量v之間的歐式距離(Euclidean distance)，也稱為2-norm
% 第一筆test資料與第一筆training資料之距離

%% 範例三、k-NN選用指令參考
%%% === sort: 排序 =====================================================%%%
A = [9 0 -7 5 3 8 -10 4 2];
[value,index] = sort(A,'ascend'); 
% 將陣列A，由小到大重新排序
% value: 由小到大重新排序的值
% index: value中每個值在A的原始位置
%%% ====================================================================%%%

%%% === mode: 眾數 =====================================================%%%
B = [3 1 3 3 2];
M = mode(B); 
% 找出陣列B中最頻繁出現的數值
%%% ====================================================================%%%









