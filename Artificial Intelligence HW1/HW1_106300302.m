clc;       % 清除command window
clear      % 清除workspace
close all  % 關閉所有figure

dataSet = load('iris.txt');
rawData = dataSet(:,1:4);    % 原始資料，75筆資料 x 4個特徵
label   = dataSet(:,5);      % 75筆資料所對應的標籤

% Scatter plot
for i=1:4
    for j=i:4
        if i==j
            continue;
        end
           
             rawData(101:150,i),rawData(101:150,j),'bo');   
          % 以plot繪圖指令分別畫出class1~3之第一與第二特徵。

        title('Scatter Plot');  % 圖名稱
        legend('class1', 'class2', 'class3'); % 類別標號說明
        xlabel(['Feature' num2str(i)]); % 特徵標號註解
        ylabel(['Feature' num2str(j)]); % 特徵標號註解
    end
end

% f1
distance1 = [];
for a=1:75
    trainSet = [rawData(  1: 25,1);...
                rawData( 51: 75,1);...
                rawData(101:125,1)]; 
                  % 選取每類別前半，合併為training set

    testSet = [rawData( 26: 50,1);...
               rawData( 76:100,1);...
               rawData(126:150,1)]; 
                  % 選取每類別後半，合併為test set

    dis = norm(testSet(a,:)-trainSet(a,:)); 
        % norm(u-v): 求出向量u與向量v之間的歐式距離(Euclidean distance)，也稱為2-norm
        % 第一筆test資料與第一筆training資料之距離
    distance1 = [distance1;dis];
end

[value,index] = sort(distance1,'ascend'); 

% f2
distance2 = [];
for a=1:75
    trainSet = [rawData(  1: 25,2);...
                rawData( 51: 75,2);...
                rawData(101:125,2)]; 
                  % 選取每類別前半，合併為training set

    testSet = [rawData( 26: 50,2);...
               rawData( 76:100,2);...
               rawData(126:150,2)]; 
                  % 選取每類別後半，合併為test set

    dis2 = norm(testSet(a,:)-trainSet(a,:)); 
        % norm(u-v): 求出向量u與向量v之間的歐式距離(Euclidean distance)，也稱為2-norm
        % 第一筆test資料與第一筆training資料之距離
    distance2 = [distance2;dis2];
end

% f3
distance3 = [];
for a=1:75
    trainSet = [rawData(  1: 25,3);...
                rawData( 51: 75,3);...
                rawData(101:125,3)]; 
                  % 選取每類別前半，合併為training set

    testSet = [rawData( 26: 50,3);...
               rawData( 76:100,3);...
               rawData(126:150,3)]; 
                  % 選取每類別後半，合併為test set

    dis3 = norm(testSet(a,:)-trainSet(a,:)); 
        % norm(u-v): 求出向量u與向量v之間的歐式距離(Euclidean distance)，也稱為2-norm
        % 第一筆test資料與第一筆training資料之距離
    distance3 = [distance3;dis3];
end

% f4
distance4 = [];
for a=1:75
    trainSet = [rawData(  1: 25,4);...
                rawData( 51: 75,4);...
                rawData(101:125,4)]; 
                  % 選取每類別前半，合併為training set

    testSet = [rawData( 26: 50,4);...
               rawData( 76:100,4);...
               rawData(126:150,4)]; 
                  % 選取每類別後半，合併為test set

    dis4 = norm(testSet(a,:)-trainSet(a,:)); 
        % norm(u-v): 求出向量u與向量v之間的歐式距離(Euclidean distance)，也稱為2-norm
        % 第一筆test資料與第一筆training資料之距離
    distance4 = [distance4;dis4];
end





