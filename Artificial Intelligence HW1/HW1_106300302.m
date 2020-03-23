clc;       % �M��command window
clear      % �M��workspace
close all  % �����Ҧ�figure

dataSet = load('iris.txt');
rawData = dataSet(:,1:4);    % ��l��ơA75����� x 4�ӯS�x
label   = dataSet(:,5);      % 75����Ʃҹ���������

% Scatter plot
for i=1:4
    for j=i:4
        if i==j
            continue;
        end
           
             rawData(101:150,i),rawData(101:150,j),'bo');   
          % �Hplotø�ϫ��O���O�e�Xclass1~3���Ĥ@�P�ĤG�S�x�C

        title('Scatter Plot');  % �ϦW��
        legend('class1', 'class2', 'class3'); % ���O�и�����
        xlabel(['Feature' num2str(i)]); % �S�x�и�����
        ylabel(['Feature' num2str(j)]); % �S�x�и�����
    end
end

% f1
distance1 = [];
for a=1:75
    trainSet = [rawData(  1: 25,1);...
                rawData( 51: 75,1);...
                rawData(101:125,1)]; 
                  % ����C���O�e�b�A�X�֬�training set

    testSet = [rawData( 26: 50,1);...
               rawData( 76:100,1);...
               rawData(126:150,1)]; 
                  % ����C���O��b�A�X�֬�test set

    dis = norm(testSet(a,:)-trainSet(a,:)); 
        % norm(u-v): �D�X�V�qu�P�V�qv�������ڦ��Z��(Euclidean distance)�A�]�٬�2-norm
        % �Ĥ@��test��ƻP�Ĥ@��training��Ƥ��Z��
    distance1 = [distance1;dis];
end

[value,index] = sort(distance1,'ascend'); 

% f2
distance2 = [];
for a=1:75
    trainSet = [rawData(  1: 25,2);...
                rawData( 51: 75,2);...
                rawData(101:125,2)]; 
                  % ����C���O�e�b�A�X�֬�training set

    testSet = [rawData( 26: 50,2);...
               rawData( 76:100,2);...
               rawData(126:150,2)]; 
                  % ����C���O��b�A�X�֬�test set

    dis2 = norm(testSet(a,:)-trainSet(a,:)); 
        % norm(u-v): �D�X�V�qu�P�V�qv�������ڦ��Z��(Euclidean distance)�A�]�٬�2-norm
        % �Ĥ@��test��ƻP�Ĥ@��training��Ƥ��Z��
    distance2 = [distance2;dis2];
end

% f3
distance3 = [];
for a=1:75
    trainSet = [rawData(  1: 25,3);...
                rawData( 51: 75,3);...
                rawData(101:125,3)]; 
                  % ����C���O�e�b�A�X�֬�training set

    testSet = [rawData( 26: 50,3);...
               rawData( 76:100,3);...
               rawData(126:150,3)]; 
                  % ����C���O��b�A�X�֬�test set

    dis3 = norm(testSet(a,:)-trainSet(a,:)); 
        % norm(u-v): �D�X�V�qu�P�V�qv�������ڦ��Z��(Euclidean distance)�A�]�٬�2-norm
        % �Ĥ@��test��ƻP�Ĥ@��training��Ƥ��Z��
    distance3 = [distance3;dis3];
end

% f4
distance4 = [];
for a=1:75
    trainSet = [rawData(  1: 25,4);...
                rawData( 51: 75,4);...
                rawData(101:125,4)]; 
                  % ����C���O�e�b�A�X�֬�training set

    testSet = [rawData( 26: 50,4);...
               rawData( 76:100,4);...
               rawData(126:150,4)]; 
                  % ����C���O��b�A�X�֬�test set

    dis4 = norm(testSet(a,:)-trainSet(a,:)); 
        % norm(u-v): �D�X�V�qu�P�V�qv�������ڦ��Z��(Euclidean distance)�A�]�٬�2-norm
        % �Ĥ@��test��ƻP�Ĥ@��training��Ƥ��Z��
    distance4 = [distance4;dis4];
end





