clc;       % 清除command window
clear      % 清除workspace
close all  % 關閉所有figure

[x1,x2,x3,x4,y] = textread('iris.txt','%f %f %f %f %f','delimiter',',');

x_train = [x1(1:25,:) x2(1:25,:) x3(1:25,:) x4(1:25,:);           
           x1(51:75,:) x2(51:75,:) x3(51:75,:) x4(51:75,:);           
           x1(101:125,:) x2(101:125,:) x3(101:125,:) x4(101:125,:)];
       
y_train = [y(1:25) y(51:75) y(101:125)]; 

x_test = [x1(26:50,:) x2(26:50,:) x3(26:50,:) x4(26:50,:);           
          x1(76:100,:) x2(76:100,:) x3(76:100,:) x4(76:100,:);           
          x1(126:150,:) x2(126:150,:) x3(126:150,:) x4(126:150,:)];
      
y_test = [y(26:50) y(76:100) y(126:150)];


[m_train,n_train] = size(x_train);

[m_test,n_test] = size(x_test);

k = [1 3];

for i = 1:m_test    
    for j = 1:m_train        
        x_d(j,:) = x_test(i,:) - x_train(j,:);        
        x_norm(j) = norm(x_d(j,:),2);    
    end
    x = [x_train  y_train  x_norm];    
    mm = n_train + 2;   
    x = sortrows(x,mm);    
    for k_num = 1:5        
        y_p = x(1:k(k_num),n_train+1);        
        y_pre(i,k_num) = mode(y_p);    
    end
end

accurancy = zeros(1,5);
for i = 1:5    
    for j = 1:m_test        
        if (y_pre(j,i)==y_test(j))            
            accurancy(1,i) = accurancy(1,i)+1;        
        else
            accurancy(1,i) = accurancy(1,i);        
        end
        
    end
    accurancy(1,i) = accurancy(1,i)/m_test;
end
figure(1)
plot(accurancy(1,:),'b');

