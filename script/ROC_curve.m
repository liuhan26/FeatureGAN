function [finalscore,labels]=ROC_curve(label_path)
fid=fopen(label_path,'rt');
label=textscan(fid,'%s %s');
lines=label{1,2};
m=length(lines);
x=evalin('base','featuregallery');
y=evalin('base','featureprobe');
[finalscore,labels]=roc(m,1,'-k',x,y,lines);
hold on;
%roc(m,0,'-b',x,y,score,lines);



function [finalscore,label]=roc(m,b,curve_color,x,y,lines)
scores=ones(m,1);
score=ones(m,1);
finalscore=ones(m,1);
ground_truth=ones(m,1);
label=ones(m,1);
pos_sum=0;
for i=1:m
    %scores(i)=pdist2(x(i,:),y(i,:),'euclidean');
    scores(i)= dot(x(i,:),y(i,:))/(norm(x(i,:))*norm(y(i,:)));
    %score(i)= dot(x2(i,:),y2(i,:))/(norm(x2(i,:))*norm(y2(i,:)));
    finalscore(i) = b * scores(i) + (1-b) * score(i);
    ground_truth(i)=str2num(lines{i,1});
    label(i)=ground_truth(i);
    if ground_truth(i)==1
        pos_sum=pos_sum+1;
    end
end
[pre,Index] = sort(finalscore);
neg_sum=m-pos_sum;
temp=[];
for i=1:m
    temp=[temp ground_truth(Index(i))];
end
ground_truth=temp;
x=zeros(m+1,1);
y=zeros(m+1,1);
auc=0;
x(1)=1;
y(1)=1;
for i=2:m
    TP=0;FP=0;
    for j=i:m     %we think >i is positive
        if ground_truth(j)==1
            TP=TP+1;
        else
            FP=FP+1;
        end
    end
    x(i)=FP/neg_sum;
    y(i)=TP/pos_sum;
    auc=auc+(y(i)+y(i-1))*(x(i-1)-x(i))/2;
end
x(m+1)=0;
y(m+1)=0;
auc=auc+y(m+1)*x(m+1)/2
hold on;
plot(x,y,curve_color);
hold off;

function []=select_hardExample(m,b,x,y,lines)
score=ones(m,1);
ground_truth=ones(m,1);
pos_sum=0;
for i=1:m
    score(i)= dot(x(i,:),y(i,:))/(norm(x(i,:))*norm(y(i,:)));
    ground_truth(i)=str2num(lines{i,1});
    if ground_truth(i)==1
        pos_sum=pos_sum+1;
    end
end
[pre,Index] = sort(score);
neg_sum=m-pos_sum;
temp=[];
for i=1:m
    temp=[temp ground_truth(Index(i))];
end
ground_truth=temp;
x=zeros(m+1,1);
y=zeros(m+1,1);
auc=0;
x(1)=1;
y(1)=1;
TP=0;FP=0;
for j=i:m     %we think >i is positive
    if ground_truth(j)==1
        TP=TP+1;
    else
        FP=FP+1;
    end
end
FPR=FP/neg_sum;
y(i)=TP/pos_sum;
x(m+1)=0;
y(m+1)=0;
auc=auc+y(m+1)*x(m+1)/2;