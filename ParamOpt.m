
clc;
clear all;
close all;

beep on;

rng('shuffle');
format longG

addpath('commons-gen');

gpu =[];

CONST = containers.Map();
CONST('gpu')=gpu;
% 计算质量密度的起止点及步长，步长越小解越好，但越费时（EOS,DFT）
CONST('st')=100;
CONST('ed')=35000;
CONST('step')=50;

% 寻找两个化学势相等的容差
CONST('epsilon1')=5e-2;
CONST('X_TOL')=1e-20;
CONST('F_TOL')=1e-4;
CONST('ZERO')=1e-10;

CONST('vl')=[];
CONST('vu')=[];

CONST('visible')=false;
CONST('P_coef')=1e6;

CONST('err_cnt')=0.25;
CONST('COLORS')=['r' 'g' 'b' 'y' 'm' 'c' 'k'];

%Lee
CONST('CO2')=[0.06386 2.4755 145.5];


iter=0;

CPUS=floor(max(maxNumCompThreads(),feature('numcores'))*0.6);

myCluster = parcluster('Processes');
delete(myCluster.Jobs);

delete(gcp('nocreate'));
parpool(CPUS);

SEGS=3;
n=1:SEGS;
N=table2array(combinations(n,n,n,n,n));
    
% *******************************************************

data = trial_data();

KEYS=data.keys();
VALS0=data.values();


VALS=reshape(cell2mat(VALS0),2,[])';
MINVALS=min(VALS);
MAXVALS=max(VALS);

MINVALS=max(MINVALS,0);
MAXVALS(MAXVALS==0) = 0.01;

VALS=normalize(VALS,'range');

CONST('KEYS')=KEYS;
CONST('VALS0')=VALS0;
CONST('VALS')=VALS;
CONST('MINVALS')=MINVALS;
CONST('MAXVALS')=MAXVALS;


pop_size = 40;
CONST('MAX_ITER') = 1000;
CONST('TRY_ITER') = 100;

fun = @foo;

lb0=[0.01,   2, 100,    -1, -400.0];
ub0=[0.10,   5, 600,     1,  400.0];


dim = length(lb0);

CONST('INSTALLATION') = min(ceil(CONST('MAX_ITER')*dim*0.2/100)*100,300);
BEST_SOLUTION=[];
try
    file='data\BEST_SOLUTION.mat';
    load(file);
catch
    BEST_SOLUTION=[];
    BEST_FITNESS=realmax;
    Z=1;
end
Z=1;
CONST('BEST_SOLUTION')=BEST_SOLUTION;

X0=[];
X0=unique([X0;BEST_SOLUTION],'rows');
CONST('X0')=X0;


FITNESS0 = zeros(size(X0,1), 1)+realmax;

for i = 1:size(X0,1)
    xx=[X0(i,1:3) CONST('CO2') X0(i,4:end)];
    FITNESS0(i) = fun(xx(i, :),CONST);
end

CONST('FITNESS0')=FITNESS0;

best_index=1;
solutions=[];

if length(FITNESS0)>0
    BEST_FITNESS=min(FITNESS0);
end

CONST('BEST_FITNESS')=BEST_FITNESS;

fitnesses=[];


h = waitbar(0, 'Starting...');

runtime_fit=[BEST_FITNESS];
runtime_z=[Z];
Z0=Z;
for Z=Z0:CPUS:size(N,1)
    pr=Z/size(N,1);
    waitbar(pr, h, sprintf('Processing %d%%', ceil(pr*100)));
    res=NaN(CPUS, dim+1);
    remains=size(N,1)-Z+1;
    
    nn=min(CPUS,remains);

    clc;
    disp(["***************",num2str(Z),"*************"]);
    figure(4);
    set(figure(4), 'Name', 'BEST_FITNESSES');
    title(runtime_fit(end));
    hold on;
    plot(runtime_z,runtime_fit);
    drawnow;
    hold on;

    parfor cpu=1:nn
        idx=N(Z+cpu-1,:);
        lb=lb0+(idx-1).*(ub0-lb0)/SEGS;
        ub=lb0+idx.*(ub0-lb0)/SEGS;

        rr=Alpha_Evolution(pop_size,dim,ub,lb,fun,CONST,Z+cpu-1);
        res(cpu,:) = rr;
    end
    res(any(isnan(res), 2),:) = [];
    res=unique(res,'rows');

    FITNESS=res(:,end);

    [BEST_FITNESS1,BEST_INDEX1]=min(FITNESS);
    BEST_SOLUTION1=res(BEST_INDEX1,1:end-1);

    if BEST_FITNESS1<10
        save(strcat('data/BETTER_SOLUTIONS_',num2str(Z),'.mat'),'Z','BEST_FITNESS1','BEST_SOLUTION1');
    end

    FITNESS0=CONST('FITNESS0');
    X0=CONST('X0');
    best_val=min(BEST_FITNESS);
    if BEST_FITNESS1<=best_val
        if BEST_FITNESS1<best_val && best_val<realmax
            BEST_FITNESS= BEST_FITNESS1;
            BEST_SOLUTION=BEST_SOLUTION1;

            FITNESS0=[FITNESS0;BEST_FITNESS];
            X0=[X0;BEST_SOLUTION];
        else
            BEST_SOLUTION=[BEST_SOLUTION;BEST_SOLUTION1];
            BEST_FITNESS= [BEST_FITNESS;BEST_FITNESS1];
            save('data/OTHER_BEST_SOLUTIONS.mat','Z','BEST_FITNESS','BEST_SOLUTION');
            FITNESS0=[FITNESS0;BEST_FITNESS1];
            X0=[X0;BEST_SOLUTION1];
        end
        figure(3);
        set(figure(3), 'Name', 'BEST_FITNESS');
        best_val=min(BEST_FITNESS);
        title(best_val);
        hold on;
        scatter(Z,best_val);
        hold on;
    end
    CONST('FITNESS0')=FITNESS0;
    CONST('X0')=X0;
    CONST('BEST_FITNESS')=BEST_FITNESS;
    CONST('BEST_SOLUTION')=BEST_SOLUTION;


    if BEST_FITNESS1<10
        runtime_fit=[runtime_fit,BEST_FITNESS1];
        runtime_z=[runtime_z,Z];
    end  
    save('data/BEST_SOLUTION.mat','Z','BEST_SOLUTION','best_val');
end

close(h);
BEST_SOLUTION

if gpuDeviceCount > 0
    reset(gpu);
end

function gbestx=Alpha_Evolution(N,D,ub,lb,Func,CONST,ZONE)
    % Algorithm Name: Alpha Evolution Algorithm (AE).
    % gbestx: The global best solution ( gbestx = [x1,x2,...,xD]).
    % gbestfitness: Record the fitness value of global best individual.
    % gbesthistory: Record the history of changes in the global optimal fitness.
    %---------------------------------------------------------------------------
    %Initialization

    MaxFEs=CONST('MAX_ITER')*D;
    FEs=0;

    X=zeros(N,D);
    R=X;
    W=X;
    newE=X;
    
    f=inf(1,N);

    X=lb+(ub-lb).*rand(N,D);

    for i=1:N
        xx=[X(i,1:3) CONST('CO2') X(i,4:end)];
        f(i)=Func(xx,CONST);
    end

    
    [gbestfitness,mi]=min(f);
    gbestx=X(mi,:);

    
    Pa=lb+(ub-lb).*rand(1,D);
    Pb=lb+(ub-lb).*rand(1,D);
    %---------------------------------------------------------------------------

    early_stop=0;
    installation=0;
    while FEs<=MaxFEs
        %Sampling evolution matrix
        if mod(FEs,10) ==0
            fprintf("%d:%s\n",ZONE,repmat('-',1,ceil(FEs/50.0)));
        end
        
        k=randi(N,N,1);
        E=X(k,:);
        [~,ind]=sort(f);
        R1=rand(N,D);
        R2=rand(N,D);

        S=randi([0,1],N,D);
        r=(ub-lb).*(2*R1.*R2-R2).*S;


        alpha=exp(log(1-FEs/MaxFEs)-(4*(FEs/MaxFEs))^2);

        ar=alpha.*r;

        for i=1:N
            cab=FEs/MaxFEs;
   
            R(i,:)=X(ind(randi([1,length(1:find(k(i)==ind))])),:);
            W(i,:)=X(ind(randi([length(1:find(k(i)==ind)),N])),:);

            if rand<0.5
                A=X(randi(N,D,1),:);
                Pa=(1-cab)*Pa+cab*diag(A)';
                Ov=Pa;
            else
                K=ceil(N*rand);
                I1=[];I1=randperm(N,K);
                w=[];w=f(I1)./sum(f(I1));
                B=[];B=X(I1,:);
                Pb=(1-cab)*Pb+cab*(w*B);
                Ov=Pb;
            end
            I2=round(rand);
            sita=[];
            sita=I2*rand(1,D)+(1-I2)*rand*2;

            newE(i,:)=Ov+ar(i,:)+sita.*(R(i,:)+E(i,:)-Ov-W(i,:));

            flagub=newE(i,:)>ub;
            if max(flagub)
                newE(i,flagub)=ub(flagub);
            end
            flaglb=newE(i,:)<lb;            
            if max(flaglb)
                newE(i,flaglb)=lb(flaglb);
            end
            xx=[newE(i,1:3) CONST('CO2') newE(i,4:end)];
            newf=Func(xx,CONST);
            FEs=FEs+1;

            %Selection
            if newf<=f(k(i))
                f(k(i))=newf;
                X(k(i),:)=newE(i,:); 
                if f(k(i))<gbestfitness
                    if f(k(i))<CONST('F_TOL')
                        if gbestfitness-f(k(i))>CONST('ZERO')
                            early_stop=1;
                        end
                    end
                    gbestfitness=f(k(i));
                    gbestx=X(k(i),:);
                    installation=0;
                else
                    installation=installation+1;
                    if installation>=CONST('INSTALLATION')
                        early_stop=1;
                    end
                end
            else
                installation=installation+1;
            end
        end
        if mod(FEs, MaxFEs/D)==0
            fprintf("        ZONE: %d, AE, FEs: %d, fitess error = %e\n",ZONE,FEs,gbestfitness);
        end
        if early_stop
            break;
        end
    end % end while
    fprintf("\n");
    fprintf("    ZONE: %d, AE, FEs: %d, fitess error = %e\n",ZONE,FEs,gbestfitness);disp(gbestx);
    gbestx=[gbestx,gbestfitness];
end

