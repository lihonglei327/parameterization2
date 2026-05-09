clc;
clear all;
close all;

global  CONST;

addpath('commons-gen','data');

gpu =[];
if ~gpuDeviceCount()
    error('No GPU');
else
    parallel.gpu.enableCUDAForwardCompatibility(true);
    gpu = gpuDevice(1);  
    fprintf('GPU: %s\n', gpu.Name);
end

gpu =[];

CONST = containers.Map();
CONST('gpu')=gpu;
CONST('P_coef')=1e6;
CONST('step')=50;
CONST('st')=100;
CONST('ed')=35000;

CONST('MAX_ITER')=1000;

CONST('epsilon')=1e-3;
CONST('zero')=1e-10;
CONST('resolution')=6;

CONST('visible')=false;
CONST('window')=5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Lee's 
CONST('CO2')=[0.06386 2.4755 145.5];
%Young's
%CONST('CO2')=[0.0471 2.88 169.2];
%Xu's
%CONST('CO2')=[2.0/44 2.79 170.5];

%Species of componets, refers to the Var. of trial_Data.m
SPECIES='PMMA';
%SPECIES='PS';
PC_SAFT_Params=[0.03865346957	2.266827298094256	186.5292968758492 CONST('CO2')	0.03243300238239573    -489.2590905984454];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vl=[];
vu=[];

data = trial_data(SPECIES);
KEYS=data.keys();
VALS=reshape(cell2mat(data.values()),2,[])';

Ts_series_trial = containers.Map;
Ts_series_comp = containers.Map;

cnt=numel(KEYS);

trial_data=[];

for i=1:cnt
    args=sscanf(KEYS{i}, '%f');
    mw1=args(1);
    mw2=args(2);    
    t=args(3);
    P=args(4);
    CONST('T')=t;
    CONST('P')=P;

    if isKey(Ts_series_trial,num2str(t))
        Ts_series_trial(num2str(t))=[Ts_series_trial(num2str(t));[mw1 mw2 P data(KEYS{i})]];
    else
        Ts_series_trial(num2str(t))=[];
        Ts_series_comp(num2str(t))=[];
    end
end

CONST('Ts_series_trial')=Ts_series_trial;
CONST('Ts_series_comp')=Ts_series_comp;
CONST('VALS')=VALS;

CONST('vl')=vl;
CONST('vu')=vu;

T0=[];
T1=[];

COLORS=['r' 'g' 'b' 'y' 'm' 'c' 'k'];

model = @foo;

lb0=[0.001,   1., 100.,    -1, -500.0];
ub0=[1.000,   10, 1000,     1,  500.0];
paramBounds=[0.01 0.10;2 5;100 600;-1 1; -500 500];
paramNames={'cc','sigma','epsilon','alpha','beta'};

param0=PC_SAFT_Params;

numSamples=50;

for m=1:3
    xx=linspace(lb0(m),ub0(m),numSamples);
    vv=[];
    v_min=inf;
    x_min=ub0(m);
    k=1;
    for x=xx
        param=[param0(1:m-1) x param0(m+1:end)];
        y=foo(param);
        vv=[vv y];
        if y<v_min || isinf(v_min)
            v_min=y;
            x_min=x;
        end
        k=k+1;
        if mod(k,50)==0
            disp(k);
        end
    end

    figure(m);    
    set(figure(m),'Name', paramNames{m});
    title(paramNames(m));
    xlim([xx(1),xx(end)]); 
    ylim([0,0.5]); 
    hold on;
    ax = gca;

    set(0, 'DefaultAxesFontName', 'Consolas'); 
    set(0, 'DefaultAxesFontSize', 15);        
    set(0, 'DefaultAxesFontWeight', 'bold');

    plot(xx,vv,'LineWidth',2);    

    scatter([param0(m)],[foo(param0)], 80,'Marker','o','DisplayName', 'Heuristic result');
    scatter([x_min],[v_min], 80,'Marker','*','DisplayName', 'Exhaustive result');       
   
    x_lim=xlim;
    y_lim=ylim;

    text(x_lim(1)+(x_lim(2)-x_lim(1))*0.8, y_lim(1)+(y_lim(2)-y_lim(1))*0.85, '* Exhaustive result','FontSize',14);
    text(x_lim(1)+(x_lim(2)-x_lim(1))*0.8, y_lim(1)+(y_lim(2)-y_lim(1))*0.90, 'o Heuristic result','FontSize',14);

    hold off; 
        
    x2=[param0(m)];
    y2=[foo(param0)];
    idx = isfinite(x2) & isfinite(y2);
    x2=x2(idx);
    y2=y2(idx);
    
    x3=[x_min];
    y3=[v_min];
    idx = isfinite(x3) & isfinite(y3);
    x3=x3(idx);
    y3=y3(idx);
    
    x_lb=min(param0(m),x_min)*0.8;
    x_ub=max(param0(m),x_min)*1.2;
    y_lb=min(foo(param0),v_min)*0.8;
    y_ub=max(foo(param0),v_min)*1.2;

    ax2=axes('Position', [0.6, 0.15, 0.30, 0.30]);
    set(ax2,'ButtonDownFcn',@(s,~)beginDrag(s));
    xlim(ax2,[x_lb, x_ub]);  
    ylim(ax2,[y_lb, y_ub]);  
    grid on;
    box on;
    hold on;

    idx = xx >= x_lb & xx <= x_ub;  
    x0=xx(idx);
    y0=vv(idx);

    plot(ax2,x0,y0);    
   
    scatter(ax2,x2,y2, 60,'Marker','o');
    scatter(ax2,x3,y3, 60,'Marker','*');
    hold off;
end

function f=foo(param)    
    global CONST;

    Ts_series_trial=CONST('Ts_series_trial');
    Ts_series_comp=CONST('Ts_series_comp');
    VALS=CONST('VALS');
    vl=CONST('vl');
    vu=CONST('vu');

    param=[param 2.0/44 2.79 170.5  -0.803052509306871	-10.0113435282662];

    KEYS=Ts_series_trial.keys();
     
    vals_fit=[];
    dd=[];
    for i=1:numel(KEYS)
        T=str2num(KEYS{i});
        args=Ts_series_trial(KEYS{i});
    
        for j=1:size(args,1)
            arg=args(j,:);
            mw1=arg(1);
            mw2=arg(2);
            P=arg(3);
            CONST('T')=T;
            pp=phasePP(mw1,mw2,T,P, param,1,CONST);
            
            Ratio=pp.xwtPP;
            idx=pp.idx;
    
            if size(Ratio,1)<1
                %found=false;
                dd=[dd,realmax];
                Ts_series_comp(KEYS{i})=[Ts_series_comp(KEYS{i});[P nan,nan]];
            else 
                VAL=VALS(i,:);
                [d,i0]=space_dist(Ratio,VAL,vl,vu);
                dd=[dd,d];
                Ts_series_comp(KEYS{i})=[Ts_series_comp(KEYS{i});[P Ratio(1,:)]];
            end
        end
    end
    
    j=1;
    for k=1:numel(KEYS)
        T=str2num(KEYS{k});
        trial_data=Ts_series_trial(KEYS{k});

        [~, idx] = sort(trial_data(:, 3));  
        trial_data = trial_data(idx, :);  
    end
    f=mean(dd);
end

function [f,i]=space_dist(r,val,vl,vu)

    if(~exist('vl','var'))
        vl=-9999999;
    end

    if(~exist('vu','var'))
        vu=9999999;
    end

    if vl~=-9999999
        r(find(r(:,1)<vl),1)=val(1);
    end

    if vu~=9999999
        r(find(r(:,2)>vu),2)=val(2);
    end

    ii=find(~isnan(val)&~isinf(val));
	r=r(:,ii);
	val=val(ii);
    
    [f,i]=min(pdist2(r,val),[],'all');

end