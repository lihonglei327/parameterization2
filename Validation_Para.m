% generate theoritical data
clc;
clear all;
close all;

% Press coef

addpath('commons-gen','data');

gpu =[];
if ~gpuDeviceCount()
    error('No GPU');
else
    parallel.gpu.enableCUDAForwardCompatibility(true);
    gpu = gpuDevice(1);  % 选择第1块GPU设备
    fprintf('GPU: %s\n', gpu.Name);
end

gpu =[];

CONST = containers.Map();
CONST('gpu')=gpu;
CONST('P_coef')=1e6;
CONST('step')=50;
CONST('st')=100;
CONST('ed')=35000;

CONST('resolution')=6;
CONST('MAX_ITER')=1000;
CONST('epsilon')=1e-3;
CONST('zero')=1e-10;

CONST('visible')=true;
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

%set the fitting parameters here.
%the first 3 parameters: m/M, epsilon, sigma
%the last 2 parameters: a and b

PC_SAFT_Params=[0.03865346957	2.266827298094256	186.5292968758492 CONST('CO2')	0.03243300238239573    -489.2590905984454];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vl=[];
vu=[];

data = trial_data(SPECIES);
KEYS=data.keys();
VALS=reshape(cell2mat(data.values()),2,[])';

N_KEYS=numel(KEYS);

Ts_series_trial = containers.Map;
Ts_series_comp = containers.Map;

trial_data=[];

for i=1:N_KEYS    
    key=KEYS{i};
    args=sscanf(key, '%f');
    mw1=args(1);
    mw2=args(2);
    P=args(4);
    T=args(3);

    if isKey(Ts_series_trial,num2str(T))
        Ts_series_trial(num2str(T))=[Ts_series_trial(num2str(T));[mw1 mw2 P data(key)]];
    else
        Ts_series_trial(num2str(T))=[mw1 mw2 P data(KEYS{i})];
        Ts_series_comp(num2str(T))=[];
    end
end

found=true;
T0=[];
T1=[];

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
        CONST('P')=P;
        pp=phasePP(mw1,mw2,T,P, PC_SAFT_Params,1,CONST);
        
        Ratio=pp.xwtPP;
        idx=pp.idx;

        if size(Ratio,1)<1
            found=false;
            dd=[dd,nan];
            Ts_series_comp(KEYS{i})=[Ts_series_comp(KEYS{i});[P nan,nan]];
        else 
            VAL=VALS(i,:);
            [d,i0]=space_dist(Ratio,VAL,vl,vu);
            dd=[dd,d];
            Ts_series_comp(KEYS{i})=[Ts_series_comp(KEYS{i});[P Ratio(1,:)]];
        end
    end
end

KEYS=sort(KEYS);
COLORS=['r' 'g' 'b' 'm' 'k' 'y' 'c'];
if found     
    figure;
    ylim([0.7 1]);
    title("Parameter fitting result"); 
    hold on;

    j=1;
    for k=1:numel(KEYS)
        T=str2num(KEYS{k});
        trial_data=Ts_series_trial(KEYS{k});

        [~, idx] = sort(trial_data(:, 3));  
        trial_data = trial_data(idx, :);  
        
        color=COLORS(k);
        %plot(trial_data(:,3),trial_data(:,4),strcat(color,'*-'),trial_data(:,3),trial_data(:,5),strcat(color,'*-'));
        plot(trial_data(:,3),trial_data(:,5),strcat(color,'*-'));
        hold on;

        comp_data=Ts_series_comp(KEYS{k});
        [~, idx] = sort(comp_data(:, 1));  
        comp_data = comp_data(idx, :);  

        %plot(comp_data(:,1),comp_data(:,2),strcat(color,'o-'),comp_data(:,1),comp_data(:,3),strcat(color,'o-'));
        plot(comp_data(:,1),comp_data(:,3),strcat(color,'o-'));        
        hold on;
    end

    legend(sort([KEYS,KEYS]),'Location','best');
    xlabel('Pressure (MPa)');
    ylabel('Ratio');
    %savefig;
   
    disp("Avg. distance");
    disp(mean(dd));
    disp("Done.");
else
    disp([T0,"losing results"]);
    disp([T1,"having results"]);
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