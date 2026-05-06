clc;
clear all;
close all;

addpath('commons-gen');

gpu =[];
if ~gpuDeviceCount()
    error('No GPU device.');
else
    parallel.gpu.enableCUDAForwardCompatibility(true);
    gpu = gpuDevice(1); 
    fprintf('Current GPU: %s\n', gpu.Name);
end

gpu =[];

CONST = containers.Map();
CONST('gpu')=gpu;
CONST('P_coef')=1e6;

CONST('step')=2e-3;
CONST('st')=0.001;
CONST('ed')=0.999;

CONST('MAX_ITER')=3000;

CONST('epsilon1')=1e-3;

CONST('visible')=false;

CONST('window')=5;
vl=[];
vu=[];

data = trial_data();
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

    key=strcat(num2str(mw1),'-',num2str(mw2));
    
    t=args(3);
    P=args(4);
    val=data(KEYS{i});
    if ~isKey(Ts_series_trial,key)
        Ts_series_trial(key)=[];
        Ts_series_comp(key)=[];
    end
    Ts_series_trial(key)=[Ts_series_trial(key);[t P val]];
end

found=true;
T0=[];
T1=[];

param=[0.0884208796677582	1.10816680103256	109.467629156393	0.00966726496023013	0.758412970051002	112.145284895758	0.960493119101414	-288.039278901308];

SERIES=Ts_series_trial.keys();
 
vals_fit=[];
dd=[];
for i=1:numel(KEYS)
    args=sscanf(KEYS{i}, '%f');
    
    mw1=args(1);
    mw2=args(2);

    key=strcat(num2str(mw1),'-',num2str(mw2));
    
    T=args(3);
    P=args(4);
    CONST('T')=T;

    pp=phasePP(mw1,mw2,T,P, param,1,CONST);
        
    Ratio=pp.xwtPP;
    idx=pp.idx;

    if size(Ratio,1)<1
        found=false;
        dd=[dd,nan];
        Ts_series_comp(key)=[Ts_series_comp(key);[T P nan,nan]];
    else 
        VAL=VALS(i,:);
        [d,i0]=space_dist(Ratio,VAL,vl,vu);
        dd=[dd,d];

        v=Ratio(i0,:);
        k=isnan(VAL);
        v(k)=nan;

        Ts_series_comp(key)=[Ts_series_comp(key);[T P v]];
    end
end

SERIES=sort(SERIES);
COLORS=['r' 'g' 'b' 'm' 'k' 'y' 'c'];
if found     
    figure;
    xlim([0 1]);
    title("Ratio");  
    hold on;

    j=1;
    for k=1:numel(SERIES)
        key=SERIES{k};
        trial_data=Ts_series_trial(key);
        
        color=COLORS(k);
        plot(trial_data(:,3),trial_data(:,1),strcat(color,'*-'),trial_data(:,4),trial_data(:,1),strcat(color,'*-'));
        hold on;

        comp_data=Ts_series_comp(key);

        plot(comp_data(:,3),comp_data(:,1),strcat(color,'o-'),comp_data(:,4),comp_data(:,1),strcat(color,'o-'));
        
        hold on;
    end

    legend(SERIES,'Location','best');
   
    disp("Distance Avg.");
    disp(mean(dd));

    disp("Done");

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