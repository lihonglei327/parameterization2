clc;
clear all;
close all;

global  CONST;

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

CONST('MAX_ITER')=1000;

CONST('epsilon1')=1e-3;

CONST('visible')=false;

CONST('window')=5;

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

CONST('Ts_series_trial')=Ts_series_trial;
CONST('Ts_series_comp')=Ts_series_comp;
CONST('VALS')=VALS;


T0=[];
T1=[];

COLORS=['r' 'g' 'b' 'y' 'm' 'c' 'k'];

model = @foo;

lb0=[0.01,   0.1, 100,    0.01,   0.1,  100,  -1, -1000.0];
ub0=[0.50,   8.0, 800,    0.50,   8.0,  800,  1,   1000.0];

paramNames={'cc1','sigma1','epsilon1','cc2','sigma2','epsilon2','alpha','beta'};
numSamples=100;

param0=[0.0884208796677582	1.10816680103256	109.467629156393	0.00966726496023013	0.758412970051002	112.145284895758	0.960493119101414	-288.039278901308];

MM=5;

for m=1:6
    paramNames{m}
    xx=linspace(lb0(m),ub0(m),MM);
    figure(m);    
    set(figure(m), 'Name', paramNames{m});
    title(paramNames(m));
    
    vv=[];
    v_min=realmax;
    x_min=realmax;
    k=0;
    for x=xx
        k=k+1
        param=[param0(1:m-1) x param0(m+1:end)];
        
        y=foo(param);
        vv=[vv y];
        if y<v_min
            v_min=y;
            x_min=x;
        end        
    end
    
    plot(xx,vv);
    hold on;
    scatter([param0(m)],[foo(param0)], 20,'Marker','*');
    hold on;
    scatter([x_min],[v_min], 20,'Marker','x');
    ylim([0 0.1]);
    xlim([lb0(m) ub0(m)]);
end

function f=foo(param)

    global CONST;

    Ts_series_trial=CONST('Ts_series_trial');
    Ts_series_comp=CONST('Ts_series_comp');
    VALS=CONST('VALS');

    KEYS=Ts_series_trial.keys();
     
    vals_fit=[];
    dd=[];
    for i=1:numel(KEYS)
        mws=sscanf(KEYS{i}, '%f');
        mw1=abs(mws(1));
        mw2=abs(mws(2));            

        args=Ts_series_trial(KEYS{i});    
        for j=1:size(args,1) 
            arg=args(j,:);
            
            T=arg(1);
            P=arg(2);
            vl=arg(3);
            vu=arg(4);

            CONST('T')=T;
            pp=phasePP(mw1,mw2,T,P, param,1,CONST);
            
            Ratio=pp.xwtPP;
            idx=pp.idx;
    
            if size(Ratio,1)<1
                dd=[dd,realmax];
                Ts_series_comp(KEYS{i})=[Ts_series_comp(KEYS{i});[P nan,nan]];
            else 
                VAL=VALS(i,:);
                [d,i0]=space_dist(Ratio,VAL,vl,vu);
                dd=[dd,d];
                Ts_series_comp(KEYS{i})=[Ts_series_comp(KEYS{i});[T P Ratio(i0)]];
            end
        end
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