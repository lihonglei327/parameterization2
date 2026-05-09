% return the super args

function INPUT=input_PPnew(T,mw1,mw2,cc1,e1,sigma1,cc2,e2,sigma2,alpha,beta,CONST)

% cc:0.004-0.028
% e:338.1-342.3
% sigma:3.236-3.28

% PS-PVME
% mw1=9.9e4;
% mw2=1e5;

%PS-PMMA
%mw1=2.4e3;
%mw2=1.1e4;

m1=mw1*cc1;
m2=mw2*cc2;
d1=sigma1*(1-0.12*exp(-3*e1/T));
d2=sigma2*(1-0.12*exp(-3*e2/T));

kij=(alpha*T+beta)/1e4;

% d1 d2  function of temperature
sigval=2.5;              % unit length
sig1=d1/sigval;
sig2=d2/sigval;              % Temperature depended
sigv1=sigma1/sigval;    % Temperature independent
sigv2=sigma2/sigval;

      


sigs=[1,sig1,sig1^2;1,sig2,sig2^2];
sig1_3=[1,sig1,sig1^2,sig1^3];
sig2_3=[1,sig2,sig2^2,sig2^3];
sig1_2_3=[pi/6*sig1^2,pi/6*sig1^3];


%       below     [ 7 X 6  matirx]
A=[0.9105631445 -0.3084016918 -0.0906148351 0.7240946941 -0.5755498075 0.0976883116
    0.6361281449 0.1860531159 0.4527842806 2.2382791861 0.6995095521 -0.2557574982
    2.6861347891 -2.5030047259 0.5962700728 -4.0025849485 3.8925673390 -9.1558561530
    -26.547362491 21.419793629 -1.7241829131 -21.003576815 -17.215471648 20.642075974
    97.759208784 -65.255885330 -4.1302112531 26.855641363 192.67226447 -38.804430052
    -159.59154087 83.318680481 13.776631870 206.55133841 -161.82646165 93.626774077
    91.297774084 -33.746922930 -8.6728470368 -355.60235612 -165.20769346 -29.666905585];


if ~isempty(CONST('gpu'))
    A=gpuArray(A);
end


B=A(:,4:6);
A=A(:,1:3);

e12=sqrt(e1*e2)*(1-kij);


%惩罚。kij太大失去意义
%{
if abs(kij)>0.1
  e12=nan;  
end
%}
%
INPUT=struct('sigval',sigval,'sig1',sig1,'sig2',sig2,'sigv1',sigv1,'sigv2',sigv2,'sigs',sigs,'sig1_3',sig1_3,'sig2_3',sig2_3,'sig1_2_3',sig1_2_3,'A',A,'B',B,...
    'T',T,'m1',m1,'m2',m2,'mw1', mw1,'mw2', mw2,'e1',e1,'e2',e2,'e12',e12);
end