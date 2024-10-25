%=====================================================================
% Programmer: Chih-Hsiang Hsieh
% E-mail: s104064515@m104.nthu.edu.tw
% Date: 2017/10/19
% -------------------------------------------------------
% Reference:
% C.-H. Lin, F. Ma, C.-Y. Chi, and C.-H. Hsieh,
% ``A convex optimization based coupled non-negative matrix factorization algorithm for hyperspectral and multispectral data fusion,"
% accepted by IEEE Trans. Geoscience and Remote Sensing, 2017.
%======================================================================
% A convex optimization based coupled NMF algorithm for hyperspectral superresolution via big data fusion
% [Z_fused,time] = ConvOptiCNMF(Yh,Ym,N,D,K)
%======================================================================
%  Input
%  Yh is low-spatial-resolution hyperspectral data cube of dimension rows_h*columns_h*M.
%       (rows_h: vertical spatial dimension; 
%        columns_h: horizontal spatial dimension; 
%        M: spectral dimension.)
%  Ym is high-spatial-resolution multispectral data cube of dimension rows_m*columns_m*Mm.
%       (rows_m: vertical spatial dimension; 
%        columns_m: horizontal spatial dimension; 
%        M_m: spectral dimension.)
%  N is the model order (should be greater than the number of endmembers).
%  D is spectral response transform matrix of dimension M_m*M.
%  K is blurring kernel of dimension k*k
%       (The first dimension of K corresponds to the vertical spatial dimension;
%        the second dimension of K corresponds to the horizontal spatial dimension.)
%----------------------------------------------------------------------
%  Output
%  Z_fused is super-resolved hyperspectral data cube of dimension rows_m*columns_m*M.
%  time is the computational time (in secs).
%========================================================================

function  [Z_fused,time] = ConvOptiCNMF(Yh,Ym,N,D,K)
t0=clock;
print_flag=0;
lambda1=0.001; lambda2=0.001; % SSD regularization parameter & L1-norm regularization parameter
eta=1; tildeeta=1; % penalty parameter in ADMM Algorithm 2 & penalty parameter in ADMM Algorithm 3
mode=1; % 0 uses Eq.(17)&(26) (for non-structured blur), 1 uses Lemmas 1 & 2 (for structured blur)
Max_iter=30; % maximum iteration of Algorithm 1
iterS=5; % maximum iteration of Algorithm 2
iterA=5; % maximum iteration of Algorithm 3

[rows_m,cols_m,bands_m]=size(Ym);
[rows_h,cols_h,M]=size(Yh);
Yh=reshape(Yh,[],M)';
[M,Lh]=size(Yh);
r1=rows_m/rows_h; % vertical dimension
r2=cols_m/cols_h; % horizontal dimension
r=max(ceil(r1),ceil(r2));
Ym=imresize(Ym,[r*rows_h,r*cols_h]);
Ym=reshape(Ym,[],bands_m)';
[Perm,g,B]=Permutation(K,r,rows_h,cols_h);
if mode==1, Ym=Ym*Perm'; end

I_M=speye(M);
P=sparse(zeros(0.5*M*N*(N-1),M*N));
x=1;
for n=1:N-1,
    e_n=speye(N);
    H_n=kron(e_n(:,n)',I_M);
    for m=n+1:N,
        P_m=kron(e_n(:,m)',I_M);
        P(x:x+M-1,:)=H_n-P_m;
        x=x+M;
    end
end
PtP=P'*P; P=[];
[A,S_h,~]=HyperCSI_int(Yh,N); % initialization by HyperCSI [43]
S=S_h*kron(speye(Lh),ones(size(g')));
for k=1:Max_iter,
    [S]=S_NonNeg_Lasso(N,Yh,Ym,A,S,D,g,B,iterS,lambda2,eta,mode); % Algorithm 2 (warm start)
    [A]=A_ICE(N,Yh,Ym,S,A,D,g,B,iterA,lambda1,tildeeta,PtP,mode); % Algorithm 3 (warm start)
end
if mode==0,
    Z_fused=A*S;
else
    Z_fused=A*S*Perm;
end
Z_fused=reshape(Z_fused',r*rows_h,r*cols_h,M);
Z_fused=imresize(Z_fused,[rows_m,cols_m]); % resize back
time=etime(clock,t0);

%% subprogram 1 (implementation of Algorithm 2, solving the non-negative LASSO)
function [S]=S_NonNeg_Lasso(N,Yh,Ym,A,S_old,D,g,B,iterS,lambda2,eta,mode)
L=size(Ym,2);
y=reshape([reshape(Yh,[],1)',reshape(Ym,[],1)'],[],1);
x=reshape(S_old,[],1);
nu=sparse(zeros(N*L,1));
if mode==1,
    C_bar1=kron(g',A)';
    C_bar2=kron(speye(size(g,1)),D*A)';
    C_bartC_bar=[C_bar1,C_bar2]*[C_bar1,C_bar2]';
    C_bar_inv=C_bartC_bar+eta*speye(size(C_bartC_bar));
    C_binv1=C_bar_inv\C_bar1;
    C_binv2=C_bar_inv\C_bar2;
    Ym_re=reshape(Ym,size(C_binv2,2),[]);
    C_bYhYm=reshape(C_binv1*Yh+C_binv2*Ym_re,[],1);
    for j=1:iterS,
        % s update
        xnu_re=reshape(eta*x-nu,size(C_bar_inv,2),[]);
        s=reshape(C_bar_inv\xnu_re,[],1)+C_bYhYm;
        % x update
        x=s+nu/eta-lambda2/eta;
        x(x<0)=0;
        % nu update
        nu=nu+eta*(s-x);
    end
else % naive version
    C=[kron(B',A)',kron(speye(L),D*A)']';
    CtC=C'*C;
    Cty=C'*y;
    Lbs=chol(CtC+eta*speye(N*L),'lower');
    for j=1:iterS,
        % s update
        q=Cty+eta*x-nu;
        s=Lbs'\(Lbs\q);
        % x update
        x=s+nu/eta-lambda2/eta;
        x(x<0)=0;
        % nu update
        nu=nu+eta*(s-x);
    end
end
S=reshape(x,N,L);
return;

%% subprogram 2 (implementation of Algorithm 3, solving the ICE-regularized problem)
function [A]=A_ICE(N,Yh,Ym,S,A_old,D,g,B,iterA,lambda1,tildeeta,PtP,mode)
M=size(Yh,1);
y=reshape([reshape(Yh,[],1)',reshape(Ym,[],1)'],[],1);
z=reshape(A_old,[],1);
nu=zeros(M*N,1);
if mode==1,
    S_3d=reshape(S,N,size(g,1),[]);
    S_perm=permute(S_3d,[1,3,2]);
    S_re=reshape(S_perm,[],size(g,1));
    Sgv=reshape(S_re*g,N,[]);
    CtCb1=kron(Sgv*Sgv',speye(M));
    DtD=sparse(D'*D);
    StS=S*S';
    CtCb2=kron(StS,DtD);
    CtC=CtCb1+CtCb2;
    CtYh=reshape((Sgv*Yh')',[],1);
    CtYm=reshape((S*(D'*Ym)')',[],1);
    Cty=CtYh+CtYm;
    CtCinv=CtC+lambda1*(PtP)+tildeeta*speye(M*N);
    for j=1:iterA,
        % a update
        a=CtCinv\(Cty+tildeeta*z-nu);
        % z update
        z=a+nu/tildeeta;
        z(z<0)=0;
        % nu update
        nu=nu+tildeeta*(a-z);
    end
else % naive version
    C=[kron((S*B)',sparse(eye(M)))' kron(S',D)']';
    CtC=C'*C;
    Cty=C'*y;
    Lba=chol(CtC+lambda1*(PtP)+tildeeta*speye(MN),'lower');
    for j=1:iterA,
        % a update
        q=Cty+tildeeta*z-nu;
        a=Lba'\(Lba\q);
        % z update
        z=a+nu/tildeeta;
        z(z<0)=0;
        % nu update
        nu=nu+tildeeta*(a-z);
    end
end
A=reshape(z,M,N);
return;

%% subprogram 3
function [Pi,gv,B] = Permutation(K,r,rows_h,cols_h)
% This function helps user automatically generate the spatial spread
% transform matrix B from the blurring kernel K, automatically permute the
% pixels in a specific order (so that our fast closed-form solutions can be
% applied), and then accordingly revise the spatial spread transform matrix.
% This function considers the following 4 types of K, and automatically adapts
% it to the desired form with the original properties (variance) remained unchanged:
%   1. K is symmetric Gaussian, and k=sqrt((rows_m*columns_m)/(rows_h*columns_h))=r (the desired form). 
%   2. K is symmetric Gaussian, but k does not equal to r.
%   3. K is not symmetric Gaussian but is uniform.
%   4. K is not symmetric Gaussian and is non-uniform.

blurkernel=size(K,1); r_square=r^2; [isgusyK,sigmaK]=isgusy(K);
if (all(K(:)==K(1))==1)
    blurring_case=0;
elseif blurkernel==r && (isgusyK==1)
    blurring_case=1;
elseif (isgusyK==1)
    blurring_case=2;
else
    blurring_case=3;
end
switch blurring_case
    case 0
        Temp=ones(r,r)/(r^2);
    case 1
        Temp=K;
    case 2
        Temp=fspecial('gaussian',[r,r],sigmaK);
    case 3
        zz=1;
        ubound=150; lbound=0.05; interval=0.05;
        a=zeros(((ubound-lbound)/interval)+1,1);
        for sigma=lbound:interval:ubound
            Temp=fspecial('gaussian',[blurkernel,blurkernel],sigma);
            a(zz)=norm(Temp-K,'fro'); zz=zz+1;
        end
        [~,index]=min(a);
        sigma=lbound+(index-1)*interval;
        Temp=fspecial('gaussian',[r,r],sigma);
end

% generate B before permutation
Hv=speye(rows_h*cols_h);
for i=1:rows_h*cols_h
    Hm=sparse(reshape(Hv(:,i),rows_h,cols_h));
    B(:,i)=sparse(reshape(kron(Hm,Temp),[],1));
end
gv=reshape(Temp,[],1);

% start permutation
Pi=sparse(zeros(r_square,r_square));
Pi=kron(ones(1,rows_h*cols_h),Pi);
Pi_temp=Pi;
vv=find(B(:,1)>0);
for j=1:length(vv)
    Pi(j,vv(j))=1;
end
for i=2:rows_h*cols_h
    vv=find(B(:,i)>0);
    Pi_new=Pi_temp;
    for j=1:length(vv)
        Pi_new(j,vv(j))=1;
    end
    Pi=[Pi;Pi_new];
end
return;

%% subprogram 4
function [E,sigma]=isgusy(K) % check if K is symmetric Gaussian
k=size(K,1);
issym=@(x)isequal(x,x.');
updown=sum(sum(abs(K-fliplr(K))));
leftright=sum(sum(abs(K-flipud(K))));
if (issym(K)==1 && updown==0 && leftright==0) % check the symmetricity
    if (mod(k,2)==0) % define the coordinate w.r.t the center
        index=(k/2)-0.5;
    else
        index=floor(k/2);
    end
    sigma=sqrt((2*(index^2)-2*(index-1)^2)/(2*log(K(2,2)/K(1,1))));
    temp=fspecial('gaussian',[k,k],sigma);
    if sum(sum(abs(K-temp)))<10^(-10)
        E=1;
    else
        E=0;
    end
else
    E=0;
end
return;

%% subprogram 5
function [A_est,S_est,time] = HyperCSI_int(X,N) % HyperCSI adapted for initialization
t0 = clock;
con_tol = 1e-8;
num_SPA_itr = N;
comm_flag = 1;
[M L] = size(X);
d = mean(X,2);
U = X-d*ones(1,L);
[eV D] = eig(U*U');
C = eV(:,M-N+2:end);
Xd = C'*(X-d*ones(1,L));
alpha_tilde = iterativeSPA(Xd,L,N, num_SPA_itr, con_tol);
for i = 1 : N
    bi_tilde(:,i) = compute_bi(alpha_tilde,i,N);
end
radius = 1e-8;
for i = 1:N-1,
    for j = i+1:N,
        dist_ai_aj(i,j) = norm(alpha_tilde(:,i)-alpha_tilde(:,j));
        if (1/2)*dist_ai_aj(i,j) < radius,
            radius = (1/2)*dist_ai_aj(i,j);
        end
    end
end
Xd_divided_idx = zeros(L,1);
radius_square = radius^2;
for i = 1:N,
    [IDX_alpha_i_tilde]= find(sum((Xd-alpha_tilde(:,i)*ones(1,L) ).^2,1) < radius_square);
    Xd_divided_idx(IDX_alpha_i_tilde) = i;
end
for i = 1:N,
    Hi_idx = setdiff([1:N],[i]);
    for k = 1:1*(N-1),
        Ri_k = Xd(:,( Xd_divided_idx == Hi_idx(k) ));
        [val idx] = max(bi_tilde(:,i)'*Ri_k);
        pi_k(:,k) = Ri_k(:,idx);
    end
    b_hat(:,i) = compute_bi([pi_k alpha_tilde(:,i)],N,N);
    h(i,1) = max(b_hat(:,i)'* pi_k(:,1));
end
for i = 1:N
    bbbb = b_hat;
    ccconst = h;
    bbbb(:,i) = []; ccconst(i) = [];
    alpha_hat(:,i) = pinv(bbbb')*ccconst;
end
A_est = C * alpha_hat + d * ones(1,N);
S_est = ( h*ones(1,L)- b_hat'*Xd   ) ./ ( (  h - sum( b_hat.*alpha_hat )' ) *ones(1,L) );
S_est(S_est<0) = 0;
time = etime(clock,t0);
return;

%% subprogram 6
function [alpha_tilde] = iterativeSPA(Xd,L,N,num_SPA_itr,con_tol) % SPA with post-processing
p = 2;
N_max = N;
A_set=[]; Xd_t = [Xd; ones(1,L)]; index = [];
[val ind] = max(sum(abs(Xd_t).^p).^(1/p));
A_set = [A_set Xd_t(:,ind)];
index = [index ind];
for i=2:N
    XX = (eye(N_max) - A_set * pinv(A_set)) * Xd_t;
    [val ind] = max(sum(abs(XX).^p).^(1/p));
    A_set = [A_set Xd_t(:,ind)];
    index = [index ind];
end
alpha_tilde = Xd(:,index);
current_vol = det( alpha_tilde(:,1:N-1) - alpha_tilde(:,N)*ones(1,N-1) );
for jjj = 1:num_SPA_itr
    for i = 1:N
        b(:,i) = compute_bi(alpha_tilde,i,N);
        b(:,i) = -b(:,i);
        [const idx] = max(b(:,i)'*Xd);
        alpha_tilde(:,i) = Xd(:,idx); 
    end
    new_vol = det( alpha_tilde(:,1:N-1) - alpha_tilde(:,N)*ones(1,N-1) );
    if (new_vol - current_vol)/current_vol  < con_tol
        break;
    end
end
return;

%% subprogram 7
function [bi] = compute_bi(A,i,N) % compute normal vectors
Hidx = setdiff([1:N],[i]);
A_Hidx = A(:,Hidx);
P = A_Hidx(:,1:N-2)-A_Hidx(:,N-1)*ones(1,N-2);
bi = A_Hidx(:,N-1)-A(:,i);
bi = (eye(N-1) - P*(pinv(P'*P))*P')*bi;
bi = bi/norm(bi);
return;