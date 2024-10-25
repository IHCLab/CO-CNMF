clear all; close all; clc;
%% Setting
load data;
load seed;
seed=seed(1);
K=fspecial('gaussian',[ratio,ratio],symvar); % Gaussian blurring kernel

[rows_h,cols_h,bands_h]=size(Yh);
Yh=reshape(Yh,[],bands_h)';
[M,L_h]=size(Yh);
SNR_Yh=35; varianc_Yh=mean(Yh.^2, 2).*(10.^(-SNR_Yh/10));
randn('state',seed); Yh=Yh+diag(sqrt(varianc_Yh))*randn(M,L_h);
Yh=reshape(Yh',rows_h,cols_h,bands_h);

[rows_m,cols_m,bands_m]=size(Ym);
Ym=reshape(Ym,[],bands_m)';
[M_m,L]=size(Ym);
SNR_Ym=30; varianc_Ym=mean(Ym.^2, 2).*(10^(-SNR_Ym/10));
randn('state',seed); Ym=Ym+diag(sqrt(varianc_Ym))*randn(M_m,L);
Ym=reshape(Ym',rows_m,cols_m,bands_m);

%% CO-CNMF
[Z_fused,time]=ConvOptiCNMF(Yh,Ym,N,D,K); % Convex Optimization based Coupled NMF (CO-CNMF)

%% Plot & Display
band_set=[61 25 13]; % RGB bands
normColor=@(R)max(min((R-mean(R(:)))/std(R(:)),2),-2)/3+0.5;

figure;
temp_show=I_REF(:,:,band_set);
temp_show=normColor(temp_show);
imshow(temp_show); xlabel('(a) (Ground Truth) Reference Image'); % ground-truth hyperspectral image

figure;
for j=1:size(band_set,2),
    temp_show2(:,:,j)=kron(Yh(:,:,band_set(j)),ones(ratio,ratio));
end
temp_show2=normColor(temp_show2);
imshow(temp_show2); xlabel('(b) Low-resolution Input Image'); % LSR hyperspectral image

figure;
temp_show=Z_fused(:,:,band_set);
temp_show=normColor(temp_show);
imshow(temp_show); xlabel('(c) Super-resolved Output Image'); % fused image

fprintf('CO-CNMF performance:\n')
tar=Z_fused;
ref=I_REF;
[~,~,bands]=size(ref);
ref=reshape(ref,[],bands);
tar=reshape(tar,[],bands);
msr=mean((ref-tar).^2,1);
max2=max(tar,[],1).^2;
out.ave=mean(10*log10(max2./msr));
fprintf('PSNR: %.2f (dB)\n',out.ave);
fprintf('TIME: %.2f (sec.)\n',time);