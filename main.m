clc;
close all;
clear all;

% Super-resolution Code by CVIP LAB, Dept of ECE, Tezpur University
p = pwd;
addpath(fullfile(p, 'ksvdbox')) % K-SVD dictionary training algorithm
addpath(fullfile(p, 'ompbox')) % Orthogonal Matching Pursuit algorithm

% my code for dictionary training
TR_IMG_PATH = 'data/train';

dict_size   = 512 ; 
lambda      = 0.1; 
patch_size  = 5; 
nSmp        = 10000;
upscale     = 2;

%patch extraction
[Xh, Xl] = patch_extract(TR_IMG_PATH, '*.png', patch_size, nSmp, upscale);

%train coupled dictionary
tic;
[Dh, Dl] = train_coupled_dict(Xh, Xl, dict_size, lambda);
dict_path = ['Dictionary/D_tris_' num2str(dict_size) '_' num2str(lambda) '_' num2str(patch_size) '.mat' ];
save(dict_path, 'Dh', 'Dl');
toc;

%Reconstruction
im_gnd = imread('Data/Test/Test1.png'); %original
[r,c]= size(im_gnd(:,:,1));
upscale = 2;
overlap =4;
maxIter = 20;

% generate low resolution counter parts
hsize = 5;
sigma = 0.5; % Whatever
kernel = fspecial('gaussian', hsize, sigma);
im_blur = imfilter(im_gnd, kernel);
im_l = imresize(im_blur,1/upscale);

% bicubic
im_b = imresize(im_l, upscale, 'bicubic');

%sparse representation
[im_h] = ScSR(im_l, upscale, Dh, Dl, lambda, overlap);
[im_h] = uint8(backprojection(im_h, im_l, maxIter));
imwrite(im_h, 'result.png');

% Display results
figure;
subplot(2,2,1); imshow(im_gnd);title('ground truth');
subplot(2,2,2); imshow(im_l);title('LR image');
subplot(2,2,3); imshow(im_b);title('bicubic');
subplot(2,2,4); imshow(im_h);title('super-resolved');

%PSNR and SSIM
psnr1 = psnr(double(im_h), double(im_gnd));
ssim1 = ssim(double(im_h), double(im_gnd));
psnr2 = psnr(double(im_b), double(im_gnd));
ssim2 = ssim(double(im_b), double(im_gnd));

%fprintf('PSNR = %0.2f\n', psnr1);
disp(['PSNR sparse = ' num2str(psnr1) ' dB']);
disp(['PSNR Bicubic = ' num2str(psnr2) ' dB']);
disp(['SSIM sparse = ' num2str(ssim1) ]);
disp(['SSIM Bicubic = ' num2str(ssim2) ]);




