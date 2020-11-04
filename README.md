# Satellite-Image-Super-resolution
*****************************************************************
* Demo Codes For Satellite Image Super-resolution Based on K-SVD 
Dictionary Training and Patch-wise Sparse Coding          
*****************************************************************

Reference:
================================================================

Mullah H.U., Deka B. (2018) A Fast Satellite Image Super-Resolution Technique Using Multicore Processing. In: Hybrid Intelligent Systems. HIS 2017. Advances in Intelligent Systems and Computing, vol 734. Springer, Cham. 

H. U. Mullah and B. Deka, "Parallel Multispectral Image Super-resolution Based on Sparse Representations," 2019 2nd International Conference on Innovations in Electronics, Signal Processing and Communication (IESC), IEEE

J. Yang et al. Image super-resolution as sparse representation of raw image patches. CVPR 2008.

================================================================

main.m: code for image super-resolution based on sparse sparse representaion

1. The  code is for upscaling factor of 2. For other magnification factors, change the upscale value. Note the code is a little different from what presented in the paper. 

2. You can run the code for any patch size, e.g 3, 5, 7, 9, etc.

3. If you want to train your own dictionary, replace the training images in subfolder "Data/Train" by yours.
