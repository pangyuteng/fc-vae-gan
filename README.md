### Fully Convolutional Variational Autoencoder.

#### tl,td;

This repo contains model definition and script train a fully convolutional variational autoendoer (FC-VAE).  As oppose to a vanilla VAE which a 2D image can be mapped to a n-d latent vector, a FC-VAE maps a 2D image to a pxqxn latent matrix.

#### Abstract

use of bayes seems like a good idea to create more generaized models.
for training adversial network seems to also be a good i dea to make learnt weights more generalized.
For VAE or VAE-GAN,  will only be able to convert images to one latent vector.  However, i hypothesize that the latent vector need not be a vector (1,n), but a matrix, i.e. pxqxn, thus mapping patches of images to a vector. 

For pixel classification, i.e. medical application like texture classification or sementic segmentation (U-net, hypercolumn style...), 

#### Introduction


##### 





### fc-vae-gan

* [x] generate example pascal dataset
    `python data/pascal.py`
* [x] train fc-vae-gan with pascal dataset
    `python fcvaegan.py`
* [x] generate latent space and perform dimension reduction
    `python gen_latent.py`
    `python train_tsne.py`
    leftovers: 
* [ ] visualize 2d mapped latent variables with using label as overlay color


```
CUDA_VISIBLE_DEVICES=0 python fcvaegan.py;CUDA_VISIBLE_DEVICES=0 python gen_latent.py;CUDA_VISIBLE_DEVICES=0 python train_tsne.py
```