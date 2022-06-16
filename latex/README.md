

We propose a self-supervised latent feature generation framework based on a combination of recent deep learning GAN techniques.

+ Variational Autoencoder
+ Self Attention
+ Patch GAN
+ Fully convolutional frameworks

We constrain the architecture to be trainable on GPUs with 12GB RAM, thus making this architecture consumer-GPU friendly.

data-generator -  list of nii.gz
data-augmentation cutout? rotate? 

physical voxel size - 2^3mm

when training - crop the input/output shape
during inference - use full image size.

x-> 64------------->                 --->64 -> x_hat
       |--32------->   z       --->32 |
            |--16-->      -->16-|


```

Autoencoding beyond pixels using a learned similarity metric
Anders Boesen Lindbo Larsen, Søren Kaae Sønderby, Hugo Larochelle, Ole Winther
https://arxiv.org/abs/1512.09300

Image-to-Image Translation with Conditional Adversarial Networks
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros
https://arxiv.org/abs/1611.07004

Self-Attention Generative Adversarial Networks
Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena
https://arxiv.org/abs/1805.08318

Fully Convolutional Networks for Semantic Segmentation
Jonathan Long, Evan Shelhamer, Trevor Darrell
https://arxiv.org/abs/1411.4038

Deep Autoencoding Models for Unsupervised Anomaly Segmentation in Brain MR Images
Christoph Baur, Benedikt Wiestler, Shadi Albarqouni, Nassir Navab
https://arxiv.org/abs/1804.04488

Latent traits of lung tissue patterns in former smokers derived by dual channel deep learning in computed tomography images
Frank Li, Jiwoong Choi, et al
https://pubmed.ncbi.nlm.nih.gov/33649381

```



thoughts:

```

Future directions: conditional vae gan, feature entanglement, domain adaptation, feature dimension reduction...


```