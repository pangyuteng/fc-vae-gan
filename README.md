# fc-vae-gan

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