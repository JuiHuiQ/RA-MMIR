
# ATGANFusion

This is Pytorch implementation of "[AT-GAN: A generative adversarial network with attention and transition for infrared and visible image fusion](https://www.sciencedirect.com/science/article/abs/pii/S156625352200255X)"



## Note


- ### *Requirements*
  *1. python 3.8.13*
  
  *2. pytorch 1.12.1*
  
  *3. xlwt 1.3.0*


- #### *Train*
  *1.  Prepare training dataset in ./Train_vi and ./Train_ir.*
  
  *2.  `python ./python/main.py`*

- #### *Test*
  *1.  Prepare test dataset in ./prepare_Dataset/XXX/vi and ./prepare_Dataset/XXX/vi.(XXX is datasetname)*
  
  *2.  Generate fused RGB image*
   `python test_color.py`
  
  *2.  Generate fused Gray image*
  `python test_gray.py`



## Acknowledgment
***Many thanks to the following implementation for improving the ATGANFusion:***

- [DenseFuse](https://github.com/hli1221/imagefusion_densefuse)
- [FusionGAN](https://github.com/jiayi-ma/FusionGAN) 
- [PMGI](https://github.com/HaoZhang1018/PMGI_AAAI2020)
- [U2Fusion](https://github.com/hanna-xu/U2Fusion)
- [GANMcC](https://github.com/HaoZhang1018/GANMcC)
- [Res2Fusion](https://github.com/Zhishe-Wang/Res2Fusion)
- [RFN_Nest](https://github.com/hli1221/imagefusion-rfn-nest)
- [SeAFusion](https://github.com/Linfeng-Tang/SeAFusion)
- [SwinFuse](https://github.com/search?q=SwinFuse)
- [SwinFusion](https://github.com/Linfeng-Tang/SwinFusion)
- [TarDAL](https://github.com/JinyuanLiu-CV/TarDAL)


## Citation
***If this work is helpful to you, please cite it as:***

```
@article{rao2023gan,
  title={AT-GAN: A generative adversarial network with attention and transition for infrared and visible image fusion},
  author={Rao, Yujing and Wu, Dan and Han, Mina and Wang, Ting and Yang, Yang and Lei, Tao and Zhou, Chengjiang and Bai, Haicheng and Xing, Lin},
  journal={Information Fusion},
  volume={92},
  pages={336--349},
  year={2023},
  publisher={Elsevier}
}
```

