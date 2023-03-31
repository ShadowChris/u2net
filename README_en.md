# U^2-Net - Portrait matting
This repository explores possibilities of using the original u^2-net model for portrait matting. The training code is rewritten and optimised for performance and reliability:
- allow mixed precision training
- allow channels_last to increase performance on tensor cores (initial observations show a performance increase of roughly 80%)
- prefetch batches to the GPU
- more efficient tensor conversations
- configuration management with hydra
- tensorboard logging
- augmentations, dataset weights, suppert for aisegment rgb masks, ...
- streamlit demo frontend using pymatting for foreground estimations
- a lot more

I changed the loss function of the original proposal to allow two different kind of masks as ground truth: rough masks and refined masks where refined masks can be for example alpha mattes synthesized using pymatting (my experiments were quite successful). This however is just a proof of concept and the refined masks itself are quite noisy. The different losses are in fact conflicting a bit to each other however initial results show quite interesting results.

## Example & pretrained model

![](/docs/sample_1.jpeg) ![](/docs/sample_1_mask.jpeg) ![](/docs/sample_1_blended.jpeg)

---

The model is able to pickup details quite precisely (e.g. hair regions) with very little overshoot however it also shows weaknesses of this training approach e.g. the confusion at the left shoulder. The model has only been trained for a couple of epochs on a synthesized dataset and is not ready for production use. However feel free to re-use it for any similar experiments.

The provided checkpoint is compatible with the original u^2-net implementation and code.

[OneDrive](https://1drv.ms/u/s!Am46hiIqzupmhq5V_Zt36ACGjmI6Xg?e=WPhq7y)

## Citation
```
@InProceedings{Qin_2020_PR,
title = {U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection},
author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar and Jagersand, Martin},
journal = {Pattern Recognition},
volume = {106},
pages = {107404},
year = {2020}
}
```
