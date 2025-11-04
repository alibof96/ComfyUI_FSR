# ComfyUI_FlashVSR
[FlashVSR](https://github.com/OpenImagingLab/FlashVSR): Towards Real-Time Diffusion-Based Streaming Video Super-Resolution,this node ,you can use it in comfyUI

# Update
* **Replaced Block-Sparse-Attention with Sparse_Sage**: Now supports RTX 50 series GPUs (sm_75 to sm_120) without needing to compile custom kernels. This removes the dependency on the difficult-to-compile Block-Sparse-Attention library.
* add full mode [lightx2v vae encoder](https://huggingface.co/lightx2v/Autoencoders/tree/main) support（only lightvaew2_1.pth,taew2_1.pth,lighttaew2_1.pth） and [Wan2.1-VAE-upscale2x](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x) support    
* 新增lightx2v 加速vae decoder支持和Wan2.1-VAE-upscale2x 放大decoder支持，只是在full 模式下有效，light的加速模型目前只支持（lightvaew2_1.pth  #32.2M,taew2_1.pth,lighttaew2_1.pth） 三个文件 
# Tips
*  满足部分网友需要超分单张图片的奇怪要求,默认输出25帧1秒的视频，详见示例，现在使用 Sparse_Sage 实现，支持 RTX 50 系列 GPU (sm_75 到 sm_120)，无需编译自定义内核； 
*  同步tiny的专属long模式  
*  新增切片视频路径加载节点，输入保存切片视频的路径，开启自动推理，即可推理完路径所有视频； 
*  修复输入图像归一化处理错误导致无法复现官方的问题，分离decoder，新增关键点模型卸载和OOM处理，包括处理超长视频向量的OOM，同步官方local range的修改，新增小波模式下的加减帧处理（项目一作大佬提的）；
*  local_range=7这个是会最清晰，local_range=11会比较稳定，color fix 推荐用小波（没重影）； 
*  方法是基于现有prompt.pt训练的，所以外置cond没有必要已经去掉，新增tile 和 color fix 选项，tile关闭质量更高，需要VRam更高，corlor fix对于非模糊图片可以试试。修复图片索引数不足的错误。  
*  Choice vae infer full mode ，encoder infer tiny mode 选择vae跑full模式 效果最好，tiny则是速度，数据集基于4倍训练，所以1 scale是不推荐的；  
*  如果觉得项目有用，请给官方项目[FlashVSR](https://github.com/OpenImagingLab/FlashVSR) 打星； if you Like it ， star the official project [link](https://github.com/OpenImagingLab/FlashVSR)

  
1.Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_FlashVSR

```

2.requirements  
----

```
pip install -r requirements.txt
```

**Note:** This version uses Sparse_Sage instead of Block-Sparse-Attention, which means:
- ✅ No need to compile custom kernels
- ✅ Works on RTX 50 series GPUs (sm_75 to sm_120)
- ✅ Easier installation with just `pip install -r requirements.txt`
- The implementation uses Triton for efficient sparse attention without custom CUDA kernels

3.checkpoints 
----

* 3.1 [FlashVSR](https://huggingface.co/JunhaoZhuang/FlashVSR/tree/main)   all checkpoints 所有模型，vae 用常规的wan2.1  
* 3.2 emb  [posi_prompt.pth](https://github.com/OpenImagingLab/FlashVSR/tree/main/examples/WanVSR/prompt_tensor)  4M而已
* 3.3 [lightvaew2_1.pth](https://huggingface.co/lightx2v/Autoencoders/tree/main) and [diffusion_pytorch_model.safetensors](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x/tree/main/diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1)
  
```
├── ComfyUI/models/FlashVSR
|     ├── LQ_proj_in.ckpt
|     ├── TCDecoder.ckpt
|     ├── diffusion_pytorch_model_streaming_dmd.safetensors
|     ├── posi_prompt.pth
├── ComfyUI/models/vae
|        ├──Wan2.1_VAE.pth
|        ├──lightvaew2_1.pth  #32.2M  or taew2_1.pth,lighttaew2_1.pth
|        ├──Wan2.1_VAE_upscale2x_imageonly_real_v1_diff.safetensors  # rename from diffusion_pytorch_model.safetensors
```
  

# Example
* upscale2x and ligth lightvaew2_1.pth
![](https://github.com/smthemex/ComfyUI_FlashVSR/blob/main/example_workflows/example_decoder.png)
* single image VSR 
![](https://github.com/smthemex/ComfyUI_FlashVSR/blob/main/example_workflows/example_s.png)
* full old node 
![](https://github.com/smthemex/ComfyUI_FlashVSR/blob/main/example_workflows/example18.png)
* tiny new
![](https://github.com/smthemex/ComfyUI_FlashVSR/blob/main/example_workflows/example1022.png)
* video files loop
![](https://github.com/smthemex/ComfyUI_FlashVSR/blob/main/example_workflows/example.png)

# Acknowledgements
[DiffSynth Studio](https://github.com/modelscope/DiffSynth-Studio)  
[Sparse_SageAttention_API](https://github.com/jt-zhang/Sparse_SageAttention_API) - Sparse attention implementation  
[ComfyUI-FlashVSR_Ultra_Fast](https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast) - Reference implementation  
[taehv](https://github.com/madebyollin/taehv)  

# Citation
```
@misc{zhuang2025flashvsrrealtimediffusionbasedstreaming,
      title={FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution}, 
      author={Junhao Zhuang and Shi Guo and Xin Cai and Xiaohui Li and Yihao Liu and Chun Yuan and Tianfan Xue},
      year={2025},
      eprint={2510.12747},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.12747}, 
}

```
lightx2v
```
@misc{lightx2v,
 author = {LightX2V Contributors},
 title = {LightX2V: Light Video Generation Inference Framework},
 year = {2025},
 publisher = {GitHub},
 journal = {GitHub repository},
 howpublished = {\url{https://github.com/ModelTC/lightx2v}},
}
```
