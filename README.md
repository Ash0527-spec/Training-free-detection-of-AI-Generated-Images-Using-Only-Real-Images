# Training-free-detection-of-AI-Generated-Images-Using-Only-Real-Images

The rapid advancement of image generation technologies has led to highly realistic synthetic images, whose spread poses a threat to public trust and security. In real-world scenarios, where the generative model behind a fake image is unknown, an effective detector should be capable of identifying out-of-distribution generated images. Current methods typically focus on identifying common artifacts across different generative models. However, these methods often erase a substantial portion of image information, resulting in detection failures even when fakes are visually distinguishable. Our experiments show that autoregressive features are effective for detecting generated images, allowing us to eliminate the reliance on artifacts and preserve the image information. Building on this finding, we propose Real-based Autoregressive Image Detector (RAID), a training-free method that relies exclusively on real images. We generate anomalous samples by reconstructing real images with a variational autoencoder (VAE), rather than using AI-generated images. RAID achieves an average accuracy of 93.07\% across 18 generative models, surpassing the SoTA by 2.36\%. RAID demonstrates robust performance under various perturbations, with a particularly notable 20.02\% improvement in detection accuracy under common JPEG compression compared to the SoTA. 


## Feature Extraction
We use **AIMv1** to extract image features. The original code can be found at [AIM GitHub repository](https://github.com/apple/ml-aim). We made certain modifications to simplify the environment configuration.

You can use `resultx_important_indices.npy` to select features for SVM

```bash

# Download the AIM model checkpoint for the backbone weights
wget -O aim_3b_5bimgs_attnprobe_backbone.pth <url of aim_3b_5bimgs_attnprobe_backbone> # the url can be found at https://github.com/apple/ml-aim/tree/main/aim-v1

# Download the second AIM model checkpoint for the head layer weights (best layers)
wget -O aim_3b_5bimgs_attnprobe_head_best_layers.pth <url of aim_3b_5bimgs_attnprobe_head_best_layers> # # the url can be found at https://github.com/apple/ml-aim/tree/main/aim-v1

python feature_extract.py --input_path [your_image_root] --output_path [your_npy_root] --backbone_ckpt_path ./aim_3b_5bimgs_attnprobe_backbone.pth --head_ckpt_path ./aim_3b_5bimgs_attnprobe_head_best_layers.pth
```

## Reconstruct Real Images
We use vae in stable diffusion to reconstruct images. You can also use other generative models.
```bash
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4

# you can also use HF-mirror to clone
# git clone https://hf-mirror.com/CompVis/stable-diffusion-v1-4

python reconstruct.py --repo_id ./stable-diffusion-v1-4 --input_dir [your_real_image] --output_dir [your_reconstruct_dir]
```

## Train and Test
We have extracted features of 10,000 real images and their reconstructed images, which are stored in `./data` and can be used directly for training. 

We provide test image features (.npy) in the following links:
- Baidu Netdisk: [Download test image features from BaiduNetdisk](https://pan.baidu.com/s/1jFyLZ8sFNh2pN-sX9qUQDg?pwd=23pz).
- Google Drive: [Download test image features from Google Drive](https://drive.google.com/file/d/1nsrkyfLX9dqtW7xnwTrQQhCSA3U2kgvj/view?usp=sharing).

The test images are from this [GitHub repository](https://github.com/Ekko-zn/AIGCDetectBenchmark) and can be download from [BaiduNetdisk](https://pan.baidu.com/s/1dZz7suD-X5h54wCC9SyGBA?pwd=l30u#list/path=%2F).
The SDXL and Flux-generated images constructed by us can be downloaded from [BaiduNetdisk](https://pan.baidu.com/s/1SOvaJULLTvnIQG5EVb67Aw?pwd=htr6) or [Google Drive](https://drive.google.com/file/d/1oIw0dWOWA8xrg9VlzYc_a8nnO4frsCuw/view?usp=sharing)

The origin image of .npy can be downloader from [BaiduNetdisk](https://pan.baidu.com/s/1secHnpVj0_a82vP17MCClQ?pwd=nw1h) or [Google Drive](https://drive.google.com/file/d/146GQNq3zrLIApzDFvEXx9xRc0iFC2mGs/view?usp=drive_link)
```bash
# Train
python train_val.py \
  --train_path /path/to/train_data \
  --test_root /path/to/test_data \
  --result_file /path/to/results.txt \
  --kernel rbf \
  --probability \
  --n_jobs -1 \
  --verbose
```

If you want to train on your own dataset, you can reconstruct real images using `reconstruct.py` and extract features using `feature_extract.py`
```bash
python reconstruct.py --repo_id ./stable-diffusion-v1-4 --input_dir [your_real_image_dir] --output_dir [your_reconstruct_image_dir]
python feature_extract.py --input_path [your_image_root] --output_path [your_npy_root] --backbone_ckpt_path ./aim_3b_5bimgs_attnprobe_backbone.pth --head_ckpt_path ./aim_3b_5bimgs_attnprobe_head_best_layers.pth
```














