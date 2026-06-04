import bisect
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import clip
import random
import torch
import pickle
import os

class ImagePaths(Dataset):
    def __init__(self, training_images_list_file, transform, use_precomputed_features=False, feature_cache_dir=None):
        # 数据集路径设置
        if 'oxford' in training_images_list_file:
            dataset = 'oxford'
            self.root = '/root/autodl-tmp/DiT/dataset/flower'
        elif 'cub' in training_images_list_file:
            dataset = 'cub'
            self.root = '/disk/yesenmao/ldm/RAT_diffusion/dataset/cub/'
        elif 'coco' in training_images_list_file:
            dataset = 'coco'
            self.root = '/disk/yesenmao/ldm/RAT_diffusion/dataset/coco/'
        elif 'tangka' in training_images_list_file:
            dataset = 'tangka'
            self.root = '/root/autodl-tmp/DiT/dataset/tangka'
        else:
            raise ValueError(f"Unknown dataset in {training_images_list_file}")

        # 加载数据
        with open(training_images_list_file, 'rb') as f:
            self.caption = pickle.load(f)
        self._length = len(self.caption)
        
        # 图像预处理
        self.preprocessor = transform
        
        # 文本处理相关
        self.use_precomputed_features = use_precomputed_features
        self.feature_cache_dir = feature_cache_dir
        
        if not use_precomputed_features:
            # 只在需要实时tokenize时加载CLIP模型
            device = "cpu"
            self.clip_model, _ = clip.load("ViT-B/32", device=device)
            self.clip_model.eval()
        
        # 如果使用预计算特征，检查缓存目录
        if use_precomputed_features and feature_cache_dir:
            os.makedirs(feature_cache_dir, exist_ok=True)
            self.feature_cache = {}
            
            # 预加载已存在的特征
            for i in range(self._length):
                cache_path = os.path.join(feature_cache_dir, f"{i}.pt")
                if os.path.exists(cache_path):
                    self.feature_cache[i] = torch.load(cache_path)

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = self.preprocessor(image)
        return image

    def __getitem__(self, i):
        cap = self.caption[i]
        path = os.path.join(self.root, cap[0])
        image = self.preprocess_image(path)
        
        if self.use_precomputed_features:
            # 使用预计算特征
            if i in self.feature_cache:
                text_features = self.feature_cache[i]
            else:
                # 实时计算并缓存
                captions = cap[1]
                index = random.randint(0, len(captions)-1)
                caption = captions[index]
                
                with torch.no_grad():
                    text_input = clip.tokenize([caption], truncate=True)
                    text_features = self.clip_model.encode_text(text_input).squeeze(0)
                
                if self.feature_cache_dir:
                    cache_path = os.path.join(self.feature_cache_dir, f"{i}.pt")
                    torch.save(text_features, cache_path)
                    self.feature_cache[i] = text_features
        else:
            # 实时tokenize文本
            captions = cap[1]
            index = random.randint(0, len(captions)-1)
            caption = captions[index]
            text_features = clip.tokenize([caption], truncate=True).squeeze(0)
        
        return image, text_features