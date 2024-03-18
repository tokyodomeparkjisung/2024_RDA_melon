import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image

# 이미지 로드 (두 개의 로컬 이미지 파일 경로 사용)
image_path1 = "/Users/yun-seungcho/Downloads/stitching_data4/testing/input1/000000.jpg"
image_path2 = "/Users/yun-seungcho/Downloads/stitching_data4/testing/input2/000000.jpg"
image1 = Image.open(image_path1).convert("RGB")
image2 = Image.open(image_path2).convert("RGB")

# 이미지의 높이와 너비 추출
image_array = np.array(image1)
img = torch.tensor(image_array)
#print(img)

img_h, img_w, _ = img.size() # 텐서의 크기는 (높이, 너비, 채널의 수 = 3)

#print(img_h, img_w)

# 사전 학습된 Vision Transformer 모델과 특징 추출기 초기화
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-384')
model = ViTModel.from_pretrained('google/vit-base-patch16-384')

# 두 이미지에서 특징 추출
inputs = feature_extractor(images=[image1, image2], return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

# `last_hidden_states`는 모델의 최종 특징 벡터를 포함하고 있습니다.
# 이 벡터를 다양한 작업에 사용할 수 있습니다.
# 여기서 last_hidden_states의 차원은 [2, sequence_length, hidden_size]가 됩니다.

class TransformationRegressor(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(TransformationRegressor, self).__init__()
        # feature_dim: 특징 벡터의 차원
        # output_dim: 출력 차원
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 2, 1024),  # 두 특징 벡터를 결합
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, feature_vec1, feature_vec2):
        # feature_vec1과 feature_vec2는 두 이미지의 특징 벡터
        x = torch.cat((feature_vec1, feature_vec2), dim=1)  # 특징 벡터 결합
        x = self.fc(x)  # 변환 행렬 예측
        return x


feature_vec = last_hidden_states.flatten(start_dim = 1)
feature_dimension = len(feature_vec[0])

# 모델 초기화 (아핀 변환을 위해 output_dim=6)
model = TransformationRegressor(feature_dim=feature_dimension, output_dim=6)

feature_vec1 = feature_vec[0]
feature_vec2 = feature_vec[1]

feature_vec1 = feature_vec[0].unsqueeze(0)  # [hidden_size] -> [1, hidden_size]
feature_vec2 = feature_vec[1].unsqueeze(0)  # [hidden_size] -> [1, hidden_size]

# 변환 행렬 예측
predicted_transformation = model(feature_vec1, feature_vec2)
print(predicted_transformation)

