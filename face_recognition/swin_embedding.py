import timm
import torch
import cv2
import numpy as np
import os

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    num_classes=0
)

# Thử nạp trọng số siêu cấp (LFW Pre-trained) trước
LFW_WEIGHTS = "database/lfw_swin.pth"
CUSTOM_WEIGHTS = "database/custom_swin.pth"

if os.path.exists(LFW_WEIGHTS):
    try:
        model.load_state_dict(torch.load(LFW_WEIGHTS, map_location="cpu"), strict=False)
        print("Đã nạp được trọng số TỐI THƯỢNG (LFW Pre-trained)!")
    except Exception as e:
        print(f"Lỗi khi nạp LFW weights: {e}")
elif os.path.exists(CUSTOM_WEIGHTS):
    try:
        model.load_state_dict(torch.load(CUSTOM_WEIGHTS, map_location="cpu"), strict=False)
        print("Đã nạp được trọng số Swin Transformer tùy chỉnh (Fine-tuned local)!")
    except Exception as e:
        print(f"Lỗi khi nạp trọng số tùy chỉnh: {e}. Hệ thống sẽ tiếp tục dùng weights mặc định.")

model.eval()

model.eval()

def get_embedding(face):
    # 1. OpenCV mặc định là BGR, nhưng Swin Transformer học mặt người chuẩn RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # 2. Thay đổi size về 224x224 (Bắt buộc cho Swin_tiny_224)
    face = cv2.resize(face, (224, 224))

    # 3. Đưa dải giá trị màu (0 -> 255) về dải tỷ lệ (0.0 -> 1.0)
    face = face / 255.0

    # 5. Cấu trúc lại bộ nhớ từ (Cao, Rộng, Kênh) sang (Kênh, Cao, Rộng)
    face = np.transpose(face, (2, 0, 1))

    # 6. Biến đổi mảng thành Tensor của PyTorch và thêm chiều Batch
    face = torch.tensor(face).float().unsqueeze(0)

    # 7. Tính toán trích xuất đặc trưng khuôn mặt (Không update gradient)
    with torch.no_grad():
        embedding = model(face)

    return embedding.numpy()