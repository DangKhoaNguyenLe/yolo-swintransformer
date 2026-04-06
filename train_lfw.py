import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import timm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

try:
    from sklearn.datasets import fetch_lfw_people
except ImportError:
    print("Vui lòng cài đặt scikit-learn: pip install scikit-learn")
    exit(1)

# --- Cấu hình ---
MODEL_SAVE_PATH = "database/lfw_swin.pth"
METRICS_SAVE_PATH = "lfw_training_metrics.png"
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4

class LFWDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Ảnh từ sklearn có shape (h, w, 3) và dải giá trị float 0-255
        img = self.images[idx]
        # Ép kiểu về uint8
        img = np.clip(img, 0, 255).astype(np.uint8)
        # Chuyển thành PIL Image
        img_pil = Image.fromarray(img)
        
        label = self.labels[idx]

        if self.transform:
            img_pil = self.transform(img_pil)

        return img_pil, label

def main():
    print("=" * 60)
    print("BẮT ĐẦU TIỀN HUẤN LUYỆN (PRE-TRAIN) TRÊN TẬP LFW")
    print("=" * 60)
    print("Đang tải dữ liệu Labeled Faces in the Wild (LFW)...")
    print("Lưu ý: Quá trình này sẽ mất một chút thời gian (sẽ down khoảng ~200MB nếu là lần đầu).")

    # Lấy những người có tối thiểu 20 ảnh để giảm thiểu các class quá hiếm
    lfw_people = fetch_lfw_people(min_faces_per_person=20, color=True, resize=1.0)
    X = lfw_people.images
    y = lfw_people.target
    target_names = lfw_people.target_names
    num_classes = len(target_names)

    print(f"Đã tải được {X.shape[0]} tấm ảnh khuôn mặt từ {num_classes} nhân vật.")

    # Khởi tạo data transform chuẩn Swin_tiny_224
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = LFWDataset(X, y, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # Khởi tạo mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nĐang huấn luyện sức mạnh trên thiết bị: {device}")
    
    # Nạp mô hình có sẵn trọng số
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=num_classes)
    model = model.to(device)

    # Khởi tạo optimizer và loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Mảng để lưu lại lịch sử
    train_loss_history = []
    train_acc_history = []

    # Tiến hành Pre-Train
    print("\n Bắt đầu quá trình Khổ Luyện (Pre-Training)...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        # Thanh tiến trình
        loop = tqdm(dataloader, leave=False, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(dtype=torch.long, device=device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_preds.double() / total_samples
        
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.item())

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

    # Lưu lại trọng số tối thượng
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nĐã lưu trọng số Face-Expert tại: {MODEL_SAVE_PATH}")

    # Vẽ và xuất biểu đồ
    plt.figure(figsize=(10, 5))

    # Biểu đồ Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPOCHS+1), train_loss_history, marker='o', color='purple', label='LFW Train Loss')
    plt.title('Biểu đồ Loss LFW (CrossEntropy)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Biểu đồ Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, NUM_EPOCHS+1), train_acc_history, marker='o', color='green', label='LFW Train Accuracy')
    plt.title('Biểu đồ Accuracy (Độ chính xác)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(METRICS_SAVE_PATH)
    plt.close()
    
    print(f"Đã xuất tập tin biểu đồ đánh giá tại: {METRICS_SAVE_PATH}")
    print("CHÚC MỪNG MÔ HÌNH SWIN CỦA BẠN ĐÃ TRỞ THÀNH CHUYÊN GIA GƯƠNG MẶT!")

if __name__ == "__main__":
    main()
