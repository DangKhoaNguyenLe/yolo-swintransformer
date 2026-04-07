import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import timm
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Cấu hình Mặc định ---
MODEL_SAVE_PATH = "database/custom_swin.pth"
METRICS_SAVE_PATH = "custom_training_metrics.png"
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 3e-4

def main(dataset_path):
    print("=" * 60)
    print(" BẮT ĐẦU HUẤN LUYỆN SWIN TRANSFORMER VỚI CUSTOM DATASET")
    print("=" * 60)
    
    if not os.path.exists(dataset_path):
        print(f" LỖI: Không tìm thấy thư mục dataset tại {dataset_path}")
        return

    # 1. Tăng cường dữ liệu (Data Augmentation thông minh hơn)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomRotation(degrees=15),          # Xoay ảnh nhẹ
        transforms.ColorJitter(brightness=0.3,          # Thay đổi độ sáng, tương phản
                               contrast=0.3, 
                               saturation=0.3, 
                               hue=0.1), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])

    print(f"Đang tải dữ liệu từ {dataset_path}...")
    dataset = ImageFolder(root=dataset_path, transform=transform_train)
    num_classes = len(dataset.classes)

    print(f"Đã tìm thấy {len(dataset)} ảnh thuộc {num_classes} nhãn/người.")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)

    # 2. Khởi tạo mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nSử dụng thiết bị huấn luyện: {device}")
    
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=num_classes)
    model = model.to(device)

    # 3. Tối ưu hóa (Optimizer & Scheduler)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Cosine Annealing giảm từ từ learning rate về mức thấp
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    train_loss_history = []
    train_acc_history = []
    
    best_loss = float('inf')
    best_acc = 0.0

    print("\n Bắt đầu quá trình Huấn Luyện...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        loop = tqdm(dataloader, leave=False, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(torch.long, device=device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels)
            total_samples += labels.size(0)

            # Lấy learning rate hiện hành
            current_lr = optimizer.param_groups[0]['lr']
            loop.set_postfix(loss=loss.item(), lr=current_lr)
            
        scheduler.step()

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_preds.double() / total_samples
        
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.item())

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f} - LR: {current_lr:.6f}")

        # 4. Save Best Model
        if epoch_acc >= best_acc and epoch_loss < best_loss:
            best_acc = epoch_acc
            best_loss = epoch_loss
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f" --> Đã lưu phiên bản mô hình tốt nhất (Best Model)!")

    print(f"\n Lưu tại đường dẫn: {MODEL_SAVE_PATH}")

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPOCHS+1), train_loss_history, marker='o', color='purple', label='Train Loss')
    plt.title('Biểu đồ Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, NUM_EPOCHS+1), train_acc_history, marker='o', color='green', label='Train Accuracy')
    plt.title('Biểu đồ Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(METRICS_SAVE_PATH)
    plt.close()
    
    print(f"\n Đã xuất biểu đồ đánh giá tại: {METRICS_SAVE_PATH}")
    print(" HOÀN TẤT HUẤN LUYỆN!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Swin Transformer trên Custom Dataset")
    parser.add_argument("--dataset_path", type=str, default="dataset/MaskedFace-Net", help="Đường dẫn đến thư mục dataset (mỗi class 1 thư mục con)")
    args = parser.parse_args()
    
    main(args.dataset_path)
