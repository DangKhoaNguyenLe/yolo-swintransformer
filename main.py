#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Điểm khởi đầu chính của Hệ thống Nhận diện Gương mặt.
Hỗ trợ các chức năng: Đăng ký, Nhận diện và Huấn luyện.
"""

# ── Thư viện chuẩn ──────────────────────────────────────────────
import pickle
import shutil
from pathlib import Path

# ── Thư viện bên thứ ba ─────────────────────────────────────────
import cv2
import numpy as np
# ── Module nội bộ dự án ─────────────────────────────────────────
from face_detection.detect_face import detect_face
from face_recognition.swin_embedding import get_embedding
from utils.similarity import compare


# Danh sách góc cần thu thập khi đăng ký: (tên_góc, nhãn_hiển_thị, số_ảnh_mục_tiêu)
REGISTER_ANGLES = [
    ("CENTERED", "Nhìn thẳng vào camera", 3),
    ("RIGHT",    "Quay đầu sang PHẢI",    3),
    ("LEFT",     "Quay đầu sang TRÁI",    3),
    ("UP",       "Ngẩng đầu lên",          2),
    ("DOWN",     "Cúi đầu xuống",          2),
]

class HeadPoseDetector:
    """Phát hiện góc xoay đầu chuẩn xác bằng OpenCV Haarcascades.
    Sử dụng cascade đệm sẵn của OpenCV để định vị chính xác KHUÔN MẶT,
    khắc phục lỗi của YOLO khi nó nhận diện cả người làm lệch trọng tâm góc.
    """
    def __init__(self):
        # Nạp cascade nhận mặt thẳng và mặt nghiêng (profile)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        # Thêm cascade dò mắt để bắt cực chuẩn góc Cúi/Ngẩng
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def get_angle(self, frame: np.ndarray) -> tuple[str, tuple]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # 1. Thử phát hiện mặt trực diện (Dành cho góc thẳng, lên, xuống)
        front_faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(front_faces) > 0:
            x, y, w, h = front_faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            
            # CHIẾN LƯỢC MỚI CHỐNG LỖI GÓC TRÊN DƯỚI: Sử dụng định vị MẮT
            # Khi ngẩng CẰM lên: Vùng mặt dưới (cằm/cổ) chiếm diện tích 
            # -> mắt bị đẩy sát lên nóc khung hình.
            # Khi cúi XUỐNG: Trán và đỉnh đầu lồi ra 
            # -> mắt bị đẩy sệ xuống dưới đáy khung hình.
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
            if len(eyes) >= 1:
                # Lấy Y trung bình của tất cả mắt tìm thấy (chuẩn hóa theo chiều cao mặt)
                avg_eye_y_norm = sum([ey + eh/2.0 for ex, ey, ew, eh in eyes]) / len(eyes) / h
                
                # Thông thường mắt nằm ở khoảng 40% - 45% từ đỉnh mặt.
                if avg_eye_y_norm < 0.38: return "UP", (x, y, w, h)
                if avg_eye_y_norm > 0.48: return "DOWN", (x, y, w, h)
                return "CENTERED", (x, y, w, h)

            # DỰ PHÒNG: Nếu người dùng đeo kính loá hoặc nhắm mắt -> Dùng Canny
            edges = cv2.Canny(roi_gray, 50, 150)
            M = cv2.moments(edges)
            if M["m00"] != 0:
                cY = int(M["m01"] / M["m00"])
                norm_y = (cY / h) * 2 - 1 
                # Chỉnh lại biên độ siêu nhạy cho AI dự phòng
                if norm_y < -0.05: return "UP", (x, y, w, h)
                if norm_y >  0.06: return "DOWN", (x, y, w, h)
            return "CENTERED", (x, y, w, h)

            
        # 2. Nếu không thấy mặt trực diện -> Tìm góc quay ngang (Nghiêng trái/phải)
        # Profile cascade mặc định của OpenCV bắt mặt quay sang một bên. 
        # Do frame đã được flip(1) thành mirror camera, ta đảo lại nhãn cho phù hợp.
        profile_left = self.profile_cascade.detectMultiScale(gray, 1.3, 5)
        if len(profile_left) > 0:
            x, y, w, h = profile_left[0]
            return "LEFT", (x, y, w, h)
            
        # Lật lại ảnh để bắt mạn ngược lại
        flipped_gray = cv2.flip(gray, 1)
        profile_right = self.profile_cascade.detectMultiScale(flipped_gray, 1.3, 5)
        if len(profile_right) > 0:
            x, y, w, h = profile_right[0]
            # Convert lại tọa độ sau khi lật
            img_w = gray.shape[1]
            actual_x = img_w - (x + w)
            return "RIGHT", (actual_x, y, w, h)
            
        return "NONE", None




class FaceRecognitionSystem:
    """Hệ thống nhận diện gương mặt đa góc độ."""

    # ── Hằng số mặc định ────────────────────────────────────────
    DATABASE_PATH   = "database/embeddings.pkl"  # Đường dẫn file lưu embeddings
    DATASET_PATH    = "dataset/persons"           # Thư mục chứa ảnh từng người
    RECOG_THRESHOLD = 0.6                         # Ngưỡng tương đồng để xác nhận nhận diện
    CAPTURE_INTERVAL = 2   # Chụp ảnh mỗi N frame khi phát hiện đúng góc

    def __init__(self):
        # Khởi tạo đường dẫn từ hằng số (dễ ghi đè trong subclass)
        self.database_path = self.DATABASE_PATH
        self.dataset_path  = self.DATASET_PATH
        self.recognition_threshold = self.RECOG_THRESHOLD

        # Tải database embeddings từ file (nếu đã có)
        self.database: dict = self._load_database()

    # ════════════════════════════════════════════════════════════
    # Quản lý Database
    # ════════════════════════════════════════════════════════════

    def _load_database(self) -> dict:
        """Tải database embeddings từ file pickle.

        Returns:
            dict: {tên_người: [embedding, ...]} hoặc {} nếu chưa có.
        """
        db_path = Path(self.database_path)
        if not db_path.exists():
            return {}

        try:
            # Dùng context manager để đảm bảo file được đóng đúng cách
            with open(db_path, "rb") as f:
                data = pickle.load(f)
            print(f"Đã tải database: {len(data)} người")
            return data
        except Exception as e:
            print(f"Lỗi khi tải database: {e}")
            return {}

    def _save_database(self) -> None:
        """Lưu database embeddings ra file pickle."""
        db_path = Path(self.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(db_path, "wb") as f:
                pickle.dump(self.database, f)
            print("Database đã được lưu!")
        except Exception as e:
            print(f"Lỗi khi lưu database: {e}")

    # ════════════════════════════════════════════════════════════
    # Đăng ký gương mặt
    # ════════════════════════════════════════════════════════════

    def register_face(self) -> None:
        """Đăng ký gương mặt mới với hướng dẫn đa góc (dùng đếm ngược tự động).

        Người dùng được yêu cầu lần lượt hướng mặt theo từng góc.
        Ảnh được tự động chụp mỗi ``CAPTURE_INTERVAL`` frame khi phát hiện có mặt.
        Không cần thư viện ngoài — chỉ dùng detect_face() sẵn có trong project.
        """
        print("\n" + "=" * 50)
        print("ĐĂNG KÝ GƯƠNG MẶT MỚI")
        print("=" * 50)

        name = input("Nhập tên của bạn: ").strip()
        if not name:
            print("Tên không được để trống!")
            return

        # Tạo thư mục lưu ảnh cho người dùng
        save_dir = Path(self.dataset_path) / name
        save_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Không thể mở webcam!")
            return

        detector = HeadPoseDetector()
        summary   = []      # Lưu kết quả để in tổng kết cuối
        cancelled = False   # Cờ báo người dùng nhấn ESC

        print("\nBẮT ĐẦU ĐĂNG KÝ KHUÔN MẶT")
        print("   Hệ thống sẽ chỉ chụp khi bạn quay đúng góc yêu cầu!")
        print("   ESC: dừng  |  SPACE: bỏ qua góc hiện tại")

        for angle_name, instruction, target in REGISTER_ANGLES:
            captured      = 0   
            correct_angle_frames = 0

            print(f"\nXin mời {instruction} - cần {target} ảnh")

            while captured < target:
                ret, frame = cap.read()
                if not ret:
                    print("Lỗi đọc frame từ webcam!")
                    cancelled = True
                    break

                frame = cv2.flip(frame, 1)   
                h, w = frame.shape[:2]

                # Lấy góc mặt hiện tại và vị trí cắt mặt
                current_angle, face_box = detector.get_angle(frame)
                
                # Logic màu viền màn hình (Gọn gàng nhất)
                # Đen = không có mặt, Vàng = có mặt nhưng sai góc, Xanh lá = đúng chuẩn
                border_color = (0, 0, 0)
                if current_angle != "NONE":
                    border_color = (0, 200, 255) # Màu Vàng BGR
                    
                if current_angle == angle_name and face_box is not None:
                    border_color = (0, 200, 0)   # Xanh lá BGR
                    correct_angle_frames += 1
                    if correct_angle_frames % self.CAPTURE_INTERVAL == 0:
                        x, y, box_w, box_h = face_box
                        
                        # Cắt vừa đủ khuôn mặt (Thêm nhỉnh viền margin 25% để ảnh có cả cằm và đỉnh đầu)
                        margin_x = int(box_w * 0.25)
                        margin_y = int(box_h * 0.25)
                        
                        crop_x1 = max(0, x - margin_x)
                        crop_y1 = max(0, y - margin_y)
                        crop_x2 = min(w, x + box_w + margin_x)
                        crop_y2 = min(h, y + box_h + int(margin_y * 1.5)) # Xuống sâu hơn chút lấy cằm rõ
                        
                        face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        
                        # Chỉ lưu nếu cắt thành công không bị rỗng
                        if face_crop.size > 0:
                            img_path = save_dir / f"{angle_name}_{captured}.jpg"
                            cv2.imwrite(str(img_path), face_crop)
                            captured += 1
                            print(f"  Đã chụp góc {angle_name}: {captured}/{target}")
                else:
                    correct_angle_frames = 0 

                # Vẽ viền màn hình dày nhạy màu
                cv2.rectangle(frame, (0, 0), (w, h), border_color, thickness=20)
                
                # Chữ gọn nhẹ ở góc nhắc người dùng cần làm gì:
                text_info = f"Goc can thu: {angle_name} ({captured}/{target})"
                cv2.putText(frame, text_info, (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                # Ghi chú góc hiện đang quét
                cv2.putText(frame, f"Hien tai he thong thay: {current_angle}", (30, 85), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

                cv2.imshow("Dang Ky Guong Mat - Khung Viec", frame)
                key = cv2.waitKey(1)

                if key == 27:   # ESC 
                    cancelled = True
                    break
                if key == 32:   # SPACE 
                    break

            summary.append((instruction, captured))
            if cancelled:
                break

        # ── Dọn dẹp ────────────────────────────────────────────
        cap.release()
        cv2.destroyAllWindows()

        # ── Tổng kết ────────────────────────────────────────────
        total = sum(c for _, c in summary)
        print(f"\nĐã đăng ký {total} ảnh cho '{name}'")
        print("\nChi tiết:")
        for label, count in summary:
            if count > 0:
                print(f"   {label}: {count} ảnh")

        if total > 0:
            # Tự động huấn luyện ngay sau khi đăng ký xong
            self.train_embeddings()


    # Huấn luyện Embeddings
    # ════════════════════════════════════════════════════════════

    def train_embeddings(self) -> None:
        """Sinh embedding từ toàn bộ ảnh trong dataset và lưu vào database.

        Quét qua từng thư mục người dùng trong dataset_path, dùng mô hình
        Swin Transformer để tạo vector đặc trưng, rồi lưu vào file pickle.
        """
        print("\n" + "=" * 50)
        print("HUẤN LUYỆN EMBEDDINGS")
        print("=" * 50)
        print("Đang xử lý ảnh và tạo embeddings...")

        self.database = {}      # Xóa dữ liệu cũ trước khi huấn luyện lại
        total_images  = 0

        dataset_root = Path(self.dataset_path)
        if not dataset_root.exists():
            print(f"Thư mục dataset không tồn tại: {dataset_root}")
            return

        for person_dir in dataset_root.iterdir():
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            embeddings: list = []
            img_count  = 0

            for img_path in person_dir.iterdir():
                # Chỉ xử lý các định dạng ảnh phổ biến
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue

                try:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue

                    faces = detect_face(image)
                    if not faces:
                        print(f"Không tìm thấy mặt trong {img_path.name}")
                        continue

                    for (x1, y1, x2, y2) in faces:
                        face_crop  = image[y1:y2, x1:x2]
                        embedding  = get_embedding(face_crop)
                        embeddings.append(embedding)
                        img_count += 1

                except Exception as e:
                    print(f"Lỗi xử lý {img_path.name}: {e}")

            if embeddings:
                self.database[person_name] = embeddings
                total_images += img_count
                print(f"{person_name}: {img_count} embeddings")

        self._save_database()
        print(f"\nHuấn luyện hoàn tất! Tổng cộng {total_images} embeddings")

    # ════════════════════════════════════════════════════════════
    # Nhận diện Gương mặt
    # ════════════════════════════════════════════════════════════

    def _find_best_match(self, embedding: np.ndarray) -> tuple[str, float]:
        """Tìm người khớp nhất trong database với embedding đầu vào.

        Args:
            embedding: Vector đặc trưng của khuôn mặt cần nhận diện.

        Returns:
            (tên, điểm_tương_đồng): tên là "Unknown" nếu dưới ngưỡng.
        """
        best_name  = "Unknown"
        best_score = 0.0

        for person, emb_list in self.database.items():
            # Tìm embedding gần nhất trong danh sách của người này
            score = max((compare(embedding, db_emb) for db_emb in emb_list),
                        default=0.0)
            if score > best_score:
                best_score = score
                best_name  = person

        # Áp ngưỡng – dưới ngưỡng → không nhận ra
        if best_score < self.recognition_threshold:
            best_name = "Unknown"

        return best_name, best_score

    def recognize_faces(self) -> None:
        """Nhận diện gương mặt theo thời gian thực từ webcam.

        Mỗi frame sẽ:
        1. Phát hiện tất cả khuôn mặt.
        2. Tính embedding và so sánh với database.
        3. Hiển thị tên + điểm tương đồng lên frame.
        """
        print("\n" + "=" * 50)
        print("NHẬN DIỆN GƯƠNG MẶT")
        print("=" * 50)

        if not self.database:
            print("Database trống! Vui lòng đăng ký gương mặt trước.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Không thể mở webcam!")
            return

        print("\nNhấn 'ESC' để dừng")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Lỗi khi đọc frame từ webcam!")
                break

            faces = detect_face(frame)

            for (x1, y1, x2, y2) in faces:
                face_crop = frame[y1:y2, x1:x2]
                try:
                    emb              = get_embedding(face_crop)
                    name, best_score = self._find_best_match(emb)

                    # Màu xanh nếu nhận ra, đỏ nếu không
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label = f"{name} ({best_score:.2f})" if name != "Unknown" else name
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                except Exception as e:
                    print(f"Lỗi nhận diện: {e}")

            cv2.imshow("Nhận diện Gương mặt", frame)

            if cv2.waitKey(1) == 27:    # ESC – thoát
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Kết thúc nhận diện")

    # ════════════════════════════════════════════════════════════
    # Quản lý Người dùng
    # ════════════════════════════════════════════════════════════

    def view_registered_persons(self) -> None:
        """Hiển thị danh sách tất cả người dùng đã đăng ký kèm thống kê."""
        print("\n" + "=" * 50)
        print("DANH SÁCH NGƯỜI ĐÃ ĐĂNG KÝ")
        print("=" * 50)

        dataset_root = Path(self.dataset_path)
        if not dataset_root.exists():
            print("Thư mục dataset không tồn tại!")
            return

        # Lấy danh sách thư mục con (mỗi thư mục = một người)
        persons = sorted(p.name for p in dataset_root.iterdir() if p.is_dir())

        if not persons:
            print("Chưa có ai đăng ký!")
            return

        print(f"\nTổng cộng: {len(persons)} người\n")
        for i, person in enumerate(persons, start=1):
            person_dir  = dataset_root / person
            # Đếm số file ảnh
            img_count   = sum(1 for f in person_dir.iterdir()
                              if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
            db_count    = len(self.database.get(person, []))

            print(f"  {i}. {person}")
            print(f"     - Ảnh       : {img_count}")
            print(f"     - Embeddings: {db_count}")

    def delete_person(self) -> None:
        """Xóa toàn bộ dữ liệu (ảnh + embedding) của một người dùng."""
        print("\n" + "=" * 50)
        print("XÓA NGƯỜI DÙNG")
        print("=" * 50)

        dataset_root = Path(self.dataset_path)
        persons = [p.name for p in dataset_root.iterdir() if p.is_dir()]

        if not persons:
            print("Chưa có ai đăng ký!")
            return

        self.view_registered_persons()
        name = input("\nNhập tên người dùng cần xóa: ").strip()

        person_dir = dataset_root / name
        if person_dir.exists():
            shutil.rmtree(person_dir)           # Xóa toàn bộ thư mục ảnh

            if name in self.database:
                del self.database[name]          # Xóa embedding khỏi database
                self._save_database()

            print(f"Đã xóa {name}")
        else:
            print(f"Không tìm thấy '{name}'")

    # ════════════════════════════════════════════════════════════
    # Giao diện Menu
    # ════════════════════════════════════════════════════════════

    def _display_menu(self) -> None:
        """In menu chính ra terminal."""
        print("\n" + "=" * 50)
        print("HỆ THỐNG NHẬN DIỆN GƯƠNG MẶT")
        print("=" * 50)
        print("1. Đăng ký gương mặt mới")
        print("2. Huấn luyện embeddings")
        print("3. Nhận diện gương mặt")
        print("4. Xem danh sách người dùng")
        print("5. Xóa người dùng")
        print("0. Thoát chương trình")
        print("=" * 50)

    def run(self) -> None:
        """Vòng lặp chính của chương trình – xử lý lựa chọn menu."""
        # Bảng ánh xạ lựa chọn → phương thức tương ứng
        actions = {
            "1": self.register_face,
            "2": self.train_embeddings,
            "3": self.recognize_faces,
            "4": self.view_registered_persons,
            "5": self.delete_person,
        }

        while True:
            self._display_menu()
            choice = input("Chọn chức năng (0-5): ").strip()

            if choice == "0":
                print("\nCảm ơn bạn đã sử dụng! Tạm biệt!")
                break

            action = actions.get(choice)
            if action:
                action()
            else:
                print("Lựa chọn không hợp lệ! Vui lòng thử lại.")


# ── Điểm khởi chạy chương trình ─────────────────────────────────
if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()
