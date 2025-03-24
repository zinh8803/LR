import pandas as pd
import numpy as np
from faker import Faker
import random

# Khởi tạo Faker với ngôn ngữ tiếng Việt
fake = Faker('vi_VN')

# Hàm tạo dữ liệu giả lập
def generate_student_data(n_students=100):
    data = []
    for _ in range(n_students):
        name = fake.name()
        mssv = f"2020{random.randint(10000, 99999)}"
        performance = random.random()
        health = random.randint(1, 3)
        location = random.random()

        # Tạo danh sách điểm và số tín chỉ cho mỗi môn
        scores = {}
        credits_per_subject = []  # Lưu số tín chỉ của từng môn
        for sem in range(1, 5):  # 4 kỳ
            for sub in range(1, 6):  # 5 môn mỗi kỳ
                subject_credits = random.randint(2, 4)  # Số tín chỉ ngẫu nhiên từ 2 đến 4
                initial_score = np.clip(np.random.normal(performance * 10, 2), 0, 10)
                final_score = initial_score if initial_score >= 5 else np.clip(np.random.normal(initial_score + 1, 1), 0, 10)
                # Chuyển đổi điểm sang thang 4 (nhân với 0.4)
                scaled_score = round(final_score * 0.4, 2)  # Làm tròn 2 chữ số
                scores[f"mon{sub}_ky{sem}"] = scaled_score
                credits_per_subject.append(subject_credits)

        # Tính tổng số môn và tín chỉ (dựa trên ngưỡng 5 của thang 10)
        total_subjects = len(scores)  # Tổng số môn (20 môn)
        passed_subjects = sum(1 for score in [v / 0.4 for v in scores.values()] if score >= 5)  # Chuyển ngược về thang 10 để kiểm tra
        failed_subjects = total_subjects - passed_subjects

        # Tính số tín chỉ dựa trên từng môn
        credits_accumulated = sum(credits_per_subject[i] for i in range(total_subjects) if [v / 0.4 for v in scores.values()][i] >= 5)
        credits_failed = sum(credits_per_subject[i] for i in range(total_subjects) if [v / 0.4 for v in scores.values()][i] < 5)
        retake_count = failed_subjects  # Giả định học lại 1 lần cho mỗi môn rớt

        # Xác định khoảng cách và thời gian di chuyển
        if location <= 0.25:
            distance = "gần"
            travel_time = random.uniform(10, 30)
        elif location <= 0.5:
            distance = "trung bình"
            travel_time = random.uniform(30, 50)
        elif location <= 0.75:
            distance = "xa"
            travel_time = random.uniform(50, 90)
        else:
            distance = "rất xa"
            travel_time = random.uniform(90, 120)

        residence_probs = {"ktx": 0.5 if distance in ["xa", "rất xa"] else 0.2,
                           "ở nhà": 0.7 if distance == "gần" else 0.3,
                           "ở trọ": 0.4}
        residence = random.choices(["ktx", "ở nhà", "ở trọ"], 
                                   weights=[residence_probs[r] for r in ["ktx", "ở nhà", "ở trọ"]], k=1)[0]

        # Nếu ở KTX, khoảng cách là "gần" và thời gian di chuyển bằng 0
        if residence == "ktx":
            distance = "gần"
            travel_time = 0

        # Tính xác suất nghỉ học
        dropout_prob = 1 / (1 + np.exp(5 * performance + 0.5 * health - 0.01 * travel_time - 
                                       0.02 * credits_failed - 0.1 * retake_count + 0.01 * credits_accumulated - 2))
        dropout = 1 if random.random() < dropout_prob else 0
        dropout_status = "Có" if dropout == 1 else "Không"  # Cột hiển thị

        record = {"Tên": name, "MSSV": mssv, "Sức khỏe": health, "Khoảng cách": distance, 
                  "Thời gian di chuyển": round(travel_time, 2), "Cư trú": residence, 
                  "Số tín chỉ tích lũy": credits_accumulated, "Số tín chỉ rớt": credits_failed, 
                  "Số lần học lại": retake_count, "Nghỉ học": dropout, "Trạng thái nghỉ học": dropout_status}
        record.update(scores)
        data.append(record)

    return pd.DataFrame(data)

# Tạo và lưu dữ liệu vào file Excel
if __name__ == "__main__":
    df = generate_student_data(100)
    df.to_excel("student_data.xlsx", index=False, engine='openpyxl')
    print("Đã tạo và lưu 100 bản ghi vào 'student_data.xlsx'.")