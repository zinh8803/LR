import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Hàm tải dữ liệu từ file Excel
def load_data(filename):
    return pd.read_excel(filename, engine='openpyxl')

# Hàm xử lý dữ liệu
def preprocess_data(df):
    X = df.drop(columns=["Tên", "MSSV", "Nghỉ học"])
    X = pd.get_dummies(X, columns=["Khoảng cách", "Cư trú"], drop_first=True)
    y = df["Nghỉ học"]
    return X, y

# Hàm huấn luyện mô hình
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Hàm chính
if __name__ == "__main__":
    # Tải dữ liệu từ file Excel
    df = load_data("student_data.xlsx")

    # Xử lý dữ liệu
    X, y = preprocess_data(df)

    # Sử dụng KFold để dự đoán xác suất
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    all_probabilities = []

    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = train_model(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        for idx, prob in zip(test_idx, probabilities):
            all_probabilities.append((idx, prob))

    # Tạo danh sách xác suất cho từng sinh viên
    student_probabilities = [0] * len(X)
    for idx, prob in all_probabilities:
        student_probabilities[idx] = prob

    # Phân loại sinh viên dựa trên ngưỡng 0.5
    student_predictions = [1 if prob >= 0.5 else 0 for prob in student_probabilities]

    # Tính phần trăm
    dropout_percentage = (sum(student_predictions) / len(student_predictions)) * 100
    print(f"Phần trăm sinh viên được dự đoán nghỉ học: {dropout_percentage:.2f}%")

    # Thêm kiểm tra tỷ lệ thực tế trong dữ liệu
    actual_dropout_percentage = (df["Nghỉ học"].sum() / len(df)) * 100
    print(f"Phần trăm sinh viên thực tế nghỉ học trong dữ liệu: {actual_dropout_percentage:.2f}%")

    # Liệt kê sinh viên có nguy cơ nghỉ học
    df_with_prob = df[["Tên", "MSSV"]].copy()
    df_with_prob["Xác suất nghỉ học"] = student_probabilities
    df_with_prob["Dự đoán nghỉ học"] = student_predictions

    # Sử dụng loc để lọc sinh viên được dự đoán nghỉ học
    dropout_students = df_with_prob.loc[df_with_prob["Dự đoán nghỉ học"] == 1].sort_values(by="Xác suất nghỉ học", ascending=False)
    print("\nDanh sách sinh viên được dự đoán nghỉ học:")
    print(dropout_students[["Tên", "MSSV", "Xác suất nghỉ học"]])

    # Vẽ biểu đồ cột
    num_dropout = sum(student_predictions)
    num_not_dropout = len(student_predictions) - num_dropout
    plt.figure(figsize=(8, 6))
    plt.bar(["Nghỉ học", "Không nghỉ học"], [num_dropout, num_not_dropout], color=['red', 'green'])
    plt.title("Dự đoán sinh viên nghỉ học")
    plt.ylabel("Số lượng sinh viên")
    plt.show()

    # Vẽ biểu đồ tròn
    plt.figure(figsize=(8, 6))
    plt.pie([num_dropout, num_not_dropout], labels=["Nghỉ học", "Không nghỉ học"], autopct='%1.1f%%', colors=['red', 'green'])
    plt.title("Phần trăm sinh viên nghỉ học")
    plt.show()