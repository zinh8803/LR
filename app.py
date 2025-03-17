import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend Agg để tránh lỗi Tkinter
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, url_for, redirect
import io
import base64
import secrets
import os

# Tạo khóa bí mật ngẫu nhiên
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Khóa bí mật 32 ký tự

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

# Hàm phân tích dữ liệu và tạo kết quả
def analyze_dropout(df, threshold=0.5):
    X, y = preprocess_data(df)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    all_probabilities = []

    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = train_model(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        for idx, prob in zip(test_idx, probabilities):
            all_probabilities.append((idx, prob))

    student_probabilities = [0] * len(X)
    for idx, prob in all_probabilities:
        student_probabilities[idx] = prob

    student_predictions = [1 if prob >= threshold else 0 for prob in student_probabilities]
    dropout_percentage = (sum(student_predictions) / len(student_predictions)) * 100
    actual_dropout_percentage = (df["Nghỉ học"].sum() / len(df)) * 100
    not_dropout_percentage = 100 - dropout_percentage

    df_with_prob = df.copy()
    df_with_prob["Xác suất nghỉ học (%)"] = [round(prob * 100, 6) for prob in student_probabilities]
    df_with_prob["Dự đoán nghỉ học"] = student_predictions

    dropout_students = df_with_prob.loc[df_with_prob["Dự đoán nghỉ học"] == 1][["Tên", "MSSV", "Xác suất nghỉ học (%)"]].sort_values(by="Xác suất nghỉ học (%)", ascending=False)
    not_dropout_students = df_with_prob.loc[df_with_prob["Dự đoán nghỉ học"] == 0][["Tên", "MSSV", "Xác suất nghỉ học (%)"]].sort_values(by="Xác suất nghỉ học (%)", ascending=True)

    # Tạo biểu đồ cột
    plt.figure(figsize=(8, 6))
    plt.bar(["Nghỉ học", "Không nghỉ học"], [sum(student_predictions), len(student_predictions) - sum(student_predictions)], color=['red', 'green'])
    plt.title("Dự đoán sinh viên nghỉ học")
    plt.ylabel("Số lượng sinh viên")
    bar_img = io.BytesIO()
    plt.savefig(bar_img, format='png')
    bar_img.seek(0)
    bar_plot_url = base64.b64encode(bar_img.getvalue()).decode('utf8')
    plt.close()

    # Tạo biểu đồ tròn
    plt.figure(figsize=(8, 6))
    plt.pie([sum(student_predictions), len(student_predictions) - sum(student_predictions)], labels=["Nghỉ học", "Không nghỉ học"], autopct='%1.1f%%', colors=['red', 'green'])
    plt.title("Phần trăm sinh viên nghỉ học")
    pie_img = io.BytesIO()
    plt.savefig(pie_img, format='png')
    pie_img.seek(0)
    pie_plot_url = base64.b64encode(pie_img.getvalue()).decode('utf8')
    plt.close()

    return (dropout_percentage, actual_dropout_percentage, not_dropout_percentage, 
            dropout_students, not_dropout_students, bar_plot_url, pie_plot_url, df_with_prob)

# Route trang chính
@app.route('/', methods=['GET', 'POST'])
def index():
    # Ưu tiên đọc file được tải lên nếu có
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        threshold = float(request.form.get('threshold', 0.5))
        if file and file.filename.endswith('.xlsx'):
            try:
                df = pd.read_excel(file, engine='openpyxl')
            except Exception as e:
                return f"Lỗi khi đọc file Excel: {str(e)}", 500
        else:
            return "Vui lòng tải lên file Excel hợp lệ (.xlsx).", 400
    else:
        # Nếu không có file được tải lên, đọc từ student_data.xlsx
        try:
            df = pd.read_excel("student_data.xlsx", engine='openpyxl')
            threshold = float(request.form.get('threshold', 0.5)) if request.method == 'POST' else 0.5
        except FileNotFoundError:
            return "Không tìm thấy file student_data.xlsx. Vui lòng tải lên file Excel.", 404
        except Exception as e:
            return f"Lỗi khi đọc file Excel: {str(e)}", 500

    # Phân tích dữ liệu
    dropout_percentage, actual_dropout_percentage, not_dropout_percentage, dropout_students, not_dropout_students, bar_plot_url, pie_plot_url, df_with_prob = analyze_dropout(df, threshold)

    # Tạo HTML cho danh sách với liên kết đúng cú pháp
    dropout_students_html = dropout_students.to_html(index=False, escape=False)
    not_dropout_students_html = not_dropout_students.to_html(index=False, escape=False)
    for mssv in dropout_students['MSSV'].astype(str):
        dropout_students_html = dropout_students_html.replace(f'>{mssv}<', f'><a href="{url_for("student_detail", mssv=mssv)}">{mssv}</a><')
    for mssv in not_dropout_students['MSSV'].astype(str):
        not_dropout_students_html = not_dropout_students_html.replace(f'>{mssv}<', f'><a href="{url_for("student_detail", mssv=mssv)}">{mssv}</a><')

    return render_template('index.html', 
                           dropout_percentage=dropout_percentage, 
                           actual_dropout_percentage=actual_dropout_percentage, 
                           not_dropout_percentage=not_dropout_percentage,
                           dropout_students=dropout_students_html, 
                           not_dropout_students=not_dropout_students_html,
                           bar_plot_url=bar_plot_url, 
                           pie_plot_url=pie_plot_url,
                           threshold=threshold)

# Route hiển thị thông tin chi tiết của sinh viên
@app.route('/student/<mssv>')
def student_detail(mssv):
    try:
        df = pd.read_excel("student_data.xlsx", engine='openpyxl')
        
        # Chuyển MSSV trong Excel và trong URL thành chuỗi để so sánh
        df["MSSV"] = df["MSSV"].astype(str)
        student = df[df["MSSV"] == mssv]  # Tìm sinh viên theo MSSV
    except Exception as e:
        return f"Lỗi khi đọc file Excel: {str(e)}", 500
    
    if student.empty:
        return "Không tìm thấy sinh viên với MSSV này.", 404
    
    student_info = student.iloc[0].to_dict()
    
    return render_template('student_detail.html', student_info=student_info)


if __name__ == '__main__':
    app.run(debug=True)
