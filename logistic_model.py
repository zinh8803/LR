from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Hàm tải dữ liệu (giả định đã có)
def load_data(filename):
    return pd.read_csv(filename)

# Hàm xử lý dữ liệu (giả định đã có)
def preprocess_data(df):
    X = df.drop(columns=["Tên", "MSSV", "Nghỉ học"])
    X = pd.get_dummies(X, columns=["Khoảng cách", "Cư trú"], drop_first=True)
    y = df["Nghỉ học"]
    return X, y

# Hàm huấn luyện mô hình (giả định đã có)
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Hàm dự đoán (giả định đã có)
def predict(model, X_test):
    return model.predict(X_test)

# Load data
df = load_data("student_data.csv")

# Preprocess data
X, y = preprocess_data(df)

# Initialize KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store predicted probabilities
all_probabilities = []

# Perform cross-validation
for train_idx, test_idx in kfold.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train model on current fold
    model = train_model(X_train, y_train)
    
    # Make probability predictions on test set of current fold
    probabilities = model.predict_proba(X_test)[:, 1]  # Probability of dropout
    
    # Store probabilities with corresponding student indices
    for idx, prob in zip(test_idx, probabilities):
        all_probabilities.append((idx, prob))

# Create a list of predicted probabilities for each student
student_probabilities = [0] * len(X)
for idx, prob in all_probabilities:
    student_probabilities[idx] = prob

# Classify students based on threshold 0.5
student_predictions = [1 if prob >= 0.5 else 0 for prob in student_probabilities]

# Calculate percentage
dropout_percentage = (sum(student_predictions) / len(student_predictions)) * 100

# List students predicted to dropout
dropout_students = df.loc[student_predictions == 1, ["Tên", "MSSV"]]

# Print results
print(f"Percentage of students predicted to dropout: {dropout_percentage:.2f}%")
print("Students predicted to dropout:")
print(dropout_students)

# Create a bar chart
num_dropout = sum(student_predictions)
num_not_dropout = len(student_predictions) - num_dropout
plt.bar(["Nghỉ học", "Không nghỉ học"], [num_dropout, num_not_dropout])
plt.title("Dự đoán sinh viên nghỉ học")
plt.show()

# Optional: Histogram of probabilities
plt.hist(student_probabilities, bins=20)
plt.title("Phân phối xác suất nghỉ học")
plt.show()

# Optional: List top N students by probability
df_with_prob = df[["Tên", "MSSV"]].copy()
df_with_prob["Dropout Probability"] = student_probabilities
df_with_prob.sort_values(by="Dropout Probability", ascending=False, inplace=True)
N = 10
top_N_students = df_with_prob.head(N)
print("Top", N, "sinh viên có nguy cơ nghỉ học cao nhất:")
print(top_N_students)