import tkinter as tk
from tkinter import messagebox
import pandas as pd
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("spam.csv", encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text

df['message'] = df['message'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

nb_model = MultinomialNB()
lr_model = LogisticRegression()

nb_model.fit(X_train_vec, y_train)
lr_model.fit(X_train_vec, y_train)

def predict_message():
    message = text_entry.get("1.0", tk.END).strip()

    if message == "":
        messagebox.showwarning("Warning", "Please enter a message!")
        return

    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])

    selected_model = model_var.get()

    if selected_model == "Naive Bayes":
        prediction = nb_model.predict(vectorized)[0]
    else:
        prediction = lr_model.predict(vectorized)[0]

    if prediction == 1:
        result_label.config(text="SPAM ❌", fg="red")
    else:
        result_label.config(text="HAM ✅", fg="green")

root = tk.Tk()
root.title("Spam Email Classifier")
root.geometry("600x500")
root.config(bg="#1e1e2f")

title = tk.Label(root, text="Spam Email Classifier",
                 font=("Helvetica", 20, "bold"),
                 bg="#1e1e2f", fg="white")
title.pack(pady=20)

text_entry = tk.Text(root, height=8, width=50,
                     font=("Arial", 12))
text_entry.pack(pady=10)

model_var = tk.StringVar(value="Naive Bayes")

frame = tk.Frame(root, bg="#1e1e2f")
frame.pack(pady=10)

nb_radio = tk.Radiobutton(frame, text="Naive Bayes",
                         variable=model_var, value="Naive Bayes",
                         font=("Arial", 12),
                         bg="#1e1e2f", fg="white",
                         selectcolor="#333")
nb_radio.grid(row=0, column=0, padx=20)

lr_radio = tk.Radiobutton(frame, text="Logistic Regression",
                         variable=model_var, value="Logistic Regression",
                         font=("Arial", 12),
                         bg="#1e1e2f", fg="white",
                         selectcolor="#333")
lr_radio.grid(row=0, column=1, padx=20)

predict_btn = tk.Button(root, text="Classify Message",
                        command=predict_message,
                        font=("Arial", 14, "bold"),
                        bg="#4CAF50", fg="white",
                        padx=10, pady=5)
predict_btn.pack(pady=20)

result_label = tk.Label(root, text="",
                        font=("Arial", 16, "bold"),
                        bg="#1e1e2f")
result_label.pack(pady=10)

root.mainloop()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("\n📊 Running Model Evaluation...\n")

nb_preds = nb_model.predict(X_test_vec)
lr_preds = lr_model.predict(X_test_vec)

def evaluate_model(name, y_true, y_pred):
    print(f"🔹 {name} Performance:\n")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("-" * 40)

    return acc, prec, rec, f1

nb_metrics = evaluate_model("Naive Bayes", y_test, nb_preds)
lr_metrics = evaluate_model("Logistic Regression", y_test, lr_preds)

def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_confusion(y_test, nb_preds, "Naive Bayes Confusion Matrix")
plot_confusion(y_test, lr_preds, "Logistic Regression Confusion Matrix")

labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

nb_values = list(nb_metrics)
lr_values = list(lr_metrics)

x = range(len(labels))

plt.figure()
plt.plot(x, nb_values, marker='o', label='Naive Bayes')
plt.plot(x, lr_values, marker='o', label='Logistic Regression')

plt.xticks(x, labels)
plt.title("Model Performance Comparison")
plt.legend()
plt.grid()

plt.show()
