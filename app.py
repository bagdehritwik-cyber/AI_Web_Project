from flask import Flask, render_template, request, send_file, url_for, redirect
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
from datetime import datetime
import os
import webbrowser
from threading import Timer

app = Flask(__name__)

# ---------------- DATASET ----------------
symptoms_list = [
    "fever", "cough", "headache", "vomiting", "fatigue",
    "chest pain", "breathlessness", "sneezing",
    "runny nose", "joint pain"
]

X = np.array([
    [1,1,1,0,1,0,0,0,0,0],
    [0,1,0,0,0,0,0,1,1,0],
    [1,0,1,1,1,0,0,0,0,1],
    [0,0,0,0,1,1,1,0,0,0],
    [0,1,0,0,0,0,1,0,0,0],
])

y = np.array(["Flu","Common Cold","Dengue","Heart Disease","Asthma"])

le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = RandomForestClassifier()
model.fit(X, y_encoded)

latest_data = {}

# ---------------- HOME ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    global latest_data
    result = None

    if request.method == "POST":
        name = request.form.get("name")
        selected = request.form.getlist("symptoms")

        if not name or not selected:
            result = "Please enter name and select symptoms."
        else:
            input_data = np.zeros(len(symptoms_list))
            for sym in selected:
                input_data[symptoms_list.index(sym)] = 1

            input_data = input_data.reshape(1, -1)
            probabilities = model.predict_proba(input_data)[0]
            top3 = np.argsort(probabilities)[-3:][::-1]

            result = []
            for idx in top3:
                disease = le.inverse_transform([idx])[0]
                confidence = round(probabilities[idx] * 100, 2)
                result.append((disease, confidence))

            latest_data = {
                "name": name,
                "symptoms": selected,
                "result": result
            }

    return render_template("index.html",
                           symptoms=symptoms_list,
                           result=result)

# ---------------- PDF DOWNLOAD ----------------
@app.route("/download")
def download():
    global latest_data

    if not latest_data:
        return redirect(url_for("home"))

    if not os.path.exists("Reports"):
        os.makedirs("Reports")

    name = latest_data["name"]
    symptoms = latest_data["symptoms"]
    result = latest_data["result"]

    file_path = f"Reports/{name.replace(' ','_')}_report.pdf"

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Disease Prediction Report", ln=True, align="C")

    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Patient Name: {name}", ln=True)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    pdf.ln(5)
    pdf.multi_cell(0, 8, "Selected Symptoms:\n" + ", ".join(symptoms))

    pdf.ln(5)
    pdf.multi_cell(0, 8, "Top Predictions:")

    for disease, confidence in result:
        pdf.multi_cell(0, 8, f"{disease} - {confidence}%")

    pdf.ln(10)
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(0, 8,
        "Disclaimer: This report is AI-generated and not a medical diagnosis."
    )

    pdf.output(file_path)

    return send_file(file_path, as_attachment=True)

# ---------------- AUTO OPEN ----------------
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=True)