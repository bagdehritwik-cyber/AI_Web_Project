from flask import Flask, render_template, request, send_file, url_for, redirect
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from datetime import datetime
import os
import io

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

# ---------------- HOME ----------------
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

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph("<b>Disease Prediction Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Patient Name: {latest_data['name']}", styles["Normal"]))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("<b>Selected Symptoms:</b>", styles["Heading3"]))
    elements.append(Paragraph(", ".join(latest_data["symptoms"]), styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("<b>Top Predictions:</b>", styles["Heading3"]))

    for disease, confidence in latest_data["result"]:
        elements.append(Paragraph(f"{disease} - {confidence}%", styles["Normal"]))

    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(
        "Disclaimer: This report is AI-generated and not a medical diagnosis.",
        styles["Italic"]
    ))

    doc.build(elements)
    buffer.seek(0)

    return send_file(buffer,
                     as_attachment=True,
                     download_name="disease_report.pdf",
                     mimetype="application/pdf")

# ---------------- RUN FOR RENDER ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
