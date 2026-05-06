from flask import Flask, render_template, request, redirect, session, url_for, make_response
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import os
from io import StringIO

app = Flask(__name__)
app.secret_key = "vlsi_secret_key_2026"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# DATABASE
# =========================
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    email TEXT UNIQUE,
    password TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS user_settings (
    email TEXT UNIQUE,
    username TEXT,
    theme TEXT,
    graph_view TEXT,
    report_mode TEXT,
    profile_image TEXT
)
""")
conn.commit()

try:
    cursor.execute("ALTER TABLE user_settings ADD COLUMN profile_image TEXT")
    conn.commit()
except:
    pass

# =========================
# ML DATASET
# =========================
np.random.seed(42)
n = 5000

alpha = np.random.uniform(0.1, 1.0, n)
C = np.random.uniform(1, 10, n)              # pF
load_C = np.random.uniform(0.5, 5, n)        # pF
V = np.random.uniform(0.8, 1.5, n)           # V
f = np.random.uniform(50, 500, n)            # MHz
temp = np.random.uniform(20, 100, n)         # °C
I_leak = np.random.uniform(0.001, 0.01, n)   # µA
gates = np.random.randint(1000, 10000, n)

P_dynamic = alpha * (C + load_C) * (V ** 2) * f
P_leakage = V * I_leak * (1 + 0.01 * temp)
P_total = P_dynamic + P_leakage

data = pd.DataFrame({
    "Alpha": alpha,
    "Capacitance": C,
    "Load_Capacitance": load_C,
    "Voltage": V,
    "Frequency": f,
    "Temperature": temp,
    "Leakage_Current": I_leak,
    "Gates": gates,
    "Power": P_total
})

X = data.drop("Power", axis=1)
y = data["Power"]

# =========================
# MODELS
# =========================
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
rf_model.fit(X, y)
rf_score = r2_score(y, rf_model.predict(X))

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
gb_model.fit(X, y)
gb_score = r2_score(y, gb_model.predict(X))

if gb_score > rf_score:
    best_model = gb_model
    best_model_name = "Gradient Boosting"
else:
    best_model = rf_model
    best_model_name = "Random Forest"

print(f"✔ Best model selected: {best_model_name}")
print(f"Random Forest R2: {rf_score:.6f}")
print(f"Gradient Boosting R2: {gb_score:.6f}")

# =========================
# HELPERS
# =========================
def get_user_settings(email):
    cursor.execute("""
        SELECT username, theme, graph_view, report_mode, profile_image
        FROM user_settings WHERE email=?
    """, (email,))
    row = cursor.fetchone()

    if row:
        return {
            "username": row[0],
            "theme": row[1],
            "graph_view": row[2],
            "report_mode": row[3],
            "profile_image": row[4]
        }

    username = email.split("@")[0]
    cursor.execute("""
        INSERT OR IGNORE INTO user_settings
        (email, username, theme, graph_view, report_mode, profile_image)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (email, username, "light", "Voltage vs Power", "Enabled", ""))
    conn.commit()

    return {
        "username": username,
        "theme": "light",
        "graph_view": "Voltage vs Power",
        "report_mode": "Enabled",
        "profile_image": ""
    }


def create_voltage_graph(alpha_val, cap_val, load_cap_val, voltage_val, freq_val, temp_val, leak_val, gates_val):
    voltages = np.linspace(0.8, 1.5, 25)
    powers_mw = []

    for v in voltages:
        temp_input = pd.DataFrame([[
            alpha_val, cap_val, load_cap_val, v,
            freq_val, temp_val, leak_val, gates_val
        ]], columns=X.columns)

        power = best_model.predict(temp_input)[0]
        powers_mw.append(power / 1000)

    plt.figure(figsize=(7, 4))
    plt.plot(voltages, powers_mw, marker='o', linewidth=2, color="#2563eb")
    plt.title("Voltage (v) vs Power Analysis (mW)")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Predicted Power (mW)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    graph_path = os.path.join("static", "voltage_graph.png")
    plt.savefig(graph_path)
    plt.close()


# =========================
# LOGIN
# =========================
@app.route("/", methods=["GET", "POST"])
def login():
    message = ""

    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        cursor.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
        user = cursor.fetchone()

        if user:
            session.clear()
            session["user"] = email
            return redirect(url_for("dashboard"))
        else:
            message = "Invalid email or password"

    return render_template("login.html", message=message)


# =========================
# REGISTER
# =========================
@app.route("/register", methods=["GET", "POST"])
def register():
    message = ""

    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        try:
            cursor.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, password))
            conn.commit()

            username = email.split("@")[0]
            cursor.execute("""
                INSERT OR IGNORE INTO user_settings
                (email, username, theme, graph_view, report_mode, profile_image)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (email, username, "light", "Voltage vs Power", "Enabled", ""))
            conn.commit()

            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            message = "Email already registered"

    return render_template("register.html", message=message)


# =========================
# SETTINGS
# =========================
@app.route("/settings", methods=["GET", "POST"])
def settings():
    if "user" not in session:
        return redirect(url_for("login"))

    email = session["user"]
    settings_data = get_user_settings(email)
    message = ""

    if request.method == "POST":
        username = request.form["username"]
        theme = request.form["theme"]
        graph_view = request.form["graph_view"]
        report_mode = request.form["report_mode"]

        profile_image = settings_data["profile_image"]
        file = request.files.get("profile_image")

        if file and file.filename:
            filename = email.replace("@", "_").replace(".", "_") + "_" + file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            profile_image = "uploads/" + filename

        cursor.execute("""
            UPDATE user_settings
            SET username=?, theme=?, graph_view=?, report_mode=?, profile_image=?
            WHERE email=?
        """, (username, theme, graph_view, report_mode, profile_image, email))
        conn.commit()

        settings_data = get_user_settings(email)
        message = "Settings updated successfully"

    return render_template("settings.html", user=email, settings_data=settings_data, message=message)


# =========================
# DASHBOARD
# =========================
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    email = session["user"]
    settings_data = get_user_settings(email)

    result = None
    recommendations = []
    insights = []
    graph_ready = False

    if request.method == "POST":
        if "clear_analysis" in request.form:
            session.pop("last_result", None)
            session.pop("last_recommendations", None)
            session.pop("last_insights", None)
            return redirect(url_for("dashboard"))

        alpha_val = float(request.form["alpha"])
        cap_val = float(request.form["cap"])
        load_cap_val = float(request.form["load_cap"])
        voltage_val = float(request.form["voltage"])
        freq_val = float(request.form["freq"])
        temp_val = float(request.form["temp"])
        leak_val = float(request.form["leak"])
        gates_val = int(request.form["gates"])

        user_df = pd.DataFrame([[
            alpha_val, cap_val, load_cap_val, voltage_val,
            freq_val, temp_val, leak_val, gates_val
        ]], columns=X.columns)

        pred_power = best_model.predict(user_df)[0]

        ideal_power = alpha_val * (cap_val + load_cap_val) * (voltage_val ** 2) * freq_val
        error = abs(ideal_power - pred_power)

        dynamic_power = ideal_power
        leakage_power = voltage_val * leak_val * (1 + 0.01 * temp_val)
        total_estimated = dynamic_power + leakage_power
        total_estimated_mw = total_estimated / 1000

        test_voltage = min(voltage_val + 0.1, 1.5)

        test_df = pd.DataFrame([[
            alpha_val, cap_val, load_cap_val, test_voltage,
            freq_val, temp_val, leak_val, gates_val
        ]], columns=X.columns)

        new_power = best_model.predict(test_df)[0]
        power_change = new_power - pred_power

        # =========================
        # COLOR GRADING LOGIC
        # =========================
        if total_estimated_mw > 3:
            status = "High Power"
            status_color = "#ef4444"
            recommendations = [
                "Reduce supply voltage",
                "Reduce operating frequency",
                "Optimize capacitance values"
            ]
        elif total_estimated_mw > 1.5:
            status = "Moderate Power"
            status_color = "#f59e0b"
            recommendations = [
                "Improve circuit efficiency",
                "Slightly reduce switching activity",
                "Check voltage and frequency trade-off"
            ]
        else:
            status = "Low Power"
            status_color = "#22c55e"
            recommendations = [
                "Design is power efficient",
                "Suitable for low-power operation",
                "You can safely explore performance tuning"
            ]

        if dynamic_power > leakage_power:
            insights.append("Dynamic power dominates the design")
        else:
            insights.append("Leakage power dominates the design")

        if power_change > 0:
            insights.append("Power increases with voltage, so the design is voltage sensitive")

        if error < 100:
            insights.append("Prediction is accurate for this input")
        else:
            insights.append("Prediction deviation exists, but trend analysis remains useful")

        result = {
            "model_used": "",
            "predicted_power": f"{pred_power / 1000:.2f} mW",
            "ideal_power": f"{ideal_power / 1000:.2f} mW",
            "error": f"{error / 1000:.2f} mW",
            "dynamic_power": f"{dynamic_power / 1000:.2f} mW",
            "leakage_power": f"{leakage_power / 1000:.4f} mW",
            "total_estimated": f"{total_estimated_mw:.2f} mW",
            "total_estimated_value": round(total_estimated_mw, 2),
            "status": status,
            "status_color": status_color,
            "new_power": f"{new_power / 1000:.2f} mW",
            "power_change": f"{power_change / 1000:.2f} mW"
        }

        session["last_result"] = result
        session["last_recommendations"] = recommendations
        session["last_insights"] = insights

        create_voltage_graph(alpha_val, cap_val, load_cap_val, voltage_val, freq_val, temp_val, leak_val, gates_val)
        graph_ready = True

    return render_template(
        "dashboard.html",
        user=email,
        settings_data=settings_data,
        result=result,
        recommendations=recommendations,
        insights=insights,
        graph_ready=graph_ready
    )


# =========================
# DOWNLOAD REPORT
# =========================
@app.route("/download-report")
def download_report():
    if "user" not in session or "last_result" not in session:
        return redirect(url_for("dashboard"))

    result = session["last_result"]
    recommendations = session.get("last_recommendations", [])
    insights = session.get("last_insights", [])

    report = StringIO()
    report.write("AI VLSI POWER ANALYSIS REPORT\n")
    report.write("=" * 40 + "\n\n")

    report.write("RESULTS\n")
    for k, v in result.items():
        if k not in ["model_used", "status_color", "total_estimated_value"]:
            report.write(f"{k}: {v}\n")

    report.write("\nRECOMMENDATIONS\n")
    for r in recommendations:
        report.write(f"- {r}\n")

    report.write("\nENGINEERING INSIGHTS\n")
    for i in insights:
        report.write(f"- {i}\n")

    response = make_response(report.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=vlsi_analysis_report.txt"
    response.headers["Content-type"] = "text/plain"
    return response


# =========================
# HELP
# =========================
@app.route("/help")
def help_page():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("help.html", chat_question=None, chat_answer=None)


@app.route("/ask", methods=["POST"])
def ask():
    if "user" not in session:
        return redirect(url_for("login"))

    question = request.form["question"].strip().lower()

    if "accuracy" in question or "r2" in question:
        answer = "Model accuracy depends on dataset quality, selected features, and ML model performance."
    elif "feature importance" in question:
        answer = "Feature importance shows which input parameter affects prediction the most."
    elif "reduce power" in question or "power reduce" in question or "high power" in question:
        answer = "To reduce power, lower voltage, lower frequency, and optimize capacitance."
    elif "voltage" in question:
        answer = "Voltage has major impact because dynamic power depends on voltage square."
    elif "frequency" in question:
        answer = "Frequency affects dynamic power. Higher frequency usually increases power."
    elif "capacitance" in question:
        answer = "Capacitance directly affects dynamic power."
    elif "temperature" in question or "leakage" in question:
        answer = "Temperature mainly affects leakage power."
    elif "sensitivity" in question:
        answer = "Sensitivity analysis shows how output power changes when voltage changes."
    elif "dataset" in question:
        answer = "The dataset is generated using simplified VLSI power equations."
    elif "formula" in question:
        answer = "Total power is calculated as dynamic power plus leakage power."
    elif "units" in question or "unit" in question:
        answer = "Inputs use pF, V, MHz, °C, µA, and power outputs are displayed in mW."
    elif "color" in question or "status" in question:
        answer = "Power status is color graded: green for low power, orange for moderate power, and red for high power."
    else:
        answer = "You can ask about voltage, frequency, accuracy, units, dataset, formula, recommendations, and power status."

    return render_template("help.html", chat_question=question, chat_answer=answer)


# =========================
# LOGOUT
# =========================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True, port=5001)