import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==============================
# DATASET GENERATION
# ==============================
np.random.seed(42)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

n = 5000

alpha = np.random.uniform(0.1, 1.0, n)
C = np.random.uniform(1, 10, n)
load_C = np.random.uniform(0.5, 5, n)
V = np.random.uniform(0.8, 1.5, n)
f = np.random.uniform(50, 500, n)
temp = np.random.uniform(20, 100, n)
I_leak = np.random.uniform(0.001, 0.01, n)
gates = np.random.randint(1000, 10000, n)

# Dynamic and leakage power
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

# ==============================
# FEATURE ENGINEERING
# ==============================
data["Switching_Power_Component"] = alpha * (C + load_C) * (V ** 2) * f
data["Leakage_Power_Component"] = V * I_leak * (1 + 0.01 * temp)

print("✔ Dataset generated successfully")
data.to_csv("dataset.csv", index=False)

# ==============================
# MACHINE LEARNING PART
# ==============================
X = data[
    [
        "Alpha",
        "Capacitance",
        "Load_Capacitance",
        "Voltage",
        "Frequency",
        "Temperature",
        "Leakage_Current",
        "Gates",
        "Switching_Power_Component",
        "Leakage_Power_Component"
    ]
]
y = data["Power"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 1. Linear Regression
# ------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

lr_r2 = r2_score(y_test, lr_predictions)
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_mse = mean_squared_error(y_test, lr_predictions)

print("\n[ LINEAR REGRESSION EVALUATION ]")
print("R2 Score:", lr_r2)
print("MAE:", lr_mae)
print("MSE:", lr_mse)

# ------------------------------
# 2. Random Forest
# ------------------------------
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)

print("\n[ RANDOM FOREST EVALUATION ]")
print("R2 Score:", rf_r2)
print("MAE:", rf_mae)
print("MSE:", rf_mse)

# ------------------------------
# 3. Gradient Boosting
# ------------------------------
gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)

gb_r2 = r2_score(y_test, gb_predictions)
gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_mse = mean_squared_error(y_test, gb_predictions)

print("\n[ GRADIENT BOOSTING EVALUATION ]")
print("R2 Score:", gb_r2)
print("MAE:", gb_mae)
print("MSE:", gb_mse)

# ==============================
# MODEL COMPARISON
# ==============================
model_scores = {
    "Linear Regression": lr_r2,
    "Random Forest": rf_r2,
    "Gradient Boosting": gb_r2
}

best_model_name = max(model_scores, key=model_scores.get)

if best_model_name == "Linear Regression":
    best_model = lr_model
elif best_model_name == "Random Forest":
    best_model = rf_model
else:
    best_model = gb_model

print("\n[ MODEL COMPARISON ]")
for model_name, score in model_scores.items():
    print(f"{model_name}: {score:.6f}")

print(f"\n✔ Best Model Selected: {best_model_name}")

# ==============================
# USER INPUT
# ==============================
def get_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("❌ Invalid input! Please enter a number.")

print("\n--- ENTER YOUR CIRCUIT PARAMETERS ---")

alpha_in = get_input("Enter Alpha (0–1): ")
cap = get_input("Enter Capacitance: ")
load_cap = get_input("Enter Load Capacitance: ")
voltage = get_input("Enter Voltage: ")
freq = get_input("Enter Frequency: ")
temp_in = get_input("Enter Temperature: ")
leak = get_input("Enter Leakage Current: ")
gates_in = int(get_input("Enter Number of Gates: "))

switching_component = alpha_in * (cap + load_cap) * (voltage ** 2) * freq
leakage_component = voltage * leak * (1 + 0.01 * temp_in)

user_df = pd.DataFrame([[
    alpha_in,
    cap,
    load_cap,
    voltage,
    freq,
    temp_in,
    leak,
    gates_in,
    switching_component,
    leakage_component
]], columns=X.columns)

predicted_power = best_model.predict(user_df)

# ==============================
# REPORT
# ==============================
print("\n" + "=" * 60)
print("        ⚡ AI VLSI POWER ANALYSIS REPORT")
print("=" * 60)

ideal_power = alpha_in * (cap + load_cap) * (voltage ** 2) * freq
pred_power = predicted_power[0]
error = abs(ideal_power - pred_power)

print("\n[ CORE RESULTS ]")
print(f"{'Best Model Used':<25}: {best_model_name}")
print(f"{'Predicted Power':<25}: {pred_power:.2f}")
print(f"{'Ideal Power':<25}: {ideal_power:.2f}")
print(f"{'Prediction Error':<25}: {error:.2f}")

dynamic_power = ideal_power
leakage_power = leak * temp_in * gates_in
total_power = dynamic_power + leakage_power

print("\n[ POWER BREAKDOWN ]")
print(f"{'Dynamic Power':<25}: {dynamic_power:.2f}")
print(f"{'Leakage Power':<25}: {leakage_power:.2f}")
print(f"{'Total Estimated':<25}: {total_power:.2f}")

test_voltage = voltage + 0.1
test_switching_component = alpha_in * (cap + load_cap) * (test_voltage ** 2) * freq
test_leakage_component = test_voltage * leak * (1 + 0.01 * temp_in)

test_input = pd.DataFrame([[
    alpha_in,
    cap,
    load_cap,
    test_voltage,
    freq,
    temp_in,
    leak,
    gates_in,
    test_switching_component,
    test_leakage_component
]], columns=X.columns)

new_power = best_model.predict(test_input)[0]
change = new_power - pred_power

print("\n[ SENSITIVITY ANALYSIS ]")
print(f"{'Voltage Change (+0.1)':<25}: {test_voltage:.2f}")
print(f"{'New Power':<25}: {new_power:.2f}")
print(f"{'Power Change':<25}: {change:.2f}")

print("\n[ DESIGN STATUS ]")
if pred_power > 3000:
    status = "HIGH POWER ⚠️"
elif pred_power > 1500:
    status = "MODERATE POWER ⚡"
else:
    status = "LOW POWER ✅"

print(f"{'System Status':<25}: {status}")

print("\n[ RECOMMENDATIONS ]")
recommendations = []

if pred_power > 3000:
    if voltage > 1.2:
        recommendations.append("Reduce voltage")
    if freq > 300:
        recommendations.append("Reduce frequency")
    if cap + load_cap > 10:
        recommendations.append("Optimize capacitance")
    if alpha_in > 0.7:
        recommendations.append("Reduce switching activity")

elif pred_power > 1500:
    recommendations.append("Optimize efficiency")
    if freq > 200:
        recommendations.append("Slightly reduce frequency")

else:
    if freq < 250:
        recommendations.append("Increase frequency for performance")
    if voltage < 1.1:
        recommendations.append("Increase voltage slightly")
    if alpha_in < 0.5:
        recommendations.append("Increase switching activity")

    recommendations.append("Design allows performance improvement")

if len(recommendations) == 0:
    print("✔ No major changes required")
else:
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

print("\n[ ENGINEERING INSIGHTS ]")
if dynamic_power > leakage_power:
    print("• Dynamic power dominates → focus on voltage & frequency")
else:
    print("• Leakage power dominates → focus on gate design")

if change > 0:
    print("• Power increases with voltage → voltage sensitive design")

if error < 100:
    print("• ML model accuracy is high")
else:
    print("• Model deviation observed")

print("\n" + "=" * 60)

# ==============================
# COMBINED GRAPHS
# ==============================
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Feature importance only if model supports it
if hasattr(best_model, "feature_importances_"):
    importance = best_model.feature_importances_
    features = X.columns
    sorted_idx = importance.argsort()

    axs[0].barh(features[sorted_idx], importance[sorted_idx])
    axs[0].set_title(f"Feature Importance ({best_model_name})")
    axs[0].set_xlabel("Importance")
else:
    axs[0].text(
        0.5, 0.5, "Feature importance not available",
        ha="center", va="center", fontsize=12
    )
    axs[0].set_title(f"Feature Importance ({best_model_name})")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

# Dynamic voltage vs power graph
voltages = np.linspace(0.8, 1.5, 25)
powers = []

for v in voltages:
    temp_switching = alpha_in * (cap + load_cap) * (v ** 2) * freq
    temp_leakage = v * leak * (1 + 0.01 * temp_in)

    temp_input = pd.DataFrame([[
        alpha_in,
        cap,
        load_cap,
        v,
        freq,
        temp_in,
        leak,
        gates_in,
        temp_switching,
        temp_leakage
    ]], columns=X.columns)

    p = best_model.predict(temp_input)[0]
    powers.append(p)

axs[1].plot(voltages, powers, marker='o')
axs[1].set_title("Voltage vs Power")
axs[1].set_xlabel("Voltage")
axs[1].set_ylabel("Power")

plt.tight_layout()
plt.savefig("analysis_graphs.png")

print("\n📊 All graphs saved as 'analysis_graphs.png'")
plt.show()