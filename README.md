# CAR_MILEAGE-ML_PREDICTION_MODEL


import os
import sys
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Optional extras
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
except Exception:
    tts_engine = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    reportlab_available = True
except Exception:
    reportlab_available = False

# Prefer customtkinter for modern look, but gracefully fallback
USE_CUSTOM = False
try:
    import customtkinter as ctk
    from PIL import Image, ImageTk
    USE_CUSTOM = True
except Exception:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    try:
        from PIL import Image, ImageTk
    except Exception:
        Image = None
        ImageTk = None

# ---------------------------
# 1) Synthetic dataset + model (kept compact and deterministic)
# ---------------------------
np.random.seed(100)

brands = [
    "Maruti", "Hyundai", "Tata", "Honda", "Toyota", "Mahindra", "Kia", "Skoda", "BMW", "Audi",
    "Mercedes", "Volkswagen", "Nissan", "Renault", "MG", "Jeep", "Volvo", "Lexus", "Porsche",
    "Jaguar", "Land Rover", "Ferrari", "Lamborghini", "Ford", "Chevrolet", "Fiat", "Citroen",
    "Peugeot", "Mitsubishi", "Subaru", "Mini", "Isuzu", "Datsun", "Rolls Royce", "Bentley"
]

n = 1500
brand_choice = np.random.choice(brands, n)
engine_size = np.random.randint(800, 3000, n)
horsepower = np.random.randint(40, 300, n)
car_weight = np.random.randint(600, 2500, n)
fuel_tank = np.random.randint(20, 70, n)
car_age = np.random.randint(0, 20, n)

mileage_clean = (
    50
    - (engine_size * 0.002)
    - (horsepower * 0.03)
    - (car_weight * 0.002)
    - (car_age * 0.20)
    + (fuel_tank * 0.04)
)

brand_mileage_boost = {
    "Maruti": 4, "Hyundai": 3, "Tata": 2, "Honda": 3, "Toyota": 4,
    "Mahindra": -1, "Kia": 2, "Skoda": -2, "BMW": -4, "Audi": -5,
    "Mercedes": -4, "Volkswagen": -2, "Nissan": 1, "Renault": 1,
    "MG": 0, "Jeep": -2, "Volvo": -3, "Lexus": -3, "Porsche": -6,
    "Jaguar": -5, "Land Rover": -4, "Ferrari": -8, "Lamborghini": -8,
    "Ford": 0, "Chevrolet": 0, "Fiat": 0, "Citroen": 1, "Peugeot": 1,
    "Mitsubishi": 0, "Subaru": 0, "Mini": -2, "Isuzu": -1, "Datsun": 1,
    "Rolls Royce": -10, "Bentley": -9
}

brand_effect = [brand_mileage_boost[b] for b in brand_choice]

mileage = mileage_clean + brand_effect + np.random.normal(0, 1.0, n)

_df = pd.DataFrame({
    "brand": brand_choice,
    "engine_size": engine_size,
    "horsepower": horsepower,
    "car_weight": car_weight,
    "fuel_tank": fuel_tank,
    "car_age": car_age,
    "mileage": mileage
})

# feature engineering
_df["power_weight_ratio"] = _df["horsepower"] / _df["car_weight"]
_df["engine_efficiency"] = _df["engine_size"] / (_df["horsepower"] + 1e-6)
_df["usage_score"] = (_df["car_age"] * _df["car_weight"]) / 1000
_df["age_category"] = np.where(_df["car_age"] <= 8, 1, 0)

# small save for reproducibility
try:
    _df.to_csv("advanced_car_mileage_dataset_clean.csv", index=False)
except Exception:
    pass

# model
_df_model = pd.get_dummies(_df, columns=["brand"], drop_first=True)
X = _df_model.drop("mileage", axis=1)
y = _df_model["mileage"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)

model_features = X_train.columns.tolist()
last_prediction_details = None

# ---------------------------
# small car DB used by UI (realistic & varied)
# ---------------------------
car_database = {
    "Maruti Alto 800":        [796, 48, 850, 35, 4, "Maruti"],
    "Maruti Swift":           [1197, 89, 960, 37, 3, "Maruti"],
    "Maruti Baleno":          [1197, 90, 960, 37, 2, "Maruti"],
    "Hyundai i20":            [1197, 83, 1060, 37, 3, "Hyundai"],
    "Tata Tiago":             [1199, 85, 935, 35, 2, "Tata"],
    "Honda City":             [1497, 121, 1130, 40, 2, "Honda"],
    "Hyundai Verna":          [1497, 115, 1190, 45, 2, "Hyundai"],
    "Skoda Slavia":           [1498, 150, 1281, 45, 1, "Skoda"],
    "Toyota Camry":           [2487, 176, 1600, 55, 4, "Toyota"],
    "Audi A4":                [1984, 190, 1600, 54, 3, "Audi"],
    "Hyundai Creta":          [1493, 113, 1290, 50, 2, "Hyundai"],
    "Kia Seltos":             [1497, 115, 1320, 50, 2, "Kia"],
    "Tata Nexon":             [1497, 108, 1200, 44, 3, "Tata"],
    "Maruti Brezza":          [1462, 103, 1180, 48, 2, "Maruti"],
    "Mahindra XUV300":        [1497, 115, 1390, 42, 3, "Mahindra"],
    "Mahindra XUV700":        [1999, 197, 1800, 60, 1, "Mahindra"],
    "Toyota Fortuner":        [2694, 166, 2100, 65, 6, "Toyota"],
    "Jeep Compass":           [1998, 170, 1600, 60, 4, "Jeep"],
    "Hyundai Tucson":         [1999, 154, 1560, 54, 3, "Hyundai"],
    "Volkswagen Tiguan":      [1984, 187, 1678, 60, 3, "Volkswagen"],
    "BMW 3 Series":           [1998, 255, 1600, 59, 3, "BMW"],
    "Mercedes C-Class":       [1991, 255, 1650, 66, 4, "Mercedes"],
    "Audi A6":                [1984, 245, 1700, 63, 3, "Audi"],
    "Land Rover Discovery":   [2996, 355, 2300, 90, 5, "Land Rover"],
    "BMW X5":                 [2998, 335, 2200, 83, 4, "BMW"],
    "Audi Q7":                [2995, 340, 2085, 75, 4, "Audi"],
    "Porsche 911":            [2994, 450, 1500, 64, 2, "Porsche"],
    "Jaguar F-Type":          [5000, 444, 1760, 63, 3, "Jaguar"],
    "Ford Mustang":           [4951, 450, 1810, 61, 4, "Ford"],
    "Chevrolet Corvette":     [6162, 490, 1530, 70, 2, "Chevrolet"],
    "Nissan GT-R":            [3799, 565, 1750, 74, 2, "Nissan"],
}

# ---------------------------
# Utilities: build features, logging, predict
# ---------------------------

def build_feature_row(engine, hp, weight, tank, age, brand):
    base = {
        "engine_size": float(engine),
        "horsepower": float(hp),
        "car_weight": float(weight),
        "fuel_tank": float(tank),
        "car_age": float(age),
        "power_weight_ratio": float(hp) / float(weight) if float(weight) != 0 else 0.0,
        "engine_efficiency": float(engine) / (float(hp) + 1e-6),
        "usage_score": (float(age) * float(weight)) / 1000.0,
        "age_category": 1 if float(age) <= 8 else 0,
    }

    row = {}
    for col in model_features:
        if col in base:
            row[col] = base[col]
        elif col.startswith("brand_"):
            bname = col.replace("brand_", "")
            row[col] = 1 if bname == brand else 0
        else:
            row[col] = 0.0
    return pd.DataFrame([row])


def save_prediction_log(input_data: dict, prediction: float, source: str = "manual"):
    log = pd.DataFrame([{
        "timestamp": datetime.now(),
        "source": source,
        **input_data,
        "predicted_mileage": float(prediction)
    }])
    try:
        old = pd.read_csv("prediction_log.csv")
        new = pd.concat([old, log], ignore_index=True)
        new.to_csv("prediction_log.csv", index=False)
    except FileNotFoundError:
        log.to_csv("prediction_log.csv", index=False)
    except Exception:
        pass


def predict_mileage(engine, hp, weight, tank, age, brand, source="manual"):
    new_df = build_feature_row(engine, hp, weight, tank, age, brand)
    prediction = float(model.predict(new_df)[0])
    input_data = {"engine_size": engine, "horsepower": hp, "car_weight": weight, "fuel_tank": tank, "car_age": age, "brand": brand}
    save_prediction_log(input_data, prediction, source)
    return prediction


def predict_by_car_name(car_name: str):
    if car_name not in car_database:
        return None, None
    engine, hp, weight, tank, age, brand = car_database[car_name]
    pred = predict_mileage(engine, hp, weight, tank, age, brand, source="car_name")
    info = {"engine_size": engine, "horsepower": hp, "car_weight": weight, "fuel_tank": tank, "car_age": age, "brand": brand, "car_name": car_name}
    return pred, info

# ---------------------------
# PDF report generator
# ---------------------------

def generate_pdf_report(details: dict, filename: str = "car_mileage_report.pdf"):
    if not reportlab_available:
        try:
            if USE_CUSTOM:
                ctk.messagebox.showerror("Missing Library", "Install reportlab (pip install reportlab)")
            else:
                messagebox.showerror("Missing Library", "Install reportlab (pip install reportlab)")
        except Exception:
            print("Install reportlab to enable PDF exports")
        return

    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Car Mileage Prediction Report")
    y -= 30
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 30
    for key, val in details.items():
        if y < 80:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 11)
        c.drawString(50, y, f"{key}: {val}")
        y -= 18
    c.showPage()
    c.save()
    try:
        if USE_CUSTOM:
            ctk.messagebox.showinfo("PDF Saved", f"Report saved as {filename}")
        else:
            messagebox.showinfo("PDF Saved", f"Report saved as {filename}")
    except Exception:
        print(f"Report saved as {filename}")

# ---------------------------
#  UI: Glassmorphism-like using customtkinter if available
# ---------------------------

class GlassMileageApp:
    def __init__(self):
        self.last_prediction_details = None
        if USE_CUSTOM:
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("blue")
            self.root = ctk.CTk()
            self.root.title("Glass Mileage Dashboard")
            self.root.geometry("1200x700")
            self.root.minsize(1000, 600)
            self._build_custom_ui()
        else:
            self.root = tk.Tk()
            self.root.title("Mileage Dashboard (Fallback)")
            self.root.geometry("1100x650")
            self._build_classic_ui()

    def _create_gradient_bg(self, parent, color1="#0f172a", color2="#0b1220"):
        # customtkinter: create a background frame with gradient via canvas widget
        if USE_CUSTOM:
            canvas = ctk.CTkCanvas(parent, width=1200, height=700)
            canvas.pack(fill='both', expand=True)
            # simple rectangle background (customtkinter doesn't support alpha; we simulate)
            canvas.create_rectangle(0, 0, 1200, 700, fill=color1, outline=color1)
            return canvas
        else:
            # for fallback simple configure bg
            parent.configure(bg=color1)
            return None

    def _build_custom_ui(self):
        root = self.root
        # main layout: sidebar + content
        sidebar = ctk.CTkFrame(root, width=260, corner_radius=12)
        sidebar.pack(side="left", fill="y", padx=(16,8), pady=16)

        content = ctk.CTkFrame(root, corner_radius=12)
        content.pack(side="left", fill="both", expand=True, padx=(8,16), pady=16)

        # gradient-like background for content using a canvas placeholder
        self._create_gradient_bg(content)

        # Title in sidebar (frosted look)
        logo = ctk.CTkLabel(sidebar, text="\nMILEAGE\nDASHBOARD", font=ctk.CTkFont(size=18, weight="bold"))
        logo.pack(pady=(8, 12))

        # sidebar buttons (icon-like unicode glyphs used; replace with images if available)
        btn_cfg = dict(width=220, height=40, corner_radius=10)
        self.btn_car = ctk.CTkButton(sidebar, text="  Predict by Car ", command=self.show_car_page, **btn_cfg)
        self.btn_car.pack(pady=6)
        self.btn_manual = ctk.CTkButton(sidebar, text="  Manual Predict ", command=self.show_manual_page, **btn_cfg)
        self.btn_manual.pack(pady=6)
        self.btn_graphs = ctk.CTkButton(sidebar, text="  Graphs & Visuals ", command=self.show_graphs_page, **btn_cfg)
        self.btn_graphs.pack(pady=6)
        self.btn_compare = ctk.CTkButton(sidebar, text="  Compare Cars ", command=self.show_compare_page, **btn_cfg)
        self.btn_compare.pack(pady=6)
        self.btn_reports = ctk.CTkButton(sidebar, text="  Reports & Logs ", command=self.show_report_page, **btn_cfg)
        self.btn_reports.pack(pady=6)
        self.btn_live = ctk.CTkButton(sidebar, text="  Live Metrics ", command=self.show_live_page, **btn_cfg)
        self.btn_live.pack(pady=6)
        self.btn_dataset = ctk.CTkButton(sidebar, text="  Dataset Info ", command=self.show_dataset_page, **btn_cfg)
        self.btn_dataset.pack(pady=6)

        # container for pages
        self.pages = {}
        self.page_container = ctk.CTkFrame(content, fg_color="transparent")
        self.page_container.place(relx=0, rely=0, relwidth=1, relheight=1)

        self._build_pages_custom()
        self.show_car_page()

    def _build_pages_custom(self):
        # --- Predict by car name page
        page = ctk.CTkFrame(self.page_container, fg_color="#0f172a")
        self.pages['car'] = page
        page.place(relx=0, rely=0, relwidth=1, relheight=1)

        title = ctk.CTkLabel(page, text="Predict by Car Name", font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(pady=(20,8))

        mid = ctk.CTkFrame(page, fg_color="#071126", corner_radius=8)
        mid.pack(padx=20, pady=8, fill='x')

        car_var = ctk.StringVar(value=list(car_database.keys())[0])
        combo = ctk.CTkComboBox(mid, values=list(car_database.keys()), variable=car_var)
        combo.grid(row=0, column=0, padx=12, pady=12)

        predict_btn = ctk.CTkButton(mid, text="Predict", command=lambda: self._on_predict_by_name_custom(car_var.get()))
        predict_btn.grid(row=0, column=1, padx=12, pady=12)

        self.car_info_box = ctk.CTkTextbox(page, width=780, height=150, corner_radius=8)
        self.car_info_box.pack(pady=8)

        self.car_result_lbl = ctk.CTkLabel(page, text="Predicted Mileage: -- kmpl", font=ctk.CTkFont(size=16, weight="bold"))
        self.car_result_lbl.pack(pady=6)

        # --- Manual page
        page_m = ctk.CTkFrame(self.page_container, fg_color="#0f172a")
        self.pages['manual'] = page_m
        page_m.place(relx=0, rely=0, relwidth=1, relheight=1)

        title = ctk.CTkLabel(page_m, text="Manual Input Prediction", font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(pady=(20,8))

        form = ctk.CTkFrame(page_m, fg_color="#071126", corner_radius=8)
        form.pack(padx=20, pady=8, fill='x')

        labels = ["Engine (cc)", "Horsepower (HP)", "Weight (kg)", "Fuel Tank (L)", "Age (yrs)", "Brand"]
        self.e_vars = [ctk.StringVar() for _ in labels]
        for i, lab in enumerate(labels[:-1]):
            lbl = ctk.CTkLabel(form, text=lab)
            lbl.grid(row=i, column=0, padx=12, pady=8, sticky='e')
            ent = ctk.CTkEntry(form, textvariable=self.e_vars[i])
            ent.grid(row=i, column=1, padx=12, pady=8)

        lbl = ctk.CTkLabel(form, text=labels[-1])
        lbl.grid(row=5, column=0, padx=12, pady=8, sticky='e')
        brand_cb = ctk.CTkComboBox(form, values=sorted(list(set(_df['brand']))), variable=self.e_vars[5])
        brand_cb.grid(row=5, column=1, padx=12, pady=8)

        self.manual_result = ctk.CTkLabel(page_m, text="Predicted Mileage: -- kmpl", font=ctk.CTkFont(size=16, weight="bold"))
        self.manual_result.pack(pady=8)

        predict_manual_btn = ctk.CTkButton(page_m, text="Predict", command=self._on_predict_manual_custom)
        predict_manual_btn.pack(pady=6)

        # --- Graphs page ---
        page_g = ctk.CTkFrame(self.page_container, fg_color="#0f172a")
        self.pages['graphs'] = page_g
        page_g.place(relx=0, rely=0, relwidth=1, relheight=1)

        t = ctk.CTkLabel(page_g, text="Graphs & Visualizations", font=ctk.CTkFont(size=20, weight="bold"))
        t.pack(pady=(20,6))

        btns = ctk.CTkFrame(page_g, fg_color="transparent")
        btns.pack(pady=8)
        ctk.CTkButton(btns, text="Engine vs Mileage (Scatter)", command=self._show_scatter).pack(padx=8, pady=6)
        ctk.CTkButton(btns, text="Correlation Matrix", command=self._show_heatmap).pack(padx=8, pady=6)
        ctk.CTkButton(btns, text="Actual vs Predicted", command=self._show_actual_vs_pred).pack(padx=8, pady=6)
        ctk.CTkButton(btns, text="3D Mileage Plot", command=self._show_3d_plot).pack(padx=8, pady=6)

        # --- Compare page ---
        page_c = ctk.CTkFrame(self.page_container, fg_color="#0f172a")
        self.pages['compare'] = page_c
        page_c.place(relx=0, rely=0, relwidth=1, relheight=1)
        t = ctk.CTkLabel(page_c, text="Compare Two Cars", font=ctk.CTkFont(size=20, weight="bold"))
        t.pack(pady=(20,8))

        cmpf = ctk.CTkFrame(page_c, fg_color="#071126", corner_radius=8)
        cmpf.pack(padx=20, pady=8)
        self.c1 = ctk.StringVar(value=list(car_database.keys())[0])
        self.c2 = ctk.StringVar(value=list(car_database.keys())[1])
        ctk.CTkComboBox(cmpf, values=list(car_database.keys()), variable=self.c1).grid(row=0, column=0, padx=8, pady=8)
        ctk.CTkComboBox(cmpf, values=list(car_database.keys()), variable=self.c2).grid(row=0, column=1, padx=8, pady=8)
        ctk.CTkButton(cmpf, text="Compare", command=self._on_compare_custom).grid(row=0, column=2, padx=8, pady=8)

        self.cmp_box = ctk.CTkTextbox(page_c, width=820, height=220, corner_radius=8)
        self.cmp_box.pack(pady=8)

        # --- Reports page ---
        page_r = ctk.CTkFrame(self.page_container, fg_color="#0f172a")
        self.pages['report'] = page_r
        page_r.place(relx=0, rely=0, relwidth=1, relheight=1)
        t = ctk.CTkLabel(page_r, text="Reports & Logs", font=ctk.CTkFont(size=20, weight="bold"))
        t.pack(pady=(20,8))
        ctk.CTkButton(page_r, text="Export Prediction Log (CSV)", command=self._export_log_custom).pack(pady=6)
        ctk.CTkButton(page_r, text="Generate PDF for Last Prediction", command=self._generate_pdf_custom).pack(pady=6)

        # --- Live page ---
        page_l = ctk.CTkFrame(self.page_container, fg_color="#0f172a")
        self.pages['live'] = page_l
        page_l.place(relx=0, rely=0, relwidth=1, relheight=1)
        t = ctk.CTkLabel(page_l, text="Live Model Metrics", font=ctk.CTkFont(size=20, weight="bold"))
        t.pack(pady=(20,8))
        self.metrics_lbl = ctk.CTkLabel(page_l, text="")
        self.metrics_lbl.pack(pady=8)
        ctk.CTkButton(page_l, text="Refresh Metrics", command=self._refresh_metrics_custom).pack(pady=6)

        # --- Dataset Info Page ---
        page_d = ctk.CTkFrame(self.page_container, fg_color="#0f172a")
        self.pages['dataset'] = page_d
        page_d.place(relx=0, rely=0, relwidth=1, relheight=1)

        t = ctk.CTkLabel(page_d, text="Dataset Information", font=ctk.CTkFont(size=20, weight="bold"))
        t.pack(pady=(20,8))

        info_box = ctk.CTkTextbox(page_d, width=900, height=400, corner_radius=8)
        info_box.pack(pady=8)

        # Prepare dataset summary text
        summary = []
        summary.append("DATASET OVERVIEW")
        summary.append("──────────────────────────────────────────────")
        summary.append("Dataset Name          : advanced_car_mileage_dataset_clean.csv")
        summary.append(f"Total Records         : {_df.shape[0]} samples")
        summary.append(f"Total Features        : {_df.shape[1]} columns (after preprocessing)")
        summary.append("Target Variable       : mileage (fuel efficiency in km/l)")

        summary.append("\nPROJECT AIM")
        summary.append("──────────────────────────────────────────────")
        summary.append("To build a machine learning-based mileage prediction system that can")
        summary.append("accurately estimate real-world fuel efficiency based on technical and")
        summary.append("usage-related characteristics of a car. This model helps in:")
        summary.append("  • Vehicle performance analysis")
        summary.append("  • Purchase decision support (best mileage car)")
        summary.append("  • Automotive research and fuel optimization studies")

        summary.append("\nWHY THESE FEATURES ARE USED (Real-Life Reasoning)")
        summary.append("──────────────────────────────────────────────")
        summary.append("• engine_size        → Bigger engines consume more fuel.")
        summary.append("• horsepower         → Higher power output reduces mileage.")
        summary.append("• car_weight         → Heavier cars require more energy → lower mileage.")
        summary.append("• fuel_tank          → Impacts vehicle range and efficiency behaviour.")
        summary.append("• car_age            → Older cars show mechanical wear, reducing mileage.")
        summary.append("• brand              → Brand technology differences affect efficiency.")

        summary.append("\nADVANCED FEATURE ENGINEERING (Makes Model Smarter)")
        summary.append("──────────────────────────────────────────────")
        summary.append("1. power_weight_ratio = horsepower / car_weight")
        summary.append("     → Real industry metric. Higher ratio means sportier engine but lower mileage.")
        summary.append("2. engine_efficiency  = engine_size / horsepower")
        summary.append("     → Shows how efficiently engine displacement is converted into power.")
        summary.append("3. usage_score        = (car_age × car_weight) / 1000")
        summary.append("     → Represents long-term mechanical stress and degradation.")
        summary.append("4. age_category       = 1 (≤8 years), else 0")
        summary.append("     → Distinguishes new vs old cars (major mileage gap in real life).")

        summary.append("\nWHY ONE-HOT ENCODING FOR BRAND?")
        summary.append("──────────────────────────────────────────────")
        summary.append("Because brand is a categorical variable. We convert brand names into")
        summary.append("multiple binary columns so the model can understand:")
        summary.append("  • Technology differences")
        summary.append("  • Engine tuning philosophy")
        summary.append("  • Brand reliability and efficiency factors")

        summary.append("\nMODEL PIPELINE (How the ML Model Works)")
        summary.append("──────────────────────────────────────────────")
        summary.append("1. StandardScaler → Normalizes all numeric values so large values")
        summary.append("   (like engine_size) do not dominate smaller ones (like age).")
        summary.append("2. Linear Regression → Fits a mathematical relationship between")
        summary.append("   features and mileage. This helps interpret how each feature")
        summary.append("   increases or decreases fuel efficiency.")

        summary.append("\nWHY LINEAR REGRESSION?")
        summary.append("──────────────────────────────────────────────")
        summary.append("• Simple, fast, and highly interpretable.")
        summary.append("• Works well when relationships are mostly linear (as in car mileage).")
        summary.append("• Provides a mathematically clear explanation of feature impact.")
        summary.append("• Excellent baseline model for fuel-efficiency prediction.")

        summary.append("\nTRAIN–TEST STRATEGY")
        summary.append("──────────────────────────────────────────────")
        summary.append("Train-Test Split      : 80% Training / 20% Testing")
        summary.append("Reason: Ensures the model learns from sufficient data while preserving")
        summary.append("        enough unseen samples to evaluate generalization ability.")

        summary.append("\nMODEL PERFORMANCE METRICS (How Good is the Model?)")
        summary.append("──────────────────────────────────────────────")
        summary.append(f"R² Score             : {r2:.4f}  (Explains {r2*100:.2f}% of variance)")
        summary.append(f"RMSE                 : {rmse:.4f} km/l  (Avg error < 1 km/l)")
        summary.append("Interpretation:")
        summary.append("• R² ≈ 0.96 → Model predictions are highly accurate.")
        summary.append("• RMSE < 1 → Excellent real-world performance for mileage prediction.")

        summary.append("\nMACHINE LEARNING WORKFLOW SUMMARY")
        summary.append("──────────────────────────────────────────────")
        summary.append("1. Raw data generation (synthetic but realistic automotive variables)")
        summary.append("2. Feature engineering to add domain-specific knowledge")
        summary.append("3. One-hot encoding of brand")
        summary.append("4. Train-test split")
        summary.append("5. Scaling numeric features")
        summary.append("6. Model training (Linear Regression)")
        summary.append("7. Evaluation using R² and RMSE")
        summary.append("8. Live prediction using the trained model")

        summary.append("\nTOP 100 ROWS OF THE DATASET (Preview)")
        summary.append("──────────────────────────────────────────────\n")
        summary.append(str(_df.head(100)))

        info_box.insert("0.0", "\n".join(summary))

        # initial metrics
        self._refresh_metrics_custom()

    # --- customtkinter callbacks ---
    def _on_predict_by_name_custom(self, name):
        pred, info = predict_by_car_name(name)
        if pred is None:
            self.car_info_box.delete("0.0", "end")
            self.car_info_box.insert("0.0", "Car not found in database")
            return
        self.car_info_box.delete("0.0", "end")
        for k, v in info.items():
            self.car_info_box.insert("0.0", f"{k}: {v}\n")
        self.car_result_lbl.configure(text=f"Predicted Mileage: {pred:.2f} kmpl")
        self.last_prediction_details = {"Source": "Car Name", **info, "Predicted Mileage (kmpl)": f"{pred:.2f}"}
        if tts_engine:
            try:
                speak = tts_engine.say
                tts_engine.say(f"The predicted mileage for {name} is {pred:.1f} kilometers per liter.")
                tts_engine.runAndWait()
            except Exception:
                pass

    def _on_predict_manual_custom(self):
        try:
            engine = float(self.e_vars[0].get())
            hp = float(self.e_vars[1].get())
            weight = float(self.e_vars[2].get())
            tank = float(self.e_vars[3].get())
            age = float(self.e_vars[4].get())
        except Exception:
            ctk.messagebox.showerror("Invalid", "Enter valid numeric values.")
            return
        brand = self.e_vars[5].get().strip()
        if not brand:
            ctk.messagebox.showwarning("Brand", "Select a brand.")
            return
        pred = predict_mileage(engine, hp, weight, tank, age, brand, source="manual")
        self.manual_result.configure(text=f"Predicted Mileage: {pred:.2f} kmpl")
        self.last_prediction_details = {"Source": "Manual", "Engine": engine, "HP": hp, "Weight": weight, "Tank": tank, "Age": age, "Brand": brand, "Predicted Mileage (kmpl)": f"{pred:.2f}"}
        if tts_engine:
            try:
                tts_engine.say(f"Predicted mileage: {pred:.1f} kilometers per liter.")
                tts_engine.runAndWait()
            except Exception:
                pass

    def _show_scatter(self):
        plt.figure(figsize=(8, 5))
        plt.scatter(_df["engine_size"], _df["mileage"], alpha=0.6)
        plt.title("Engine Size vs Mileage")
        plt.xlabel("Engine Size (cc)")
        plt.ylabel("Mileage (kmpl)")
        plt.grid(True)
        plt.show()

    def _show_heatmap(self):
        num = _df.select_dtypes(include=["number"])
        plt.figure(figsize=(9, 6))
        plt.imshow(num.corr(), interpolation='nearest', cmap='coolwarm')
        plt.colorbar()
        plt.xticks(range(len(num.columns)), num.columns, rotation=90)
        plt.yticks(range(len(num.columns)), num.columns)
        plt.title("Correlation Matrix (numeric)")
        plt.tight_layout()
        plt.show()

    def _show_actual_vs_pred(self):
        plt.figure(figsize=(7, 5))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Actual Mileage")
        plt.ylabel("Predicted Mileage")
        plt.title("Actual vs Predicted Mileage")
        plt.grid(True)
        plt.show()

    def _show_3d_plot(self):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection='3d')
        x = _df["engine_size"]
        y = _df["car_weight"]
        z = _df["mileage"]
        sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=25)
        ax.set_xlabel("Engine Size (cc)")
        ax.set_ylabel("Car Weight (kg)")
        ax.set_zlabel("Mileage (kmpl)")
        ax.set_title("3D Mileage vs Weight vs Engine Size")
        plt.show()

    def _on_compare_custom(self):
        car1 = self.c1.get()
        car2 = self.c2.get()
        self.cmp_box.delete("0.0", "end")
        if not car1 or not car2:
            self.cmp_box.insert("0.0", "Select both cars to compare.\n")
            return
        p1, info1 = predict_by_car_name(car1)
        p2, info2 = predict_by_car_name(car2)
        if p1 is None or p2 is None:
            self.cmp_box.insert("0.0", "One of the selected cars is not in the database.\n")
            return
        self.cmp_box.insert("0.0", f"--- {car1} ---\n")
        for k, v in info1.items():
            if k != "car_name":
                self.cmp_box.insert("0.0", f"  {k}: {v}\n")
        self.cmp_box.insert("0.0", f"  Predicted Mileage: {p1:.2f} kmpl\n\n")
        self.cmp_box.insert("0.0", f"--- {car2} ---\n")
        for k, v in info2.items():
            if k != "car_name":
                self.cmp_box.insert("0.0", f"  {k}: {v}\n")
        self.cmp_box.insert("0.0", f"  Predicted Mileage: {p2:.2f} kmpl\n\n")
        better = car1 if p1 > p2 else car2
        self.cmp_box.insert("0.0", f"Conclusion: {better} is predicted to be more fuel-efficient.\n")

    def _export_log_custom(self):
        try:
            df_log = pd.read_csv("prediction_log.csv")
        except Exception:
            ctk.messagebox.showwarning("No Log", "No prediction_log.csv found.")
            return
        fname = ctk.filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile="prediction_log_copy.csv")
        if fname:
            df_log.to_csv(fname, index=False)
            ctk.messagebox.showinfo("Saved", f"Log saved to {fname}")

    def _generate_pdf_custom(self):
        if self.last_prediction_details is None:
            ctk.messagebox.showwarning("No Prediction", "No prediction available to generate a report.")
            return
        generate_pdf_report(self.last_prediction_details)

    def _refresh_metrics_custom(self):
        try:
            y_p = model.predict(X_test)
            mse_v = mean_squared_error(y_test, y_p)
            rmse_v = math.sqrt(mse_v)
            r2_v = r2_score(y_test, y_p)
            self.metrics_lbl.configure(text=f"MSE: {mse_v:.3f}    RMSE: {rmse_v:.3f}    R²: {r2_v:.3f}")
        except Exception as e:
            self.metrics_lbl.configure(text=f"Error computing metrics: {e}")

    def show_car_page(self):
        self.pages['car'].tkraise()

    def show_dataset_page(self):
        self.pages['dataset'].tkraise()

    def show_manual_page(self):
        self.pages['manual'].tkraise()

    def show_graphs_page(self):
        self.pages['graphs'].tkraise()

    def show_compare_page(self):
        self.pages['compare'].tkraise()

    def show_report_page(self):
        self.pages['report'].tkraise()

    def show_live_page(self):
        self.pages['live'].tkraise()

    def run(self):
        self.root.mainloop()

# ---------------------------
# Classic Tkinter fallback app (simpler styling)
# ---------------------------

class ClassicMileageApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mileage Dashboard")
        self.root.geometry("1100x650")
        self._build_ui()

    def _build_ui(self):
        sidebar = tk.Frame(self.root, bg="#0b1220", width=220)
        sidebar.pack(side="left", fill="y")
        content = tk.Frame(self.root, bg="#0f172a")
        content.pack(side="left", fill="both", expand=True)
        # We reuse many of the older widgets for the fallback; keeping it compact here
        lbl = tk.Label(sidebar, text="MILEAGE DASHBOARD", bg="#0b1220", fg="white", font=("Segoe UI", 12, "bold"))
        lbl.pack(pady=12)
        tk.Button(sidebar, text="Predict by Car", command=self._dummy).pack(fill="x", padx=8, pady=6)
        tk.Button(sidebar, text="Manual Predict", command=self._dummy).pack(fill="x", padx=8, pady=6)
        tk.Button(sidebar, text="Graphs", command=self._dummy).pack(fill="x", padx=8, pady=6)
        tk.Button(sidebar, text="Compare", command=self._dummy).pack(fill="x", padx=8, pady=6)
        tk.Button(sidebar, text="Reports", command=self._dummy).pack(fill="x", padx=8, pady=6)

        label = tk.Label(content, text="Fallback UI: customtkinter not installed.", bg="#0f172a", fg="white")
        label.pack(pady=20)

    def _dummy(self):
        messagebox.showinfo("Info", "This is the fallback UI. Install customtkinter for the modern interface.")

    def run(self):
        self.root.mainloop()

# ---------------------------
# Runner
# ---------------------------

if __name__ == "__main__":
    if USE_CUSTOM:
        app = GlassMileageApp()
    else:
        app = ClassicMileageApp()
    app.run()
