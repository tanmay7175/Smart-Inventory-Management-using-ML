from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import os
import xgboost as xgb
import warnings
from stable_baselines3 import PPO

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# ✅ Load AI Models
xgb_model = None
ppo_model = None

try:
    if os.path.exists("inventory_model.pkl"):
        with open("inventory_model.pkl", "rb") as f:
            xgb_model = pickle.load(f)  # XGBoost for Demand Prediction
    else:
        print("❌ Error: 'inventory_model.pkl' not found!")

    if os.path.exists("ppo_inventory.zip"):
        ppo_model = PPO.load("ppo_inventory.zip")  # PPO for Inventory Optimization
    else:
        print("❌ Error: 'ppo_inventory.zip' not found!")

    print("✅ AI Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# ✅ Load CSV Data & Rename Columns
datasets = {
    "Sales": "SalesFINAL12312016.csv",
    "Purchases": "PurchasesFINAL12312016.csv",
    "Suppliers": "InvoicePurchases12312016.csv"
}
dataframes = {}

try:
    for name, file in datasets.items():
        if os.path.exists(file):
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()  # Trim whitespace from column names
            if "InventoryId" in df.columns:
                df.rename(columns={"InventoryId": "Inventory_ID"}, inplace=True)
            df["Inventory_ID"] = df["Inventory_ID"].astype(str).str.strip()  # Ensure correct type
            dataframes[name] = df
        else:
            print(f"⚠️ Warning: {file} not found! Skipping {name} dataset.")
    print("✅ Datasets loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading datasets: {e}")

@app.route('/')
def home():
    return render_template('home.html')

def calculate_eoq(demand, ordering_cost=50, holding_cost=10):
    """Economic Order Quantity (EOQ) Calculation"""
    try:
        demand = max(demand, 1)  # Ensure positive demand
        return round(np.sqrt((2 * demand * ordering_cost) / holding_cost), 2)
    except Exception:
        return "EOQ Calculation Failed"

def optimize_inventory(inventory_id):
    """Optimize Inventory using PPO Model"""
    try:
        if ppo_model is None:
            return "Inventory Optimization Error: PPO Model Not Loaded"

        sales_data = dataframes.get("Sales")
        if sales_data is None or "SalesQuantity" not in sales_data.columns:
            return "Inventory Optimization Error: Missing Sales Data"

        # ✅ Fetch past sales and stock levels
        sales_quantity = sales_data[sales_data["Inventory_ID"] == inventory_id]["SalesQuantity"].sum()
        total_sales = sales_data["SalesQuantity"].sum()
        total_stock = sales_data["SalesQuantity"].max()

        if total_sales == 0 or total_stock == 0:
            return "No Sales Data for Optimization"

        normalized_sales = sales_quantity / total_sales
        normalized_inventory = sales_quantity / total_stock
        normalized_end_inventory = normalized_inventory  # Placeholder if no end inventory data

        # ✅ Ensure PPO model gets (1,3) input shape
        state_features = np.array([[normalized_sales, normalized_inventory, normalized_end_inventory]], dtype=np.float32)
        
        # ✅ PPO Model Prediction
        action, _ = ppo_model.predict(state_features, deterministic=True)
        return round(float(action[0]), 2)
    
    except Exception as e:
        return f"Inventory Optimization Error: {str(e)}"

def process_inventory_data(inventory_id):
    """Process inventory data and return analysis"""
    result = {}
    inventory_id = str(inventory_id).strip()

    # ✅ Predict Demand using XGBoost
    demand_prediction = 0
    try:
        if xgb_model is not None:
            sales_data = dataframes.get("Sales")
            past_sales = sales_data[sales_data["Inventory_ID"] == inventory_id]["SalesQuantity"].sum() if sales_data is not None else 0
            inventory_numeric = np.array([[hash(inventory_id) % 10000, past_sales]], dtype=np.float32)

            demand_prediction = xgb_model.predict(inventory_numeric)[0]
            result["Predicted Demand"] = round(float(demand_prediction), 2)
        else:
            result["Predicted Demand"] = "XGBoost Model Not Loaded"
    except Exception as e:
        result["Predicted Demand"] = f"Error: {str(e)}"

    # ✅ Calculate EOQ
    result["EOQ"] = calculate_eoq(demand_prediction)

    # ✅ Check Stock Availability
    sales_data = dataframes.get("Sales")
    result["Stock Available"] = "Not Available"
    if sales_data is not None and "SalesQuantity" in sales_data.columns:
        stock_info = sales_data[sales_data["Inventory_ID"] == inventory_id]
        result["Stock Available"] = stock_info["SalesQuantity"].sum() if not stock_info.empty else "Not Available"

    # ✅ Optimize Inventory using PPO
    result["Optimized Restocking Quantity"] = optimize_inventory(inventory_id)

    # ✅ Restocking Alert
    stock_available = result.get("Stock Available", 0)
    if isinstance(stock_available, (int, float)) and stock_available < result["EOQ"]:
        result["Restocking Alert"] = "Restocking Required"
    else:
        result["Restocking Alert"] = "Sufficient Stock"

    # ✅ Major Suppliers
    supplier_data = dataframes.get("Suppliers")
    result["Major Suppliers"] = "No Supplier Data"
    if supplier_data is not None and "VendorName" in supplier_data.columns:
        supplier_info = supplier_data[supplier_data["Inventory_ID"] == inventory_id]
        result["Major Suppliers"] = list(supplier_info["VendorName"].unique()) if not supplier_info.empty else "No Supplier Data"

    # ✅ Total Sales Revenue
    result["Total Revenue"] = "No Sales Data"
    revenue_column = "SalesDollars" if "SalesDollars" in sales_data.columns else "Revenue"
    if sales_data is not None and revenue_column in sales_data.columns:
        revenue_info = sales_data[sales_data["Inventory_ID"] == inventory_id]
        result["Total Revenue"] = round(float(revenue_info[revenue_column].sum()), 2) if not revenue_info.empty else "No Sales Data"

    # ✅ Top 5 Selling Items
    result["Top 5 Selling Items"] = "No Data"
    if sales_data is not None and "SalesQuantity" in sales_data.columns:
        top_selling = sales_data.groupby("Inventory_ID")["SalesQuantity"].sum().nlargest(5)
        result["Top 5 Selling Items"] = dict(top_selling.items())

    return result

@app.route('/ai/<page>')
def render_ai_page(page):
    inventory_id = request.args.get('inventory_id')
    if not inventory_id:
        return "Error: Inventory ID is required!", 400

    result = process_inventory_data(inventory_id)
    return render_template(f"{page}.html", result=result, inventory_id=inventory_id)
