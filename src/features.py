import pandas as pd
import datetime as dt

# ---- Step 1: Load the cleaned data ----
print("Loading cleaned data...")
df = pd.read_csv("data/cleaned_data.csv")

# ---- Step 2: Create a TotalPrice column (Quantity x UnitPrice) ----
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# ---- Step 3: Convert InvoiceDate to an actual date type ----
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# ---- Step 4: Set a reference date (1 day after the last purchase in the data) ----
reference_date = df["InvoiceDate"].max() + dt.timedelta(days=1)
print(f"Reference date for Recency calculation: {reference_date}")

# ---- Step 5: Group by customer and calculate RFM ----
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,   # Recency
    "InvoiceNo": "nunique",                                      # Frequency
    "TotalPrice": "sum"                                          # Monetary
})

# ---- Step 6: Rename columns clearly ----
rfm.rename(columns={
    "InvoiceDate": "Recency",
    "InvoiceNo": "Frequency",
    "TotalPrice": "Monetary"
}, inplace=True)

print(f"RFM table shape: {rfm.shape}")
print(rfm.head())

# ---- Step 7: Save the RFM table ----
rfm.to_csv("data/rfm.csv")
print("RFM data saved to data/rfm.csv")