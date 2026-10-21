import pandas as pd

# ---- Step 1: Load the raw data ----
print("Loading raw data...")
df = pd.read_excel("data/Online Retail.xlsx")
print(f"Raw data shape: {df.shape}")

# ---- Step 2: Remove rows with missing CustomerID ----
df = df.dropna(subset=["CustomerID"])
print(f"After removing missing CustomerID: {df.shape}")

# ---- Step 3: Remove cancelled orders (InvoiceNo starting with 'C') ----
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
print(f"After removing cancelled orders: {df.shape}")

# ---- Step 4: Remove invalid quantities and prices ----
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
print(f"After removing invalid quantity/price: {df.shape}")

# ---- Step 5: Remove duplicate rows ----
df = df.drop_duplicates()
print(f"After removing duplicates: {df.shape}")

# ---- Step 6: Save the cleaned data ----
df.to_csv("data/cleaned_data.csv", index=False)
print("Cleaned data saved to data/cleaned_data.csv")