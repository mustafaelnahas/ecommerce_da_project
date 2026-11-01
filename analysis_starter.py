import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Import three related tables
customers = pd.read_csv("customers_raw.csv")
orders = pd.read_csv("orders_raw.csv")
items = pd.read_csv("order_items_raw.csv")

# Inspect
for name, table in [("customers",customers),("orders",orders),("items",items)]:
    print(f"\n{name}: shape={table.shape}")
    print(table.head())
    print(table.info())
    print(table.isna().sum())
    print("Duplicates:",table.duplicated().sum())

# NumPy summary
sales = pd.to_numeric(items["Line_Sales_EGP"],errors="coerce").dropna().to_numpy()
print("Median line sales:",np.median(sales))
print("95th percentile:",np.percentile(sales,95))

# Clean each table separately
customers=customers.drop_duplicates()
orders=orders.drop_duplicates()
items=items.drop_duplicates()
orders["Order_Date"]=pd.to_datetime(orders["Order_Date"],errors="coerce")
orders["Delivery_Date"]=pd.to_datetime(orders["Delivery_Date"],errors="coerce")
items.loc[items["Quantity"]<=0,"Quantity"]=np.nan
items.loc[items["Unit_Price_EGP"]<=0,"Unit_Price_EGP"]=np.nan
items.loc[(items["Discount_Rate"]<0)|(items["Discount_Rate"]>.8),"Discount_Rate"]=np.nan
items["Discount_Rate"]=items["Discount_Rate"].fillna(items["Discount_Rate"].median())
items["Quantity"]=items["Quantity"].fillna(items["Quantity"].median())
items["Unit_Price_EGP"]=items["Unit_Price_EGP"].fillna(items.groupby("Category")["Unit_Price_EGP"].transform("median"))
items["Line_Sales_EGP"]=(items["Quantity"]*items["Unit_Price_EGP"]*(1-items["Discount_Rate"])).round(2)
items["Line_Profit_EGP"]=items["Line_Sales_EGP"]-items["Quantity"]*items["Unit_Cost_EGP"]

# Aggregate items to one row per order before merging
order_totals=items.groupby("Order_ID",as_index=False).agg(
    Revenue=("Line_Sales_EGP","sum"),Profit=("Line_Profit_EGP","sum"),Units=("Quantity","sum"))
orders=orders.merge(order_totals,on="Order_ID",how="left")
orders=orders.merge(customers[["Customer_ID","City","Customer_Segment"]],on="Customer_ID",how="left")

# Example customer/order analysis
print(orders.groupby("Sales_Channel").agg(
    Orders=("Order_ID","nunique"),Revenue=("Revenue","sum")).round(2))
print(orders.groupby("City").agg(
    Orders=("Order_ID","nunique"),Revenue=("Revenue","sum")).round(2))

# Matplotlib
orders.groupby("Sales_Channel")["Revenue"].sum().sort_values().plot(
    kind="barh",title="Revenue by sales channel")
plt.tight_layout()
plt.show()

# Seaborn
sns.boxplot(data=items,x="Category",y="Line_Profit_EGP")
plt.xticks(rotation=25)
plt.title("Profit distribution by category")
plt.tight_layout()
plt.show()

# Plotly
monthly=orders.assign(Month=orders["Order_Date"].dt.to_period("M").astype(str)).groupby(
    "Month",as_index=False).agg(Revenue=("Revenue","sum"))
px.line(monthly,x="Month",y="Revenue",markers=True,title="Monthly revenue trend").show()
