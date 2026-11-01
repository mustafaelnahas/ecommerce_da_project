import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="E-commerce Customer & Orders Analysis",layout="wide")
st.title("E-commerce Customer & Orders Analysis")
st.caption("Challenge-based graduation project • Synthetic e-commerce data • EGP")

@st.cache_data
def load_tables():
    customers=pd.read_csv("customers_raw.csv")
    orders=pd.read_csv("orders_raw.csv")
    items=pd.read_csv("order_items_raw.csv")
    return customers,orders,items

customers_raw,orders_raw,items_raw=load_tables()
customer_count_before=len(customers_raw); order_count_before=len(orders_raw); item_count_before=len(items_raw)

customers=customers_raw.drop_duplicates().copy()
orders=orders_raw.drop_duplicates().copy()
items=items_raw.drop_duplicates().copy()
customers["Signup_Date"]=pd.to_datetime(customers["Signup_Date"],errors="coerce")
orders["Order_Date"]=pd.to_datetime(orders["Order_Date"],errors="coerce")
orders["Delivery_Date"]=pd.to_datetime(orders["Delivery_Date"],errors="coerce")

for col in ["Quantity","Unit_Price_EGP","Discount_Rate","Unit_Cost_EGP","Line_Sales_EGP","Line_Cost_EGP","Line_Profit_EGP"]:
    items[col]=pd.to_numeric(items[col],errors="coerce")
items.loc[items["Quantity"]<=0,"Quantity"]=np.nan
items.loc[items["Unit_Price_EGP"]<=0,"Unit_Price_EGP"]=np.nan
items.loc[(items["Discount_Rate"]<0)|(items["Discount_Rate"]>0.8),"Discount_Rate"]=np.nan
items["Discount_Rate"]=items["Discount_Rate"].fillna(items["Discount_Rate"].median())
items["Quantity"]=items["Quantity"].fillna(items["Quantity"].median())
items["Unit_Price_EGP"]=items["Unit_Price_EGP"].fillna(items.groupby("Category")["Unit_Price_EGP"].transform("median"))
items["Line_Sales_EGP"]=(items["Quantity"]*items["Unit_Price_EGP"]*(1-items["Discount_Rate"])).round(2)
items["Line_Cost_EGP"]=(items["Quantity"]*items["Unit_Cost_EGP"]).round(2)
items["Line_Profit_EGP"]=(items["Line_Sales_EGP"]-items["Line_Cost_EGP"]).round(2)

orders["Shipping_Fee_EGP"]=pd.to_numeric(orders["Shipping_Fee_EGP"],errors="coerce")
orders["Shipping_Fee_EGP"]=orders["Shipping_Fee_EGP"].fillna(orders["Shipping_Fee_EGP"].median())
orders["Payment_Method"]=orders["Payment_Method"].fillna("Unknown")
customers["City"]=customers["City"].fillna("Unknown")
customers["Customer_Segment"]=customers["Customer_Segment"].fillna("Unknown")

# One row per order in order-level metrics; line items remain at item grain.
order_totals=items.groupby("Order_ID",as_index=False).agg(
    Order_Revenue_EGP=("Line_Sales_EGP","sum"),
    Order_Cost_EGP=("Line_Cost_EGP","sum"),
    Order_Profit_EGP=("Line_Profit_EGP","sum"),
    Units=("Quantity","sum")
)
orders=orders.merge(order_totals,on="Order_ID",how="left")
orders["Order_Month"]=orders["Order_Date"].dt.to_period("M").astype(str)
orders["Delivery_Days"]=(orders["Delivery_Date"]-orders["Order_Date"]).dt.total_seconds()/86400
orders.loc[(orders["Delivery_Days"]<0)|(orders["Delivery_Days"]>30),"Delivery_Days"]=np.nan
orders=orders.merge(customers[["Customer_ID","City","Customer_Segment"]],on="Customer_ID",how="left")

st.sidebar.header("Filters")
channels=["All"]+sorted(orders["Sales_Channel"].dropna().unique().tolist())
channel=st.sidebar.selectbox("Sales channel",channels)
cities=["All"]+sorted(orders["City"].dropna().unique().tolist())
city=st.sidebar.selectbox("City",cities)
segments=["All"]+sorted(orders["Customer_Segment"].dropna().unique().tolist())
segment=st.sidebar.selectbox("Customer segment",segments)
statuses=["All"]+sorted(orders["Order_Status"].dropna().unique().tolist())
status=st.sidebar.selectbox("Order status",statuses)
f=orders.copy()
if channel!="All": f=f[f["Sales_Channel"]==channel]
if city!="All": f=f[f["City"]==city]
if segment!="All": f=f[f["Customer_Segment"]==segment]
if status!="All": f=f[f["Order_Status"]==status]

delivered=f[f["Order_Status"]=="Delivered"]
revenue=delivered["Order_Revenue_EGP"].sum()
profit=delivered["Order_Profit_EGP"].sum()
order_count=f["Order_ID"].nunique()
aov=revenue/delivered["Order_ID"].nunique() if delivered["Order_ID"].nunique() else 0
customers_count=f["Customer_ID"].nunique()
returning_rate=(f["Customer_ID"].value_counts().gt(1).mean()*100) if len(f) else 0
c1,c2,c3,c4,c5=st.columns(5)
c1.metric("Orders",f"{order_count:,}")
c2.metric("Delivered revenue",f"{revenue:,.0f} EGP")
c3.metric("Delivered profit",f"{profit:,.0f} EGP")
c4.metric("Avg. order value",f"{aov:,.0f} EGP")
c5.metric("Unique customers",f"{customers_count:,}")

st.subheader("Order and revenue overview")
left,right=st.columns(2)
with left:
    monthly=f.groupby("Order_Month",as_index=False).agg(Orders=("Order_ID","nunique"))
    fig=px.line(monthly,x="Order_Month",y="Orders",markers=True,title="Orders by month")
    st.plotly_chart(fig,use_container_width=True)
with right:
    status_summary=f.groupby("Order_Status",as_index=False).agg(Orders=("Order_ID","nunique"))
    fig=px.bar(status_summary,x="Order_Status",y="Orders",color="Order_Status",title="Orders by status")
    st.plotly_chart(fig,use_container_width=True)

st.subheader("Products and customers")
left,right=st.columns(2)
with left:
    item_filtered=items[items["Order_ID"].isin(f["Order_ID"])]
    cat_summary=item_filtered.groupby("Category",as_index=False).agg(
        Revenue=("Line_Sales_EGP","sum"),Profit=("Line_Profit_EGP","sum"),Units=("Quantity","sum"))
    fig=px.bar(cat_summary,x="Category",y="Revenue",color="Category",title="Revenue by category")
    st.plotly_chart(fig,use_container_width=True)
with right:
    city_summary=f.groupby("City",as_index=False).agg(
        Revenue=("Order_Revenue_EGP","sum"),Orders=("Order_ID","nunique"))
    fig=px.scatter(city_summary,x="Orders",y="Revenue",size="Orders",color="City",
                   hover_name="City",title="City: orders vs revenue")
    st.plotly_chart(fig,use_container_width=True)

st.subheader("Customer and order behavior")
customer_order_counts=f.groupby("Customer_ID",as_index=False).agg(Order_Count=("Order_ID","nunique"))
fig=px.histogram(customer_order_counts,x="Order_Count",nbins=15,title="Orders per customer in filtered data")
st.plotly_chart(fig,use_container_width=True)

with st.expander("Data quality checks"):
    st.write("Rows before duplicate removal",{"customers":customer_count_before,"orders":order_count_before,"items":item_count_before})
    st.write("Rows after duplicate removal",{"customers":len(customers),"orders":len(orders),"items":len(items)})
    st.write("Missing values in customer table"); st.dataframe(customers.isna().sum().rename("Missing").to_frame())
    st.write("Missing values in order table"); st.dataframe(orders_raw.isna().sum().rename("Missing").to_frame())
    st.write("Missing values in item table"); st.dataframe(items.isna().sum().rename("Missing").to_frame())
    st.caption("Keep table grain clear: order-level KPIs should use one row per order; item-level analysis should use order items. Avoid double-counting revenue after joins.")

with st.expander("Explore filtered orders"):
    st.dataframe(f,use_container_width=True)
    st.download_button("Download filtered orders",f.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_orders.csv",mime="text/csv")

st.subheader("Suggested business actions")
st.markdown("""
- Separate delivered revenue from cancelled and returned orders; report status mix independently.
- Prioritize categories by profit as well as revenue and units sold.
- Identify cities with many orders but lower revenue per order for targeted offers or assortment review.
- Encourage repeat purchase with retention campaigns only after checking customer order history and margins.
- Verify unusual prices, discounts, and duplicate keys before using the results for real decisions.
""")
