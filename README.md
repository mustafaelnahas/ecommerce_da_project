# E-commerce Customer & Orders Analysis — Graduation Project

## Scenario
An online retailer wants to understand order trends, customer repeat behavior, category profitability, sales channels, and order status. The dataset is synthetic and intended for training.

## Files
- `customers_raw.csv`: customer-level table.
- `orders_raw.csv`: one row per order.
- `order_items_raw.csv`: one row per product line in an order.
- `app.py`: interactive Streamlit dashboard.
- `analysis_starter.py`: guided examples for NumPy, Pandas, Matplotlib, Seaborn, Plotly, and table joins.
- `trainer_guide.pdf`: challenge-based two-hour lesson plan and hints.
- `requirements.txt`: required packages.

## Run
`pip install -r requirements.txt`
`streamlit run app.py`

Keep all three CSV files beside `app.py`.

## Critical modeling rule
Orders and order items have different grains. Aggregate order items to one row per Order_ID before joining totals to the orders table. Never sum order-level revenue after joining it to multiple item rows, or it will be double-counted.
