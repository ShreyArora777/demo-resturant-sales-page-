# Demo Restaurant Sales Page

A simple Flask + Bootstrap app to manage, visualize, and predict restaurant sales.  
Built with PostgreSQL, Pandas, and a touch of machine learning.

---

## Features
- Add daily sales from the webpage  
- View all sales in a table  
- Graph sales trends  
- Predict next 7 days of sales with ML  

---

## Tech Stack
**Frontend:** HTML, CSS, Bootstrap 5  
**Backend:** Python (Flask), PostgreSQL  
**ML/Data:** Pandas, Matplotlib, Scikit-learn  

---

## Setup
```bash
git clone https://github.com/ShreyArora777/demo-resturant-sales-page-.git
cd demo-resturant-sales-page-

python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# Create 'momos' database & run:
psql -d momos -f db/schema.sql

cd backend
python app.py
