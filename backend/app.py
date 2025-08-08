from flask import Flask, jsonify, request, send_from_directory, abort
import os
import psycopg2
import pandas as pd
from io import StringIO
from datetime import date, timedelta
import logging
from functools import wraps
from contextlib import contextmanager
import numpy as np
from sklearn.linear_model import Ridge


class Config:
    DB = {
        "host": os.getenv("DB_HOST", "localhost"),
        "database": os.getenv("DB_NAME", "momos"),
        "user": os.getenv("DB_USER", "shreyarora"),
        "password": os.getenv("DB_PASSWORD", "")
    }
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
    MAX_CSV_ROWS = 10000
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


app = Flask(__name__, static_folder=None)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection():
    """Context manager for database connections with proper error handling."""
    connection = None
    try:
        connection = psycopg2.connect(**Config.DB)
        yield connection
    except psycopg2.Error as e:
        logger.error(f"Database error: {e}")
        if connection:
            connection.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if connection:
            connection.rollback()
        raise
    finally:
        if connection:
            connection.close()

# ---------- Utility Functions ----------
def clean_item(name: str) -> str:
    """Clean and normalize item names."""
    if not name:
        return ""
    name = name.strip().lower()
  
    name_mapping = {
        "momu": "momo",
        "mo mo": "momo",
        "momoz": "momo"
    }
    return name_mapping.get(name, name)

def validate_numeric_input(value, field_name, min_value=0, allow_zero=False):
    """Validate numeric input with proper error messages."""
    try:
        num_value = float(value) if field_name == "price" else int(value)
        if not allow_zero and num_value <= min_value:
            raise ValueError(f"{field_name} must be greater than {min_value}")
        elif allow_zero and num_value < min_value:
            raise ValueError(f"{field_name} must be at least {min_value}")
        return num_value
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid {field_name}: {str(e)}")

# ---------- Error Handlers ----------
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(psycopg2.Error)
def handle_db_error(error):
    logger.error(f"Database error: {error}")
    return jsonify({"error": "Database operation failed"}), 500

# ---------- Request Validation Decorator ----------
def validate_json(required_fields=None):
    """Decorator to validate JSON requests."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
            
            data = request.get_json(silent=True)
            if not data:
                return jsonify({"error": "Invalid JSON"}), 400
            
            if required_fields:
                missing = [field for field in required_fields if not data.get(field)]
                if missing:
                    return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


@app.route("/")
def index():
    """Serve the main HTML file."""
    try:
        path = os.path.join(Config.FRONTEND_DIR, "index.html")
        if not os.path.exists(path):
            logger.warning("Frontend index.html not found")
            abort(404)
        return send_from_directory(Config.FRONTEND_DIR, "index.html")
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        abort(500)

@app.route("/<path:asset>")
def static_files(asset):
    """Serve static assets with security checks."""
    try:
        # Security: prevent directory traversal
        if ".." in asset or asset.startswith("/"):
            abort(403)
        
        path = os.path.join(Config.FRONTEND_DIR, asset)
        if not os.path.exists(path) or not os.path.isfile(path):
            abort(404)
        
        return send_from_directory(Config.FRONTEND_DIR, asset)
    except Exception as e:
        logger.error(f"Error serving static file {asset}: {e}")
        abort(500)

# ---------- API: Sales Management ----------
@app.route("/api/sale", methods=["POST"])
@validate_json(required_fields=["branch", "item", "quantity", "price"])
def add_sale():
    """Add a single sale record."""
    try:
        data = request.get_json()
        
        # Validate and clean input
        branch = data["branch"].strip()
        item = clean_item(data["item"])
        quantity = validate_numeric_input(data["quantity"], "quantity", min_value=0)
        price = validate_numeric_input(data["price"], "price", min_value=0)
        
        if not branch or not item:
            return jsonify({"error": "Branch and item cannot be empty"}), 400
        
        total = quantity * price
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO sales (branch, item, quantity, price, total)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id, sale_date;
                """, (branch, item, quantity, price, total))
                new_id, sale_date = cur.fetchone()
                conn.commit()
        
        logger.info(f"Sale added: ID {new_id}, Branch: {branch}, Item: {item}")
        return jsonify({
            "success": True, 
            "id": new_id, 
            "sale_date": str(sale_date),
            "total": total
        }), 201
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error adding sale: {e}")
        return jsonify({"error": "Failed to add sale"}), 500

@app.route("/api/upload_csv", methods=["POST"])
def upload_csv():
    """Upload sales data via CSV file."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > Config.MAX_FILE_SIZE:
            return jsonify({"error": "File too large (max 5MB)"}), 400
        
        # Read and parse CSV
        content = file.read().decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(content))
        
        if len(df) > Config.MAX_CSV_ROWS:
            return jsonify({"error": f"Too many rows (max {Config.MAX_CSV_ROWS})"}), 400
        
        # Validate CSV structure
        df.columns = [c.lower().strip() for c in df.columns]
        required_columns = {"branch", "item", "quantity", "price"}
        
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            return jsonify({"error": f"Missing columns: {', '.join(missing)}"}), 400
        
        # Clean and validate data
        df["branch"] = df["branch"].astype(str).str.strip()
        df["item"] = df["item"].astype(str).apply(clean_item)
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
        
        # Filter valid rows
        initial_count = len(df)
        df = df[
            (df["branch"] != "") & 
            (df["item"] != "") & 
            (df["quantity"] > 0) & 
            (df["price"] > 0)
        ]
        
        if df.empty:
            return jsonify({"error": "No valid rows found in CSV"}), 400
        
        df["total"] = df["quantity"] * df["price"]
        
        # Insert data
        rows = list(df[["branch", "item", "quantity", "price", "total"]].itertuples(
            index=False, name=None
        ))
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany("""
                    INSERT INTO sales (branch, item, quantity, price, total)
                    VALUES (%s, %s, %s, %s, %s);
                """, rows)
                conn.commit()
        
        skipped = initial_count - len(df)
        logger.info(f"CSV upload: {len(rows)} rows inserted, {skipped} rows skipped")
        
        return jsonify({
            "success": True,
            "inserted": len(rows),
            "skipped": skipped,
            "message": f"Successfully imported {len(rows)} sales records"
        }), 201
        
    except pd.errors.EmptyDataError:
        return jsonify({"error": "CSV file is empty"}), 400
    except pd.errors.ParserError as e:
        return jsonify({"error": f"CSV parsing failed: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"CSV upload error: {e}")
        return jsonify({"error": "Failed to process CSV file"}), 500

# ---------- API: Data Retrieval ----------
@app.route("/api/sales/full", methods=["GET"])
def full_sales():
    """Get recent sales records with pagination support."""
    try:
        limit = min(int(request.args.get("limit", 500)), 1000)  # Max 1000 records
        offset = max(int(request.args.get("offset", 0)), 0)
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, branch, item, quantity, price, total, sale_date
                    FROM sales
                    ORDER BY sale_date DESC, id DESC
                    LIMIT %s OFFSET %s;
                """, (limit, offset))
                rows = cur.fetchall()
        
        return jsonify({
            "success": True,
            "data": [
                {
                    "id": r[0],
                    "branch": r[1],
                    "item": r[2],
                    "quantity": int(r[3]),
                    "price": float(r[4]),
                    "total": float(r[5]),
                    "sale_date": str(r[6])
                }
                for r in rows
            ],
            "count": len(rows)
        })
        
    except ValueError:
        return jsonify({"error": "Invalid pagination parameters"}), 400
    except Exception as e:
        logger.error(f"Error fetching sales: {e}")
        return jsonify({"error": "Failed to fetch sales data"}), 500

@app.route("/api/sales/series", methods=["GET"])
def sales_series():
    """Get historical daily revenue time series."""
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query("""
                SELECT sale_date, SUM(total) as revenue
                FROM sales
                GROUP BY sale_date
                ORDER BY sale_date;
            """, conn)
        
        if df.empty:
            return jsonify({"success": True, "data": []})
        
        data = [
            {
                "date": str(row["sale_date"]),
                "revenue": float(row["revenue"])
            }
            for _, row in df.iterrows()
        ]
        
        return jsonify({"success": True, "data": data})
        
    except Exception as e:
        logger.error(f"Error fetching sales series: {e}")
        return jsonify({"error": "Failed to fetch sales series"}), 500

@app.route("/api/sales/forecast", methods=["GET"])
def forecast():
    """Generate ML-based revenue forecast using Ridge regression."""
    try:
        days = min(int(request.args.get("days", 7)), 30)  # Max 30 days forecast
        
        with get_db_connection() as conn:
            df = pd.read_sql_query("""
                SELECT sale_date, SUM(total) AS revenue
                FROM sales
                GROUP BY sale_date
                ORDER BY sale_date;
            """, conn)
        
        if df.empty:
            # No historical data - return zero forecast
            start = date.today() + timedelta(days=1)
            forecast_data = [
                {
                    "date": str(start + timedelta(days=i)),
                    "revenue": 0.0,
                    "confidence": "low"
                }
                for i in range(days)
            ]
            return jsonify({
                "success": True,
                "data": forecast_data,
                "model": "no_data_fallback"
            })
        
        # Prepare time series data
        df["sale_date"] = pd.to_datetime(df["sale_date"])
        df = df.sort_values("sale_date").reset_index(drop=True)
        
        # Simple mean-based forecast for insufficient data
        if len(df) < 14:
            base_revenue = float(df["revenue"].mean())
            start = date.today() + timedelta(days=1)
            forecast_data = [
                {
                    "date": str(start + timedelta(days=i)),
                    "revenue": round(base_revenue, 2),
                    "confidence": "low"
                }
                for i in range(days)
            ]
            return jsonify({
                "success": True,
                "data": forecast_data,
                "model": "mean_fallback"
            })
        
        # Feature engineering for ML model
        df["day_idx"] = (df["sale_date"] - df["sale_date"].min()).dt.days
        df["dow"] = df["sale_date"].dt.dayofweek  # 0=Monday, 6=Sunday
        
        # Create day-of-week dummy variables
        dow_dummies = pd.get_dummies(df["dow"], prefix="dow", drop_first=True)
        
        # Combine features
        X = pd.concat([df[["day_idx"]], dow_dummies], axis=1).values
        y = df["revenue"].values
        
        # Train Ridge regression model
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X, y)
        
        # Generate future predictions
        last_date = df["sale_date"].iloc[-1].date()
        last_idx = df["day_idx"].iloc[-1]
        
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        future_day_idx = np.array([last_idx + i + 1 for i in range(days)])
        
        # Create future day-of-week features
        future_dow = [d.weekday() for d in future_dates]
        future_dow_df = pd.DataFrame({"dow": future_dow})
        future_dow_dummies = pd.get_dummies(future_dow_df["dow"], prefix="dow", drop_first=True)
        
        # Align columns with training data
        for col in dow_dummies.columns:
            if col not in future_dow_dummies.columns:
                future_dow_dummies[col] = 0
        future_dow_dummies = future_dow_dummies.reindex(columns=dow_dummies.columns, fill_value=0)
        
        # Prepare future feature matrix
        X_future = np.column_stack([future_day_idx, future_dow_dummies.values])
        
        # Make predictions
        predictions = model.predict(X_future)
        predictions = np.maximum(predictions, 0)  # Ensure non-negative values
        
        # Calculate confidence based on model performance
        train_score = model.score(X, y)
        confidence = "high" if train_score > 0.8 else "medium" if train_score > 0.5 else "low"
        
        forecast_data = [
            {
                "date": str(date_val),
                "revenue": round(float(pred), 2),
                "confidence": confidence
            }
            for date_val, pred in zip(future_dates, predictions)
        ]
        
        return jsonify({
            "success": True,
            "data": forecast_data,
            "model": "ridge_regression",
            "model_score": round(train_score, 3),
            "training_days": len(df)
        })
        
    except ValueError:
        return jsonify({"error": "Invalid forecast parameters"}), 400
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return jsonify({"error": "Failed to generate forecast"}), 500

# ---------- Health Check ----------
@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        
        return jsonify({
            "status": "healthy",
            "database": "connected",
            "timestamp": str(date.today())
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "database": "disconnected",
            "error": "Database connection failed"
        }), 500

# ---------- Application Entry Point ----------
if __name__ == "__main__":
    logger.info("Starting Flask application...")
    
    # Verify database connection on startup
    try:
        with get_db_connection() as conn:
            logger.info("Database connection verified")
    except Exception as e:
        logger.error(f"Failed to connect to database on startup: {e}")
        exit(1)
    
    app.run(debug=False, host="0.0.0.0", port=5000)
