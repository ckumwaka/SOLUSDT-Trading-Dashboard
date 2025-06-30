"""
main.py
========
Desktop application entry point for the SOL/USDT trading dashboard.
"""
import sys
import threading
import webview
import time
import logging
from flask import Flask, render_template, jsonify, request
from trading_logic import market_analyzer
from websocket_manager import ws_manager
from config import *
import os

# Ensure console uses UTF-8 so ✅/✓ characters don’t crash logging
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
dashboard_state = {
    "last_update": None,
    "connection_status": "Connecting...",
    "real_time_enabled": False,
    "analysis_cache": None
}

@app.after_request
def add_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route("/")
def dashboard():
    try:
        if (
            dashboard_state["analysis_cache"] is None or
            time.time() - dashboard_state["last_update"] > CACHE_TIMEOUT
        ):
            result = market_analyzer.analyze_market(DEFAULT_SYMBOL)
            dashboard_state["analysis_cache"] = result
            dashboard_state["last_update"] = time.time()
        else:
            result = dashboard_state["analysis_cache"]

        if ws_manager.is_running:
            dashboard_state["connection_status"] = "Connected"
            result["real_time_price"] = ws_manager.get_latest_price()
            result["price_change"] = ws_manager.get_price_change()
            result["volume_spike"] = ws_manager.get_volume_spike()
        else:
            dashboard_state["connection_status"] = "Disconnected"

        if result.get("decision") in ["INSUFFICIENT_DATA", "ANALYSIS_ERROR"]:
            return render_template(
                "error.html",
                error=result.get("error", "Unknown error"),
                result=result
            )

        return render_template(
            "dashboard.html",
            result=result,
            connection_status=dashboard_state["connection_status"],
            real_time_enabled=dashboard_state["real_time_enabled"],
            processing_time="< 100ms",
            data_points=MAX_DATA_POINTS
        )

    except Exception as e:
        logger.exception("Dashboard error")
        return render_template(
            "error.html",
            error=f"System error: {str(e)}",
            result={"time": time.strftime("%Y-%m-%d %H:%M:%S")}
        )

@app.route("/api/data")
def api_data():
    try:
        result = market_analyzer.analyze_market(DEFAULT_SYMBOL)
        if ws_manager.is_running:
            result["real_time_price"] = ws_manager.get_latest_price()
            result["price_change"] = ws_manager.get_price_change()
        return jsonify(result)
    except Exception as e:
        logger.error(f"API data error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/toggle_realtime", methods=["POST"])
def toggle_realtime():
    try:
        if dashboard_state["real_time_enabled"]:
            ws_manager.stop()
            dashboard_state["real_time_enabled"] = False
            status = "disabled"
        else:
            ws_manager.start()
            dashboard_state["real_time_enabled"] = True
            status = "enabled"
        return jsonify({
            "status": status,
            "real_time_enabled": dashboard_state["real_time_enabled"]
        })
    except Exception as e:
        logger.error(f"Toggle realtime error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/model_info")
def model_info():
    try:
        from ml_model import ml_model
        return jsonify(ml_model.get_model_info())
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    })

def setup_websocket_callbacks():
    def on_connection(data):
        dashboard_state["connection_status"] = data["status"].title()
    def on_trade(data):
        if dashboard_state["analysis_cache"]:
            dashboard_state["analysis_cache"]["real_time_price"] = data["price"]
    def on_error(data):
        dashboard_state["connection_status"] = "Error"
    ws_manager.register_callback("connection", on_connection)
    ws_manager.register_callback("trade", on_trade)
    ws_manager.register_callback("error", on_error)

def run_flask():
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=DEBUG_MODE,
        use_reloader=False,
        threaded=True
    )

def wait_for_server():
    import requests
    for _ in range(30):
        try:
            if requests.get(f"http://{FLASK_HOST}:{FLASK_PORT}/health").status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
    return False

def main():
    logger.info("Starting SOL/USDT Trading Dashboard...")
    setup_websocket_callbacks()

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    if not wait_for_server():
        logger.error("Flask server failed to start")
        return

    webview.create_window(
        title="SOL/USDT Trading Dashboard",
        url=f"http://{FLASK_HOST}:{FLASK_PORT}",
        width=1400,
        height=900,
        resizable=True,
        confirm_close=True,
        background_color="#0f172a"
    )
    webview.start(debug=DEBUG_MODE)

if __name__ == "__main__":
    main()
