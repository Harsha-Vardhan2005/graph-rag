"""
FinReflectKG Web Demo Launcher
Starts the FastAPI backend and opens the interactive dashboard in your browser.
"""

import os
import sys
import time
import webbrowser
import threading

# Suppress TensorFlow and PyTorch verbose warnings for clean terminal
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        
    port = 8000
    host = "127.0.0.1"
    url = f"http://{host}:{port}"
    
    print("=" * 70)
    print("  FinReflectKG: Agentic GraphRAG Interactive Web Showcase")
    print(f"  Starting local server at: {url}")
    print("=" * 70)
    print("Loading models and Knowledge Graph index (takes ~4-5s)...")

    import uvicorn
    from UI_Demo.backend import app

    def open_browser():
        time.sleep(1.5)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(app, host=host, port=port, log_level="info")
