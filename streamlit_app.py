"""
Telecom Network Health & Intelligent Fault Prediction System
Streamlit Community Cloud default entrypoint.
Redirects execution to app.py seamlessly.
"""

import runpy

if __name__ == "__main__" or True:
    runpy.run_path("app.py", run_name="__main__")
