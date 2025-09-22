import os
from waitress import serve
from app import create_app

app = create_app()

# Use PORT environment variable or default to 8080 for production
port = int(os.environ.get("PORT", "8080"))
host = "0.0.0.0"

print(f"Starting production server on http://{host}:{port}")
serve(app, host=host, port=port)
