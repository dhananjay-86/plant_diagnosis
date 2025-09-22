import os, json, requests
PORT = int(os.environ.get("PORT", "5007"))
BASE = f"http://127.0.0.1:{PORT}"
for ep in ["/health", "/labels", "/model_info"]:
    try:
        r = requests.get(BASE+ep, timeout=5)
        print(ep, r.status_code, (r.text[:200] + ("..." if len(r.text)>200 else "")))
    except Exception as e:
        print(ep, "ERR", e)
