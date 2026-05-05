from __future__ import annotations

import argparse

from flask import Flask, jsonify, request

from numerical_methods import NumericalError, calculate


app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "service": "numerical-methods-api"})


@app.route("/api/calculate", methods=["POST", "OPTIONS"])
def calculate_route():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    method = payload.get("method")
    params = payload.get("params") or {}

    try:
        result = calculate(method, params)
    except NumericalError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive API guard
        return jsonify({"ok": False, "error": f"Unexpected server error: {exc}"}), 500

    return jsonify({"ok": True, "data": result})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=5050, type=int)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
