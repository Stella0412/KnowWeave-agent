from __future__ import annotations

import argparse
import json

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", action="append", required=True)
    parser.add_argument("--service-url", default="http://127.0.0.1:8200")
    args = parser.parse_args()
    response = httpx.post(f"{args.service_url}/classify", json={"texts": args.text}, timeout=30)
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
