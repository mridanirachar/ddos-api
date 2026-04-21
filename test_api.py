import argparse
import json
import random
import requests


def test(api_url: str):
    print(f"\nTesting API: {api_url}")

    # Health
    r = requests.get(f"{api_url}/health")
    dim = r.json().get("input_dim", 52)

    # Valid input
    features = [random.uniform(0, 1) for _ in range(dim)]
    r = requests.post(f"{api_url}/predict", json={"features": features})
    print("Valid:", r.status_code, r.json())

    # ❗ Wrong length (now should fail)
    features = [1, 2, 3]
    r = requests.post(f"{api_url}/predict", json={"features": features})
    print("Wrong length:", r.status_code, r.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    args = parser.parse_args()
    test(args.api)