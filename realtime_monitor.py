"""
realtime_monitor.py — LOCAL packet monitor
------------------------------------------
Captures live packets, extracts simplified features,
and sends them to your DEPLOYED cloud API for classification.

WHY this is separate from the cloud:
  - Scapy requires raw socket access (needs root/admin + local OS)
  - Cloud VMs (Render, Railway, etc.) block raw socket access entirely
  - The correct architecture is: Local sensor → REST API → Cloud model

Usage:
    # Point to your deployed Render API:
    python realtime_monitor.py --api https://ddos-detection-api.onrender.com

    # Or test locally:
    python realtime_monitor.py --api http://localhost:8000

Requirements:
    pip install scapy requests
    Run as root/admin (required for raw sockets)
"""

import argparse
import time
import json
import logging
from collections import defaultdict, deque

import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ── Feature extraction ────────────────────────────────────────────────────────
# CIC-IDS2017 uses flow-level statistics (mean/std of many packets per flow).
# In a live demo we approximate with per-packet features.
# This is a known limitation — document it clearly in your report.

FLOW_WINDOW = 20   # aggregate this many packets per "flow" before predicting

flow_buffer: dict[str, deque] = defaultdict(lambda: deque(maxlen=FLOW_WINDOW))


def extract_features(packet) -> list[float] | None:
    """
    Extract simplified per-packet features.
    Returns None if packet is not IP.
    """
    try:
        from scapy.all import IP, TCP, UDP

        if not packet.haslayer(IP):
            return None

        ip   = packet[IP]
        size = len(packet)

        features = [
            float(size),                                # total packet size
            float(ip.ttl),                              # time to live
            float(ip.len),                              # IP length
            float(ip.proto),                            # protocol number
            float(packet.time % 1),                     # sub-second timestamp
            float(int(packet.haslayer(TCP))),           # is TCP
            float(int(packet.haslayer(UDP))),           # is UDP
        ]

        if packet.haslayer(TCP):
            tcp = packet[TCP]
            features += [
                float(tcp.sport),
                float(tcp.dport),
                float(tcp.flags),                       # SYN/ACK/FIN etc.
                float(tcp.window),
            ]
        elif packet.haslayer(UDP):
            udp = packet[UDP]
            features += [float(udp.sport), float(udp.dport), 0.0, 0.0]
        else:
            features += [0.0, 0.0, 0.0, 0.0]

        return features

    except Exception as e:
        logger.debug("Feature extraction error: %s", e)
        return None


def flow_key(packet) -> str:
    """Group packets into flows by (src_ip, dst_ip, protocol)."""
    try:
        from scapy.all import IP
        ip = packet[IP]
        return f"{ip.src}→{ip.dst}:{ip.proto}"
    except Exception:
        return "unknown"


def aggregate_flow(features_list: list[list[float]], target_dim: int = 52) -> list[float]:
    """
    Aggregate a window of packet feature vectors into one flow feature vector
    using mean + std (mimics what CIC-IDS2017 flow statistics do).
    Pads/truncates to target_dim.
    """
    arr  = np.array(features_list, dtype=np.float32)
    mean = arr.mean(axis=0).tolist()
    std  = arr.std(axis=0).tolist()
    agg  = mean + std   # length = 2 * n_packet_features

    # Pad or truncate
    if len(agg) < target_dim:
        agg = agg + [0.0] * (target_dim - len(agg))
    else:
        agg = agg[:target_dim]
    return agg


# ── API client ────────────────────────────────────────────────────────────────

def call_api(api_url: str, features: list[float]) -> dict | None:
    try:
        resp = requests.post(
            f"{api_url}/predict",
            json={"features": features},
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        logger.error("Cannot reach API at %s — is it running?", api_url)
        return None
    except Exception as e:
        logger.warning("API call failed: %s", e)
        return None


# ── Packet handler ────────────────────────────────────────────────────────────

def make_handler(api_url: str, target_dim: int):
    packet_count = [0]

    def handle(packet):
        packet_count[0] += 1
        feats = extract_features(packet)
        if feats is None:
            return

        key = flow_key(packet)
        flow_buffer[key].append(feats)

        # Only predict once we have enough packets for a flow window
        if len(flow_buffer[key]) < FLOW_WINDOW:
            return

        agg = aggregate_flow(list(flow_buffer[key]), target_dim=target_dim)
        result = call_api(api_url, agg)
        flow_buffer[key].clear()   # reset window

        if result:
            label   = result.get("prediction", "?")
            conf    = result.get("confidence", 0)
            unknown = result.get("flagged_unknown", False)
            tag     = " ⚠ UNKNOWN/SUSPICIOUS" if unknown else ""
            icon    = "🔴" if label != "BENIGN" else "🟢"
            print(f"{icon} [{key}]  {label}  (conf={conf:.2f}){tag}")

    return handle


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live DDoS monitor — sends flows to cloud API")
    parser.add_argument("--api",    default="http://localhost:8000",
                        help="Base URL of your deployed API")
    parser.add_argument("--iface",  default=None,
                        help="Network interface to sniff (default: system default)")
    parser.add_argument("--dim",    type=int, default=52,
                        help="Feature dimension expected by model (default: 52)")
    parser.add_argument("--count",  type=int, default=0,
                        help="Number of packets to capture (0 = infinite)")
    args = parser.parse_args()

    # Check API health before starting
    try:
        resp = requests.get(f"{args.api}/health", timeout=5)
        health = resp.json()
        logger.info("API health: %s", json.dumps(health, indent=2))
        if health.get("input_dim"):
            args.dim = health["input_dim"]
            logger.info("Using model input_dim=%d from API", args.dim)
    except Exception as e:
        logger.warning("Could not reach API health endpoint: %s", e)
        logger.warning("Continuing anyway — predictions will fail if API is down.")

    logger.info("Starting packet capture → %s  (Ctrl+C to stop)", args.api)
    logger.info("NOTE: This requires root/admin privileges for raw socket access.")

    try:
        from scapy.all import sniff
        sniff(
            iface=args.iface,
            prn=make_handler(args.api, args.dim),
            store=False,
            count=args.count if args.count > 0 else 0,
        )
    except PermissionError:
        logger.error("Permission denied. Run as root: sudo python realtime_monitor.py")
    except KeyboardInterrupt:
        logger.info("Stopped.")


if __name__ == "__main__":
    main()
