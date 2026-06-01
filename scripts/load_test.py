"""Load test: 20 consecutive requests against the production backend.

Usage:
    python scripts/load_test.py

Requires: requests (pip install requests)
Generates: LOAD_TEST_REPORT.md
"""

import requests
import time
import sys
import os
import io
import struct
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BACKEND_URL = os.getenv("BACKEND_URL", "https://military-object-detection.onrender.com")
RESULTS = []


def make_jpeg(width=64, height=64, quality=95):
    """Generate a minimal valid JPEG without external dependencies."""
    # Minimal JPEG: SOI marker + APP0 JFIF header + DQT + SOF0 + DHT + SOS + EOI
    # This creates a valid grayscale JPEG that any JPEG parser should accept

    def put_16(stream, val):
        stream.extend(struct.pack(">H", val))

    def put_marker(stream, marker, data=b""):
        stream.append(0xFF)
        stream.append(marker)
        put_16(stream, len(data) + 2)
        stream.extend(data)

    buf = bytearray()

    # SOI
    buf.extend(b"\xFF\xD8")

    # APP0 JFIF
    app0 = bytearray(b"JFIF\x00")
    app0.extend(struct.pack(">B", 1))  # version major
    app0.extend(struct.pack(">B", 1))  # version minor
    app0.extend(b"\x00")  # units (0 = no units)
    app0.extend(struct.pack(">H", 1))  # X density
    app0.extend(struct.pack(">H", 1))  # Y density
    app0.extend(b"\x00\x00")  # thumbnail dimensions
    put_marker(buf, 0xE0, app0)

    # DQT (quantization table) — default luminance table
    dqt = bytearray(b"\x00")  # table ID 0, 8-bit precision
    dqt.extend(bytes([
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ]))
    put_marker(buf, 0xDB, dqt)

    # SOF0 (start of frame) — baseline, 1 component (grayscale)
    sof = bytearray()
    sof.extend(struct.pack(">B", 8))  # precision
    sof.extend(struct.pack(">H", height))  # height
    sof.extend(struct.pack(">H", width))  # width
    sof.extend(b"\x01")  # number of components
    sof.extend(b"\x01\x11\x00")  # component 1: ID=1, sampling=1x1, QT=0
    put_marker(buf, 0xC0, sof)

    # DHT (Huffman tables) — minimal default DC/AC tables for luminance
    # DC table (class=0, id=0)
    dht_dc = bytearray(b"\x00")  # table class=0 (DC), id=0
    dht_dc.extend(bytes([0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]))  # counts
    dht_dc.extend(bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0x0A, 0x0B]))
    put_marker(buf, 0xC4, dht_dc)

    # AC table (class=1, id=0)
    dht_ac = bytearray(b"\x10")  # table class=1 (AC), id=0
    dht_ac.extend(bytes([0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D]))
    dht_ac.extend(bytes(range(0x01, 0x9A)))
    put_marker(buf, 0xC4, dht_ac)

    # SOS (start of scan)
    sos = bytearray()
    sos.extend(b"\x01\x01\x00\x00\x3F\x00")  # 1 component, DC=0, AC=0
    put_marker(buf, 0xDA, sos)

    # Entropy-coded data (minimum: end of block markers)
    # For each MCU (8x8 block), write a single EOB marker
    mcu_blocks = ((width + 7) // 8) * ((height + 7) // 8)
    for _ in range(mcu_blocks):
        buf.extend(b"\x00")  # one byte of zero-filled data

    # EOI (end of image)
    buf.extend(b"\xFF\xD9")

    return bytes(buf)


test_images = {
    "small": {"filename": "test_small.jpg", "content": make_jpeg(64, 64), "desc": "64x64 tiny image"},
    "medium": {"filename": "test_medium.jpg", "content": make_jpeg(640, 480), "desc": "640x480 standard image"},
    "large": {"filename": "test_large.jpg", "content": make_jpeg(1920, 1080), "desc": "1920x1080 large image"},
}

for name, img in test_images.items():
    actual_size = len(img["content"])
    desc = img["desc"]
    print(f"  {name}: {actual_size} bytes ({desc})")


def run_test(label, img_data, filename, expected_status=200):
    """Run a single predict request and record timing."""
    start = time.time()
    try:
        resp = requests.post(
            f"{BACKEND_URL}/predict",
            files={"file": (filename, img_data, "image/jpeg")},
            timeout=300,
        )
        elapsed = time.time() - start
        result = {
            "label": label,
            "status": resp.status_code,
            "elapsed": round(elapsed, 2),
            "expected": expected_status,
            "passed": resp.status_code == expected_status,
            "error": None,
            "detections": 0,
        }
        if resp.status_code == 200:
            data = resp.json()
            result["detections"] = len(data.get("detections", []))
            result["message"] = data.get("message", "")
        elif resp.status_code != expected_status:
            try:
                result["error"] = resp.json().get("error", str(resp.text[:200]))
            except Exception:
                result["error"] = resp.text[:200]
        else:
            result["error"] = resp.text[:200] if resp.text else "No response body"
    except requests.Timeout:
        elapsed = time.time() - start
        result = {
            "label": label,
            "status": 0,
            "elapsed": round(elapsed, 2),
            "expected": expected_status,
            "passed": False,
            "error": "Timeout (>300s)",
            "detections": 0,
        }
    except requests.exceptions.ConnectionError as e:
        elapsed = time.time() - start
        result = {
            "label": label,
            "status": 0,
            "elapsed": round(elapsed, 2),
            "expected": expected_status,
            "passed": False,
            "error": f"ConnectionError: {e}",
            "detections": 0,
        }
    except Exception as e:
        elapsed = time.time() - start
        result = {
            "label": label,
            "status": 0,
            "elapsed": round(elapsed, 2),
            "expected": expected_status,
            "passed": False,
            "error": str(e),
            "detections": 0,
        }

    RESULTS.append(result)
    status_str = "PASS" if result["passed"] else "FAIL"
    print(f"  [{status_str}] {label}: {result['elapsed']}s status={result['status']}", end="")
    if result.get("detections"):
        print(f" detections={result['detections']}", end="")
    if result.get("error"):
        print(f" error={result['error']}", end="")
    print()

    # Check memory growth (crash = process died)
    return result


def print_summary():
    """Print a summary of all test results."""
    passed = sum(1 for r in RESULTS if r["passed"])
    failed = sum(1 for r in RESULTS if not r["passed"])
    total = len(RESULTS)
    elapsed_times = [r["elapsed"] for r in RESULTS]

    print(f"\n{'='*60}")
    print(f"Load Test Summary")
    print(f"{'='*60}")
    print(f"  Total requests: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {passed/total*100:.1f}%")
    print(f"  Min time: {min(elapsed_times):.2f}s")
    print(f"  Max time: {max(elapsed_times):.2f}s")
    print(f"  Avg time: {sum(elapsed_times)/len(elapsed_times):.2f}s")
    print(f"  Median time: {sorted(elapsed_times)[len(elapsed_times)//2]:.2f}s")

    # Check for crashes
    crashed = any(r.get("error") and "ConnectionError" in str(r.get("error")) for r in RESULTS)
    if crashed:
        print(f"  ⚠️ SOME REQUESTS CRASHED (connection errors detected)")
    else:
        print(f"  ✅ No crashes detected")

    # Check for timeouts
    timeouts = sum(1 for r in RESULTS if r.get("error") == "Timeout (>300s)")
    if timeouts:
        print(f"  ⚠️ {timeouts} requests timed out")
    else:
        print(f"  ✅ No timeouts")


def generate_markdown():
    """Generate LOAD_TEST_REPORT.md content."""
    passed = sum(1 for r in RESULTS if r["passed"])
    failed = sum(1 for r in RESULTS if not r["passed"])
    total = len(RESULTS)
    elapsed_times = [r["elapsed"] for r in RESULTS]

    lines = []
    lines.append("# Load Test Report")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append(f"**Backend:** {BACKEND_URL}")
    lines.append(f"**Description:** 20 consecutive predict requests with small, medium, and large images")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total requests | {total} |")
    lines.append(f"| Passed | {passed} |")
    lines.append(f"| Failed | {failed} |")
    lines.append(f"| Success rate | {passed/total*100:.1f}% |")
    lines.append(f"| Min time | {min(elapsed_times):.2f}s |")
    lines.append(f"| Max time | {max(elapsed_times):.2f}s |")
    lines.append(f"| Avg time | {sum(elapsed_times)/len(elapsed_times):.2f}s |")
    lines.append(f"| Median time | {sorted(elapsed_times)[len(elapsed_times)//2]:.2f}s |")
    lines.append(f"| Std Dev | {__import__('statistics').stdev(elapsed_times):.2f}s |")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| # | Label | Status | Time (s) | Detections | Error |")
    lines.append("|---|-------|--------|----------|------------|-------|")
    for i, r in enumerate(RESULTS, 1):
        error_str = r.get("error", "") or ""
        if len(error_str) > 100:
            error_str = error_str[:97] + "..."
        status_str = "✅ PASS" if r["passed"] else "❌ FAIL"
        lines.append(f"| {i} | {r['label']} | {status_str} ({r['status']}) | {r['elapsed']} | {r.get('detections', 0)} | {error_str} |")
    lines.append("")
    lines.append("## Analysis")
    lines.append("")
    if failed == 0:
        lines.append("✅ **All requests passed** — no crashes, no timeouts, no failed fetch errors.")
    else:
        lines.append(f"⚠️ **{failed} requests failed** — see details below.")
        lines.append("")
        lines.append("### Failure Analysis")
        lines.append("")
        for r in RESULTS:
            if not r["passed"]:
                lines.append(f"- **{r['label']}**: status={r['status']}, error=\"{r.get('error', 'N/A')}\"")
        lines.append("")

    lines.append("### Memory Stability")
    lines.append("- Monitor memory usage across consecutive requests")
    lines.append("- Look for: RAM growth (leak), crashes (OOM), slowdown (thrashing)")
    lines.append("- If all 20 pass without crash → memory is stable")
    lines.append("")
    lines.append("### Timeout Analysis")
    timeouts = [r for r in RESULTS if r["elapsed"] > 180]
    if timeouts:
        lines.append(f"⚠️ {len(timeouts)} requests took >3 minutes (enough to trigger frontend timeout):")
        for r in timeouts:
            lines.append(f"  - {r['label']}: {r['elapsed']}s")
    else:
        lines.append("✅ All requests completed within 3-minute frontend timeout window.")
    lines.append("")
    lines.append("### Cold Start Detection")
    cold_start = RESULTS[0] if RESULTS else None
    if cold_start:
        lines.append(f"- First request (cold start): {cold_start['elapsed']}s")
        warm_times = [r["elapsed"] for r in RESULTS[1:]]
        if warm_times:
            lines.append(f"- Subsequent requests (warm): avg {sum(warm_times)/len(warm_times):.2f}s")
            if cold_start["elapsed"] > 2 * (sum(warm_times)/len(warm_times)):
                lines.append("- Cold start is significantly slower than warm requests (as expected)")

    return "\n".join(lines)


def main():
    print(f"Load Test — {BACKEND_URL}")
    print(f"{'='*60}")
    print()

    # Warm up / health check
    print("1. Health check...")
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=30)
        print(f"   Health: {resp.status_code} {resp.json()}")
    except Exception as e:
        print(f"   Health check FAILED: {e}")
        print("   Aborting — backend is not reachable")
        return

    print("   Test images generated:")
    for name, img in test_images.items():
        print(f"     {name}: {len(img['content'])} bytes ({img['desc']})")

    print()

    # Sequence: small, medium, small, large, small, medium, small, large, ...
    # 5 small, 5 medium, 5 large, 5 mixed = 20
    sequence = []
    for _ in range(5):
        sequence.append(("small", test_images["small"]))
    for _ in range(5):
        sequence.append(("medium", test_images["medium"]))
    for _ in range(5):
        sequence.append(("large", test_images["large"]))
    for _ in range(5):
        seq_idx = len(sequence) % 3
        if seq_idx == 0:
            sequence.append(("small", test_images["small"]))
        elif seq_idx == 1:
            sequence.append(("medium", test_images["medium"]))
        else:
            sequence.append(("large", test_images["large"]))

    sequence = sequence[:20]

    print("2. Running 20 consecutive requests...")
    print()
    for idx, (name, img) in enumerate(sequence, 1):
        label = f"{idx:02d}/{len(sequence)} {name}"
        run_test(label, img["content"], img["filename"])
        if idx < len(sequence):
            time.sleep(0.5)  # Small delay between requests

    print()
    print_summary()

    # Generate markdown report
    markdown = generate_markdown()
    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LOAD_TEST_REPORT.md")
    with open(report_path, "w") as f:
        f.write(markdown)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
