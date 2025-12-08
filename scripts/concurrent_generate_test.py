import argparse
import http.client
import json
import threading
import time
from typing import Any, Dict, List


HOST = "127.0.0.1"
PORT = 9000
PATH = "/generate"


def build_payload(prompt: str, steps: int, guidance: float, height: int, width: int) -> Dict[str, Any]:
    return {
        "prompt": prompt,
        "negative_prompt": "",
        "steps": steps,
        "guidance": guidance,
        "height": height,
        "width": width,
    }


def send_request(idx: int, payload: Dict[str, Any], results: List[Dict[str, Any]]):
    start = time.time()
    entry: Dict[str, Any] = {"idx": idx}
    try:
        conn = http.client.HTTPConnection(HOST, PORT, timeout=300)
        body = json.dumps(payload).encode("utf-8")
        conn.request("POST", PATH, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
        entry.update(
            {
                "status": resp.status,
                "duration": round(time.time() - start, 2),
                "bytes": len(data),
            }
        )
        try:
            parsed = json.loads(data.decode("utf-8"))
            entry["has_error"] = "error" in parsed
        except Exception:
            entry["has_error"] = True
    except Exception as exc:  # noqa: BLE001
        entry.update({"error": str(exc), "duration": round(time.time() - start, 2)})
    finally:
        results.append(entry)


def main():
    parser = argparse.ArgumentParser(description="Fire concurrent /generate requests for quick load testing.")
    parser.add_argument("-c", "--concurrency", type=int, default=3, help="Number of parallel requests to send")
    parser.add_argument("-s", "--steps", type=int, default=2, help="Inference steps per request")
    parser.add_argument("-g", "--guidance", type=float, default=0.0, help="Guidance scale")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("-p", "--prompt", type=str, default="test concurrent request", help="Prompt to use")
    args = parser.parse_args()

    payload = build_payload(args.prompt, args.steps, args.guidance, args.height, args.width)

    threads = []
    results: List[Dict[str, Any]] = []
    start = time.time()
    for i in range(args.concurrency):
        t = threading.Thread(target=send_request, args=(i, payload, results))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    total = round(time.time() - start, 2)
    print(f"Sent {args.concurrency} concurrent requests in {total}s")
    for item in sorted(results, key=lambda x: x["idx"]):
        print(item)


if __name__ == "__main__":
    main()
