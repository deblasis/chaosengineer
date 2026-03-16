# tests/e2e/test_bus_integration.py
"""End-to-end integration test for the message bus.

Requires the bus binary to be built: cd bus && go build -o chaos-bus .
Skipped automatically if the binary is not found.
"""
import json
import subprocess
import time
from pathlib import Path

import pytest

from chaosengineer.metrics.logger import Event
from chaosengineer.metrics.publisher import EventPublisher


def find_bus_binary() -> Path | None:
    repo_root = Path(__file__).parent.parent.parent
    binary = repo_root / "bus" / "chaos-bus"
    if binary.is_file():
        return binary
    return None


BUS_BINARY = find_bus_binary()

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(BUS_BINARY is None, reason="bus binary not built"),
]


@pytest.fixture()
def bus_server(tmp_path):
    """Start the bus binary and yield (bus_url, events_path)."""
    events_path = tmp_path / "events.jsonl"
    proc = subprocess.Popen(
        [str(BUS_BINARY), "--port", "0", "--output-file", str(events_path),
         "--shutdown-delay", "1s"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    port_line = proc.stdout.readline()
    port_data = json.loads(port_line)
    bus_url = f"http://127.0.0.1:{port_data['port']}"

    # Wait for healthy
    import urllib.request
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{bus_url}/healthz", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("Bus did not become healthy")

    yield bus_url, events_path

    proc.terminate()
    proc.wait(timeout=10)


class TestBusIntegration:
    def test_publish_and_read_events(self, bus_server, tmp_path):
        bus_url, events_path = bus_server

        pub = EventPublisher(bus_url=bus_url, fallback_path=events_path)

        # Publish a sequence of events
        pub.log(Event(event="run_started", data={
            "run_id": "run-e2e",
            "workload": "test",
            "budget": {},
            "baseline": {"commit": "HEAD", "metric_value": 1.0},
            "mode": "sequential",
            "metric_direction": "lower",
            "workload_spec_hash": "abc",
        }))
        pub.log(Event(event="worker_completed", data={
            "experiment_id": "exp-0-0",
            "dimension": "lr",
            "metric": 2.5,
            "cost_usd": 0.12,
        }))
        pub.log(Event(event="run_completed", data={
            "reason": "all_dimensions_explored",
        }))

        # Wait for file writer to flush
        time.sleep(0.5)

        # Read events back from the JSONL file (written by bus)
        events = pub.read_events()
        assert len(events) == 3
        assert events[0]["event"] == "run_started"
        assert events[0]["run_id"] == "run-e2e"
        assert events[1]["event"] == "worker_completed"
        assert events[1]["metric"] == 2.5
        assert events[2]["event"] == "run_completed"

    def test_poll_commands_empty(self, bus_server):
        """Verify poll_commands returns empty when no commands are queued."""
        bus_url, events_path = bus_server

        pub = EventPublisher(bus_url=bus_url, fallback_path=events_path)
        pub.log(Event(event="run_started", data={"run_id": "run-cmd"}))

        # No commands have been queued, poll should return empty
        assert pub.poll_commands() == []

        # Note: The gRPC → command queue → poll path is tested in the
        # Go integration tests (Task 7). Python cannot easily call
        # Connect gRPC without a dedicated client library.
