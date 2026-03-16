"""Tests for EventBridge."""
import queue
import threading

import pytest

from chaosengineer.tui.bridge import EventBridge


class TestEventBridgePublish:
    def test_publish_stores_in_ring_buffer(self):
        bridge = EventBridge(capacity=5)
        bridge.publish({"event": "run_started", "run_id": "r1"})
        assert len(bridge.snapshot()) == 1
        assert bridge.snapshot()[0]["event"] == "run_started"

    def test_ring_buffer_evicts_oldest(self):
        bridge = EventBridge(capacity=3)
        for i in range(5):
            bridge.publish({"event": f"e{i}"})
        snap = bridge.snapshot()
        assert len(snap) == 3
        assert [e["event"] for e in snap] == ["e2", "e3", "e4"]


class TestEventBridgeSubscribe:
    def test_subscriber_receives_live_events(self):
        bridge = EventBridge()
        q = bridge.subscribe()
        bridge.publish({"event": "test"})
        event = q.get(timeout=1.0)
        assert event["event"] == "test"

    def test_unsubscribe_stops_delivery(self):
        bridge = EventBridge()
        q = bridge.subscribe()
        bridge.unsubscribe(q)
        bridge.publish({"event": "test"})
        assert q.empty()

    def test_slow_subscriber_drops_events(self):
        bridge = EventBridge()
        q = bridge.subscribe()
        # Fill the queue (maxsize=500)
        for i in range(600):
            bridge.publish({"event": f"e{i}"})
        # Queue should have 500 (first 500), rest dropped
        assert q.qsize() == 500


class TestEventBridgeSnapshot:
    def test_snapshot_returns_copy(self):
        bridge = EventBridge()
        bridge.publish({"event": "a"})
        snap = bridge.snapshot()
        bridge.publish({"event": "b"})
        assert len(snap) == 1  # copy, not live reference


class TestEventBridgeThreadSafety:
    def test_concurrent_publish_and_snapshot(self):
        bridge = EventBridge(capacity=100)
        errors = []

        def publisher():
            try:
                for i in range(200):
                    bridge.publish({"event": f"e{i}"})
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(200):
                    bridge.snapshot()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=publisher), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
