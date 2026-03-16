// bus/internal/filewriter_test.go
package internal

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	chaosv1 "chaos-bus/gen/chaos/v1"

	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func TestFileWriter_OutputFormat(t *testing.T) {
	broker := NewBroker()
	path := t.TempDir() + "/events.jsonl"

	cancel := StartFileWriter(broker, path)
	defer cancel()

	// Publish an event with known data
	ts := time.Date(2026, 3, 16, 14, 23, 45, 123456000, time.UTC)
	data, _ := structpb.NewStruct(map[string]any{
		"experiment_id": "exp-0-0",
		"dimension":     "lr",
		"params":        map[string]any{"lr": 0.01},
		"metric":        2.5,
		"cost_usd":      0.12,
	})
	event := &chaosv1.Event{
		EventType: "worker_completed",
		RunId:     "run-abc",
		Ts:        timestamppb.New(ts),
		Data:      data,
	}
	broker.Publish(event)

	// Give file writer time to flush
	time.Sleep(100 * time.Millisecond)

	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read file: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(string(content)), "\n")
	if len(lines) != 1 {
		t.Fatalf("got %d lines, want 1", len(lines))
	}

	var record map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &record); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	// Verify fields
	if record["event"] != "worker_completed" {
		t.Errorf("event: got %v", record["event"])
	}
	if record["run_id"] != "run-abc" {
		t.Errorf("run_id: got %v", record["run_id"])
	}
	if record["experiment_id"] != "exp-0-0" {
		t.Errorf("experiment_id: got %v", record["experiment_id"])
	}
	if record["metric"] != 2.5 {
		t.Errorf("metric: got %v", record["metric"])
	}

	// Verify timestamp format matches Python isoformat()
	tsStr, ok := record["ts"].(string)
	if !ok {
		t.Fatal("ts not a string")
	}
	if tsStr != "2026-03-16T14:23:45.123456+00:00" {
		t.Errorf("ts: got %s, want 2026-03-16T14:23:45.123456+00:00", tsStr)
	}

	// Verify nested data preserved
	params, ok := record["params"].(map[string]any)
	if !ok {
		t.Fatal("params not a map")
	}
	if params["lr"] != 0.01 {
		t.Errorf("params.lr: got %v", params["lr"])
	}
}

func TestFileWriter_NoRunID(t *testing.T) {
	broker := NewBroker()
	path := t.TempDir() + "/events.jsonl"

	cancel := StartFileWriter(broker, path)
	defer cancel()

	event := &chaosv1.Event{
		EventType: "iteration_started",
		Ts:        timestamppb.Now(),
		Data:      &structpb.Struct{},
	}
	broker.Publish(event)
	time.Sleep(100 * time.Millisecond)

	content, _ := os.ReadFile(path)
	var record map[string]any
	json.Unmarshal([]byte(strings.TrimSpace(string(content))), &record)

	// run_id should be absent when empty
	if _, exists := record["run_id"]; exists {
		t.Error("run_id should not be present when empty")
	}
}

func TestFileWriter_MultipleEvents(t *testing.T) {
	broker := NewBroker()
	path := t.TempDir() + "/events.jsonl"

	cancel := StartFileWriter(broker, path)
	defer cancel()

	for i := 0; i < 5; i++ {
		broker.Publish(makeEvent("test", "run-1"))
	}
	time.Sleep(200 * time.Millisecond)

	content, _ := os.ReadFile(path)
	lines := strings.Split(strings.TrimSpace(string(content)), "\n")
	if len(lines) != 5 {
		t.Errorf("got %d lines, want 5", len(lines))
	}
}

func TestFileWriter_SurvivesWriteError(t *testing.T) {
	broker := NewBroker()
	// Use a path that doesn't exist — file writer should log warning, not panic
	cancel := StartFileWriter(broker, "/nonexistent/dir/events.jsonl")
	defer cancel()

	// Publish should not panic
	broker.Publish(makeEvent("test", "run-1"))
	time.Sleep(100 * time.Millisecond)
}
