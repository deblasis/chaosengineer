// bus/internal/publisher_test.go
package internal

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestPublishHandler_ValidEvent(t *testing.T) {
	broker := NewBroker()
	handler := NewPublishHandler(broker, nil)

	body := `{"event":"worker_completed","run_id":"run-abc","experiment_id":"exp-0","metric":2.5}`
	req := httptest.NewRequest(http.MethodPost, "/publish", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("got status %d, want 200", w.Code)
	}

	// Verify event was published to broker
	ch, cancel := broker.Subscribe("")
	defer cancel()

	e, ok := recvTimeout(ch, time.Second)
	if !ok {
		t.Fatal("no event in broker")
	}
	if e.EventType != "worker_completed" {
		t.Errorf("event_type: got %s, want worker_completed", e.EventType)
	}
	if e.RunId != "run-abc" {
		t.Errorf("run_id: got %s, want run-abc", e.RunId)
	}
	if e.Ts == nil {
		t.Error("timestamp should be set")
	}

	// Data should contain remaining fields
	data := e.Data.AsMap()
	if data["experiment_id"] != "exp-0" {
		t.Errorf("experiment_id: got %v, want exp-0", data["experiment_id"])
	}
	if data["metric"] != 2.5 {
		t.Errorf("metric: got %v, want 2.5", data["metric"])
	}
}

func TestPublishHandler_TimestampPreserved(t *testing.T) {
	broker := NewBroker()
	handler := NewPublishHandler(broker, nil)

	ts := "2026-03-16T14:00:00.000000+00:00"
	body := `{"event":"test","ts":"` + ts + `"}`
	req := httptest.NewRequest(http.MethodPost, "/publish", bytes.NewBufferString(body))
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	ch, cancel := broker.Subscribe("")
	defer cancel()
	e, ok := recvTimeout(ch, time.Second)
	if !ok {
		t.Fatal("no event")
	}

	got := e.Ts.AsTime()
	if got.Year() != 2026 || got.Month() != 3 || got.Day() != 16 {
		t.Errorf("timestamp not preserved: %v", got)
	}
}

func TestPublishHandler_MissingEventField(t *testing.T) {
	broker := NewBroker()
	handler := NewPublishHandler(broker, nil)

	body := `{"run_id":"run-1","metric":1.0}`
	req := httptest.NewRequest(http.MethodPost, "/publish", bytes.NewBufferString(body))
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("got status %d, want 400", w.Code)
	}
}

func TestPublishHandler_MalformedJSON(t *testing.T) {
	broker := NewBroker()
	handler := NewPublishHandler(broker, nil)

	req := httptest.NewRequest(http.MethodPost, "/publish", bytes.NewBufferString("{bad"))
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("got status %d, want 400", w.Code)
	}
}

func TestPublishHandler_MethodNotAllowed(t *testing.T) {
	broker := NewBroker()
	handler := NewPublishHandler(broker, nil)

	req := httptest.NewRequest(http.MethodGet, "/publish", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("got status %d, want 405", w.Code)
	}
}

func TestPublishHandler_UpdatesCommandQueueCurrentRun(t *testing.T) {
	broker := NewBroker()
	queue := NewCommandQueue()
	handler := NewPublishHandler(broker, queue)

	body := `{"event":"run_started","run_id":"run-xyz"}`
	req := httptest.NewRequest(http.MethodPost, "/publish", bytes.NewBufferString(body))
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if queue.CurrentRun() != "run-xyz" {
		t.Errorf("got %s, want run-xyz", queue.CurrentRun())
	}
}

func TestPublishHandler_NestedData(t *testing.T) {
	broker := NewBroker()
	handler := NewPublishHandler(broker, nil)

	payload := map[string]any{
		"event":  "worker_completed",
		"params": map[string]any{"lr": 0.01, "batch_size": 32},
		"metric": 2.5,
	}
	body, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/publish", bytes.NewBuffer(body))
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	ch, cancel := broker.Subscribe("")
	defer cancel()
	e, _ := recvTimeout(ch, time.Second)

	data := e.Data.AsMap()
	params, ok := data["params"].(map[string]any)
	if !ok {
		t.Fatalf("params not a map: %T", data["params"])
	}
	if params["lr"] != 0.01 {
		t.Errorf("lr: got %v, want 0.01", params["lr"])
	}
}
