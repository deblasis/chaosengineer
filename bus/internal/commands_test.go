// bus/internal/commands_test.go
package internal

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCommandQueue_PushAndDrain(t *testing.T) {
	q := NewCommandQueue()

	cmd1, _ := json.Marshal(map[string]string{"command": "pause", "run_id": "run-1"})
	cmd2, _ := json.Marshal(map[string]any{"command": "extend_budget", "add_cost_usd": 10.0})

	q.Push(cmd1)
	q.Push(cmd2)

	cmds := q.Drain()
	if len(cmds) != 2 {
		t.Fatalf("got %d commands, want 2", len(cmds))
	}

	// Drain again — should be empty
	cmds2 := q.Drain()
	if len(cmds2) != 0 {
		t.Errorf("got %d commands after drain, want 0", len(cmds2))
	}
}

func TestCommandQueue_DrainEmpty(t *testing.T) {
	q := NewCommandQueue()
	cmds := q.Drain()
	if cmds != nil {
		t.Errorf("expected nil, got %v", cmds)
	}
}

func TestCommandQueue_SetCurrentRun(t *testing.T) {
	q := NewCommandQueue()
	q.SetCurrentRun("run-abc")
	if q.CurrentRun() != "run-abc" {
		t.Errorf("got %s, want run-abc", q.CurrentRun())
	}
}

func TestCommandsHandler_ReturnsDrainedCommands(t *testing.T) {
	q := NewCommandQueue()
	cmd, _ := json.Marshal(map[string]string{"command": "pause"})
	q.Push(cmd)

	handler := NewCommandsHandler(q)
	req := httptest.NewRequest(http.MethodGet, "/commands", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("got status %d, want 200", w.Code)
	}

	body, _ := io.ReadAll(w.Body)
	var cmds []json.RawMessage
	if err := json.Unmarshal(body, &cmds); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if len(cmds) != 1 {
		t.Errorf("got %d commands, want 1", len(cmds))
	}
}

func TestCommandsHandler_EmptyArray(t *testing.T) {
	q := NewCommandQueue()
	handler := NewCommandsHandler(q)

	req := httptest.NewRequest(http.MethodGet, "/commands", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	body, _ := io.ReadAll(w.Body)
	var cmds []json.RawMessage
	json.Unmarshal(body, &cmds)
	if len(cmds) != 0 {
		t.Errorf("got %d commands, want 0", len(cmds))
	}
}

func TestCommandsHandler_MethodNotAllowed(t *testing.T) {
	q := NewCommandQueue()
	handler := NewCommandsHandler(q)

	req := httptest.NewRequest(http.MethodPost, "/commands", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("got status %d, want 405", w.Code)
	}
}
