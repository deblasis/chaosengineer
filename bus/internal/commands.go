// bus/internal/commands.go
package internal

import (
	"encoding/json"
	"net/http"
	"sync"
)

// CommandQueue accumulates commands from gRPC RPCs and serves them
// to the coordinator via GET /commands.
type CommandQueue struct {
	mu         sync.Mutex
	queue      []json.RawMessage
	currentRun string
}

func NewCommandQueue() *CommandQueue {
	return &CommandQueue{}
}

func (q *CommandQueue) SetCurrentRun(runID string) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.currentRun = runID
}

func (q *CommandQueue) CurrentRun() string {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.currentRun
}

// Push adds a command to the queue.
func (q *CommandQueue) Push(cmd json.RawMessage) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.queue = append(q.queue, cmd)
}

// Drain returns all queued commands and clears the queue.
// Returns nil if empty.
func (q *CommandQueue) Drain() []json.RawMessage {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.queue) == 0 {
		return nil
	}
	cmds := q.queue
	q.queue = nil
	return cmds
}

// NewCommandsHandler returns an HTTP handler for GET /commands.
// Returns the drained queue as a JSON array.
func NewCommandsHandler(queue *CommandQueue) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		cmds := queue.Drain()
		if cmds == nil {
			cmds = []json.RawMessage{}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(cmds)
	}
}
