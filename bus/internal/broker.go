// bus/internal/broker.go
package internal

import (
	"sync"

	chaosv1 "chaos-bus/gen/chaos/v1"
)

// Broker manages an in-memory replay buffer and fans out events to subscribers.
type Broker struct {
	mu          sync.RWMutex
	buffer      []*chaosv1.Event
	subscribers map[chan *chaosv1.Event]struct{}
	currentRun  string
}

func NewBroker() *Broker {
	return &Broker{
		subscribers: make(map[chan *chaosv1.Event]struct{}),
	}
}

// Publish sends an event to all subscribers and appends to the replay buffer.
// Resets the buffer on "run_started" events.
func (b *Broker) Publish(event *chaosv1.Event) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if event.EventType == "run_started" {
		b.buffer = b.buffer[:0]
		b.currentRun = event.RunId
	}
	b.buffer = append(b.buffer, event)

	for ch := range b.subscribers {
		select {
		case ch <- event:
		default:
			// Slow subscriber — drop event rather than block others
		}
	}
}

// Subscribe returns a channel that receives all buffered events (replay)
// followed by live events. Call the returned cancel function to unsubscribe.
func (b *Broker) Subscribe(runID string) (<-chan *chaosv1.Event, func()) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Size channel to fit replay + headroom for live events
	bufSize := len(b.buffer) + 256
	ch := make(chan *chaosv1.Event, bufSize)

	// Replay buffered events while holding lock (no concurrent publishes)
	for _, event := range b.buffer {
		if runID == "" || event.RunId == runID {
			ch <- event
		}
	}

	b.subscribers[ch] = struct{}{}

	cancel := func() {
		b.mu.Lock()
		defer b.mu.Unlock()
		if _, ok := b.subscribers[ch]; ok {
			delete(b.subscribers, ch)
			close(ch)
		}
	}

	return ch, cancel
}

// CurrentRun returns the run_id from the most recent run_started event.
func (b *Broker) CurrentRun() string {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.currentRun
}
