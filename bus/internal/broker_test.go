// bus/internal/broker_test.go
package internal

import (
	"testing"
	"time"

	chaosv1 "chaos-bus/gen/chaos/v1"

	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func makeEvent(eventType, runID string) *chaosv1.Event {
	return &chaosv1.Event{
		EventType: eventType,
		RunId:     runID,
		Ts:        timestamppb.Now(),
		Data:      &structpb.Struct{},
	}
}

func recvTimeout(ch <-chan *chaosv1.Event, timeout time.Duration) (*chaosv1.Event, bool) {
	select {
	case e := <-ch:
		return e, true
	case <-time.After(timeout):
		return nil, false
	}
}

func TestBroker_ReplayOnSubscribe(t *testing.T) {
	b := NewBroker()
	b.Publish(makeEvent("run_started", "run-1"))
	b.Publish(makeEvent("worker_completed", "run-1"))

	ch, cancel := b.Subscribe("")
	defer cancel()

	e1, ok := recvTimeout(ch, time.Second)
	if !ok {
		t.Fatal("timeout waiting for first replayed event")
	}
	if e1.EventType != "run_started" {
		t.Errorf("got %s, want run_started", e1.EventType)
	}

	e2, ok := recvTimeout(ch, time.Second)
	if !ok {
		t.Fatal("timeout waiting for second replayed event")
	}
	if e2.EventType != "worker_completed" {
		t.Errorf("got %s, want worker_completed", e2.EventType)
	}
}

func TestBroker_LiveDelivery(t *testing.T) {
	b := NewBroker()
	ch, cancel := b.Subscribe("")
	defer cancel()

	b.Publish(makeEvent("test_event", "run-1"))

	e, ok := recvTimeout(ch, time.Second)
	if !ok {
		t.Fatal("timeout waiting for live event")
	}
	if e.EventType != "test_event" {
		t.Errorf("got %s, want test_event", e.EventType)
	}
}

func TestBroker_BufferResetsOnRunStarted(t *testing.T) {
	b := NewBroker()
	b.Publish(makeEvent("old_event", "run-1"))
	b.Publish(makeEvent("run_started", "run-2"))
	b.Publish(makeEvent("new_event", "run-2"))

	ch, cancel := b.Subscribe("")
	defer cancel()

	// Should only get run_started and new_event (old_event was cleared)
	var types []string
	for i := 0; i < 2; i++ {
		e, ok := recvTimeout(ch, time.Second)
		if !ok {
			t.Fatalf("timeout at event %d", i)
		}
		types = append(types, e.EventType)
	}

	if types[0] != "run_started" || types[1] != "new_event" {
		t.Errorf("got %v, want [run_started, new_event]", types)
	}
}

func TestBroker_MultipleSubscribers(t *testing.T) {
	b := NewBroker()
	ch1, cancel1 := b.Subscribe("")
	defer cancel1()
	ch2, cancel2 := b.Subscribe("")
	defer cancel2()

	b.Publish(makeEvent("broadcast", "run-1"))

	for i, ch := range []<-chan *chaosv1.Event{ch1, ch2} {
		e, ok := recvTimeout(ch, time.Second)
		if !ok {
			t.Fatalf("subscriber %d: timeout", i)
		}
		if e.EventType != "broadcast" {
			t.Errorf("subscriber %d: got %s, want broadcast", i, e.EventType)
		}
	}
}

func TestBroker_CurrentRun(t *testing.T) {
	b := NewBroker()
	if b.CurrentRun() != "" {
		t.Errorf("expected empty, got %s", b.CurrentRun())
	}

	b.Publish(makeEvent("run_started", "run-abc"))
	if b.CurrentRun() != "run-abc" {
		t.Errorf("expected run-abc, got %s", b.CurrentRun())
	}
}

func TestBroker_CancelUnsubscribes(t *testing.T) {
	b := NewBroker()
	ch, cancel := b.Subscribe("")
	cancel()

	// Channel should be closed after cancel
	_, open := <-ch
	if open {
		t.Error("expected channel to be closed after cancel")
	}

	// Double cancel should not panic
	cancel()

	// Publishing after cancel should not panic
	b.Publish(makeEvent("after_cancel", "run-1"))
}
