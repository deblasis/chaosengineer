// bus/internal/server.go
package internal

import (
	"context"
	"encoding/json"

	chaosv1 "chaos-bus/gen/chaos/v1"

	"connectrpc.com/connect"
)

// BusServer implements the BusService Connect gRPC interface.
type BusServer struct {
	Broker *Broker
	Queue  *CommandQueue
}

func NewBusServer(broker *Broker, queue *CommandQueue) *BusServer {
	return &BusServer{Broker: broker, Queue: queue}
}

// Subscribe streams events to the client. Replays buffered events first,
// then delivers live events until the client disconnects.
func (s *BusServer) Subscribe(
	ctx context.Context,
	req *connect.Request[chaosv1.SubscribeRequest],
	stream *connect.ServerStream[chaosv1.Event],
) error {
	runID := req.Msg.RunId
	ch, cancel := s.Broker.Subscribe(runID)
	defer cancel()

	for {
		select {
		case <-ctx.Done():
			return nil
		case event, ok := <-ch:
			if !ok {
				return nil
			}
			if err := stream.Send(event); err != nil {
				return err
			}
		}
	}
}

// PauseRun queues a pause command for the coordinator to pick up.
func (s *BusServer) PauseRun(
	ctx context.Context,
	req *connect.Request[chaosv1.PauseRunRequest],
) (*connect.Response[chaosv1.PauseRunResponse], error) {
	runID := req.Msg.RunId
	if runID == "" {
		runID = s.Queue.CurrentRun()
	}
	cmd, _ := json.Marshal(map[string]string{
		"command": "pause",
		"run_id":  runID,
	})
	s.Queue.Push(cmd)
	return connect.NewResponse(&chaosv1.PauseRunResponse{}), nil
}

// ExtendBudget queues a budget extension command for the coordinator.
func (s *BusServer) ExtendBudget(
	ctx context.Context,
	req *connect.Request[chaosv1.ExtendBudgetRequest],
) (*connect.Response[chaosv1.ExtendBudgetResponse], error) {
	runID := req.Msg.RunId
	if runID == "" {
		runID = s.Queue.CurrentRun()
	}
	cmd, _ := json.Marshal(map[string]any{
		"command":          "extend_budget",
		"run_id":           runID,
		"add_cost_usd":     req.Msg.AddCostUsd,
		"add_experiments":  req.Msg.AddExperiments,
		"add_time_seconds": req.Msg.AddTimeSeconds,
	})
	s.Queue.Push(cmd)
	return connect.NewResponse(&chaosv1.ExtendBudgetResponse{}), nil
}
