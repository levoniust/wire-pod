package wirepod_ttr

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/fforchino/vector-go-sdk/pkg/vector"
	"github.com/fforchino/vector-go-sdk/pkg/vectorpb"
	"github.com/kercre123/wire-pod/chipper/pkg/logger"
	"github.com/kercre123/wire-pod/chipper/pkg/vars"
	"github.com/sashabaranov/go-openai"
)

// vectorBody implements AgentBody with the Vector SDK.
type vectorBody struct {
	robot *vector.Vector
}

func degToRad(d float64) float32 { return float32(d * math.Pi / 180) }

func actionErr(res *vectorpb.ActionResult, err error) error {
	if err != nil {
		return err
	}
	if res != nil && res.Code != vectorpb.ActionResult_ACTION_RESULT_SUCCESS {
		return errors.New(strings.ToLower(strings.TrimPrefix(res.Code.String(), "ACTION_RESULT_")))
	}
	return nil
}

func (b vectorBody) State(ctx context.Context) (string, error) {
	var parts []string
	if bat, err := b.robot.Conn.BatteryState(ctx, &vectorpb.BatteryStateRequest{}); err == nil {
		parts = append(parts, "battery "+strings.ToLower(strings.TrimPrefix(bat.BatteryLevel.String(), "BATTERY_LEVEL_")))
		if bat.IsOnChargerPlatform {
			parts = append(parts, "on the charger")
		}
	}
	// Pose, head, lift and status flags only come through the event stream; read one robot_state event.
	sctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	stream, err := b.robot.Conn.EventStream(sctx, &vectorpb.EventRequest{
		ListType: &vectorpb.EventRequest_WhiteList{WhiteList: &vectorpb.FilterList{List: []string{"robot_state"}}},
	})
	if err == nil {
		for {
			ev, err := stream.Recv()
			if err != nil {
				break
			}
			rs := ev.GetEvent().GetRobotState()
			if rs == nil {
				continue
			}
			parts = append(parts,
				fmt.Sprintf("head %.0f degrees", float64(rs.HeadAngleRad)*180/math.Pi),
				fmt.Sprintf("lift %.0f%% up", clamp((float64(rs.LiftHeightMm)-32)/60*100, 0, 100)))
			flags := map[vectorpb.RobotStatus]string{
				vectorpb.RobotStatus_ROBOT_STATUS_IS_PICKED_UP:      "picked up",
				vectorpb.RobotStatus_ROBOT_STATUS_IS_BEING_HELD:     "being held",
				vectorpb.RobotStatus_ROBOT_STATUS_CLIFF_DETECTED:    "CLIFF DETECTED",
				vectorpb.RobotStatus_ROBOT_STATUS_IS_CARRYING_BLOCK: "carrying the cube",
			}
			for flag, name := range flags {
				if rs.Status&uint32(flag) != 0 {
					parts = append(parts, name)
				}
			}
			if rs.ProxData != nil && rs.ProxData.FoundObject && !rs.ProxData.IsLiftInFov {
				parts = append(parts, fmt.Sprintf("distance sensor: something %d mm ahead", rs.ProxData.DistanceMm))
			}
			break
		}
	}
	if len(parts) == 0 {
		return "", errors.New("couldn't read the robot state")
	}
	return strings.Join(parts, ", "), nil
}

func (b vectorBody) Look(ctx context.Context) ([]byte, error) {
	resp, err := b.robot.Conn.CaptureSingleImage(ctx, &vectorpb.CaptureSingleImageRequest{})
	if err != nil {
		return nil, err
	}
	if len(resp.Data) == 0 {
		return nil, errors.New("camera returned no image")
	}
	return resp.Data, nil
}

func (b vectorBody) Say(ctx context.Context, text string) error {
	return DoSayText(text, b.robot)
}

func (b vectorBody) Animate(ctx context.Context, emotion string) error {
	for _, a := range animationMap {
		if a[0] == emotion {
			_, err := b.robot.Conn.PlayAnimation(ctx, &vectorpb.PlayAnimationRequest{Animation: &vectorpb.Animation{Name: a[1]}, Loops: 1})
			return err
		}
	}
	return errors.New("unknown emotion " + emotion)
}

func (b vectorBody) Drive(ctx context.Context, mm float64) error {
	resp, err := b.robot.Conn.DriveStraight(ctx, &vectorpb.DriveStraightRequest{SpeedMmps: 80, DistMm: float32(mm), NumRetries: 1})
	if err != nil {
		return err
	}
	return actionErr(resp.Result, nil)
}

func (b vectorBody) Turn(ctx context.Context, degrees float64) error {
	resp, err := b.robot.Conn.TurnInPlace(ctx, &vectorpb.TurnInPlaceRequest{
		AngleRad: degToRad(degrees), SpeedRadPerSec: math.Pi / 2, AccelRadPerSec2: 10, NumRetries: 1,
	})
	if err != nil {
		return err
	}
	return actionErr(resp.Result, nil)
}

func (b vectorBody) SetHead(ctx context.Context, degrees float64) error {
	resp, err := b.robot.Conn.SetHeadAngle(ctx, &vectorpb.SetHeadAngleRequest{AngleRad: degToRad(degrees), MaxSpeedRadPerSec: 10, AccelRadPerSec2: 10})
	if err != nil {
		return err
	}
	return actionErr(resp.Result, nil)
}

func (b vectorBody) SetLift(ctx context.Context, fraction float64) error {
	// Vector's lift travels from 32 mm (down) to 92 mm (up).
	resp, err := b.robot.Conn.SetLiftHeight(ctx, &vectorpb.SetLiftHeightRequest{HeightMm: float32(32 + 60*fraction), MaxSpeedRadPerSec: 10, AccelRadPerSec2: 10})
	if err != nil {
		return err
	}
	return actionErr(resp.Result, nil)
}

func (b vectorBody) DriveOnCharger(ctx context.Context) error {
	resp, err := b.robot.Conn.DriveOnCharger(ctx, &vectorpb.DriveOnChargerRequest{})
	if err != nil {
		return err
	}
	if resp.Result != vectorpb.BehaviorResults_BEHAVIOR_COMPLETE_STATE {
		return errors.New("couldn't get onto the charger")
	}
	return nil
}

// llmClient builds the chat client and model name from the knowledge graph settings.
func llmClient() (*openai.Client, string, error) {
	k := vars.APIConfig.Knowledge
	switch k.Provider {
	case "openai":
		model := openai.GPT4oMini
		if m := strings.TrimSpace(k.Model); m != "" {
			model = m
		}
		return openai.NewClient(k.Key), model, nil
	case "together":
		conf := openai.DefaultConfig(k.Key)
		conf.BaseURL = "https://api.together.xyz/v1"
		return openai.NewClientWithConfig(conf), k.Model, nil
	case "custom":
		conf := openai.DefaultConfig(k.Key)
		conf.BaseURL = k.Endpoint
		return openai.NewClientWithConfig(conf), k.Model, nil
	}
	return nil, "", errors.New("agent mode needs the OpenAI, Together or custom LLM provider")
}

var (
	agentMu      sync.Mutex
	agentCancels = map[string]context.CancelFunc{}
)

// StopAgentTask cancels the running task on a robot, if any. It reports whether one was running.
func StopAgentTask(esn string) bool {
	agentMu.Lock()
	defer agentMu.Unlock()
	if cancel, ok := agentCancels[esn]; ok {
		cancel()
		return true
	}
	return false
}

// AgentTaskRunning reports whether a task is running on a robot.
func AgentTaskRunning(esn string) bool {
	agentMu.Lock()
	defer agentMu.Unlock()
	_, ok := agentCancels[esn]
	return ok
}

// StartAgentTask runs a task in the background: it takes behavior control, runs the loop, and releases
// control at the end. Touching Vector or saying the wake word stops it. Only one task runs per robot.
func StartAgentTask(robot *vector.Vector, goal string) error {
	if !vars.APIConfig.Knowledge.AgentEnable {
		return errors.New("agent mode is turned off in the knowledge graph settings")
	}
	goal = strings.TrimSpace(goal)
	if goal == "" {
		return errors.New("the task is empty")
	}
	client, model, err := llmClient()
	if err != nil {
		return err
	}
	esn := robot.Cfg.SerialNo
	ctx, cancel := context.WithCancel(context.Background())
	agentMu.Lock()
	if _, busy := agentCancels[esn]; busy {
		agentMu.Unlock()
		cancel()
		return errors.New("a task is already running on this robot")
	}
	agentCancels[esn] = cancel
	agentMu.Unlock()

	go func() {
		defer func() {
			agentMu.Lock()
			delete(agentCancels, esn)
			agentMu.Unlock()
			cancel()
		}()
		release, err := takeBehaviorControl(ctx, robot)
		if err != nil {
			logger.LogUI("Agent (" + esn + "): couldn't take behavior control: " + err.Error())
			return
		}
		defer release()

		// Reuse the knowledge graph interrupter: touch or wake word cancels the task.
		// Buffered so the interrupter never blocks if it fires after the task has ended.
		interrupted := make(chan bool, 1)
		stopInterrupter := make(chan bool)
		go InterruptKGSimWhenTouchedOrWaked(robot, interrupted, stopInterrupter)
		go func() {
			select {
			case <-interrupted:
				logger.LogUI("Agent (" + esn + "): stopped by touch or wake word")
				cancel()
			case <-ctx.Done():
			}
		}()
		defer func() {
			select {
			case stopInterrupter <- true:
			case <-time.After(time.Second):
			}
		}()

		logger.LogUI("Agent (" + esn + ") task: " + goal)
		summary, err := RunAgent(ctx, client, vectorBody{robot: robot}, AgentConfig{
			Model:    model,
			MaxSteps: agentDefaultSteps,
			Log:      func(s string) { logger.LogUI("Agent (" + esn + ") " + s) },
		}, goal)
		if err != nil {
			logger.LogUI("Agent (" + esn + ") ended: " + err.Error())
			return
		}
		logger.LogUI("Agent (" + esn + ") done: " + summary)
	}()
	return nil
}

// takeBehaviorControl asks for control at OVERRIDE_BEHAVIORS and returns a function that releases it.
func takeBehaviorControl(ctx context.Context, robot *vector.Vector) (func(), error) {
	r, err := robot.Conn.BehaviorControl(ctx)
	if err != nil {
		return nil, err
	}
	err = r.Send(&vectorpb.BehaviorControlRequest{RequestType: &vectorpb.BehaviorControlRequest_ControlRequest{
		ControlRequest: &vectorpb.ControlRequest{Priority: vectorpb.ControlRequest_OVERRIDE_BEHAVIORS},
	}})
	if err != nil {
		return nil, err
	}
	for {
		resp, err := r.Recv()
		if err != nil {
			return nil, err
		}
		if resp.GetControlGrantedResponse() != nil {
			break
		}
	}
	return func() {
		r.Send(&vectorpb.BehaviorControlRequest{RequestType: &vectorpb.BehaviorControlRequest_ControlRelease{
			ControlRelease: &vectorpb.ControlRelease{},
		}})
		r.CloseSend()
	}, nil
}
