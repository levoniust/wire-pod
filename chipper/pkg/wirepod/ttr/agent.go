package wirepod_ttr

// Agent mode: the LLM drives Vector in a loop. Each step it sees the robot's state (and a camera
// image when it asks for one), calls tools (drive, turn, look, say, ...), gets the results back and
// decides what to do next, until it calls finish or runs out of steps or time.

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai"
)

// AgentBody is what the agent loop needs from a robot. vectorBody implements it with the SDK; tests use a fake.
type AgentBody interface {
	// State returns a short plain-text description (battery, on charger, picked up, cliff, head, lift).
	State(ctx context.Context) (string, error)
	// Look returns a JPEG from the camera.
	Look(ctx context.Context) ([]byte, error)
	Say(ctx context.Context, text string) error
	Animate(ctx context.Context, emotion string) error
	Drive(ctx context.Context, mm float64) error
	Turn(ctx context.Context, degrees float64) error
	SetHead(ctx context.Context, degrees float64) error
	SetLift(ctx context.Context, fraction float64) error
	DriveOnCharger(ctx context.Context) error
}

// AgentLLM is the one call the loop makes; *openai.Client satisfies it.
type AgentLLM interface {
	CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error)
}

type AgentConfig struct {
	Model    string
	MaxSteps int
	Timeout  time.Duration
	// Log receives one line per step, for the wire-pod log page.
	Log func(string)
}

const (
	agentMaxDriveMm   = 300
	agentMinHeadDeg   = -22
	agentMaxHeadDeg   = 45
	agentDefaultSteps = 15
)

const agentSystemPrompt = `You are controlling an Anki Vector robot (a small robot about 10 cm long that usually sits on a table) to carry out a task for the user.
You work in a loop: call one or more tools, read their results, then decide the next step. You get the robot's state after every step.
Rules:
- Move in small steps (at most 300 mm per drive) and check the result before moving on. Vector can fall off tables; if a cliff is detected, back up and turn away.
- Use look to see through the camera before and after moving toward something. The camera points where the head points; head angle 0 is level, 45 is up, -22 is down at the floor.
- Use say for short spoken updates (one sentence). Don't narrate every step.
- When the task is done, or impossible, call finish with one short sentence for Vector to say.`

var agentTools = []openai.Tool{
	agentTool("look", "Take a picture with the camera. The image is added to the conversation.", nil, nil),
	agentTool("drive", "Drive straight. Positive is forward, negative is backward.",
		map[string]any{"distance_mm": map[string]any{"type": "number", "description": "-300 to 300"}}, []string{"distance_mm"}),
	agentTool("turn", "Turn in place. Positive turns left, negative turns right.",
		map[string]any{"degrees": map[string]any{"type": "number", "description": "-180 to 180"}}, []string{"degrees"}),
	agentTool("set_head", "Move the head (and camera).",
		map[string]any{"degrees": map[string]any{"type": "number", "description": "-22 (down) to 45 (up)"}}, []string{"degrees"}),
	agentTool("set_lift", "Move the lift (arms).",
		map[string]any{"height": map[string]any{"type": "number", "description": "0 (down) to 1 (up)"}}, []string{"height"}),
	agentTool("say", "Say something out loud.",
		map[string]any{"text": map[string]any{"type": "string"}}, []string{"text"}),
	agentTool("animate", "Show an emotion with an animation.",
		map[string]any{"emotion": map[string]any{"type": "string", "enum": agentEmotions()}}, []string{"emotion"}),
	agentTool("drive_on_charger", "Find the charger and drive onto it.", nil, nil),
	agentTool("finish", "End the task.",
		map[string]any{"summary": map[string]any{"type": "string", "description": "one short sentence for Vector to say"}}, []string{"summary"}),
}

func agentTool(name, desc string, props map[string]any, required []string) openai.Tool {
	if props == nil {
		props = map[string]any{}
	}
	params := map[string]any{"type": "object", "properties": props}
	if required != nil {
		params["required"] = required
	}
	return openai.Tool{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: name, Description: desc, Parameters: params}}
}

func agentEmotions() []string {
	var out []string
	for _, a := range animationMap {
		out = append(out, a[0])
	}
	return out
}

func clamp(v, lo, hi float64) float64 {
	if math.IsNaN(v) {
		return 0
	}
	return math.Max(lo, math.Min(hi, v))
}

// agentToolResult runs one tool call. It returns the text for the tool message, an image to show the
// model (look only), and whether the task is finished.
func agentToolResult(ctx context.Context, body AgentBody, call openai.ToolCall) (result string, image []byte, finished bool) {
	var args map[string]any
	if strings.TrimSpace(call.Function.Arguments) != "" {
		if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
			return "error: arguments are not valid JSON", nil, false
		}
	}
	num := func(key string) (float64, bool) {
		v, ok := args[key].(float64)
		return v, ok
	}
	str := func(key string) string {
		s, _ := args[key].(string)
		return strings.TrimSpace(s)
	}
	done := func(err error, okText string) string {
		if err != nil {
			return "error: " + err.Error()
		}
		return okText
	}

	switch call.Function.Name {
	case "look":
		img, err := body.Look(ctx)
		if err != nil {
			return "error: " + err.Error(), nil, false
		}
		return "picture taken; it follows in the next message", img, false
	case "drive":
		mm, ok := num("distance_mm")
		if !ok {
			return "error: distance_mm is required", nil, false
		}
		mm = clamp(mm, -agentMaxDriveMm, agentMaxDriveMm)
		return done(body.Drive(ctx, mm), fmt.Sprintf("drove %.0f mm", mm)), nil, false
	case "turn":
		deg, ok := num("degrees")
		if !ok {
			return "error: degrees is required", nil, false
		}
		deg = clamp(deg, -180, 180)
		return done(body.Turn(ctx, deg), fmt.Sprintf("turned %.0f degrees", deg)), nil, false
	case "set_head":
		deg, ok := num("degrees")
		if !ok {
			return "error: degrees is required", nil, false
		}
		deg = clamp(deg, agentMinHeadDeg, agentMaxHeadDeg)
		return done(body.SetHead(ctx, deg), fmt.Sprintf("head at %.0f degrees", deg)), nil, false
	case "set_lift":
		h, ok := num("height")
		if !ok {
			return "error: height is required", nil, false
		}
		h = clamp(h, 0, 1)
		return done(body.SetLift(ctx, h), fmt.Sprintf("lift at %.2f", h)), nil, false
	case "say":
		text := str("text")
		if text == "" {
			return "error: text is required", nil, false
		}
		return done(body.Say(ctx, text), "said it"), nil, false
	case "animate":
		return done(body.Animate(ctx, str("emotion")), "animation played"), nil, false
	case "drive_on_charger":
		return done(body.DriveOnCharger(ctx), "on the charger"), nil, false
	case "finish":
		if s := str("summary"); s != "" {
			body.Say(ctx, s)
		}
		return "finished", nil, true
	}
	return "error: unknown tool " + call.Function.Name, nil, false
}

// observation is the user message sent after every step: the robot state, plus the newest picture if there is one.
func agentObservation(ctx context.Context, body AgentBody, prefix string, image []byte) openai.ChatCompletionMessage {
	state, err := body.State(ctx)
	if err != nil {
		state = "unknown (" + err.Error() + ")"
	}
	text := prefix + "Robot state: " + state
	if image == nil {
		return openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: text}
	}
	return openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, MultiContent: []openai.ChatMessagePart{
		{Type: openai.ChatMessagePartTypeText, Text: text},
		{Type: openai.ChatMessagePartTypeImageURL, ImageURL: &openai.ChatMessageImageURL{
			URL:    "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(image),
			Detail: openai.ImageURLDetailLow,
		}},
	}}
}

// dropOldImages keeps only the newest picture in the conversation, so long tasks don't grow without bound.
func dropOldImages(msgs []openai.ChatCompletionMessage) {
	seen := false
	for i := len(msgs) - 1; i >= 0; i-- {
		if len(msgs[i].MultiContent) == 0 {
			continue
		}
		if !seen {
			seen = true
			continue
		}
		text := ""
		for _, p := range msgs[i].MultiContent {
			if p.Type == openai.ChatMessagePartTypeText {
				text = p.Text
			}
		}
		msgs[i].MultiContent = nil
		msgs[i].Content = text + " [older picture removed]"
	}
}

// RunAgent runs the loop until the model calls finish (or answers without tools), the step limit or
// timeout is hit, or ctx is cancelled (for example when Vector is touched). It returns a short summary.
func RunAgent(ctx context.Context, llm AgentLLM, body AgentBody, cfg AgentConfig, goal string) (string, error) {
	if cfg.MaxSteps <= 0 {
		cfg.MaxSteps = agentDefaultSteps
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 3 * time.Minute
	}
	if cfg.Log == nil {
		cfg.Log = func(string) {}
	}
	ctx, cancel := context.WithTimeout(ctx, cfg.Timeout)
	defer cancel()

	msgs := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: agentSystemPrompt},
		agentObservation(ctx, body, "Task: "+goal+"\n", nil),
	}
	for step := 1; step <= cfg.MaxSteps; step++ {
		resp, err := llm.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
			Model:    cfg.Model,
			Messages: msgs,
			Tools:    agentTools,
		})
		if err != nil {
			if ctx.Err() != nil {
				return "", ctx.Err()
			}
			return "", fmt.Errorf("LLM error: %w", err)
		}
		if len(resp.Choices) == 0 {
			return "", errors.New("LLM returned no choices")
		}
		msg := resp.Choices[0].Message
		msgs = append(msgs, msg)
		if len(msg.ToolCalls) == 0 {
			// No tool call: the model is done talking. Say what it said and stop.
			text := strings.TrimSpace(msg.Content)
			cfg.Log(fmt.Sprintf("step %d: answered without a tool: %s", step, text))
			if text != "" {
				body.Say(ctx, text)
			}
			return text, nil
		}
		var image []byte
		for _, call := range msg.ToolCalls {
			if ctx.Err() != nil {
				return "", ctx.Err()
			}
			result, img, finished := agentToolResult(ctx, body, call)
			cfg.Log(fmt.Sprintf("step %d: %s(%s) -> %s", step, call.Function.Name, call.Function.Arguments, result))
			msgs = append(msgs, openai.ChatCompletionMessage{Role: openai.ChatMessageRoleTool, ToolCallID: call.ID, Content: result})
			if finished {
				var a struct {
					Summary string `json:"summary"`
				}
				json.Unmarshal([]byte(call.Function.Arguments), &a)
				return a.Summary, nil
			}
			if img != nil {
				image = img
			}
		}
		left := cfg.MaxSteps - step
		msgs = append(msgs, agentObservation(ctx, body, fmt.Sprintf("Step %d done, %d steps left.\n", step, left), image))
		dropOldImages(msgs)
	}
	body.Say(ctx, "I ran out of steps for that task.")
	return "", fmt.Errorf("stopped after %d steps", cfg.MaxSteps)
}
