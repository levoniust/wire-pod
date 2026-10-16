package wirepod_ttr

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/kercre123/wire-pod/chipper/pkg/vars"
	"github.com/sashabaranov/go-openai"
)

type fakeBody struct {
	calls []string
	said  []string
}

func (f *fakeBody) State(ctx context.Context) (string, error) { return "battery nominal", nil }
func (f *fakeBody) Look(ctx context.Context) ([]byte, error) {
	f.calls = append(f.calls, "look")
	return []byte{0xff, 0xd8, 0xff}, nil
}
func (f *fakeBody) Say(ctx context.Context, text string) error {
	f.said = append(f.said, text)
	return nil
}
func (f *fakeBody) Animate(ctx context.Context, e string) error { return nil }
func (f *fakeBody) Drive(ctx context.Context, mm float64) error {
	f.calls = append(f.calls, "drive")
	return nil
}
func (f *fakeBody) Turn(ctx context.Context, d float64) error {
	f.calls = append(f.calls, "turn")
	return errors.New("cancelled")
}
func (f *fakeBody) SetHead(ctx context.Context, d float64) error { return nil }
func (f *fakeBody) SetLift(ctx context.Context, h float64) error { return nil }
func (f *fakeBody) DriveOnCharger(ctx context.Context) error     { return nil }

// scriptedLLM answers each request with the next message and records what it was sent.
type scriptedLLM struct {
	replies  []openai.ChatCompletionMessage
	requests []openai.ChatCompletionRequest
}

func (s *scriptedLLM) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
	cp := req
	cp.Messages = append([]openai.ChatCompletionMessage(nil), req.Messages...)
	s.requests = append(s.requests, cp)
	if len(s.replies) == 0 {
		return openai.ChatCompletionResponse{}, errors.New("script ran out")
	}
	m := s.replies[0]
	s.replies = s.replies[1:]
	return openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: m}}}, nil
}

func toolCall(id, name, args string) openai.ToolCall {
	return openai.ToolCall{ID: id, Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: name, Arguments: args}}
}

func assistant(calls ...openai.ToolCall) openai.ChatCompletionMessage {
	return openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, ToolCalls: calls}
}

func TestAgentDrivesLooksAndFinishes(t *testing.T) {
	llm := &scriptedLLM{replies: []openai.ChatCompletionMessage{
		assistant(toolCall("1", "drive", `{"distance_mm": 5000}`), toolCall("2", "look", `{}`)),
		assistant(toolCall("3", "turn", `{"degrees": 90}`)),
		assistant(toolCall("4", "finish", `{"summary": "Found it."}`)),
	}}
	body := &fakeBody{}
	var logs []string
	summary, err := RunAgent(context.Background(), llm, body, AgentConfig{Model: "m", Log: func(s string) { logs = append(logs, s) }}, "find the cube")
	if err != nil || summary != "Found it." {
		t.Fatalf("summary %q, err %v", summary, err)
	}
	if strings.Join(body.calls, ",") != "drive,look,turn" {
		t.Fatalf("calls %v", body.calls)
	}
	if len(body.said) != 1 || body.said[0] != "Found it." {
		t.Fatalf("said %v", body.said)
	}
	// The drive was clamped to the safe maximum.
	if !strings.Contains(logs[0], "drove 300 mm") {
		t.Fatalf("log %q", logs[0])
	}
	// The second request answers both tool calls and then shows the picture with the state.
	msgs := llm.requests[1].Messages
	last := msgs[len(msgs)-1]
	if msgs[len(msgs)-3].ToolCallID != "1" || msgs[len(msgs)-2].ToolCallID != "2" {
		t.Fatalf("tool results out of order: %+v", msgs)
	}
	if len(last.MultiContent) != 2 || last.MultiContent[1].ImageURL == nil || !strings.HasPrefix(last.MultiContent[1].ImageURL.URL, "data:image/jpeg;base64,") {
		t.Fatalf("expected an image observation, got %+v", last)
	}
	// A failing action is reported back to the model instead of stopping the task.
	msgs = llm.requests[2].Messages
	if msgs[len(msgs)-2].Content != "error: cancelled" {
		t.Fatalf("expected the turn error to reach the model, got %q", msgs[len(msgs)-2].Content)
	}
	if len(llm.requests[0].Tools) == 0 || llm.requests[0].Model != "m" {
		t.Fatalf("tools/model not sent")
	}
}

func TestAgentPlainAnswerEndsTask(t *testing.T) {
	llm := &scriptedLLM{replies: []openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleAssistant, Content: "I can't do that."}}}
	body := &fakeBody{}
	summary, err := RunAgent(context.Background(), llm, body, AgentConfig{}, "fly")
	if err != nil || summary != "I can't do that." || len(body.said) != 1 {
		t.Fatalf("summary %q err %v said %v", summary, err, body.said)
	}
}

func TestAgentStepLimit(t *testing.T) {
	var replies []openai.ChatCompletionMessage
	for i := 0; i < 5; i++ {
		replies = append(replies, assistant(toolCall("x", "drive", `{"distance_mm": 10}`)))
	}
	body := &fakeBody{}
	_, err := RunAgent(context.Background(), &scriptedLLM{replies: replies}, body, AgentConfig{MaxSteps: 3}, "wander")
	if err == nil || len(body.calls) != 3 {
		t.Fatalf("err %v calls %v", err, body.calls)
	}
}

func TestAgentStopsWhenCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	llm := &scriptedLLM{replies: []openai.ChatCompletionMessage{assistant(toolCall("1", "drive", `{"distance_mm": 10}`))}}
	body := &fakeBody{}
	if _, err := RunAgent(ctx, llm, body, AgentConfig{}, "go"); !errors.Is(err, context.Canceled) {
		t.Fatalf("err %v", err)
	}
	if len(body.calls) != 0 {
		t.Fatalf("moved after cancel: %v", body.calls)
	}
}

func TestAgentBadArguments(t *testing.T) {
	res, _, _ := agentToolResult(context.Background(), &fakeBody{}, toolCall("1", "drive", `{nope`))
	if !strings.HasPrefix(res, "error:") {
		t.Fatal(res)
	}
	res, _, _ = agentToolResult(context.Background(), &fakeBody{}, toolCall("1", "fly", `{}`))
	if res != "error: unknown tool fly" {
		t.Fatal(res)
	}
}

func TestDropOldImagesKeepsNewest(t *testing.T) {
	body := &fakeBody{}
	msgs := []openai.ChatCompletionMessage{
		agentObservation(context.Background(), body, "a ", []byte{1}),
		agentObservation(context.Background(), body, "b ", []byte{2}),
	}
	dropOldImages(msgs)
	if msgs[0].MultiContent != nil || !strings.Contains(msgs[0].Content, "older picture removed") || len(msgs[1].MultiContent) != 2 {
		t.Fatalf("%+v", msgs)
	}
}

func TestDoTaskCommandOnlyWhenEnabled(t *testing.T) {
	old := vars.APIConfig.Knowledge
	defer func() { vars.APIConfig.Knowledge = old }()
	vars.APIConfig.Knowledge.CommandsEnable = true

	vars.APIConfig.Knowledge.AgentEnable = false
	if strings.Contains(CreatePrompt("p", "m", false), "doTask") {
		t.Fatal("doTask offered while agent mode is off")
	}
	for _, a := range GetActionsFromString("Sure. {{doTask||find the cube}}") {
		if a.Action == ActionDoTask {
			t.Fatal("doTask accepted while agent mode is off")
		}
	}

	vars.APIConfig.Knowledge.AgentEnable = true
	if !strings.Contains(CreatePrompt("p", "m", false), "doTask") {
		t.Fatal("doTask missing from the prompt")
	}
	acts := GetActionsFromString("Sure. {{doTask||find the cube}}")
	if len(acts) < 2 || acts[1].Action != ActionDoTask || acts[1].Parameter != "find the cube" {
		t.Fatalf("%+v", acts)
	}
}

// The real client against a fake OpenAI-compatible server: checks the requests serialize and tool calls round-trip.
func TestAgentWithOpenAIClient(t *testing.T) {
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		json.NewDecoder(r.Body).Decode(&req)
		bodies = append(bodies, req)
		w.Header().Set("Content-Type", "application/json")
		if len(bodies) == 1 {
			io.WriteString(w, `{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"look","arguments":"{}"}}]}}]}`)
			return
		}
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"c2","type":"function","function":{"name":"finish","arguments":"{\"summary\":\"done\"}"}}]}}]}`)
	}))
	defer srv.Close()
	conf := openai.DefaultConfig("key")
	conf.BaseURL = srv.URL + "/v1"
	summary, err := RunAgent(context.Background(), openai.NewClientWithConfig(conf), &fakeBody{}, AgentConfig{Model: "m"}, "look")
	if err != nil || summary != "done" {
		t.Fatalf("summary %q err %v", summary, err)
	}
	if len(bodies) != 2 || bodies[0]["tools"] == nil {
		t.Fatalf("requests: %v", bodies)
	}
	msgs := bodies[1]["messages"].([]any)
	if tool := msgs[len(msgs)-2].(map[string]any); tool["role"] != "tool" || tool["tool_call_id"] != "c1" {
		t.Fatalf("tool message: %v", tool)
	}
	if obs := msgs[len(msgs)-1].(map[string]any); obs["content"] == nil {
		t.Fatalf("observation: %v", obs)
	}
}
