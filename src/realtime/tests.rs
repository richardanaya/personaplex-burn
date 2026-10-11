//! Scripted clients. No weights and no GPU.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde_json::json;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;

use futures_util::{SinkExt, StreamExt};

use super::bridge::{Activity, Bridge, RealtimeEngine};
use super::codec::{self, f32_to_pcm16};
use super::serve_connection;

struct Script {
    rate: u32,
    steps: Vec<Vec<Activity>>,
    pushes: usize,
    interrupts: Arc<AtomicUsize>,
    instructions: String,
    voice: String,
}

impl Script {
    fn new(steps: Vec<Vec<Activity>>) -> Self {
        Self {
            rate: 24_000,
            steps,
            pushes: 0,
            interrupts: Arc::new(AtomicUsize::new(0)),
            instructions: String::new(),
            voice: String::new(),
        }
    }
}

impl RealtimeEngine for Script {
    fn sample_rate(&self) -> u32 {
        self.rate
    }

    fn apply_session(&mut self, instructions: &str, voice: &str) -> Result<(), String> {
        self.instructions = instructions.to_string();
        self.voice = voice.to_string();
        Ok(())
    }

    fn push_audio(&mut self, _mono: &[f32], _sample_rate: u32) -> Result<Vec<Activity>, String> {
        let step = self.pushes;
        self.pushes += 1;
        Ok(self.steps.get(step).cloned().unwrap_or_default())
    }

    fn interrupt(&mut self) -> Result<(), String> {
        self.interrupts.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn types_of(events: &[serde_json::Value]) -> Vec<&str> {
    events
        .iter()
        .filter_map(|event| event.get("type").and_then(|kind| kind.as_str()))
        .collect()
}

fn assert_order(events: &[serde_json::Value], want: &[&str]) {
    let names = types_of(events);
    let mut from = 0usize;
    for kind in want {
        let Some(at) = names[from..].iter().position(|name| name == kind) else {
            panic!("missing {kind} in {names:?} starting at {from}");
        };
        from += at + 1;
    }
}

fn pcm_b64(samples: &[f32]) -> String {
    STANDARD.encode(f32_to_pcm16(samples))
}

fn append(samples: &[f32]) -> String {
    json!({
        "type": "input_audio_buffer.append",
        "audio": pcm_b64(samples),
    })
    .to_string()
}

fn tone(n: usize) -> Vec<Activity> {
    vec![
        Activity::Audio(vec![0.25; n.clamp(1, 480)]),
        Activity::Transcript(" hi".to_string()),
    ]
}

#[test]
fn scripted_client_emits_deltas_then_done() {
    let mut bridge = Bridge::new(Script::new(vec![tone(480), Vec::new()]));
    let hello = bridge.hello();
    assert_order(&hello, &["session.created", "conversation.created"]);

    let updated = bridge.handle_text(
        &json!({
            "type": "session.update",
            "session": {
                "voice": "leo",
                "instructions": "You are a wise teacher.",
                "turn_detection": { "type": "server_vad", "threshold": 0.3, "silence_duration_ms": 500 },
                "audio": {
                    "input": { "format": { "type": "audio/pcm", "rate": 24000 } },
                    "output": { "format": { "type": "audio/pcm", "rate": 24000 } }
                }
            }
        })
        .to_string(),
    );
    assert_eq!(types_of(&updated), vec!["session.updated"]);
    assert_eq!(updated[0]["session"]["voice"], "leo");
    assert_eq!(
        updated[0]["session"]["instructions"],
        "You are a wise teacher."
    );

    let loud = vec![0.8f32; 2_400];
    let started = bridge.handle_text(&append(&loud));
    assert_order(
        &started,
        &[
            "input_audio_buffer.speech_started",
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_audio.delta",
            "response.output_audio_transcript.delta",
        ],
    );
    assert_eq!(
        started
            .iter()
            .find(|e| e["type"] == "response.output_audio_transcript.delta")
            .unwrap()["delta"],
        " hi"
    );
    let delta = started
        .iter()
        .find(|event| event["type"] == "response.output_audio.delta")
        .unwrap();
    let bytes = STANDARD.decode(delta["delta"].as_str().unwrap()).unwrap();
    let decoded = codec::pcm16_to_f32(&bytes);
    assert!(codec::rms(&decoded) > 0.05);

    let silence = vec![0.0f32; 14_400];
    let done = bridge.handle_text(&append(&silence));
    assert_order(
        &done,
        &[
            "input_audio_buffer.speech_stopped",
            "input_audio_buffer.committed",
            "response.output_audio.done",
            "response.output_audio_transcript.done",
            "response.done",
        ],
    );
    assert!(types_of(&done).iter().all(|kind| *kind != "error"));
}

#[test]
fn pi_spacexai_shapes_and_barge_in() {
    let interrupts = Arc::new(AtomicUsize::new(0));
    let mut engine = Script::new(vec![tone(480), tone(480), tone(480)]);
    engine.interrupts = Arc::clone(&interrupts);
    let mut bridge = Bridge::new(engine);

    // The object pi-spacexai sends from realtime-voice.ts sessionUpdate().
    let update = json!({
        "type": "session.update",
        "session": {
            "voice": "leo",
            "instructions": "You are the voice co-pilot for a coding session.",
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.3,
                "silence_duration_ms": 500
            },
            "audio": {
                "input": {
                    "format": { "type": "audio/pcm", "rate": 24000 },
                    "transcription": { "model": "grok-transcribe" }
                },
                "output": { "format": { "type": "audio/pcm", "rate": 24000 } }
            },
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "send_task",
                        "description": "Request a job.",
                        "parameters": {
                            "type": "object",
                            "properties": { "request": { "type": "string" } },
                            "required": ["request"]
                        }
                    }
                },
                { "type": "function", "function": { "name": "search_conversations", "parameters": { "type": "object" } } },
                { "type": "function", "function": { "name": "end_the_call", "parameters": { "type": "object" } } },
                { "type": "function", "function": { "name": "read_background_updates", "parameters": { "type": "object" } } }
            ]
        }
    });
    let updated = bridge.handle_text(&update.to_string());
    assert_eq!(types_of(&updated), vec!["session.updated"]);
    assert_eq!(updated[0]["session"]["voice"], "leo");
    assert_eq!(
        updated[0]["session"]["turn_detection"]["type"],
        "server_vad"
    );
    assert_eq!(
        updated[0]["session"]["audio"]["input"]["format"]["type"],
        "audio/pcm"
    );
    assert_eq!(
        updated[0]["session"]["audio"]["input"]["format"]["rate"],
        24000
    );
    assert_eq!(
        updated[0]["session"]["audio"]["input"]["transcription"]["model"],
        "grok-transcribe"
    );
    assert_eq!(updated[0]["session"]["tools"].as_array().unwrap().len(), 4);
    assert!(types_of(&updated).iter().all(|kind| *kind != "error"));

    let loud = vec![0.8f32; 2_400];
    let started = bridge.handle_text(&append(&loud));
    assert_order(
        &started,
        &[
            "input_audio_buffer.speech_started",
            "response.created",
            "response.output_audio.delta",
            "response.output_audio_transcript.delta",
        ],
    );
    let item_id = started
        .iter()
        .find(|event| event["type"] == "response.output_audio.delta")
        .unwrap()["item_id"]
        .as_str()
        .unwrap()
        .to_string();

    // Model keeps talking through the user's pause, so the response stays open.
    let silence = vec![0.0f32; 14_400];
    let still = bridge.handle_text(&append(&silence));
    assert!(types_of(&still).contains(&"input_audio_buffer.speech_stopped"));
    assert!(types_of(&still).iter().all(|kind| *kind != "response.done"));
    assert!(types_of(&still)
        .iter()
        .all(|kind| *kind != "response.cancelled"));

    let barge = bridge.handle_text(&append(&loud));
    assert_order(
        &barge,
        &[
            "input_audio_buffer.speech_started",
            "conversation.item.truncated",
            "response.cancelled",
        ],
    );
    assert_eq!(interrupts.load(Ordering::SeqCst), 1);
    assert!(types_of(&barge)
        .iter()
        .all(|kind| *kind != "response.created"));

    // pi-spacexai sends these itself when it sees speech_started.
    let truncated = bridge.handle_text(
        &json!({
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": 0,
            "audio_end_ms": 80
        })
        .to_string(),
    );
    assert_eq!(types_of(&truncated), vec!["conversation.item.truncated"]);
    assert_eq!(truncated[0]["item_id"], item_id);
    let cancel = bridge.handle_text(r#"{"type":"response.cancel"}"#);
    assert_eq!(types_of(&cancel), vec!["error"]);
    assert!(cancel[0]["error"]["message"]
        .as_str()
        .unwrap()
        .to_ascii_lowercase()
        .contains("no active response"));
}

#[test]
fn text_only_input_and_tools_are_rejected_without_dropping_the_session() {
    let mut bridge = Bridge::new(Script::new(Vec::new()));
    bridge.handle_text(
        &json!({
            "type": "session.update",
            "session": { "instructions": "hi", "turn_detection": { "type": "server_vad" } }
        })
        .to_string(),
    );
    let text = bridge.handle_text(
        &json!({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": "Hello!" }]
            }
        })
        .to_string(),
    );
    assert!(text[0]["error"]["message"]
        .as_str()
        .unwrap()
        .contains("text-only input"));
    let tool = bridge.handle_text(
        &json!({
            "type": "conversation.item.create",
            "item": { "type": "function_call_output", "call_id": "upd_1", "output": "{}" }
        })
        .to_string(),
    );
    assert!(tool[0]["error"]["message"]
        .as_str()
        .unwrap()
        .contains("function calls are not supported"));
    let again = bridge.handle_text(
        &json!({
            "type": "session.update",
            "session": { "instructions": "still here" }
        })
        .to_string(),
    );
    assert_eq!(types_of(&again), vec!["session.updated"]);
}

#[test]
fn manual_turns_wait_for_response_create_and_g711_is_loud_enough() {
    let mut bridge = Bridge::new(Script::new(vec![tone(480), Vec::new()]));
    bridge.handle_text(
        &json!({
            "type": "session.update",
            "session": {
                "turn_detection": null,
                "audio": {
                    "input": { "format": { "type": "audio/pcmu" } },
                    "output": { "format": { "type": "audio/pcm", "rate": 24000 } }
                }
            }
        })
        .to_string(),
    );
    let loud: Vec<f32> = (0..160)
        .map(|i| if i % 2 == 0 { 0.8 } else { -0.8 })
        .collect();
    let bytes = codec::f32_to_mulaw(&loud);
    let hidden = bridge.handle_text(
        &json!({
            "type": "input_audio_buffer.append",
            "audio": STANDARD.encode(bytes),
        })
        .to_string(),
    );
    assert!(types_of(&hidden)
        .iter()
        .all(|kind| *kind != "response.created"));
    assert!(types_of(&hidden)
        .iter()
        .all(|kind| *kind != "input_audio_buffer.speech_started"));
    let created = bridge.handle_text(r#"{"type":"response.create"}"#);
    assert_order(
        &created,
        &[
            "response.created",
            "response.output_audio.delta",
            "response.output_audio_transcript.delta",
        ],
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn websocket_replay_of_pi_append_receives_ga_deltas() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        serve_connection(
            stream,
            Bridge::new(Script::new(vec![tone(480), Vec::new()])),
        )
        .await
        .unwrap();
    });
    let (mut socket, _) = tokio_tungstenite::connect_async(format!(
        "ws://{addr}/v1/realtime?model=grok-voice-latest"
    ))
    .await
    .unwrap();
    let created = next_json(&mut socket).await;
    let conversation = next_json(&mut socket).await;
    assert_eq!(created["type"], "session.created");
    assert_eq!(created["session"]["model"], "grok-voice-latest");
    assert_eq!(conversation["type"], "conversation.created");
    socket
        .send(Message::Text(
            json!({
                "type": "session.update",
                "session": {
                    "voice": "eve",
                    "instructions": "Be brief.",
                    "turn_detection": { "type": "server_vad", "threshold": 0.3, "silence_duration_ms": 200 },
                    "audio": {
                        "input": { "format": { "type": "audio/pcm", "rate": 24000 } },
                        "output": { "format": { "type": "audio/pcm", "rate": 24000 } }
                    }
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
    let updated = next_json(&mut socket).await;
    assert_eq!(updated["type"], "session.updated");
    assert_eq!(updated["session"]["voice"], "eve");
    socket
        .send(Message::Text(append(&vec![0.8f32; 2_400]).into()))
        .await
        .unwrap();
    let mut saw_audio = false;
    let mut saw_transcript = false;
    for _ in 0..12 {
        let event = next_json(&mut socket).await;
        match event["type"].as_str() {
            Some("response.output_audio.delta") => saw_audio = true,
            Some("response.output_audio_transcript.delta") => {
                assert_eq!(event["delta"], " hi");
                saw_transcript = true;
            }
            _ => {}
        }
        if saw_audio && saw_transcript {
            break;
        }
    }
    assert!(saw_audio && saw_transcript);
}

async fn next_json(
    socket: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) -> serde_json::Value {
    let message = socket.next().await.unwrap().unwrap();
    let Message::Text(text) = message else {
        panic!("expected text");
    };
    serde_json::from_str(text.as_str()).unwrap()
}
