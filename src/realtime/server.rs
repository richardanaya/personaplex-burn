//! One WebSocket connection speaking the realtime protocol.

use std::sync::{Arc, Mutex};

use futures_util::{SinkExt, StreamExt};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::Message;

use super::bridge::{Bridge, RealtimeEngine};
use crate::error::{Error, Result};

/// Accept one upgraded stream and run a session until the socket closes.
///
/// The caller supplies the engine. Inference runs inside `block_in_place`, so
/// this must be polled on a multi-thread Tokio runtime.
#[allow(clippy::result_large_err)]
pub async fn serve_connection<S, E>(stream: S, bridge: Bridge<E>) -> Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
    E: RealtimeEngine,
{
    let model = Arc::new(Mutex::new(String::new()));
    let model_for_handshake = Arc::clone(&model);
    let mut socket =
        tokio_tungstenite::accept_hdr_async(stream, move |request: &Request, response| {
            capture_model(&model_for_handshake, request);
            accept_client_secret(request, response)
        })
        .await
        .map_err(|err| Error::session(err.to_string()))?;

    let mut bridge = bridge;
    if let Ok(guard) = model.lock() {
        bridge.set_advertised_model(guard.clone());
    }
    for event in bridge.hello() {
        send_json(&mut socket, &event).await?;
    }
    while let Some(message) = socket.next().await {
        let message = message.map_err(|err| Error::session(err.to_string()))?;
        match message {
            Message::Text(text) => {
                let (events, binary) = tokio::task::block_in_place(|| {
                    let events = bridge.handle_text(text.as_str());
                    let binary = bridge.take_binary();
                    (events, binary)
                });
                for event in events {
                    send_json(&mut socket, &event).await?;
                }
                for frame in binary {
                    send_binary(&mut socket, frame).await?;
                }
            }
            Message::Binary(bytes) => {
                let (events, binary) = tokio::task::block_in_place(|| {
                    let events = bridge.handle_binary(&bytes);
                    let binary = bridge.take_binary();
                    (events, binary)
                });
                for event in events {
                    send_json(&mut socket, &event).await?;
                }
                for frame in binary {
                    send_binary(&mut socket, frame).await?;
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
    Ok(())
}

fn capture_model(slot: &Mutex<String>, request: &Request) {
    let Some(query) = request.uri().query() else {
        return;
    };
    for pair in query.split('&') {
        let mut parts = pair.splitn(2, '=');
        if parts.next() == Some("model") {
            if let Some(value) = parts.next() {
                if let Ok(mut guard) = slot.lock() {
                    *guard = value.to_string();
                }
            }
        }
    }
}

#[allow(clippy::result_large_err)]
fn accept_client_secret(
    request: &Request,
    mut response: Response,
) -> std::result::Result<Response, tokio_tungstenite::tungstenite::handshake::server::ErrorResponse>
{
    let Some(header) = request.headers().get("Sec-WebSocket-Protocol") else {
        return Ok(response);
    };
    let Ok(text) = header.to_str() else {
        return Ok(response);
    };
    if let Some(protocol) = text
        .split(',')
        .map(str::trim)
        .find(|protocol| protocol.starts_with("xai-client-secret."))
    {
        if let Ok(value) = HeaderValue::from_str(protocol) {
            response
                .headers_mut()
                .insert("Sec-WebSocket-Protocol", value);
        }
    }
    Ok(response)
}

async fn send_json<S>(
    socket: &mut tokio_tungstenite::WebSocketStream<S>,
    event: &serde_json::Value,
) -> Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    socket
        .send(Message::Text(event.to_string().into()))
        .await
        .map_err(|err| Error::session(err.to_string()))
}

async fn send_binary<S>(
    socket: &mut tokio_tungstenite::WebSocketStream<S>,
    bytes: Vec<u8>,
) -> Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    socket
        .send(Message::Binary(bytes.into()))
        .await
        .map_err(|err| Error::session(err.to_string()))
}
