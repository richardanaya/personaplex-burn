//! One connection: the realtime WebSocket, or the page that talks to it.
//!
//! [`serve_connection`] is the protocol. [`serve_with_page`] only decides
//! whether this TCP socket is `GET /` or an upgrade on `/v1/realtime`.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::Message;

use super::bridge::{Bridge, RealtimeEngine};
use crate::error::{Error, Result};

/// Page at `/` and `/index.html`. WebSocket upgrade at `/v1/realtime`.
///
/// The upgrade is not consumed here. [`serve_connection`] still performs the
/// handshake, so the protocol stays in one place.
pub async fn serve_with_page<E: RealtimeEngine>(
    mut stream: TcpStream,
    page: &str,
    bridge: Bridge<E>,
) -> Result<()> {
    let head = peek_headers(&stream).await?;
    let Some((method, target)) = request_target(&head) else {
        return write_bytes(
            &mut stream,
            400,
            "Bad Request",
            "text/plain; charset=utf-8",
            b"bad request\n",
        )
        .await;
    };
    let path = path_only(target);
    if is_websocket_upgrade(&head) {
        if method.eq_ignore_ascii_case("GET") && path == "/v1/realtime" {
            return serve_connection(stream, bridge).await;
        }
        return write_bytes(
            &mut stream,
            404,
            "Not Found",
            "text/plain; charset=utf-8",
            b"not found\n",
        )
        .await;
    }
    drain_headers(&mut stream).await?;
    if path == "/v1/realtime" {
        return write_bytes(
            &mut stream,
            426,
            "Upgrade Required",
            "text/plain; charset=utf-8",
            b"WebSocket endpoint. Connect a realtime client to /v1/realtime.\n",
        )
        .await;
    }
    if path != "/" && path != "/index.html" {
        return write_bytes(
            &mut stream,
            404,
            "Not Found",
            "text/plain; charset=utf-8",
            b"not found\n",
        )
        .await;
    }
    if !method.eq_ignore_ascii_case("GET") && !method.eq_ignore_ascii_case("HEAD") {
        return write_bytes(
            &mut stream,
            405,
            "Method Not Allowed",
            "text/plain; charset=utf-8",
            b"method not allowed\n",
        )
        .await;
    }
    let body = if method.eq_ignore_ascii_case("HEAD") {
        b""
    } else {
        page.as_bytes()
    };
    write_bytes(&mut stream, 200, "OK", "text/html; charset=utf-8", body).await
}

/// Wait until the HTTP head has arrived. Bytes stay in the socket.
async fn peek_headers(stream: &TcpStream) -> Result<String> {
    let mut buf = vec![0u8; 8192];
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let n = stream
            .peek(&mut buf)
            .await
            .map_err(|err| Error::session(err.to_string()))?;
        if n == 0 {
            return Err(Error::session("connection closed before a request"));
        }
        if buf[..n].windows(4).any(|w| w == b"\r\n\r\n") || n == buf.len() {
            return Ok(String::from_utf8_lossy(&buf[..n]).into_owned());
        }
        if tokio::time::Instant::now() >= deadline {
            return Ok(String::from_utf8_lossy(&buf[..n]).into_owned());
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn drain_headers(stream: &mut TcpStream) -> Result<()> {
    let mut buf = Vec::with_capacity(1024);
    let mut tmp = [0u8; 1024];
    loop {
        let n = stream
            .read(&mut tmp)
            .await
            .map_err(|err| Error::session(err.to_string()))?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.windows(4).any(|w| w == b"\r\n\r\n") || buf.len() > 65_536 {
            break;
        }
    }
    Ok(())
}

fn request_target(head: &str) -> Option<(&str, &str)> {
    let line = head.split_once('\n')?.0.trim_end_matches('\r');
    let mut parts = line.split_whitespace();
    let method = parts.next()?;
    let target = parts.next()?;
    Some((method, target))
}

fn path_only(target: &str) -> &str {
    let path = target
        .split_once('?')
        .map(|(path, _)| path)
        .unwrap_or(target);
    path.split_once('#').map(|(path, _)| path).unwrap_or(path)
}

fn is_websocket_upgrade(head: &str) -> bool {
    head.lines().any(|line| {
        let Some((name, value)) = line.split_once(':') else {
            return false;
        };
        name.trim().eq_ignore_ascii_case("upgrade")
            && value.to_ascii_lowercase().contains("websocket")
    })
}

async fn write_bytes(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    content_type: &str,
    body: &[u8],
) -> Result<()> {
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\n\r\n",
        body.len()
    );
    stream
        .write_all(head.as_bytes())
        .await
        .map_err(|err| Error::session(err.to_string()))?;
    if !body.is_empty() {
        stream
            .write_all(body)
            .await
            .map_err(|err| Error::session(err.to_string()))?;
    }
    let _ = stream.shutdown().await;
    Ok(())
}

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
