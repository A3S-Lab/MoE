use std::pin::Pin;
use std::task::{Context, Poll};

use a3s_power::error::Result as PowerResult;
use futures::Stream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

#[derive(Debug)]
pub(crate) struct GenerationEvent {
    pub text: String,
    pub token_id: Option<u32>,
    pub done: bool,
    pub done_reason: Option<String>,
    pub prompt_tokens: Option<u32>,
    pub prompt_eval_duration_ns: Option<u64>,
}

pub(crate) struct GenerationHandle {
    pub receiver: mpsc::Receiver<PowerResult<GenerationEvent>>,
    pub cancellation: CancellationToken,
}

pub(crate) struct CancellableStream<T> {
    inner: ReceiverStream<T>,
    cancellation: CancellationToken,
}

impl<T> CancellableStream<T> {
    pub(crate) fn new(receiver: mpsc::Receiver<T>, cancellation: CancellationToken) -> Self {
        Self {
            inner: ReceiverStream::new(receiver),
            cancellation,
        }
    }
}

impl<T> Stream for CancellableStream<T> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(context)
    }
}

impl<T> Drop for CancellableStream<T> {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}
