mod postgres;
mod registry;
use futures::{FutureExt, StreamExt, future::BoxFuture, stream::BoxStream, Stream};

use crate::{LanguageModelCompletionError, LanguageModelCompletionEvent, LanguageModelRequest, LanguageModelRequestMessage, LanguageModelToolUse, Role, TokenUsage};
use gpui::Global;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

pub use postgres::PostgresDatabaseClient;
// pub use example::run_message_handler_example;
pub use registry::{
    MessageHandlerConfig, MessageHandlerRegistry, create_conversation_id, get_message_handler,
    get_message_handler_async, init_message_handler,
};

/// Message types compatible with LangGraph's data model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    HumanMessage,
    AiMessage,
    SystemMessage,
    ToolMessage,
}

/// Content types for messages
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum MessageContent {
    Text(String),
    Thinking {
        text: String,
        signature: Option<String>,
    },
    ToolUse {
        name: String,
        input: String,
    },
    ToolCall {
        tool_calls: Vec<ToolCallContent>,
    },
    ToolResult {
        tool_name: String,
        result: serde_json::Value,
    },
    /// Multiple tool calls in a single message (used by some providers)
    ToolCalls {
        tool_calls: Vec<ToolCallContent>,
    },
}

/// Tool call content structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallContent {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Base message structure compatible with LangGraph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub message_type: MessageType,
    pub content: Vec<MessageContent>,
    pub timestamp: u64,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Interface for database operations
pub trait DatabaseClient: Send + Sync {
    async fn save_append_messages(&self, message: Vec<Message>, thread_id: &str);
}

/// Message handler for interfacing with LangGraph and database storage
pub struct AiMessageHandler {
    database_client: Option<Arc<PostgresDatabaseClient>>,
}

pub trait MessageHandlerTrait: Send + Sync {}

impl MessageHandlerTrait for AiMessageHandler {}

impl Global for AiMessageHandler {}

pub fn peek_db<T>(stream: T,
                  message_handler: Option<Arc<AiMessageHandler>>,
                  thread_id: String) -> T
where T: Stream<Item=Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>
{
    if let Some(handler) = message_handler {
        let stream = AiMessageHandler::inspect_stream(stream, handler.clone(), thread_id.clone());
        stream
    } else {
        stream
    }
}

impl AiMessageHandler {
    pub fn new(database_client: Option<Arc<PostgresDatabaseClient>>) -> Self {
        Self { database_client }
    }

    pub async fn save_completion_req(
        &self,
        request_message: &LanguageModelRequest,
        thread_id: &str,
    ) {
        let collected = request_message.messages.iter()
            .flat_map(|r| Self::map_from_completion_request(r).into_iter())
            .collect::<Vec<Message>>();
        self.save_append_messages(collected, thread_id).await;
    }

    pub async fn save_completion_request(
        &self,
        request_message: &LanguageModelRequestMessage,
        thread_id: &str,
    ) {
        if let Some(msg) = Self::map_from_completion_request(request_message) {
            self.save_append_messages(vec![msg], thread_id).await;
        }
    }


    pub async fn save_completion_event(
        &self,
        request_message: &LanguageModelCompletionEvent,
        thread_id: &str,
    ) {
        if let Some(msg) = Self::map_from_completion_event(request_message, thread_id) {
            self.save_append_messages(vec![msg], thread_id).await;
        }
    }

    pub fn map_from_completion_request(
        request_message: &LanguageModelRequestMessage,
    ) -> Option<Message> {
        let content = serde_json::to_string(&request_message.content).unwrap();
        match &request_message.role {
            Role::User => Some(Message {
                id: uuid::Uuid::new_v4().to_string(),
                message_type: MessageType::HumanMessage,
                content: vec![MessageContent::Text(content.clone())],
                timestamp: Self::now_ts(),
                metadata: HashMap::new(),
            }),
            Role::System => Some(Message {
                id: uuid::Uuid::new_v4().to_string(),
                message_type: MessageType::SystemMessage,
                content: vec![MessageContent::Text(content)],
                timestamp: Self::now_ts(),
                metadata: HashMap::new(),
            }),
            Role::Assistant => Some(Message {
                id: uuid::Uuid::new_v4().to_string(),
                message_type: MessageType::AiMessage,
                content: vec![MessageContent::Text(content)],
                timestamp: Self::now_ts(),
                metadata: HashMap::new(),
            })
        }
    }

    pub fn map_from_completion_event(
        request_message: &LanguageModelCompletionEvent,
        thread_id: &str,
    ) -> Option<Message> {
        match request_message {
            LanguageModelCompletionEvent::StatusUpdate { .. } => None,
            LanguageModelCompletionEvent::StartMessage { .. } => None,
            LanguageModelCompletionEvent::Text(text) => {
                Some(Message {
                    id: thread_id.to_string(),
                    message_type: MessageType::AiMessage,
                    content: vec![MessageContent::Text(text.clone())],
                    timestamp: Self::now_ts(),
                    metadata: HashMap::new(),
                })
            }
            LanguageModelCompletionEvent::Thinking { text, signature } => {
                // Save thinking message
                Some(Message {
                    id: thread_id.to_string(),
                    message_type: MessageType::AiMessage,
                    content: vec![MessageContent::Thinking {
                        text: text.clone(),
                        signature: signature.clone(),
                    }],
                    timestamp: Self::now_ts(),
                    metadata: HashMap::new(),
                })
            }
            LanguageModelCompletionEvent::Stop(_) => Some(Message {
                id: thread_id.to_string(),
                message_type: MessageType::AiMessage,
                content: vec![MessageContent::Text("STOP".to_string())],
                timestamp: Self::now_ts(),
                metadata: HashMap::new(),
            }),
            LanguageModelCompletionEvent::ToolUse(tool_use) => {
                let name = tool_use.name.as_ref().to_string();
                let input = serde_json::to_string(&tool_use.input).unwrap_or_default();

                // Create a tool message with the appropriate format
                let mut metadata = HashMap::new();
                metadata.insert(
                    "raw_input".to_string(),
                    serde_json::Value::String(tool_use.raw_input.clone()),
                );
                
                // Add is_input_complete flag to metadata
                metadata.insert(
                    "is_input_complete".to_string(),
                    serde_json::Value::Bool(tool_use.is_input_complete),
                );

                Some(Message {
                    id: tool_use.id.to_string(),
                    message_type: MessageType::ToolMessage,
                    content: vec![MessageContent::ToolUse {
                        name,
                        input,
                    }],
                    timestamp: Self::now_ts(),
                    metadata,
                })
            }
            LanguageModelCompletionEvent::UsageUpdate(token_usage) => None,
        }
    }

    fn now_ts() -> u64 {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        timestamp
    }

    /// Save a message to the database
    pub async fn save_append_messages(&self, message: Vec<Message>, thread_id: &str) {
        if let Some(db_client) = &self.database_client {
            println!("Saving appending..");
            db_client.save_append_messages(message, thread_id).await;
        }
    }

    pub fn inspect_stream<T>(s: T,
                          handler: Arc<AiMessageHandler>,
                         thread_id_clone: String) -> T
    where T: Stream<Item=Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>
    {
        s.inspect(move |result_ref| {
                let result = result_ref.clone();
                let arc = handler.clone();
                let thread_id = thread_id_clone.clone();

                if let Ok(res) = result {
                    let res = res.clone();
                    smol::spawn(async move {
                            arc.save_completion_event(&res, &thread_id).await;
                        })
                        .detach();
                }
            })
            .into_inner()
    }

}
