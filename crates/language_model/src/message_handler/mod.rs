mod postgres;
mod registry;

use crate::{LanguageModelId, RequestIds};
use futures::{Stream, StreamExt};

use crate::{
    LanguageModelCompletionError, LanguageModelCompletionEvent, LanguageModelRequest,
    LanguageModelRequestMessage, Role,
};
use enum_fields::EnumFields;
use gpui::Global;
pub use postgres::PostgresDatabaseClient;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
// pub use example::run_message_handler_example;
pub use registry::{
    MessageHandlerConfig, MessageHandlerRegistry, create_conversation_id, get_message_handler,
    get_message_handler_async, init_message_handler,
};

/// Message types compatible with LangGraph's data model
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageType {
    #[serde(rename = "human")]
    Human,
    #[serde(rename = "ai")]
    Ai,
    #[serde(rename = "system")]
    System,
    #[serde(rename = "tool")]
    Tool,
    #[serde(rename = "function")]
    Function,
}

/// Content value that can be either a single string or array of strings
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ContentValue {
    Single(String),
    Multiple(Vec<String>),
}

impl ContentValue {
    pub fn new(content: String) -> Self {
        ContentValue::Single(content)
    }

    pub fn from_vec(content: Vec<String>) -> Self {
        ContentValue::Multiple(content)
    }
}

/// Tool call content structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallContent {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Base message structure compatible with LangGraph and Java schema
#[derive(Debug, Clone, Serialize, Deserialize, EnumFields)]
#[serde(tag = "type")]
pub enum Message {
    #[serde(rename = "human")]
    Human {
        content: ContentValue,
        id: String,
        name: Option<String>,
        #[serde(default)]
        example: bool,
        #[serde(rename = "additional_kwargs", default)]
        additional_kwargs: HashMap<String, serde_json::Value>,
        #[serde(rename = "response_metadata", default)]
        response_metadata: HashMap<String, serde_json::Value>,
    },
    #[serde(rename = "ai")]
    Ai {
        content: ContentValue,
        id: String,
        name: Option<String>,
        #[serde(default)]
        example: bool,
        #[serde(rename = "invalid_tool_calls")]
        invalid_tool_calls: Option<HashMap<String, serde_json::Value>>,
        #[serde(rename = "tool_calls")]
        tool_calls: Option<HashMap<String, serde_json::Value>>,
        #[serde(rename = "additional_kwargs", default)]
        additional_kwargs: HashMap<String, serde_json::Value>,
        #[serde(rename = "response_metadata", default)]
        response_metadata: HashMap<String, serde_json::Value>,
    },
    #[serde(rename = "system")]
    System {
        content: ContentValue,
        id: String,
        name: Option<String>,
        #[serde(default)]
        example: bool,
        #[serde(rename = "additional_kwargs", default)]
        additional_kwargs: HashMap<String, serde_json::Value>,
        #[serde(rename = "response_metadata", default)]
        response_metadata: HashMap<String, serde_json::Value>,
    },
    #[serde(rename = "tool")]
    Tool {
        content: ContentValue,
        id: String,
        name: Option<String>,
        #[serde(default)]
        example: bool,
        #[serde(rename = "tool_call_id")]
        tool_call_id: Option<String>,
        #[serde(rename = "tool_name")]
        tool_name: Option<String>,
        #[serde(rename = "additional_kwargs", default)]
        additional_kwargs: HashMap<String, serde_json::Value>,
        #[serde(rename = "response_metadata", default)]
        response_metadata: HashMap<String, serde_json::Value>,
    },
    #[serde(rename = "function")]
    Function {
        content: ContentValue,
        id: String,
        name: Option<String>,
        #[serde(default)]
        example: bool,
        #[serde(rename = "function_call")]
        function_call: Option<HashMap<String, serde_json::Value>>,
        #[serde(rename = "additional_kwargs", default)]
        additional_kwargs: HashMap<String, serde_json::Value>,
        #[serde(rename = "response_metadata", default)]
        response_metadata: HashMap<String, serde_json::Value>,
    },
}

/// Interface for database operations
pub trait DatabaseClient: Send + Sync {
    async fn save_append_messages(&self, message: Vec<Message>, ids: &RequestIds);
}

/// Message handler for interfacing with LangGraph and database storage
pub struct AiMessageHandler {
    database_client: Option<Arc<PostgresDatabaseClient>>,
}

pub trait MessageHandlerTrait: Send + Sync {}

impl MessageHandlerTrait for AiMessageHandler {}

impl Global for AiMessageHandler {}

#[derive(Clone)]
pub struct LanguageModelArgs(pub LanguageModelId);

pub fn peek_db<T>(stream: T, message_handler: Option<Arc<AiMessageHandler>>, ids: RequestIds,
                  language_model_request: &LanguageModelRequest, language_id: LanguageModelArgs) -> T
where
    T: Stream<Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>,
{
    if let Some(handler) = message_handler {
        let stream = AiMessageHandler::inspect_stream(stream, handler.clone(), ids, language_model_request, language_id);
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
        ids: &RequestIds,
        language_model_args: LanguageModelArgs
    ) {
        let collected = request_message
            .messages
            .iter()
            .flat_map(|r| {
                Self::map_from_completion_request(r, ids, Some(request_message), &language_model_args).into_iter()
            })
            .collect::<Vec<Message>>();
        let _ = self.save_append_messages(collected, ids).await;
    }

    pub async fn save_completion_event(
        &self,
        request_message: &LanguageModelCompletionEvent,
        ids: &RequestIds,
        language_model_request: &LanguageModelRequest,
        language_model_args: &LanguageModelArgs
    ) {
        if let Some(msg) =
            Self::map_from_completion_event(request_message, &ids.checkpoint_id, Some(language_model_request), language_model_args)
        {
            let _ = self.save_append_messages(vec![msg], ids).await;
        }
    }

    fn build_response_metadata(
        metadata: Option<&LanguageModelRequest>,
        language_model_args: &LanguageModelArgs
    ) -> HashMap<String, serde_json::Value> {
        let mut response_metadata = HashMap::new();

        response_metadata.insert(
            "model_id".to_string(),
            serde_json::Value::from(format!("{:?}", language_model_args.0.0.to_string())));

        if let Some(meta) = metadata {
            if let Some(temperature) = meta.temperature {
                response_metadata.insert(
                    "temperature".to_string(),
                    serde_json::Value::from(temperature),
                );
            }
            if let Some(intent) = &meta.intent {
                response_metadata.insert(
                    "intent".to_string(),
                    serde_json::Value::from(format!("{:?}", intent)),
                );
            }
            if let Some(mode) = &meta.mode {
                response_metadata.insert(
                    "mode".to_string(),
                    serde_json::Value::from(format!("{:?}", mode)),
                );
            }
            if let Some(prompt_id) = &meta.prompt_id {
                response_metadata.insert(
                    "prompt_id".to_string(),
                    serde_json::Value::from(prompt_id.clone()),
                );
            }
        }
        response_metadata
    }

    pub fn map_from_completion_request(
        request_message: &LanguageModelRequestMessage,
        id: &RequestIds,
        metadata: Option<&LanguageModelRequest>,
        language_model_args: &LanguageModelArgs
    ) -> Option<Message> {
        let content = match serde_json::to_string(&request_message.content) {
            Ok(content) => content,
            Err(e) => {
                log::error!("Failed to serialize request message content: {}", e);
                String::default()
            }
        };
        let content_value = ContentValue::new(content);
        let id = id.thread_id.to_string();

        let response_metadata = Self::build_response_metadata(metadata, language_model_args);

        match &request_message.role {
            Role::User => Some(Message::Human {
                content: content_value,
                id,
                name: Some("ZedIdeAgent".to_string()),
                example: false,
                additional_kwargs: HashMap::new(),
                response_metadata,
            }),
            Role::System => Some(Message::System {
                content: content_value,
                id,
                name: Some("ZedIdeAgent".to_string()),
                example: false,
                additional_kwargs: HashMap::new(),
                response_metadata,
            }),
            Role::Assistant => Some(Message::Ai {
                content: content_value,
                id,
                name: Some("ZedIdeAgent".to_string()),
                example: false,
                invalid_tool_calls: None,
                tool_calls: None,
                additional_kwargs: HashMap::new(),
                response_metadata,
            }),
        }
    }

    pub fn map_from_completion_event(
        request_message: &LanguageModelCompletionEvent,
        thread_id: &str,
        metadata: Option<&LanguageModelRequest>,
        language_model_args: &LanguageModelArgs,
    ) -> Option<Message> {

        let response_metadata = Self::build_response_metadata(metadata, &language_model_args);
        match request_message {
            LanguageModelCompletionEvent::StatusUpdate { .. } => None,
            LanguageModelCompletionEvent::StartMessage { .. } => None,
            LanguageModelCompletionEvent::Text(text) => {
                let id = thread_id.to_string();
                Some(Message::Ai {
                    content: ContentValue::new(text.clone()),
                    id,
                    name: Some("ZedIdeAgent".to_string()),
                    example: false,
                    invalid_tool_calls: None,
                    tool_calls: None,
                    additional_kwargs: HashMap::new(),
                    response_metadata,
                })
            }
            LanguageModelCompletionEvent::Thinking { text, signature } => {
                let id = thread_id.to_string();
                let mut additional_kwargs = HashMap::new();
                additional_kwargs.insert(
                    "thinking".to_string(),
                    serde_json::Value::String(text.clone()),
                );
                if let Some(sig) = signature {
                    additional_kwargs.insert(
                        "signature".to_string(),
                        serde_json::Value::String(sig.clone()),
                    );
                }


                Some(Message::Ai {
                    content: ContentValue::new(text.clone()),
                    id,
                    name: Some("ZedIdeAgent".to_string()),
                    example: false,
                    invalid_tool_calls: None,
                    tool_calls: None,
                    additional_kwargs,
                    response_metadata,
                })
            }
            LanguageModelCompletionEvent::Stop(_) => {
                let id = thread_id.to_string();
                Some(Message::Ai {
                    content: ContentValue::new("STOP".to_string()),
                    id,
                    name: Some("ZedIdeAgent".to_string()),
                    example: false,
                    invalid_tool_calls: None,
                    tool_calls: None,
                    additional_kwargs: HashMap::new(),
                    response_metadata,
                })
            }
            LanguageModelCompletionEvent::ToolUse(tool_use) => {
                let content = match serde_json::to_string(&tool_use.input) {
                    Ok(content) => content,
                    Err(e) => {
                        log::error!("Failed to serialize tool use input: {}", e);
                        String::default()
                    }
                };
                let mut additional_kwargs = HashMap::new();
                additional_kwargs.insert(
                    "raw_input".to_string(),
                    serde_json::Value::String(tool_use.raw_input.clone()),
                );
                additional_kwargs.insert(
                    "is_input_complete".to_string(),
                    serde_json::Value::Bool(tool_use.is_input_complete),
                );

                Some(Message::Tool {
                    content: ContentValue::new(content),
                    id: tool_use.id.to_string(),
                    name: Some("ZedIdeAgent".to_string()),
                    example: false,
                    tool_call_id: Some(tool_use.id.to_string()),
                    tool_name: Some(tool_use.name.as_ref().to_string()),
                    additional_kwargs,
                    response_metadata,
                })
            }
            LanguageModelCompletionEvent::UsageUpdate(_token_usage) => None,
        }
    }

    /// Save a message to the database
    pub async fn save_append_messages(
        &self,
        messages: Vec<Message>,
        ids: &RequestIds,
    ) -> anyhow::Result<()> {
        if let Some(ref db_client) = self.database_client {
            db_client.save_append_messages(messages, ids).await;
        }
        Ok(())
    }

    pub fn inspect_stream<T>(s: T, handler: Arc<AiMessageHandler>, ids: RequestIds,
                            language_model_request: &LanguageModelRequest, language_id: LanguageModelArgs) -> T
    where
        T: Stream<Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>,
    {
        s.inspect(move |result_ref| {
            let result = result_ref;
            let arc = handler.clone();
            let ids = ids.clone();
            let language_id = language_id.clone();
            let language_model_request = language_model_request.clone();

            if let Ok(res) = result {
                let res = res.clone();
                smol::spawn(async move {
                    arc.save_completion_event(&res, &ids, &language_model_request, &language_id).await;
                })
                .detach();
            }
        })
        .into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_message_serialization_java_compatibility() {
        // Test Human message
        let human_msg = Message::Human {
            content: ContentValue::new("Hello world".to_string()),
            id: "test-id".to_string(),
            name: Some("user".to_string()),
            example: false,
            additional_kwargs: HashMap::new(),
            response_metadata: HashMap::new(),
        };

        let serialized = serde_json::to_string(&human_msg).unwrap();
        let expected = json!({
            "type": "human",
            "content": "Hello world",
            "id": "test-id",
            "name": "user",
            "example": false,
            "additional_kwargs": {},
            "response_metadata": {}
        });

        let actual: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        assert_eq!(actual, expected);

        // Test AI message with tool calls
        let mut tool_calls = HashMap::new();
        tool_calls.insert(
            "function".to_string(),
            json!({"name": "search", "args": {}}),
        );

        let ai_msg = Message::Ai {
            content: ContentValue::new("I'll search for that".to_string()),
            id: "ai-test-id".to_string(),
            name: None,
            example: false,
            invalid_tool_calls: None,
            tool_calls: Some(tool_calls),
            additional_kwargs: HashMap::new(),
            response_metadata: HashMap::new(),
        };

        let serialized = serde_json::to_string(&ai_msg).unwrap();
        let expected = json!({
            "type": "ai",
            "content": "I'll search for that",
            "id": "ai-test-id",
            "name": null,
            "example": false,
            "invalid_tool_calls": null,
            "tool_calls": {
                "function": {"name": "search", "args": {}}
            },
            "additional_kwargs": {},
            "response_metadata": {}
        });

        let actual: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        assert_eq!(actual, expected);

        // Test Tool message
        let tool_msg = Message::Tool {
            content: ContentValue::new("Search results: ...".to_string()),
            id: "tool-test-id".to_string(),
            name: None,
            example: false,
            tool_call_id: Some("call-123".to_string()),
            tool_name: Some("search".to_string()),
            additional_kwargs: HashMap::new(),
            response_metadata: HashMap::new(),
        };

        let serialized = serde_json::to_string(&tool_msg).unwrap();
        let expected = json!({
            "type": "tool",
            "content": "Search results: ...",
            "id": "tool-test-id",
            "name": null,
            "example": false,
            "tool_call_id": "call-123",
            "tool_name": "search",
            "additional_kwargs": {},
            "response_metadata": {}
        });

        let actual: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_content_value_serialization() {
        // Test single string content
        let single_content = ContentValue::Single("Hello".to_string());
        let serialized = serde_json::to_string(&single_content).unwrap();
        assert_eq!(serialized, "\"Hello\"");

        let s = serde_json::from_str::<ContentValue>(&serialized);
        assert!(s.is_ok());
        if let (ContentValue::Single(s)) = &s.as_ref().unwrap() {
            assert_eq!(s, &"Hello".to_string())
        }

        // Test multiple string content
        let multi_content = ContentValue::Multiple(vec!["Hello".to_string(), "World".to_string()]);
        let serialized = serde_json::to_string(&multi_content).unwrap();
        assert_eq!(serialized, "[\"Hello\",\"World\"]");
        let s = serde_json::from_str::<ContentValue>(&serialized);
        assert!(s.is_ok());
        if let (ContentValue::Multiple(s)) = &s.as_ref().unwrap() {
            assert_eq!(s, &vec!["Hello".to_string(), "World".to_string()]);
        }
    }
}
