mod postgres;
mod registry;

use crate::{LanguageModelId, RequestIds};
use agent_client_protocol::{self as acp, Annotations, ContentBlock, Plan, SessionUpdate};
use futures::{Stream, StreamExt, TryFutureExt};

use crate::{
    LanguageModelCompletionError, LanguageModelCompletionEvent, LanguageModelRequest,
    LanguageModelRequestMessage, Role,
};
use enum_fields::EnumFields;
use gpui::Global;
pub use postgres::PostgresDatabaseClient;
use ratelimit::Ratelimiter;
use schemars::_private::NoSerialize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::thread::sleep;
use std::time::Duration;
use serde_json::Value;
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
    fn save_append_messages(&self, message: Vec<Message>, ids: &RequestIds);
    async fn save_append_messages_async(&self, message: Vec<Message>, ids: &RequestIds);
}

/// Message handler for interfacing with LangGraph and database storage
pub struct AiMessageHandler {
    database_client: Option<Arc<PostgresDatabaseClient>>,
}

pub trait MessageHandlerTrait: Send + Sync {}

impl MessageHandlerTrait for AiMessageHandler {}

impl Global for AiMessageHandler {}

#[derive(Clone)]
pub struct LanguageModelArgs {
    pub model_id: LanguageModelId,
    pub temperature: Option<f32>,
    pub intent: Option<String>,
    pub mode: Option<String>,
    pub prompt_id: Option<String>,
}

impl LanguageModelArgs {
    pub fn new(model_id: LanguageModelId) -> Self {
        Self {
            model_id,
            temperature: None,
            intent: None,
            mode: None,
            prompt_id: None,
        }
    }

    pub fn from_request(model_id: LanguageModelId, request: &LanguageModelRequest) -> Self {
        Self {
            model_id,
            temperature: request.temperature,
            intent: request.intent.as_ref().map(|i| format!("{:?}", i)),
            mode: request.mode.as_ref().map(|m| format!("{:?}", m)),
            prompt_id: request.prompt_id.clone(),
        }
    }
}

pub fn peek_db<T>(
    stream: T,
    message_handler: Option<Arc<AiMessageHandler>>,
    ids: RequestIds,
    language_model_args: LanguageModelArgs,
) -> T
where
    T: Stream<Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>,
{
    if let Some(handler) = message_handler {
        let stream =
            AiMessageHandler::inspect_stream(stream, handler.clone(), ids, language_model_args);
        stream
    } else {
        stream
    }
}

pub struct TokenRateLimiter {
    rate_limiter: Ratelimiter,
    response_tokens_hint: u64,
}

impl TokenRateLimiter {
    pub fn new(duration: Duration, max_tokens: u64) -> Self {
        Self {
            rate_limiter: Ratelimiter::builder(max_tokens, duration)
                .max_tokens(max_tokens)
                .build()
                .unwrap(),
            response_tokens_hint: max_tokens / 16,
        }
    }

    pub fn limit(&self, request: &LanguageModelRequest) {
        if let Some(r) = request.maybe_to_value()
            && let Some(s) = r.as_str()
        {
            self.rate_limit_ser(s);

            for _ in 0..self.response_tokens_hint {
                if let Err(e) = self.rate_limiter.try_wait() {
                    sleep(e);
                }
            }
        }
    }

    fn rate_limit_ser(&self, s: &str) {
        s.split(" ").for_each(|_| {
            if let Err(e) = self.rate_limiter.try_wait() {
                log::info!("Sleeping for {}.", &e.as_secs());
                sleep(e);
            }
        });
    }

    pub fn register_response(&self, request_message: &LanguageModelCompletionEvent) {
        if let Some(r) = request_message.maybe_to_value()
            && let Some(s) = r.as_str()
        {
            self.rate_limit_ser(s);
        }
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
        language_model_args: LanguageModelArgs,
    ) {
        let collected = request_message
            .messages
            .iter()
            .flat_map(|r| {
                Self::map_from_completion_request(r, ids, &language_model_args).into_iter()
            })
            .collect::<Vec<Message>>();
        let _ = self.save_append_messages_async(collected, ids).await;
    }

    pub fn save_acp(&self, update: &acp::SessionUpdate, ids: &RequestIds) {
        if let Some(msg) = Self::map_from_acp(update, ids) {
            let _ = self.save_append_messages(vec![msg], ids);
        }
    }

    pub async fn save_completion_event(
        &self,
        request_message: &LanguageModelCompletionEvent,
        ids: &RequestIds,
        language_model_args: &LanguageModelArgs,
    ) {
        if let Some(msg) = Self::map_from_completion_event(
            request_message,
            &ids.checkpoint_id,
            language_model_args,
        ) {
            let _ = self.save_append_messages_async(vec![msg], ids).await;
        }
    }

    fn build_response_metadata(
        language_model_args: &LanguageModelArgs,
    ) -> HashMap<String, serde_json::Value> {
        let mut response_metadata = HashMap::new();

        response_metadata.insert(
            "model_id".to_string(),
            serde_json::Value::from(language_model_args.model_id.0.to_string()),
        );

        if let Some(temperature) = language_model_args.temperature {
            response_metadata.insert(
                "temperature".to_string(),
                serde_json::Value::from(temperature),
            );
        }
        if let Some(intent) = &language_model_args.intent {
            response_metadata.insert(
                "intent".to_string(),
                serde_json::Value::from(intent.clone()),
            );
        }
        if let Some(mode) = &language_model_args.mode {
            response_metadata.insert("mode".to_string(), serde_json::Value::from(mode.clone()));
        }
        if let Some(prompt_id) = &language_model_args.prompt_id {
            response_metadata.insert(
                "prompt_id".to_string(),
                serde_json::Value::from(prompt_id.clone()),
            );
        }
        response_metadata
    }

    pub fn map_from_completion_request(
        request_message: &LanguageModelRequestMessage,
        id: &RequestIds,
        language_model_args: &LanguageModelArgs,
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

        let response_metadata = Self::build_response_metadata(language_model_args);

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

    pub fn map_from_acp(update: &acp::SessionUpdate, id: &RequestIds) -> Option<Message> {
        match update {
            SessionUpdate::UserMessageChunk { content } => match content {
                ContentBlock::Text(t) => {
                    Some(Message::Human {
                        content: ContentValue::new(t.text.to_string()),
                        id: id.thread_id.to_string(),
                        name: Some("ZedIdeAgent".to_string()),
                        example: false,
                        additional_kwargs: Default::default(),
                        response_metadata: Self::_create_acp_response_metadata(t.meta.clone()),
                    })
                },
                _ => None,
            },
            SessionUpdate::AgentMessageChunk { content } => match content {
                ContentBlock::Text(t) => {
                    Some(Message::Ai {
                        content: ContentValue::new(t.text.to_string()),
                        id: id.thread_id.to_string(),
                        name: Some("ZedIdeAgent".to_string()),
                        example: false,
                        invalid_tool_calls: None,
                        tool_calls: None,
                        additional_kwargs: Default::default(),
                        response_metadata: Self::_create_acp_response_metadata(t.meta.clone()),
                    })
                },
                _ => None,
            },
            SessionUpdate::AgentThoughtChunk { content } => match content {
                ContentBlock::Text(t) => {
                    let mut additional_kwargs = HashMap::new();
                    additional_kwargs.insert(
                        "thinking".to_string(),
                        serde_json::Value::String(t.text.to_string()),
                    );
                    Some(Message::Ai {
                        content: ContentValue::new(String::default()),
                        id: id.thread_id.to_string(),
                        name: Some("ZedIdeAgent".to_string()),
                        example: false,
                        invalid_tool_calls: None,
                        tool_calls: None,
                        additional_kwargs,
                        response_metadata: Self::_create_acp_response_metadata(t.meta.clone()),
                    })
                }
                _ => None,
            },
            SessionUpdate::ToolCall(tc) => {
                let mut additional_kwargs = HashMap::new();
                if let Some(raw_input) = &tc.raw_input {
                    additional_kwargs.insert("raw_input".to_string(), raw_input.clone());
                }
                if let Some(raw_output) = &tc.raw_output {
                    additional_kwargs.insert("raw_output".to_string(), raw_output.clone());
                }

                let content = match serde_json::to_string(&tc.raw_input) {
                    Ok(content) => content,
                    Err(_) => String::default(),
                };

                Some(Message::Tool {
                    content: ContentValue::new(content),
                    id: id.thread_id.to_string(),
                    name: Some("ZedIdeAgent".to_string()),
                    example: false,
                    tool_call_id: Some(tc.id.0.to_string()),
                    tool_name: Some(tc.title.to_string()),
                    additional_kwargs,
                    response_metadata: Self::_create_acp_response_metadata(tc.meta.clone()),
                })
            }
            SessionUpdate::ToolCallUpdate(tcu) => {
                let mut additional_kwargs = HashMap::new();
                if let Some(status) = &tcu.fields.status {
                    additional_kwargs.insert(
                        "status".to_string(),
                        serde_json::Value::String(format!("{:?}", status)),
                    );
                }

                Some(Message::Tool {
                    content: ContentValue::new(String::default()),
                    id: id.thread_id.to_string(),
                    name: Some("ZedIdeAgent".to_string()),
                    example: false,
                    tool_call_id: Some(tcu.id.0.to_string()),
                    tool_name: Some(tcu.id.0.to_string()),
                    additional_kwargs,
                    response_metadata: Self::_create_acp_response_metadata(tcu.meta.clone()),
                })
            }
            SessionUpdate::Plan(p) => {
                let plan_entry = p.entries.iter()
                    .flat_map(|s| {
                        s.maybe_to_value()
                            .into_iter()
                            .flat_map(|s| s.as_str().map(|f| f.to_string()).into_iter())
                    })
                    .collect::<Vec<String>>();

                let r = Self::_create_acp_response_metadata(p.meta.clone());

                Some(Message::Ai {
                    content: ContentValue::Multiple(plan_entry),
                    id: id.session_id.to_string(),
                    name: Some("ZedIdeAgent".to_string()),
                    example: false,
                    invalid_tool_calls: None,
                    tool_calls: None,
                    additional_kwargs: Default::default(),
                    response_metadata: r,
                })
            },
            SessionUpdate::AvailableCommandsUpdate { .. } => None,
            SessionUpdate::CurrentModeUpdate { .. } => None,
        }
    }

    fn _create_annotations(option: Option<Annotations>) -> HashMap<String, Value> {
        let mut r = HashMap::new();
        // option
        //     .and_then(|f| f.as_str().map(|s| s.to_string()))
        //     .and_then(|f| {
        //         r.insert("meta".to_string(), f);
        //     });
        r
    }

    fn _create_acp_response_metadata(option: Option<Value>) -> HashMap<String, Value> {
        let mut r = HashMap::new();
        option.and_then(|f| f.as_str().map(|s| s.to_string()))
            .and_then(|f| {
                r.insert("meta".to_string(), Value::String(f))
            });

        r.insert("acp".into(), Value::String("true".into()));
        r
    }

    pub fn map_from_completion_event(
        request_message: &LanguageModelCompletionEvent,
        thread_id: &str,
        language_model_args: &LanguageModelArgs,
    ) -> Option<Message> {
        let response_metadata = Self::build_response_metadata(&language_model_args);
        match request_message {
            LanguageModelCompletionEvent::RedactedThinking { data } => {
                let id = thread_id.to_string();
                Some(Message::Ai {
                    content: ContentValue::new(data.clone()),
                    id,
                    name: Some("ZedIdeAgent".to_string()),
                    example: false,
                    invalid_tool_calls: None,
                    tool_calls: None,
                    additional_kwargs: HashMap::new(),
                    response_metadata,
                })
            }
            LanguageModelCompletionEvent::ToolUseJsonParseError { .. } => None,
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
    pub fn save_append_messages(
        &self,
        messages: Vec<Message>,
        ids: &RequestIds,
    ) -> anyhow::Result<()> {
        if let Some(ref db_client) = self.database_client {
            db_client.save_append_messages(messages, ids);
        }
        Ok(())
    }

    /// Save a message to the database
    pub async fn save_append_messages_async(
        &self,
        messages: Vec<Message>,
        ids: &RequestIds,
    ) -> anyhow::Result<()> {
        if let Some(ref db_client) = self.database_client {
            db_client.save_append_messages_async(messages, ids).await;
        }
        Ok(())
    }

    pub fn inspect_stream<T>(
        s: T,
        handler: Arc<AiMessageHandler>,
        ids: RequestIds,
        language_model_args: LanguageModelArgs,
    ) -> T
    where
        T: Stream<Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>,
    {
        s.inspect(move |result_ref| {
            let result = result_ref;
            let arc = handler.clone();
            let ids = ids.clone();
            let language_model_args = language_model_args.clone();

            if let Ok(res) = result {
                let res = res.clone();
                smol::spawn(async move {
                    arc.save_completion_event(&res, &ids, &language_model_args).await;
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
