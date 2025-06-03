use crate::message_handler::{AiMessageHandler, PostgresDatabaseClient};
use anyhow::Result;
use gpui::{App, AppContext, AsyncApp, Global, Task, UpdateGlobal};
use image::imageops::flip_horizontal;
use std::sync::Arc;
use uuid::uuid;

/// Global registry for the AiMessageHandler
#[derive(Default)]
pub struct MessageHandlerRegistry {
    message_handler: Option<Arc<AiMessageHandler>>,
}

impl Global for MessageHandlerRegistry {}

/// Configuration for the message handler database connection
#[derive(Debug, Clone)]
pub struct MessageHandlerConfig {
    /// PostgreSQL connection string
    pub postgres_connection_string: Option<String>,

    /// Whether to enable database storage
    pub enable_storage: bool,
}

impl Default for MessageHandlerConfig {
    fn default() -> Self {
        Self {
            postgres_connection_string: None,
            enable_storage: false,
        }
    }
}

/// Initialize the message handler with the given configuration
pub fn init_message_handler(config: MessageHandlerConfig, cx: &mut App) -> Task<Result<()>> {
    log::info!("Initializing connection string");

    let connection_string = match &config.postgres_connection_string {
        Some(cs) => cs.clone(),
        None => {
            // Use environment variable if available
            std::env::var("ZED_LLM_POSTGRES_URL").unwrap_or_else(|_| {
                // Create a message handler without database support
                "postgresql://postgres:postgres@localhost:5488/postgres".to_string()
            })
        }
    };

    log::info!("Initializing connection string");

    if cx.has_global::<MessageHandlerRegistry>() {
        let option = get_message_handler(cx);
        if option.as_ref().is_some() {
            if option.as_ref().unwrap().database_client.as_ref().is_some() {
                return Task::ready(Ok(()));
            }
        }
    }

    let message_handler = AiMessageHandler::new(None);

    println!("Setting global message handler");

    let mut registry = MessageHandlerRegistry::default();
    registry.message_handler = Some(Arc::new(message_handler));
    cx.set_global(registry);

    log::info!("Setting global postgres message handler");

    cx.spawn(async move |t| {
        let t: &mut AsyncApp = t;
        log::info!("Postgres Connection initializing");
        let db_client = PostgresDatabaseClient::new(&connection_string).await?;
        let out = t
            .update_global::<MessageHandlerRegistry, Result<()>>(|g, c| {
                g.message_handler =
                    Some(Arc::new(AiMessageHandler::new(Some(Arc::new(db_client)))));
                Ok(())
            })
            .inspect_err(|e| log::error!("Found err when initializing message handler: {}", e))
            .unwrap();

        out
    })
}

/// Get the message handler instance
pub fn get_message_handler(cx: &App) -> Option<Arc<AiMessageHandler>> {
    cx.global::<MessageHandlerRegistry>()
        .message_handler
        .clone()
}

/// Get the message handler instance in an async context
pub fn get_message_handler_async(cx: &App) -> Option<Arc<AiMessageHandler>> {
    cx.global::<MessageHandlerRegistry>()
        .message_handler
        .clone()
}

/// Create a conversation ID for a new conversation
pub fn create_conversation_id() -> String {
    uuid::Uuid::new_v4().to_string()
}
