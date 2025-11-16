use crate::message_handler::{AiMessageHandler, PostgresDatabaseClient};
use anyhow::Result;
use gpui::{App, AppContext, AsyncApp, Global, Task, UpdateGlobal};
use image::imageops::flip_horizontal;
use std::sync::Arc;

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

impl MessageHandlerConfig {
    pub fn parse_cxn_string(&self) -> String {
        _parse_cxn_string(self)
    }
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
pub fn init_message_handler(config: MessageHandlerConfig, cx: &mut App) {
    log::info!("Initializing connection string");

    let connection_string = config.parse_cxn_string();

    log::info!("Initializing connection string");

    if cx.has_global::<MessageHandlerRegistry>() {
        let option = get_message_handler(cx);
        if option.as_ref().is_some() {
            if option.as_ref().unwrap().database_client.as_ref().is_some() {
                return;
            }
        }
    }

    log::info!("Setting global message handler");
    log::info!("Setting global postgres message handler");
    log::info!("Postgres Connection initializing");

    smol::block_on(async move {
        let mut registry = MessageHandlerRegistry::default();

        if !config.enable_storage {
            let message_handler = Arc::new(AiMessageHandler::new(None));
            registry.message_handler = Some(message_handler);
        } else if let Ok(db_client)  = PostgresDatabaseClient::new(&connection_string).await {
            let message_handler = Arc::new(AiMessageHandler::new(Some(Arc::new(db_client))));
            registry.message_handler = Some(message_handler);
        } else {
            let message_handler = Arc::new(AiMessageHandler::new(None));
            registry.message_handler = Some(message_handler);
        }

        cx.set_global(registry);
    });
}

fn _parse_cxn_string(config: &MessageHandlerConfig) -> String {
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
    connection_string
}

pub fn create_message_handler(connection_string: MessageHandlerConfig) -> Arc<AiMessageHandler> {
    smol::block_on(async move {
        if !connection_string.enable_storage {
            Arc::new(AiMessageHandler::new(None))
        } else if let Ok(db_client)  = PostgresDatabaseClient::new(&connection_string.parse_cxn_string()).await {
            Arc::new(AiMessageHandler::new(Some(Arc::new(db_client))))
        } else {
            Arc::new(AiMessageHandler::new(None))
        }
    })
}

/// Get the message handler instance in an async context
pub fn get_message_handler(cx: &App) -> Option<Arc<AiMessageHandler>> {
    cx.global::<MessageHandlerRegistry>()
        .message_handler
        .clone()
}

pub fn get_message_handler_async(cx: &App) -> Option<Arc<AiMessageHandler>> {
    get_message_handler(cx)
}

/// Create a conversation ID for a new conversation
pub fn create_conversation_id() -> String {
    uuid::Uuid::new_v4().to_string()
}
