use crate::RequestIds;
use crate::message_handler::{DatabaseClient, Message};
use anyhow::Result;
use smol::channel::{Sender, bounded};
use sqlx::{PgPool, postgres::PgPoolOptions};
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;

pub struct WriteRequest {
    pub messages: Vec<Message>,
    pub ids: RequestIds,
    pub task_path: String,
}

/// A PostgreSQL implementation of the DatabaseClient trait
pub struct PostgresDatabaseClient {
    pool: Option<Arc<PgPool>>,
    write_sender: Option<Sender<WriteRequest>>,
}

impl PostgresDatabaseClient {
    /// Creates a new PostgreSQL database client
    pub async fn new(connection_string: &str) -> Result<Self> {
        log::info!("Connecting to postgres.");

        let connection_string_value = connection_string.to_string();
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .acquire_timeout(Duration::from_secs(10))
            .connect(&connection_string_value)
            .await;

        match pool {
            Ok(pool) => {
                log::info!("Connected to postgres... initializing schema");

                match Self::initialize_schema(&pool).await {
                    Ok(()) => {
                        let pool = Arc::new(pool);
                        let (sender, receiver) = bounded::<WriteRequest>(128);

                        let writer_pool = pool.clone();
                        smol::spawn(async move {
                            Self::background_writer(writer_pool, receiver).await;
                        })
                        .detach();

                        Ok(Self {
                            pool: Some(pool),
                            write_sender: Some(sender),
                        })
                    }
                    Err(e) => {
                        log::error!("Could not initialize schema: {:?}", e);
                        Ok(Self {
                            pool: None,
                            write_sender: None,
                        })
                    }
                }
            }
            Err(err) => {
                log::error!("Could not build the pool: {:?}", err);
                Self::debug_log(&format!("Could not build the pool: {:?}", err));
                Ok(Self {
                    pool: None,
                    write_sender: None,
                })
            }
        }
    }

    /// Background writer task that drains the channel and writes to the database
    async fn background_writer(
        pool: Arc<PgPool>,
        receiver: smol::channel::Receiver<WriteRequest>,
    ) {
        while let Ok(request) = receiver.recv().await {
            if let Err(e) = Self::execute_write(&pool, &request).await {
                log::error!("Background write failed: {}", e);
            }
        }
        log::info!("Background writer shutting down");
    }

    fn debug_log(message: &str) {
        log::info!("{}", message);

        if let Ok(path) = std::env::var("ZED_POSTGRES_DEBUG") {
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S%.3f");
                let _ = writeln!(file, "[{}] {}", timestamp, message);
            }
        }
    }

    /// Execute a single write: check offset, insert message, update offset
    pub async fn execute_write(pool: &PgPool, request: &WriteRequest) -> Result<()> {
        let json = serde_json::to_string(&request.messages)?;

        // Phase 1: Check current offset to see if this message was already saved
        let message_hash = Self::compute_message_hash(&json);

        let existing: Option<(i64,)> = sqlx::query_as(
            "SELECT last_message_offset FROM ide_checkpoint_offsets_v2 \
             WHERE thread_id = $1 AND checkpoint_id = $2",
        )
        .bind(&request.ids.thread_id)
        .bind(&request.ids.checkpoint_id)
        .fetch_optional(&*pool)
        .await?;

        let current_offset = existing.map(|(offset,)| offset).unwrap_or(0);

        // Phase 2: Insert the message into the messages table
        let maybe_new_offset: Option<i64> = sqlx::query_scalar::<_, i64>(
            "INSERT INTO ide_checkpoint_messages_v2 \
             (thread_id, checkpoint_id, prompt_id, session_id, message_hash, messages, task_path) \
             VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7) \
             ON CONFLICT (thread_id, checkpoint_id, message_hash) DO NOTHING \
             RETURNING sequence_id",
        )
        .bind(&request.ids.thread_id)
        .bind(&request.ids.checkpoint_id)
        .bind(&request.ids.prompt_id)
        .bind(&request.ids.session_id)
        .bind(message_hash)
        .bind(&json)
        .bind(&request.task_path)
        .fetch_optional(&*pool)
        .await?;

        let new_offset = match maybe_new_offset {
            Some(offset) => {
                Self::debug_log(&format!(
                    "Inserted message: thread_id={}, checkpoint_id={}, hash={}, sequence_id={}",
                    &request.ids.thread_id, &request.ids.checkpoint_id, message_hash, offset
                ));
                offset
            }
            None => {
                Self::debug_log(&format!(
                    "Skipped duplicate message: thread_id={}, checkpoint_id={}, hash={}, current_offset={}",
                    &request.ids.thread_id, &request.ids.checkpoint_id, message_hash, current_offset
                ));
                current_offset
            }
        };

        // Phase 3: Update the offset tracker
        if new_offset > current_offset {
            sqlx::query(
                "INSERT INTO ide_checkpoint_offsets_v2 (thread_id, checkpoint_id, last_message_offset) \
                 VALUES ($1, $2, $3) \
                 ON CONFLICT (thread_id, checkpoint_id) \
                 DO UPDATE SET last_message_offset = GREATEST(ide_checkpoint_offsets_v2.last_message_offset, $3)",
            )
            .bind(&request.ids.thread_id)
            .bind(&request.ids.checkpoint_id)
            .bind(new_offset)
            .execute(&*pool)
            .await?;
        }

        Ok(())
    }

    /// Compute a hash of the message content for deduplication
    pub fn compute_message_hash(json: &str) -> i64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        json.hash(&mut hasher);
        hasher.finish() as i64
    }

    /// Initialize the database schema if it doesn't exist
    pub async fn initialize_schema(pool: &PgPool) -> Result<()> {
        sqlx::raw_sql(
            r#"
CREATE TABLE IF NOT EXISTS ide_checkpoints_v2
(
    thread_id     text                  not null,
    prompt_id     text                  not null,
    session_id    text                  not null,
    checkpoint_ts text default ''::text not null,
    checkpoint_id text                  not null,
    blob          bytea                 not null,
    task_path     text default ''::text not null,
    primary key (thread_id, checkpoint_id)
);

CREATE INDEX IF NOT EXISTS ide_checkpoints_v2_thread_id_idx
    ON ide_checkpoints_v2 (thread_id);
CREATE INDEX IF NOT EXISTS ide_checkpoints_v2_thread_id_checkpoint_id_idx
    ON ide_checkpoints_v2 (thread_id, checkpoint_id);

CREATE TABLE IF NOT EXISTS ide_checkpoint_messages_v2
(
    sequence_id    BIGSERIAL             NOT NULL,
    thread_id      text                  NOT NULL,
    checkpoint_id  text                  NOT NULL,
    prompt_id      text                  NOT NULL,
    session_id     text                  NOT NULL,
    message_hash   BIGINT                NOT NULL,
    created_at     TIMESTAMPTZ           DEFAULT now() NOT NULL,
    messages       JSONB                 NOT NULL,
    task_path      text                  DEFAULT '' NOT NULL,
    PRIMARY KEY (sequence_id),
    UNIQUE (thread_id, checkpoint_id, message_hash)
);

CREATE INDEX IF NOT EXISTS ide_checkpoint_messages_v2_thread_checkpoint_idx
    ON ide_checkpoint_messages_v2 (thread_id, checkpoint_id);
CREATE INDEX IF NOT EXISTS ide_checkpoint_messages_v2_thread_checkpoint_seq_idx
    ON ide_checkpoint_messages_v2 (thread_id, checkpoint_id, sequence_id);

CREATE TABLE IF NOT EXISTS ide_checkpoint_offsets_v2
(
    thread_id           text NOT NULL,
    checkpoint_id       text NOT NULL,
    last_message_offset BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (thread_id, checkpoint_id)
);
            "#,
        )
        .execute(pool)
        .await
        .inspect_err(|e| log::error!("Found error initializing schema: {}", e))
        .map(|_| Ok(()))?
    }

    fn _parse_task_path<'a>(message: &Vec<Message>) -> &'a str {
        let task_paths = message
            .iter()
            .flat_map(|f| {
                f.response_metadata()
                    .get("intent")
                    .cloned()
                    .into_iter()
                    .flat_map(|j| j.as_str().map(|s| s.to_string()).into_iter())
            })
            .collect::<Vec<String>>();

        let mut task_path = "standard";

        if task_paths.iter().all(|t| t.eq("ThreadSummarization")) {
            task_path = "summarization";
        }

        if task_paths
            .iter()
            .all(|t| t.eq("ThreadContextSummarization"))
        {
            task_path = "context_summarization";
        }

        if !task_path.eq("summarization") && task_paths.iter().any(|t| t.eq("ThreadSummarization"))
        {
            log::error!("Found strange situation where not all were ThreadSummarization")
        }

        if !task_path.eq("context_summarization")
            && task_paths
                .iter()
                .any(|t| t.eq("ThreadContextSummarization"))
        {
            log::error!("Found strange situation where not all were ThreadContextSummarization")
        }
        task_path
    }
}

impl DatabaseClient for PostgresDatabaseClient {
    async fn save_append_messages_async(&self, message: Vec<Message>, ids: &RequestIds) {
        if message.is_empty() {
            return;
        }

        let pool = match &self.pool {
            Some(p) => p.clone(),
            None => {
                log::error!("Database pool is not initialized");
                return;
            }
        };

        let task_path = Self::_parse_task_path(&message).to_string();

        let request = WriteRequest {
            messages: message,
            ids: ids.clone(),
            task_path,
        };

        if let Err(e) = Self::execute_write(&pool, &request).await {
            log::error!("Async write failed: {}", e);
        }
    }

    fn save_append_messages(&self, message: Vec<Message>, ids: &RequestIds) {
        if message.is_empty() {
            return;
        }

        let sender = match &self.write_sender {
            Some(s) => s.clone(),
            None => {
                log::error!("Write channel is not initialized");
                return;
            }
        };

        let task_path = Self::_parse_task_path(&message).to_string();

        let request = WriteRequest {
            messages: message,
            ids: ids.clone(),
            task_path,
        };

        match sender.try_send(request) {
            Ok(()) => {}
            Err(smol::channel::TrySendError::Full(_)) => {
                log::warn!("Write channel is full, dropping message batch");
            }
            Err(smol::channel::TrySendError::Closed(_)) => {
                log::error!("Write channel is closed");
            }
        }
    }
}

#[cfg(test)]
mod test_db_client {
    use crate::message_handler::{ContentValue, Message, PostgresDatabaseClient};
    use crate::{AiMessageContent, MessageContent};
    use std::collections::HashMap;

    #[test]
    fn test_append_messages() {
        let parsed = PostgresDatabaseClient::_parse_task_path(&vec![Message::Ai {
            content: ContentValue::Single("hello".to_string()),
            id: "".to_string(),
            name: None,
            example: false,
            invalid_tool_calls: None,
            tool_calls: None,
            additional_kwargs: Default::default(),
            response_metadata: [(
                "intent".to_string(),
                serde_json::Value::String("ThreadSummarization".to_string()),
            )]
            .into_iter()
            .collect::<HashMap<String, serde_json::Value>>(),
        }]);

        assert_eq!(parsed, "summarization");
    }
}
