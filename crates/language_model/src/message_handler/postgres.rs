use crate::RequestIds;
use crate::message_handler::{DatabaseClient, Message};
use anyhow::Result;
use chrono::Utc;
use sqlx::{Connection, Executor, PgConnection, PgPool, postgres::PgPoolOptions};
use std::sync::Arc;
use std::time::Duration;

/// A PostgreSQL implementation of the DatabaseClient trait
pub struct PostgresDatabaseClient {
    pool: Option<Arc<PgPool>>,
}

impl PostgresDatabaseClient {
    /// Creates a new PostgreSQL database client
    pub async fn new(connection_string: &str) -> Result<Self> {
        log::info!("Connecting to postgres.");

        let pool = PgPoolOptions::new()
            .max_connections(5)
            .acquire_timeout(Duration::from_secs(3))
            .connect(connection_string)
            .await?;

        log::info!("Connected to postgres... initializing schema");

        // Ensure tables exist
        Self::initialize_schema(&pool).await?;

        log::info!("Initialized schema.");

        Ok(Self {
            pool: Some(Arc::new(pool)),
        })
    }

    /// Initialize the database schema if it doesn't exist
    async fn initialize_schema(pool: &PgPool) -> Result<()> {
        sqlx::raw_sql(
            r#"
create table if not exists  ide_checkpoints
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

create index if not exists  ide_checkpoints_thread_id_idx
    on ide_checkpoints (thread_id);
create index if not exists  ide_checkpoints_thread_id_checkpoint_id_idx
    on ide_checkpoints (thread_id, checkpoint_id);
            "#,
        )
        .execute(pool)
        .await
        .inspect_err(|e| log::error!("Found error initializing schema: {}", e))
        .map(|p| Ok(()))?
    }

    fn _parse_sql_query(ids: &RequestIds, json: &String, task_path: &str) -> String {
        let json = json.replace("'", "");

        let f = format!(
            r#"
                INSERT INTO ide_checkpoints (thread_id, prompt_id, session_id, checkpoint_ts, checkpoint_id, blob, task_path)
                VALUES ('{}',
                        '{}',
                        '{}',
                        now(),
                        '{}',
                        convert_to('{}', 'UTF8'),
                        '{}')
                ON CONFLICT (thread_id, checkpoint_id)
                DO UPDATE
                SET blob = convert_to(
                        (
                            (
                                COALESCE(
                                        convert_from(ide_checkpoints.blob, 'UTF8')::jsonb,
                                        '[]'::jsonb
                                ) || '{}'::jsonb
                                )::text
                            ),
                    'UTF8');
                "#,
            &ids.thread_id, &ids.prompt_id, &ids.session_id, &ids.checkpoint_id, &json, task_path, &json
        );

        log::info!("Here is sql query\n{}", &f);

        f
    }

    fn _parse_task_path<'a>(message: &Vec<Message>) -> &'a str {
        let task_paths = message.iter()
            .flat_map(|f| {
                f.response_metadata().get("intent").cloned().into_iter()
                    .flat_map(|j| j.as_str()
                        .map(|s| s.to_string())
                        .into_iter())
            })
            .collect::<Vec<String>>();

        let mut task_path = "standard";

        if task_paths.iter().all(|t| t.eq("ThreadSummarization")) {
            task_path = "summarization";
        }

        if task_paths.iter().all(|t| t.eq("ThreadContextSummarization")) {
            task_path = "context_summarization";
        }

        if !task_path.eq("summarization") && task_paths.iter().any(|t| t.eq("ThreadSummarization")) {
            log::error!("Found strange situation where not all were ThreadSummarization")
        }

        if !task_path.eq("context_summarization") && task_paths.iter().any(|t| t.eq("ThreadContextSummarization")) {
            log::error!("Found strange situation where not all were ThreadContextSummarization")
        }
        task_path
    }
}

impl DatabaseClient for PostgresDatabaseClient {
    async fn save_append_messages(&self, message: Vec<Message>, ids: &RequestIds) {
        let message_clone = message.clone();
        let pool = self.pool.clone();

        if pool.as_ref().is_none() {
            log::error!("Database pool is not initialized");
            return;
        }


        let task_path = Self::_parse_task_path(&message);

        let message_json_res = serde_json::to_string(&message_clone);

        if let Ok(json) = &message_json_res {
            let sql_res = sqlx::raw_sql(&Self::_parse_sql_query(ids, json, task_path))
                .execute(&*pool.unwrap())
                .await;

            if let Err(e) = sql_res {
                log::error!("Found sql err {}!", &e);
            }
        } else if let Err(e) = &message_json_res {
            log::error!("Found err: {}", &e);
        }
    }
}

#[cfg(test)]
mod test_db_client {
    use std::collections::HashMap;
    use crate::{AiMessageContent, MessageContent};
    use crate::message_handler::{ContentValue, Message, PostgresDatabaseClient};

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
            response_metadata: [("intent".to_string(), serde_json::Value::String("ThreadSummarization".to_string()))].into_iter().collect::<HashMap<String, serde_json::Value>>(),
        }]);

        assert_eq!(parsed, "summarization");
    }

}