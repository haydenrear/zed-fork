use crate::message_handler::{DatabaseClient, Message};
use anyhow::Result;
use chrono::Utc;
use sqlx::{PgPool, postgres::PgPoolOptions, PgConnection, Connection, Executor};
use std::sync::Arc;
use std::time::Duration;

/// A PostgreSQL implementation of the DatabaseClient trait
pub struct PostgresDatabaseClient {
    pool: Option<Arc<PgPool>>,
}

impl PostgresDatabaseClient {
    /// Creates a new PostgreSQL database client
    pub async fn new(connection_string: &str) -> Result<Self> {
        println!("Connecting");
        // tracing::Span::current().record("connection_id", format!("{}", connection_id));
        //
        // tracing::info!("connection opened");

        let pool = PgPoolOptions::new()
            .max_connections(5)
            .acquire_timeout(Duration::from_secs(3))
            .connect("postgresql://postgres:postgres@localhost:5488/postgres")
            .await?;


        println!("Connecting to postgres");

        // Ensure tables exist
        Self::initialize_schema(&pool).await?;

        println!("Initialized schema.");

        Ok(Self {
            pool: None,
        })
    }

    /// Initialize the database schema if it doesn't exist
    async fn initialize_schema(pool: &PgPool) -> Result<()> {
        sqlx::raw_sql(
            r#"
create table if not exists  ide_checkpoints
(
    thread_id     text                  not null,
    checkpoint_ts text default ''::text not null,
    checkpoint_id text                  not null,
    blob          bytea                 not null,
    task_path     text default ''::text not null,
    primary key (thread_id)
);

create index if not exists  ide_checkpoints_thread_id_idx
    on ide_checkpoints (thread_id);
            "#,
        )
        .execute(pool)
        .await?;

        Ok(())
    }

    fn _parse_sql_query(thread_id: &&String, json: &String) -> String {

        let json = json.replace("'", "");

        let f = format!(r#"
                INSERT INTO ide_checkpoints (thread_id, checkpoint_ts, checkpoint_id, blob, task_path)
                VALUES ('{}',
                        now(),
                        '{}',
                        convert_to('{}', 'UTF8'),
                        '{}')
                ON CONFLICT (thread_id)
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
                "#, &thread_id, &thread_id, &json, "standard", &json);



        println!("Here is sql query\n{}", &f);

        f
    }
}

impl DatabaseClient for PostgresDatabaseClient {
    async fn save_append_messages(&self, message: Vec<Message>, thread_id: &str) {

        let message_clone = message.clone();
        let pool = self.pool.clone();

        if pool.as_ref().is_none() {
            return;
        }

        let thread_id = &thread_id.to_string();

        let message_json_res = serde_json::to_string(&message_clone);
        println!("Performing...");
        if message_json_res.as_ref().is_ok() {
            println!("Performing... is ok...");
            let json = message_json_res.unwrap();
            let sql_res = sqlx::raw_sql(
                &Self::_parse_sql_query(&thread_id, &json)
            )
            .execute(&*pool.unwrap())
            .await;

            if sql_res.is_err() {
                println!("Found sql err {}!", &sql_res.err().unwrap());
            }

        } else {
            println!("Found err: {}", &message_json_res.err().unwrap());
        }
    }
}
