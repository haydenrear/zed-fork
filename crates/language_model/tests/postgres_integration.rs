use language_model::message_handler::{
    ContentValue, DatabaseClient, Message, PostgresDatabaseClient, WriteRequest,
};
use language_model::RequestIds;
use sqlx::postgres::PgPoolOptions;
use std::collections::HashMap;
use std::sync::{Arc, Once};
use std::time::{Duration, Instant};

fn connection_string() -> String {
    std::env::var("TEST_POSTGRES_URL")
        .unwrap_or_else(|_| "postgres://zed_test:zed_test@localhost:5877/zed_test".to_string())
}

static SCHEMA_INIT: Once = Once::new();

fn ensure_schema(pool: &sqlx::PgPool) {
    SCHEMA_INIT.call_once(|| {
        smol::block_on(async {
            PostgresDatabaseClient::initialize_schema(pool)
                .await
                .expect("Failed to initialize schema");
        });
    });
}

fn make_ids(thread_id: &str, checkpoint_id: &str) -> RequestIds {
    RequestIds {
        thread_id: thread_id.to_string(),
        checkpoint_id: checkpoint_id.to_string(),
        session_id: "test-session".to_string(),
        prompt_id: "test-prompt".to_string(),
    }
}

fn make_message(content: &str) -> Vec<Message> {
    vec![Message::Human {
        content: ContentValue::Single(content.to_string()),
        id: "msg-id".to_string(),
        name: Some("test".to_string()),
        example: false,
        additional_kwargs: HashMap::new(),
        response_metadata: HashMap::new(),
    }]
}

async fn setup_pool() -> sqlx::PgPool {
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .acquire_timeout(Duration::from_secs(10))
        .connect(&connection_string())
        .await
        .expect("Failed to connect to test postgres");

    ensure_schema(&pool);

    sqlx::raw_sql("TRUNCATE ide_checkpoint_messages_v2, ide_checkpoint_offsets_v2")
        .execute(&pool)
        .await
        .expect("Failed to truncate tables");

    pool
}

/// Verify that messages are inserted in order and sequence_ids are monotonically increasing.
#[test]
fn test_messages_inserted_in_order() {
    smol::block_on(async {
        let pool = setup_pool().await;
        let ids = make_ids("order-thread", "order-checkpoint");

        for i in 0..20 {
            let request = WriteRequest {
                messages: make_message(&format!("message {}", i)),
                ids: ids.clone(),
                task_path: "standard".to_string(),
            };
            PostgresDatabaseClient::execute_write(&pool, &request)
                .await
                .expect("write should succeed");
        }

        let rows: Vec<(i64, serde_json::Value)> = sqlx::query_as(
            "SELECT sequence_id, messages FROM ide_checkpoint_messages_v2 \
             WHERE thread_id = $1 AND checkpoint_id = $2 \
             ORDER BY sequence_id ASC",
        )
        .bind("order-thread")
        .bind("order-checkpoint")
        .fetch_all(&pool)
        .await
        .expect("query should succeed");

        assert_eq!(rows.len(), 20, "should have 20 messages");

        let mut previous_sequence_id = 0i64;
        for (index, (sequence_id, messages_json)) in rows.iter().enumerate() {
            assert!(
                *sequence_id > previous_sequence_id,
                "sequence_id should be monotonically increasing"
            );
            previous_sequence_id = *sequence_id;

            let messages_array = messages_json.as_array().expect("messages should be array");
            assert_eq!(messages_array.len(), 1);
            let content = messages_array[0]["content"]
                .as_str()
                .expect("content should be string");
            assert_eq!(
                content,
                format!("message {}", index),
                "messages should be in insertion order"
            );
        }

        // Verify offset was updated to the last sequence_id
        let offset: (i64,) = sqlx::query_as(
            "SELECT last_message_offset FROM ide_checkpoint_offsets_v2 \
             WHERE thread_id = $1 AND checkpoint_id = $2",
        )
        .bind("order-thread")
        .bind("order-checkpoint")
        .fetch_one(&pool)
        .await
        .expect("offset should exist");

        assert_eq!(
            offset.0, previous_sequence_id,
            "offset should equal the last sequence_id"
        );
    });
}

/// Verify that duplicate messages (same content, same thread/checkpoint) are deduplicated.
#[test]
fn test_deduplication_on_restart() {
    smol::block_on(async {
        let pool = setup_pool().await;
        let ids = make_ids("dedup-thread", "dedup-checkpoint");

        // Simulate first session: write 5 messages
        for i in 0..5 {
            let request = WriteRequest {
                messages: make_message(&format!("message {}", i)),
                ids: ids.clone(),
                task_path: "standard".to_string(),
            };
            PostgresDatabaseClient::execute_write(&pool, &request)
                .await
                .expect("first write should succeed");
        }

        let count_before: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM ide_checkpoint_messages_v2 \
             WHERE thread_id = $1 AND checkpoint_id = $2",
        )
        .bind("dedup-thread")
        .bind("dedup-checkpoint")
        .fetch_one(&pool)
        .await
        .expect("count query should succeed");

        assert_eq!(count_before.0, 5);

        // Simulate restart: write the same 5 messages again
        for i in 0..5 {
            let request = WriteRequest {
                messages: make_message(&format!("message {}", i)),
                ids: ids.clone(),
                task_path: "standard".to_string(),
            };
            PostgresDatabaseClient::execute_write(&pool, &request)
                .await
                .expect("duplicate write should succeed");
        }

        let count_after: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM ide_checkpoint_messages_v2 \
             WHERE thread_id = $1 AND checkpoint_id = $2",
        )
        .bind("dedup-thread")
        .bind("dedup-checkpoint")
        .fetch_one(&pool)
        .await
        .expect("count query should succeed");

        assert_eq!(
            count_after.0, 5,
            "duplicate messages should not create new rows"
        );

        // But new messages after the duplicates should still work
        let request = WriteRequest {
            messages: make_message("message 5 (new after restart)"),
            ids: ids.clone(),
            task_path: "standard".to_string(),
        };
        PostgresDatabaseClient::execute_write(&pool, &request)
            .await
            .expect("new message after restart should succeed");

        let count_final: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM ide_checkpoint_messages_v2 \
             WHERE thread_id = $1 AND checkpoint_id = $2",
        )
        .bind("dedup-thread")
        .bind("dedup-checkpoint")
        .fetch_one(&pool)
        .await
        .expect("count query should succeed");

        assert_eq!(count_final.0, 6, "new message should create a new row");
    });
}

/// Verify that the background channel queue processes messages and is non-blocking to the caller.
#[test]
fn test_queue_is_nonblocking() {
    // Run inside a smol executor so background tasks spawned by PostgresDatabaseClient are driven
    smol::block_on(smol::future::or(
        async {
            let client = PostgresDatabaseClient::new(&connection_string())
                .await
                .expect("client should connect");

            let pool = PgPoolOptions::new()
                .max_connections(1)
                .connect(&connection_string())
                .await
                .expect("pool for cleanup");
            sqlx::raw_sql("TRUNCATE ide_checkpoint_messages_v2, ide_checkpoint_offsets_v2")
                .execute(&pool)
                .await
                .expect("truncate");

            let ids = make_ids("nonblock-thread", "nonblock-checkpoint");
            let message_count = 50i64;

            // Time how long it takes to enqueue all messages (should be near-instant)
            let start = Instant::now();
            for i in 0..message_count {
                client.save_append_messages(
                    make_message(&format!("queued message {}", i)),
                    &ids,
                );
            }
            let enqueue_duration = start.elapsed();

            assert!(
                enqueue_duration < Duration::from_millis(100),
                "enqueuing {} messages took {:?}, should be < 100ms (non-blocking)",
                message_count,
                enqueue_duration
            );

            // Wait for the background writer to drain
            let deadline = Instant::now() + Duration::from_secs(30);
            loop {
                // Yield to let background writer tasks run
                smol::Timer::after(Duration::from_millis(50)).await;

                let count: (i64,) = sqlx::query_as(
                    "SELECT COUNT(*) FROM ide_checkpoint_messages_v2 \
                     WHERE thread_id = $1 AND checkpoint_id = $2",
                )
                .bind("nonblock-thread")
                .bind("nonblock-checkpoint")
                .fetch_one(&pool)
                .await
                .expect("count query");

                if count.0 == message_count {
                    break;
                }
                if Instant::now() > deadline {
                    panic!(
                        "Background writer did not drain in time: got {} of {} messages",
                        count.0, message_count
                    );
                }
            }

            // Verify ordering
            let rows: Vec<(i64,)> = sqlx::query_as(
                "SELECT sequence_id FROM ide_checkpoint_messages_v2 \
                 WHERE thread_id = $1 AND checkpoint_id = $2 \
                 ORDER BY sequence_id ASC",
            )
            .bind("nonblock-thread")
            .bind("nonblock-checkpoint")
            .fetch_all(&pool)
            .await
            .expect("fetch rows");

            assert_eq!(rows.len(), message_count as usize);
            for window in rows.windows(2) {
                assert!(
                    window[1].0 > window[0].0,
                    "sequence_ids must be strictly increasing"
                );
            }
        },
        // Ensure the executor doesn't starve: run background tasks concurrently
        async {
            smol::Timer::after(Duration::from_secs(60)).await;
            panic!("test timed out after 60s");
        },
    ));
}

/// Load test: fire many concurrent writes and verify all land correctly without errors.
#[test]
fn test_under_load() {
    smol::block_on(async {
        let pool = setup_pool().await;
        let pool = Arc::new(pool);
        let thread_count = 10;
        let messages_per_thread = 50;

        let mut tasks = Vec::new();

        for thread_index in 0..thread_count {
            let pool = pool.clone();
            let task = smol::spawn(async move {
                let ids = make_ids("load-thread", &format!("load-checkpoint-{}", thread_index));
                for message_index in 0..messages_per_thread {
                    let request = WriteRequest {
                        messages: make_message(&format!(
                            "thread {} msg {}",
                            thread_index, message_index
                        )),
                        ids: ids.clone(),
                        task_path: "standard".to_string(),
                    };
                    PostgresDatabaseClient::execute_write(&pool, &request)
                        .await
                        .expect("write under load should succeed");
                }
            });
            tasks.push(task);
        }

        for task in tasks {
            task.await;
        }

        // Verify total row count
        let total: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM ide_checkpoint_messages_v2 WHERE thread_id = $1",
        )
        .bind("load-thread")
        .fetch_one(&*pool)
        .await
        .expect("count query");

        let expected = (thread_count * messages_per_thread) as i64;
        assert_eq!(
            total.0, expected,
            "all messages should be persisted: got {}, expected {}",
            total.0, expected
        );

        // Verify each checkpoint has correct count and ordering
        for thread_index in 0..thread_count {
            let checkpoint_id = format!("load-checkpoint-{}", thread_index);
            let rows: Vec<(i64, serde_json::Value)> = sqlx::query_as(
                "SELECT sequence_id, messages FROM ide_checkpoint_messages_v2 \
                 WHERE thread_id = $1 AND checkpoint_id = $2 \
                 ORDER BY sequence_id ASC",
            )
            .bind("load-thread")
            .bind(&checkpoint_id)
            .fetch_all(&*pool)
            .await
            .expect("query per checkpoint");

            assert_eq!(
                rows.len(),
                messages_per_thread as usize,
                "checkpoint {} should have {} messages",
                checkpoint_id,
                messages_per_thread
            );

            for window in rows.windows(2) {
                assert!(
                    window[1].0 > window[0].0,
                    "sequence_ids must be strictly increasing within checkpoint {}",
                    checkpoint_id
                );
            }

            // Verify offset matches last sequence_id
            let offset: (i64,) = sqlx::query_as(
                "SELECT last_message_offset FROM ide_checkpoint_offsets_v2 \
                 WHERE thread_id = $1 AND checkpoint_id = $2",
            )
            .bind("load-thread")
            .bind(&checkpoint_id)
            .fetch_one(&*pool)
            .await
            .expect("offset should exist");

            let last_seq = rows.last().expect("should have rows").0;
            assert_eq!(
                offset.0, last_seq,
                "offset should match last sequence_id for checkpoint {}",
                checkpoint_id
            );
        }
    });
}

/// Verify that the background queue + full PostgresDatabaseClient round-trips correctly
/// with interleaved sync and async writes.
#[test]
fn test_mixed_sync_async_writes() {
    smol::block_on(smol::future::or(
        async {
            let client = PostgresDatabaseClient::new(&connection_string())
                .await
                .expect("client should connect");

            let pool = PgPoolOptions::new()
                .max_connections(1)
                .connect(&connection_string())
                .await
                .expect("pool for verification");
            sqlx::raw_sql("TRUNCATE ide_checkpoint_messages_v2, ide_checkpoint_offsets_v2")
                .execute(&pool)
                .await
                .expect("truncate");

            let ids = make_ids("mixed-thread", "mixed-checkpoint");

            // Async writes (awaited, so guaranteed committed before we check)
            for i in 0..5 {
                client
                    .save_append_messages_async(
                        make_message(&format!("async msg {}", i)),
                        &ids,
                    )
                    .await;
            }

            let count_after_async: (i64,) = sqlx::query_as(
                "SELECT COUNT(*) FROM ide_checkpoint_messages_v2 \
                 WHERE thread_id = $1 AND checkpoint_id = $2",
            )
            .bind("mixed-thread")
            .bind("mixed-checkpoint")
            .fetch_one(&pool)
            .await
            .expect("count");
            assert_eq!(count_after_async.0, 5, "async writes should be immediate");

            // Sync writes (queued via channel)
            for i in 0..5 {
                client.save_append_messages(
                    make_message(&format!("sync msg {}", i)),
                    &ids,
                );
            }

            // Wait for background queue to drain
            let deadline = Instant::now() + Duration::from_secs(30);
            loop {
                smol::Timer::after(Duration::from_millis(50)).await;

                let count: (i64,) = sqlx::query_as(
                    "SELECT COUNT(*) FROM ide_checkpoint_messages_v2 \
                     WHERE thread_id = $1 AND checkpoint_id = $2",
                )
                .bind("mixed-thread")
                .bind("mixed-checkpoint")
                .fetch_one(&pool)
                .await
                .expect("count");

                if count.0 == 10 {
                    break;
                }
                if Instant::now() > deadline {
                    panic!("did not see all 10 messages, got {}", count.0);
                }
            }
        },
        async {
            smol::Timer::after(Duration::from_secs(60)).await;
            panic!("test timed out after 60s");
        },
    ));
}
