use std::process::Command;
use std::time::Duration;
use anyhow::{Context, Result};
use bollard::Docker;
use bollard::container::{CreateContainerOptions, Config, StartContainerOptions};
use bollard::models::HostConfig;
use std::collections::HashMap;
use tokio::runtime::Runtime;

#[async_test]
async fn test_extension() {
    CdcAgentsExtension::new().await.unwrap();
}

#[test]
fn test_docker_connectivity() {
    // This test verifies if Docker is available on the system
    let output = Command::new("docker")
        .args(["info"])
        .output()
        .expect("Failed to execute docker info command");

    assert!(output.status.success(), "Docker is not available on the system");
    println!("Docker is available!");
}

#[test]
fn test_docker_run_command() {
    // This test verifies we can run a simple Docker container
    let output = Command::new("docker")
        .args(["run", "--rm", "hello-world"])
        .output()
        .expect("Failed to execute docker run command");

    assert!(output.status.success(), "Failed to run hello-world container");
    println!("Successfully ran hello-world container");
}

#[test]
fn test_docker_volume_mount() {
    // This test specifically checks the volume mounting capability as requested
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());

    let output = Command::new("docker")
        .args([
            "run",
            "--rm",
            "-v",
            &format!("{}:/host_home", home),
            "alpine",
            "ls",
            "/host_home"
        ])
        .output()
        .expect("Failed to execute docker run with volume");

    assert!(output.status.success(), "Failed to run container with volume mount");
    println!("Successfully mounted volume and executed command");
    println!("Output: {}", String::from_utf8_lossy(&output.stdout));
}

#[test]
fn test_bollard_api() {
    // Create a runtime for async operations
    let rt = Runtime::new().expect("Failed to create Tokio runtime");

    // Test using the same API that our extension uses
    rt.block_on(async {
        // Connect to Docker
        let docker = Docker::connect_with_local_defaults()
            .context("Failed to connect to Docker daemon")
            .expect("Docker connection failed");

        // Test listing images
        let images = docker.list_images(None)
            .await
            .expect("Failed to list Docker images");

        assert!(!images.is_empty(), "No Docker images found");
        println!("Successfully connected using bollard API");
        println!("Found {} images", images.len());

        // Test that we can find the alpine image (pulling it if needed)
        let alpine_exists = images.iter().any(|img| {
            img.repo_tags.as_ref().map_or(false, |tags| {
                tags.iter().any(|tag| tag.contains("alpine"))
            })
        });

        if !alpine_exists {
            println!("Pulling alpine image...");
            docker.create_image(
                Some(bollard::image::CreateImageOptions {
                    from_image: "alpine",
                    tag: "latest",
                    ..Default::default()
                }),
                None,
                None
            )
            .try_collect::<Vec<_>>()
            .await
            .expect("Failed to pull alpine image");
        }

        // Try to create and run a container
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());

        let options = Some(CreateContainerOptions {
            name: "zed_extension_test_container",
            ..Default::default()
        });

        let config = Config {
            image: Some("alpine:latest"),
            cmd: Some(vec!["ls", "/host_home"]),
            host_config: Some(HostConfig {
                binds: Some(vec![format!("{}:/host_home:ro", home)]),
                auto_remove: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };

        // Remove container if it already exists
        let _ = docker.remove_container("zed_extension_test_container", None).await;

        // Create and start container
        let container = docker.create_container(options, config)
            .await
            .expect("Failed to create test container");

        docker.start_container(&container.id, None::<StartContainerOptions<String>>)
            .await
            .expect("Failed to start test container");

        // Wait for container to finish
        let wait = docker.wait_container(&container.id, None)
            .try_collect::<Vec<_>>()
            .await
            .expect("Failed waiting for container");

        assert_eq!(wait[0].status_code, 0, "Container exited with non-zero status");

        // Get logs
        let logs = docker.logs(&container.id, None)
            .try_collect::<Vec<_>>()
            .await
            .expect("Failed to get container logs");

        assert!(!logs.is_empty(), "No logs from container");
        println!("Container executed successfully");
    });
}

#[test]
fn test_specific_docker_command() {
    // This test verifies the specific command mentioned in the request
    let home = std::env::var("HOME").unwrap_or_else(|_| "/Users".to_string());

    // First check if the mcp/git image exists
    let output = Command::new("docker")
        .args(["images", "mcp/git", "--quiet"])
        .output()
        .expect("Failed to check for mcp/git image");

    if output.stdout.is_empty() {
        println!("Warning: mcp/git image not found. This test will likely fail unless you have this image.");
        println!("If the image doesn't exist, the test will be skipped.");
        return;
    }

    // Run the specified command
    let output = Command::new("docker")
        .args([
            "run",
            "-i",
            "--rm",
            "-v",
            &format!("{}:/", home),
            "mcp/git"
        ])
        .output();

    match output {
        Ok(output) => {
            assert!(output.status.success(), "Failed to run mcp/git container");
            println!("Successfully ran mcp/git container");
            println!("Output: {}", String::from_utf8_lossy(&output.stdout));
        },
        Err(e) => {
            println!("Test skipped: {}", e);
        }
    }
}
