use std::{path::Path, sync::Arc};

use editor::Editor;
use gpui::{
    AsyncWindowContext, Context, Entity, IntoElement, ParentElement, Render, Subscription, Task,
    WeakEntity, Window, div,
};
use language::{Buffer, BufferEvent, LanguageName, Toolchain};
use project::{Project, ProjectPath, WorktreeId, toolchain_store::ToolchainStoreEvent};
use ui::{Button, ButtonCommon, Clickable, FluentBuilder, LabelSize, SharedString, Tooltip};
use workspace::{StatusItemView, Workspace, item::ItemHandle};

use crate::ToolchainSelector;

pub struct ActiveToolchain {
    active_toolchain: Option<Toolchain>,
    term: SharedString,
    workspace: WeakEntity<Workspace>,
    active_buffer: Option<(WorktreeId, WeakEntity<Buffer>, Subscription)>,
    _update_toolchain_task: Task<Option<()>>,
}

impl ActiveToolchain {
    pub fn new(workspace: &Workspace, window: &mut Window, cx: &mut Context<Self>) -> Self {
        if let Some(store) = workspace.project().read(cx).toolchain_store() {
            cx.subscribe_in(
                &store,
                window,
                |this, _, _: &ToolchainStoreEvent, window, cx| {
                    let editor = this
                        .workspace
                        .update(cx, |workspace, cx| {
                            workspace
                                .active_item(cx)
                                .and_then(|item| item.downcast::<Editor>())
                        })
                        .ok()
                        .flatten();
                    if let Some(editor) = editor {
                        this.active_toolchain.take();
                        this.update_lister(editor, window, cx);
                    }
                },
            )
            .detach();
        }
        Self {
            active_toolchain: None,
            active_buffer: None,
            term: SharedString::new_static("Toolchain"),
            workspace: workspace.weak_handle(),

            _update_toolchain_task: Self::spawn_tracker_task(window, cx),
        }
    }
    fn spawn_tracker_task(window: &mut Window, cx: &mut Context<Self>) -> Task<Option<()>> {
        cx.spawn_in(window, async move |this, cx| {
            let active_file = this
                .read_with(cx, |this, _| {
                    this.active_buffer
                        .as_ref()
                        .map(|(_, buffer, _)| buffer.clone())
                })
                .ok()
                .flatten()?;
            let workspace = this.read_with(cx, |this, _| this.workspace.clone()).ok()?;
            let language_name = active_file
                .read_with(cx, |this, _| Some(this.language()?.name()))
                .ok()
                .flatten()?;
            let term = workspace
                .update(cx, |workspace, cx| {
                    let languages = workspace.project().read(cx).languages();
                    Project::toolchain_term(languages.clone(), language_name.clone())
                })
                .ok()?
                .await?;
            let _ = this.update(cx, |this, cx| {
                this.term = term;
                cx.notify();
            });
            let (worktree_id, path) = active_file
                .update(cx, |this, cx| {
                    this.file().and_then(|file| {
                        Some((
                            file.worktree_id(cx),
                            Arc::<Path>::from(file.path().parent()?),
                        ))
                    })
                })
                .ok()
                .flatten()?;
            let toolchain =
                Self::active_toolchain(workspace, worktree_id, path, language_name, cx).await?;
            let _ = this.update(cx, |this, cx| {
                this.active_toolchain = Some(toolchain);

                cx.notify();
            });
            Some(())
        })
    }

    fn update_lister(
        &mut self,
        editor: Entity<Editor>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let editor = editor.read(cx);
        if let Some((_, buffer, _)) = editor.active_excerpt(cx) {
            if let Some(worktree_id) = buffer.read(cx).file().map(|file| file.worktree_id(cx)) {
                let subscription = cx.subscribe_in(
                    &buffer,
                    window,
                    |this, _, event: &BufferEvent, window, cx| {
                        if matches!(event, BufferEvent::LanguageChanged) {
                            this._update_toolchain_task = Self::spawn_tracker_task(window, cx);
                        }
                    },
                );
                self.active_buffer = Some((worktree_id, buffer.downgrade(), subscription));
                self._update_toolchain_task = Self::spawn_tracker_task(window, cx);
            }
        }

        cx.notify();
    }

    fn active_toolchain(
        workspace: WeakEntity<Workspace>,
        worktree_id: WorktreeId,
        relative_path: Arc<Path>,
        language_name: LanguageName,
        cx: &mut AsyncWindowContext,
    ) -> Task<Option<Toolchain>> {
        cx.spawn(async move |cx| {
            let workspace_id = workspace
                .read_with(cx, |this, _| this.database_id())
                .ok()
                .flatten()?;
            let selected_toolchain = workspace
                .update(cx, |this, cx| {
                    this.project().read(cx).active_toolchain(
                        ProjectPath {
                            worktree_id,
                            path: relative_path.clone(),
                        },
                        language_name.clone(),
                        cx,
                    )
                })
                .ok()?
                .await;
            if let Some(toolchain) = selected_toolchain {
                Some(toolchain)
            } else {
                let project = workspace
                    .read_with(cx, |this, _| this.project().clone())
                    .ok()?;
                let (toolchains, relative_path) = cx
                    .update(|_, cx| {
                        project.read(cx).available_toolchains(
                            ProjectPath {
                                worktree_id,
                                path: relative_path.clone(),
                            },
                            language_name,
                            cx,
                        )
                    })
                    .ok()?
                    .await?;
                if let Some(toolchain) = toolchains.toolchains.first() {
                    // Since we don't have a selected toolchain, pick one for user here.
                    workspace::WORKSPACE_DB
                        .set_toolchain(
                            workspace_id,
                            worktree_id,
                            relative_path.to_string_lossy().into_owned(),
                            toolchain.clone(),
                        )
                        .await
                        .ok()?;
                    project
                        .update(cx, |this, cx| {
                            this.activate_toolchain(
                                ProjectPath {
                                    worktree_id,
                                    path: relative_path,
                                },
                                toolchain.clone(),
                                cx,
                            )
                        })
                        .ok()?
                        .await;
                }

                toolchains.toolchains.first().cloned()
            }
        })
    }
}

impl Render for ActiveToolchain {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div().when_some(self.active_toolchain.as_ref(), |el, active_toolchain| {
            let term = self.term.clone();
            el.child(
                Button::new("change-toolchain", active_toolchain.name.clone())
                    .label_size(LabelSize::Small)
                    .on_click(cx.listener(|this, _, window, cx| {
                        if let Some(workspace) = this.workspace.upgrade() {
                            workspace.update(cx, |workspace, cx| {
                                ToolchainSelector::toggle(workspace, window, cx)
                            });
                        }
                    }))
                    .tooltip(Tooltip::text(format!("Select {}", &term))),
            )
        })
    }
}

impl StatusItemView for ActiveToolchain {
    fn set_active_pane_item(
        &mut self,
        active_pane_item: Option<&dyn ItemHandle>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(editor) = active_pane_item.and_then(|item| item.downcast::<Editor>()) {
            self.active_toolchain.take();
            self.update_lister(editor, window, cx);
        }
        cx.notify();
    }
}
