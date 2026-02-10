use std::sync::Arc;

use axum::http::request::Parts;
use rmcp::{
    handler::server::{
        router::{prompt::PromptRouter, tool::ToolRouter},
        wrapper::Parameters,
    },
    model::*,
    prompt, prompt_handler, prompt_router,
    service::RequestContext,
    tool, tool_handler, tool_router,
    transport::{
        streamable_http_server::{
            session::local::LocalSessionManager, tower::StreamableHttpService,
        },
        StreamableHttpServerConfig,
    },
    ErrorData as McpError, RoleServer, ServerHandler,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::RwLock;

use crate::{
    api::AppState,
    mcp::auth::{auth_context_from_parts, McpAuthContext},
    models::{ForgetMemoryRequest, GetProfileRequest, MemoryType, SearchMemoriesRequest},
};

const PROFILE_URI: &str = "supermemory://profile";
const PROJECTS_URI: &str = "supermemory://projects";

#[derive(Debug, Clone)]
struct ClientDescriptor {
    name: String,
    version: String,
}

#[derive(Clone)]
pub struct MomoMcpServer {
    state: AppState,
    tool_router: ToolRouter<Self>,
    prompt_router: PromptRouter<Self>,
    client_info: Arc<RwLock<Option<ClientDescriptor>>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct MemoryArgs {
    content: String,
    #[serde(default)]
    action: MemoryAction,
    #[serde(default)]
    container_tag: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "lowercase")]
enum MemoryAction {
    #[default]
    Save,
    Forget,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct RecallArgs {
    query: String,
    #[serde(default = "default_true")]
    include_profile: bool,
    #[serde(default)]
    container_tag: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct ListProjectsArgs {
    #[serde(default = "default_true")]
    refresh: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct WhoAmIArgs {}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct ContextPromptArgs {
    #[serde(default)]
    container_tag: Option<String>,
    #[serde(default = "default_true")]
    include_recent: bool,
}

fn default_true() -> bool {
    true
}

impl MomoMcpServer {
    pub fn new(state: AppState) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
            prompt_router: Self::prompt_router(),
            client_info: Arc::new(RwLock::new(None)),
        }
    }

    fn request_parts<'a>(&self, ctx: &'a RequestContext<RoleServer>) -> Option<&'a Parts> {
        ctx.extensions.get::<Parts>()
    }

    fn auth_context(&self, ctx: &RequestContext<RoleServer>) -> Option<McpAuthContext> {
        self.request_parts(ctx).and_then(auth_context_from_parts)
    }

    fn session_id(&self, ctx: &RequestContext<RoleServer>) -> String {
        self.request_parts(ctx)
            .and_then(|parts| {
                parts
                    .headers
                    .get("Mcp-Session-Id")
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "unknown".to_string())
    }

    fn resolve_container_tag(
        &self,
        container_tag: Option<&str>,
        ctx: &RequestContext<RoleServer>,
    ) -> Result<String, McpError> {
        let resolved = container_tag
            .map(str::trim)
            .filter(|tag| !tag.is_empty())
            .map(str::to_string)
            .or_else(|| {
                self.auth_context(ctx)
                    .and_then(|auth| auth.container_tag)
                    .map(|tag| tag.trim().to_string())
                    .filter(|tag| !tag.is_empty())
            })
            .unwrap_or_else(|| self.state.config.mcp.default_container_tag.clone());

        validate_container_tag(&resolved)?;
        Ok(resolved)
    }

    fn as_internal_error(message: &'static str, error: impl std::fmt::Display) -> McpError {
        tracing::error!(error = %error, "{message}");
        McpError::internal_error(message, None)
    }

    fn format_memory_section(results: &[crate::models::HybridSearchResult]) -> Vec<String> {
        let mut parts = Vec::new();

        if results.is_empty() {
            return parts;
        }

        parts.push("\n## Relevant Memories".to_string());
        for (index, item) in results.iter().enumerate() {
            let score = (item.similarity * 100.0).round() as i32;
            parts.push(format!("\n### Memory {} ({}% match)", index + 1, score));
            let content = item
                .memory
                .as_deref()
                .or(item.chunk.as_deref())
                .unwrap_or_default();
            parts.push(content.to_string());
        }

        parts
    }

    async fn profile_text_resource(
        &self,
        ctx: &RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        let container_tag = self.resolve_container_tag(None, ctx)?;
        let profile = self
            .state
            .db
            .get_user_profile(&container_tag, true, 50)
            .await
            .map_err(|error| Self::as_internal_error("Failed to fetch profile resource", error))?;

        let mut parts = vec![format!("# User Profile ({container_tag})\n")];

        if !profile.static_facts.is_empty() {
            parts.push("## Stable Preferences".to_string());
            for fact in profile.static_facts {
                parts.push(format!("- {}", fact.memory));
            }
        }

        if !profile.dynamic_facts.is_empty() {
            parts.push("\n## Recent Activity".to_string());
            for fact in profile.dynamic_facts {
                parts.push(format!("- {}", fact.memory));
            }
        }

        let text = if parts.len() > 1 {
            parts.join("\n")
        } else {
            "No profile yet. Start saving memories.".to_string()
        };

        Ok(ReadResourceResult {
            contents: vec![ResourceContents::text(text, PROFILE_URI)],
        })
    }

    async fn projects_resource(&self) -> Result<ReadResourceResult, McpError> {
        let mut projects = self
            .state
            .db
            .get_active_container_tags()
            .await
            .map_err(|error| Self::as_internal_error("Failed to fetch projects", error))?;

        projects.sort();
        projects.dedup();

        Ok(ReadResourceResult {
            contents: vec![ResourceContents::text(
                json!({ "projects": projects }).to_string(),
                PROJECTS_URI,
            )],
        })
    }
}

#[tool_router]
impl MomoMcpServer {
    #[tool(
        name = "memory",
        description = "Save or forget information about the user."
    )]
    async fn memory_tool(
        &self,
        Parameters(args): Parameters<MemoryArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let content = args.content.trim();
        if content.is_empty() {
            return Err(McpError::invalid_params("content cannot be empty", None));
        }
        if content.len() > 200_000 {
            return Err(McpError::invalid_params(
                "content exceeds maximum length of 200,000 characters",
                None,
            ));
        }

        let container_tag = self.resolve_container_tag(args.container_tag.as_deref(), &ctx)?;

        match args.action {
            MemoryAction::Save => {
                let created = self
                    .state
                    .memory
                    .create_memory_with_type(content, &container_tag, false, MemoryType::Fact)
                    .await
                    .map_err(|error| Self::as_internal_error("Failed to save memory", error))?;

                Ok(CallToolResult::success(vec![Content::text(format!(
                    "Saved memory (id: {}) in {} project",
                    created.id, container_tag
                ))]))
            }
            MemoryAction::Forget => {
                let search = self
                    .state
                    .search
                    .search_memories(SearchMemoriesRequest {
                        q: content.to_string(),
                        container_tag: Some(container_tag.clone()),
                        threshold: None,
                        filters: None,
                        include: None,
                        limit: Some(5),
                        rerank: None,
                        rewrite_query: None,
                    })
                    .await
                    .map_err(|error| Self::as_internal_error("Failed to search memories", error))?;

                let Some(candidate) = search.results.first() else {
                    return Ok(CallToolResult::success(vec![Content::text(
                        "No matching memory found to forget.",
                    )]));
                };

                self.state
                    .memory
                    .forget_memory(ForgetMemoryRequest {
                        id: Some(candidate.id.clone()),
                        content: None,
                        container_tag: container_tag.clone(),
                        reason: Some("Forgotten via MCP memory tool".to_string()),
                    })
                    .await
                    .map_err(|error| Self::as_internal_error("Failed to forget memory", error))?;

                let snippet = candidate
                    .memory
                    .as_deref()
                    .map(|memory| {
                        if memory.len() > 100 {
                            format!("{}...", &memory[..100])
                        } else {
                            memory.to_string()
                        }
                    })
                    .unwrap_or_else(|| candidate.id.clone());

                Ok(CallToolResult::success(vec![Content::text(format!(
                    "Forgot: \"{snippet}\" in container {container_tag}",
                ))]))
            }
        }
    }

    #[tool(
        name = "recall",
        description = "Search the user's memories and optionally include profile context."
    )]
    async fn recall_tool(
        &self,
        Parameters(args): Parameters<RecallArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let query = args.query.trim();
        if query.is_empty() {
            return Err(McpError::invalid_params("query cannot be empty", None));
        }
        if query.len() > 1_000 {
            return Err(McpError::invalid_params(
                "query exceeds maximum length of 1,000 characters",
                None,
            ));
        }

        let container_tag = self.resolve_container_tag(args.container_tag.as_deref(), &ctx)?;

        if args.include_profile {
            let profile_result = self
                .state
                .memory
                .get_profile(
                    GetProfileRequest {
                        container_tag,
                        q: Some(query.to_string()),
                        threshold: None,
                        include_dynamic: Some(true),
                        limit: Some(10),
                        compact: None,
                        generate_narrative: Some(false),
                    },
                    &self.state.search,
                )
                .await
                .map_err(|error| Self::as_internal_error("Failed to fetch profile", error))?;

            let mut parts = Vec::new();

            if !profile_result.profile.static_facts.is_empty()
                || !profile_result.profile.dynamic_facts.is_empty()
            {
                parts.push("## User Profile".to_string());

                if !profile_result.profile.static_facts.is_empty() {
                    parts.push("**Stable facts:**".to_string());
                    for fact in profile_result.profile.static_facts {
                        parts.push(format!("- {fact}"));
                    }
                }

                if !profile_result.profile.dynamic_facts.is_empty() {
                    parts.push("\n**Recent context:**".to_string());
                    for fact in profile_result.profile.dynamic_facts {
                        parts.push(format!("- {fact}"));
                    }
                }
            }

            if let Some(search_results) = profile_result.search_results {
                parts.extend(Self::format_memory_section(&search_results.results));
            }

            let output = if parts.is_empty() {
                "No memories or profile found.".to_string()
            } else {
                parts.join("\n")
            };

            return Ok(CallToolResult::success(vec![Content::text(output)]));
        }

        let search = self
            .state
            .search
            .search_memories(SearchMemoriesRequest {
                q: query.to_string(),
                container_tag: Some(container_tag),
                threshold: None,
                filters: None,
                include: None,
                limit: Some(10),
                rerank: None,
                rewrite_query: None,
            })
            .await
            .map_err(|error| Self::as_internal_error("Failed to search memories", error))?;

        if search.results.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No memories found.",
            )]));
        }

        let mut parts = vec!["## Relevant Memories".to_string()];
        for (index, memory) in search.results.iter().enumerate() {
            let score = (memory.similarity * 100.0).round() as i32;
            parts.push(format!("\n### Memory {} ({}% match)", index + 1, score));
            parts.push(memory.memory.clone().unwrap_or_default());
        }

        Ok(CallToolResult::success(vec![Content::text(
            parts.join("\n"),
        )]))
    }

    #[tool(
        name = "listProjects",
        description = "List available memory projects (container tags)."
    )]
    async fn list_projects_tool(
        &self,
        Parameters(args): Parameters<ListProjectsArgs>,
    ) -> Result<CallToolResult, McpError> {
        let _ = args.refresh;

        let mut projects = self
            .state
            .db
            .get_active_container_tags()
            .await
            .map_err(|error| Self::as_internal_error("Failed to list projects", error))?;

        projects.sort();
        projects.dedup();

        if projects.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No projects found. Memories will use the default project.",
            )]));
        }

        let project_text = format!(
            "Available projects:\n{}",
            projects
                .into_iter()
                .map(|project| format!("- {project}"))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(CallToolResult::success(vec![Content::text(project_text)]))
    }

    #[tool(
        name = "whoAmI",
        description = "Get the current authenticated user details."
    )]
    async fn who_am_i_tool(
        &self,
        Parameters(_args): Parameters<WhoAmIArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let auth = self.auth_context(&ctx);
        let client_info = self.client_info.read().await.clone();

        let payload = json!({
            "userId": auth.as_ref().map(|ctx| ctx.user_id.clone()).unwrap_or_else(|| "anonymous".to_string()),
            "email": auth.as_ref().and_then(|ctx| ctx.email.clone()),
            "name": auth.as_ref().and_then(|ctx| ctx.name.clone()),
            "client": client_info.map(|info| {
                json!({
                    "name": info.name,
                    "version": info.version,
                })
            }),
            "sessionId": self.session_id(&ctx),
        });

        Ok(CallToolResult::success(vec![Content::text(
            payload.to_string(),
        )]))
    }
}

#[prompt_router]
impl MomoMcpServer {
    #[prompt(
        name = "context",
        description = "Inject user profile and preferences as conversation context."
    )]
    async fn context_prompt(
        &self,
        Parameters(args): Parameters<ContextPromptArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<GetPromptResult, McpError> {
        let container_tag = self.resolve_container_tag(args.container_tag.as_deref(), &ctx)?;

        let profile = self
            .state
            .db
            .get_user_profile(&container_tag, args.include_recent, 50)
            .await
            .map_err(|error| Self::as_internal_error("Failed to build context prompt", error))?;

        let mut parts = vec![
            "**Important:** Whenever the user shares informative facts, preferences, personal details, or any memory-worthy information, use the `memory` tool to save it to Momo. This helps maintain context across conversations.".to_string(),
            "".to_string(),
        ];

        if !profile.static_facts.is_empty()
            || (args.include_recent && !profile.dynamic_facts.is_empty())
        {
            parts.push("## User Context".to_string());
        }

        if !profile.static_facts.is_empty() {
            parts.push("**Stable Preferences:**".to_string());
            for fact in profile.static_facts {
                parts.push(format!("- {}", fact.memory));
            }
        }

        if args.include_recent && !profile.dynamic_facts.is_empty() {
            parts.push("\n**Recent Activity:**".to_string());
            for fact in profile.dynamic_facts {
                parts.push(format!("- {}", fact.memory));
            }
        }

        let context_text = if parts.len() > 2 {
            parts.join("\n")
        } else {
            "**Important:** Whenever the user shares informative facts, preferences, personal details, or any memory-worthy information, use the `memory` tool to save it to Momo. This helps maintain context across conversations.\n\nNo user profile available yet. Start saving memories to build context.".to_string()
        };

        Ok(GetPromptResult {
            description: Some("User profile and memory context".to_string()),
            messages: vec![PromptMessage::new_text(
                PromptMessageRole::User,
                context_text,
            )],
        })
    }
}

#[tool_handler]
#[prompt_handler]
impl ServerHandler for MomoMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .enable_prompts()
                .build(),
            server_info: Implementation {
                name: "momo".to_string(),
                title: Some("Momo MCP".to_string()),
                version: env!("CARGO_PKG_VERSION").to_string(),
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "Use the memory tool to save/forget user context and recall to retrieve relevant memories."
                    .to_string(),
            ),
        }
    }

    async fn initialize(
        &self,
        request: InitializeRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<InitializeResult, McpError> {
        let mut client_info = self.client_info.write().await;
        *client_info = Some(ClientDescriptor {
            name: request.client_info.name,
            version: request.client_info.version,
        });

        Ok(self.get_info())
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParam>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        Ok(ListResourcesResult::with_all_items(vec![
            RawResource::new(PROFILE_URI, "User Profile").no_annotation(),
            RawResource::new(PROJECTS_URI, "Projects").no_annotation(),
        ]))
    }

    async fn read_resource(
        &self,
        ReadResourceRequestParam { uri, .. }: ReadResourceRequestParam,
        context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        match uri.as_str() {
            PROFILE_URI => self.profile_text_resource(&context).await,
            PROJECTS_URI => self.projects_resource().await,
            _ => Err(McpError::resource_not_found(
                "resource_not_found",
                Some(json!({ "uri": uri })),
            )),
        }
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParam>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        Ok(ListResourceTemplatesResult::with_all_items(Vec::new()))
    }
}

pub fn streamable_http_service(
    state: AppState,
) -> StreamableHttpService<MomoMcpServer, LocalSessionManager> {
    StreamableHttpService::new(
        move || Ok(MomoMcpServer::new(state.clone())),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    )
}

fn validate_container_tag(container_tag: &str) -> Result<(), McpError> {
    if container_tag.is_empty() {
        return Err(McpError::invalid_params(
            "containerTag cannot be empty",
            None,
        ));
    }

    if container_tag.len() > 128 {
        return Err(McpError::invalid_params(
            "containerTag exceeds maximum length of 128 characters",
            None,
        ));
    }

    Ok(())
}
