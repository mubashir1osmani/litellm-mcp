#!/usr/bin/env python3
import os
import time
from typing import Optional

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from litellm_mcp.indexer import search as _search_docs, DEFAULT_INDEX_DIR

load_dotenv()

LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "").rstrip("/")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")
LITELLM_EMBEDDING_MODEL = os.environ.get("LITELLM_EMBEDDING_MODEL", "")
LITELLM_DOCS_INDEX_DIR = os.environ.get("LITELLM_DOCS_INDEX_DIR", DEFAULT_INDEX_DIR)

mcp = FastMCP(
    "LiteLLM",
    instructions="Manage your self-hosted LiteLLM proxy — users, teams, projects, keys, spend, and budgets.",
)

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _headers() -> dict:
    return {"Authorization": f"Bearer {LITELLM_API_KEY}"}

def _clean(d: dict | None) -> dict:
    """Strip None values so we don't send nulls to the API."""
    return {k: v for k, v in (d or {}).items() if v is not None}

async def _get(path: str, params: dict | None = None) -> dict:
    async with httpx.AsyncClient(
        base_url=LITELLM_BASE_URL, headers=_headers(), timeout=30
    ) as c:
        r = await c.get(path, params=_clean(params))
        _raise(r)
        return r.json()

async def _post(path: str, body: dict | None = None) -> dict:
    async with httpx.AsyncClient(
        base_url=LITELLM_BASE_URL, headers=_headers(), timeout=30
    ) as c:
        r = await c.post(path, json=_clean(body))
        _raise(r)
        return r.json()

async def _delete(path: str, params: dict | None = None) -> dict:
    async with httpx.AsyncClient(
        base_url=LITELLM_BASE_URL, headers=_headers(), timeout=30
    ) as c:
        r = await c.delete(path, params=_clean(params))
        _raise(r)
        return r.json()

def _raise(r: httpx.Response) -> None:
    if r.is_error:
        raise ValueError(f"LiteLLM API {r.status_code}: {r.text}")


# ── Users ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def create_user(
    user_email: Optional[str] = None,
    user_alias: Optional[str] = None,
    user_id: Optional[str] = None,
    user_role: Optional[str] = None,
    team_id: Optional[str] = None,
    max_budget: Optional[float] = None,
    budget_duration: Optional[str] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    models: Optional[list[str]] = None,
    auto_create_key: Optional[bool] = None,
    send_invite_email: Optional[bool] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """Create a new user on the LiteLLM proxy."""
    return await _post("/user/new", locals())


@mcp.tool()
async def list_users(
    role: Optional[str] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
) -> dict:
    """List all users. Optionally filter by role (proxy_admin, internal_user, etc.)."""
    return await _get("/user/list", {"role": role, "page": page, "page_size": page_size})


@mcp.tool()
async def get_user(user_id: str) -> dict:
    """Get details for a specific user by user_id."""
    return await _get("/user/info", {"user_id": user_id})


@mcp.tool()
async def delete_user(user_ids: list[str]) -> dict:
    """Delete one or more users by their user_ids."""
    return await _post("/user/delete", {"user_ids": user_ids})


@mcp.tool()
async def get_user_daily_activity(
    user_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Get daily spend/request activity for a user. Dates in YYYY-MM-DD format."""
    return await _get(
        "/user/daily/activity",
        {"user_id": user_id, "start_date": start_date, "end_date": end_date},
    )


# ── Teams ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def create_team(
    team_alias: Optional[str] = None,
    team_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    admins: Optional[list[str]] = None,
    members: Optional[list[str]] = None,
    max_budget: Optional[float] = None,
    budget_duration: Optional[str] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    models: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
    blocked: Optional[bool] = None,
) -> dict:
    """Create a new team on the LiteLLM proxy."""
    return await _post("/team/new", locals())


@mcp.tool()
async def list_teams(
    organization_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """List all teams. Optionally filter by organization or user."""
    return await _get("/team/list", {"organization_id": organization_id, "user_id": user_id})


@mcp.tool()
async def get_team(team_id: str) -> dict:
    """Get details for a specific team."""
    return await _get("/team/info", {"team_id": team_id})


@mcp.tool()
async def update_team(
    team_id: str,
    team_alias: Optional[str] = None,
    max_budget: Optional[float] = None,
    budget_duration: Optional[str] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    models: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
    blocked: Optional[bool] = None,
) -> dict:
    """Update an existing team's settings."""
    return await _post("/team/update", locals())


@mcp.tool()
async def delete_team(team_ids: list[str]) -> dict:
    """Delete one or more teams by their team_ids."""
    return await _post("/team/delete", {"team_ids": team_ids})


@mcp.tool()
async def add_team_member(
    team_id: str,
    member: dict,
) -> dict:
    """Add a member to a team. member should be {"role": "user"|"admin", "user_id": "...", "user_email": "..."}."""
    return await _post("/team/member_add", {"team_id": team_id, "member": member})


@mcp.tool()
async def remove_team_member(
    team_id: str,
    user_id: Optional[str] = None,
    user_email: Optional[str] = None,
) -> dict:
    """Remove a member from a team by user_id or user_email."""
    return await _post(
        "/team/member_delete",
        {"team_id": team_id, "user_id": user_id, "user_email": user_email},
    )


@mcp.tool()
async def get_team_daily_activity(
    team_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Get daily spend/request activity for a team. Dates in YYYY-MM-DD format."""
    return await _get(
        "/team/daily/activity",
        {"team_id": team_id, "start_date": start_date, "end_date": end_date},
    )


# ── Projects ──────────────────────────────────────────────────────────────────

@mcp.tool()
async def create_project(
    team_id: str,
    project_alias: Optional[str] = None,
    project_id: Optional[str] = None,
    description: Optional[str] = None,
    max_budget: Optional[float] = None,
    budget_duration: Optional[str] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    models: Optional[list[str]] = None,
    soft_budget: Optional[float] = None,
    blocked: Optional[bool] = None,
    metadata: Optional[dict] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    """Create a new project. Projects belong to a team (team_id required)."""
    return await _post("/project/new", locals())


@mcp.tool()
async def list_projects() -> dict:
    """List all projects on the proxy."""
    return await _get("/project/list")


@mcp.tool()
async def get_project(project_id: str) -> dict:
    """Get details for a specific project."""
    return await _get("/project/info", {"project_id": project_id})


@mcp.tool()
async def update_project(
    project_id: str,
    project_alias: Optional[str] = None,
    description: Optional[str] = None,
    max_budget: Optional[float] = None,
    budget_duration: Optional[str] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    models: Optional[list[str]] = None,
    blocked: Optional[bool] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """Update an existing project's settings."""
    return await _post("/project/update", locals())


@mcp.tool()
async def delete_project(project_id: str) -> dict:
    """Delete a project by project_id."""
    return await _delete("/project/delete", {"project_id": project_id})


# ── Keys ──────────────────────────────────────────────────────────────────────

@mcp.tool()
async def generate_key(
    key_alias: Optional[str] = None,
    user_id: Optional[str] = None,
    team_id: Optional[str] = None,
    project_id: Optional[str] = None,
    duration: Optional[str] = None,
    max_budget: Optional[float] = None,
    budget_duration: Optional[str] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    models: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
    tags: Optional[list[str]] = None,
    key_type: Optional[str] = None,
) -> dict:
    """Generate a new API key. Returns the key — save it, it won't be shown again."""
    return await _post("/key/generate", locals())


@mcp.tool()
async def list_keys(
    user_id: Optional[str] = None,
    team_id: Optional[str] = None,
    key_alias: Optional[str] = None,
    page: Optional[int] = None,
    size: Optional[int] = None,
) -> dict:
    """List API keys with optional filters."""
    return await _get(
        "/key/list",
        {"user_id": user_id, "team_id": team_id, "key_alias": key_alias, "page": page, "size": size},
    )


@mcp.tool()
async def get_key(key: str) -> dict:
    """Get info for a specific API key."""
    return await _get("/key/info", {"key": key})


@mcp.tool()
async def delete_key(keys: list[str]) -> dict:
    """Delete one or more API keys."""
    return await _post("/key/delete", {"keys": keys})


@mcp.tool()
async def block_key(key: str) -> dict:
    """Block an API key so it can no longer make requests."""
    return await _post("/key/block", {"key": key})


@mcp.tool()
async def unblock_key(key: str) -> dict:
    """Unblock a previously blocked API key."""
    return await _post("/key/unblock", {"key": key})


# ── Spend / Usage ─────────────────────────────────────────────────────────────

@mcp.tool()
async def get_spend_logs(
    api_key: Optional[str] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    summarize: Optional[bool] = None,
) -> dict:
    """View spend logs. Filter by key, user, date range (YYYY-MM-DD), or request ID."""
    return await _get(
        "/spend/logs",
        {
            "api_key": api_key,
            "user_id": user_id,
            "request_id": request_id,
            "start_date": start_date,
            "end_date": end_date,
            "summarize": summarize,
        },
    )


@mcp.tool()
async def get_global_spend_report(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    group_by: Optional[str] = None,
    api_key: Optional[str] = None,
    internal_user_id: Optional[str] = None,
    team_id: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> dict:
    """Get aggregated global spend report. group_by can be 'team', 'user', 'key', 'model', etc."""
    return await _get(
        "/global/spend/report",
        {
            "start_date": start_date,
            "end_date": end_date,
            "group_by": group_by,
            "api_key": api_key,
            "internal_user_id": internal_user_id,
            "team_id": team_id,
            "customer_id": customer_id,
        },
    )


@mcp.tool()
async def calculate_spend(
    model: str,
    messages: Optional[list[dict]] = None,
    completion: Optional[str] = None,
) -> dict:
    """Calculate the cost of a hypothetical request without making it."""
    return await _post("/spend/calculate", {"model": model, "messages": messages, "completion": completion})


# ── Budgets ───────────────────────────────────────────────────────────────────

@mcp.tool()
async def create_budget(
    max_budget: Optional[float] = None,
    soft_budget: Optional[float] = None,
    budget_duration: Optional[str] = None,
    max_parallel_requests: Optional[int] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
) -> dict:
    """Create a reusable budget object that can be attached to users/teams/projects."""
    return await _post("/budget/new", locals())


@mcp.tool()
async def list_budgets() -> dict:
    """List all budget objects."""
    return await _get("/budget/list")


@mcp.tool()
async def get_budget(budget_id: str) -> dict:
    """Get details for a specific budget."""
    return await _post("/budget/info", {"budget_id": budget_id})


@mcp.tool()
async def update_budget(
    budget_id: str,
    max_budget: Optional[float] = None,
    soft_budget: Optional[float] = None,
    budget_duration: Optional[str] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
) -> dict:
    """Update an existing budget."""
    return await _post("/budget/update", locals())


@mcp.tool()
async def delete_budget(budget_id: str) -> dict:
    """Delete a budget object."""
    return await _post("/budget/delete", {"id": budget_id})


# ── Organizations ─────────────────────────────────────────────────────────────

@mcp.tool()
async def create_organization(
    organization_alias: str,
    organization_id: Optional[str] = None,
    max_budget: Optional[float] = None,
    budget_duration: Optional[str] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    models: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """Create a new organization (top-level grouping above teams)."""
    return await _post("/organization/new", locals())


@mcp.tool()
async def list_organizations() -> dict:
    """List all organizations."""
    return await _get("/organization/list")


@mcp.tool()
async def get_organization(organization_id: str) -> dict:
    """Get details for a specific organization."""
    return await _get("/organization/info", {"organization_id": organization_id})


@mcp.tool()
async def delete_organization(organization_id: str) -> dict:
    """Delete an organization."""
    return await _delete("/organization/delete", {"organization_id": organization_id})


@mcp.tool()
async def add_organization_member(
    organization_id: str,
    member: dict,
) -> dict:
    """Add a member to an organization. member: {"role": "org_admin"|"internal_user", "user_id": "...", "user_email": "..."}."""
    return await _post("/organization/member_add", {"organization_id": organization_id, "member": member})


# ── Models ───────────────────────────────────────────────────────────────────

@mcp.tool()
async def list_models(
    team_id: Optional[str] = None,
    scope: Optional[str] = None,
    include_metadata: Optional[bool] = None,
) -> dict:
    """List models visible to callers via /v1/models. Optionally filter by team_id or scope."""
    return await _get(
        "/v1/models",
        {"team_id": team_id, "scope": scope, "include_metadata": include_metadata},
    )


@mcp.tool()
async def list_model_deployments(litellm_model_id: Optional[str] = None) -> dict:
    """List internal model deployment config (admin view). Optionally filter by litellm_model_id."""
    return await _get("/model/info", {"litellm_model_id": litellm_model_id})


@mcp.tool()
async def list_model_groups() -> dict:
    """List model groups (public-facing model names and their underlying deployments)."""
    return await _get("/model_group/info")


@mcp.tool()
async def add_model(
    model_name: str,
    litellm_model: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider: Optional[str] = None,
    rpm: Optional[int] = None,
    tpm: Optional[int] = None,
    timeout: Optional[float] = None,
    organization: Optional[str] = None,
    region_name: Optional[str] = None,
    input_cost_per_token: Optional[float] = None,
    output_cost_per_token: Optional[float] = None,
    model_info: Optional[dict] = None,
) -> dict:
    """Add a new model deployment to the proxy.

    - model_name: public name callers use (e.g. "gpt-4o")
    - litellm_model: underlying model string (e.g. "openai/gpt-4o", "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
    - api_base: for custom / self-hosted endpoints
    """
    litellm_params: dict = {"model": litellm_model}
    for k, v in {
        "api_base": api_base,
        "api_key": api_key,
        "api_version": api_version,
        "custom_llm_provider": custom_llm_provider,
        "rpm": rpm,
        "tpm": tpm,
        "timeout": timeout,
        "organization": organization,
        "region_name": region_name,
        "input_cost_per_token": input_cost_per_token,
        "output_cost_per_token": output_cost_per_token,
    }.items():
        if v is not None:
            litellm_params[k] = v

    return await _post(
        "/model/new",
        {
            "model_name": model_name,
            "litellm_params": litellm_params,
            "model_info": model_info or {},
        },
    )


@mcp.tool()
async def update_model(
    litellm_model_id: str,
    model_name: Optional[str] = None,
    litellm_model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    rpm: Optional[int] = None,
    tpm: Optional[int] = None,
    model_info: Optional[dict] = None,
) -> dict:
    """Update an existing model deployment. litellm_model_id is the internal ID from list_models."""
    body: dict = {}
    if model_name:
        body["model_name"] = model_name
    litellm_params: dict = {}
    for k, v in {"model": litellm_model, "api_base": api_base, "api_key": api_key, "rpm": rpm, "tpm": tpm}.items():
        if v is not None:
            litellm_params[k] = v
    if litellm_params:
        body["litellm_params"] = litellm_params
    if model_info:
        body["model_info"] = model_info
    return await _post(f"/model/{litellm_model_id}/update", body)


@mcp.tool()
async def delete_model(litellm_model_id: str) -> dict:
    """Delete a model deployment by its internal litellm_model_id."""
    return await _post("/model/delete", {"id": litellm_model_id})


# ── Docs search ──────────────────────────────────────────────────────────────

@mcp.tool()
async def search_docs(
    query: str,
    n_results: int = 5,
) -> list[dict]:
    """Semantic search over the LiteLLM documentation.

    Returns the most relevant doc sections with their title, URL, similarity
    score (0–1), and a content excerpt.

    Requires LITELLM_EMBEDDING_MODEL to be set and `index-docs` to have been
    run at least once to build the local vector index.
    """
    if not LITELLM_EMBEDDING_MODEL:
        raise ValueError(
            "LITELLM_EMBEDDING_MODEL is not set. "
            "Set it to an embedding model available on your proxy (e.g. text-embedding-3-small)."
        )
    return await _search_docs(
        query=query,
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
        embedding_model=LITELLM_EMBEDDING_MODEL,
        index_dir=LITELLM_DOCS_INDEX_DIR,
        n_results=n_results,
    )


# ── Changelog resource ────────────────────────────────────────────────────────

_changelog_cache: dict = {"data": None, "ts": 0.0}
_CACHE_TTL = 300  # 5 minutes


@mcp.resource("litellm://changelog")
async def get_changelog() -> str:
    """Latest LiteLLM releases fetched from GitHub."""
    now = time.time()
    if _changelog_cache["data"] and now - _changelog_cache["ts"] < _CACHE_TTL:
        return _changelog_cache["data"]

    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            "https://api.github.com/repos/BerriAI/litellm/releases",
            params={"per_page": 10},
            headers={"Accept": "application/vnd.github+json"},
        )
        r.raise_for_status()
        releases = r.json()

    lines = []
    for rel in releases:
        lines.append(f"## {rel['tag_name']} — {rel['published_at'][:10]}")
        lines.append(f"**{rel['name']}**")
        body = (rel.get("body") or "").strip()
        if body:
            lines.append(body[:2000])
        lines.append("")

    result = "\n".join(lines)
    _changelog_cache["data"] = result
    _changelog_cache["ts"] = now
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if not LITELLM_BASE_URL:
        raise RuntimeError("LITELLM_BASE_URL is not set")
    if not LITELLM_API_KEY:
        raise RuntimeError("LITELLM_API_KEY is not set")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
