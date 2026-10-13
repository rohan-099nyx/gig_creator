"""
product_gig_api/main.py
-----------------------
FastAPI service that:
  1. Validates the incoming X-Action-Key header.
  2. Accepts a product-gig JSON payload.
  3. Commits one uniquely-named JSON file to GitHub under
     product-gigs/YYYY/MM/<slug>-<uuid>.json
  4. Returns the file path and its public GitHub URL.

Required environment variables
-------------------------------
ACTION_API_KEY  – long random secret shared with the Custom GPT action
GITHUB_TOKEN    – fine-grained PAT with Contents: Read & Write for the repo
GITHUB_REPO     – owner/repo, e.g. rohan-099nyx/self_study
GITHUB_BRANCH   – branch to commit to, e.g. main
GITHUB_DIR      – directory prefix, e.g. product-gigs
"""

from __future__ import annotations

import base64
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Product Gig GitHub API",
    version="1.0.0",
    description="Saves Custom-GPT-generated product gigs as JSON files in a GitHub repository.",
)

# ---------------------------------------------------------------------------
# Config (from environment)
# ---------------------------------------------------------------------------
ACTION_API_KEY: str = os.environ.get("ACTION_API_KEY", "")
GITHUB_TOKEN: str = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO: str = os.environ.get("GITHUB_REPO", "")
GITHUB_BRANCH: str = os.environ.get("GITHUB_BRANCH", "main")
GITHUB_DIR: str = os.environ.get("GITHUB_DIR", "product-gigs")

GITHUB_API = "https://api.github.com"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class SocialCopy(BaseModel):
    instagram: str | None = None
    facebook: str | None = None
    whatsapp: str | None = None


class EmailCopy(BaseModel):
    subject: str | None = None
    body: str | None = None


class ProductGigPayload(BaseModel):
    model_config = {"populate_by_name": True, "extra": "allow"}

    product_name: str = Field(..., description="Short product name")
    category: str | None = Field(None, description="Product category")

    # GPT sometimes sends short_title instead of sales_title — accept both
    sales_title: str | None = Field(None, alias="sales_title", description="Concise sales title")
    short_title: str | None = Field(None, description="Alias for sales_title used by some GPT versions")

    description: str = Field(..., description="Marketing listing description")

    # price can arrive as a string ("₹289") or a bare number (128)
    price: str | int | float | None = Field(None, description="Price – string or number")
    currency: str | None = Field(None, description="Currency code e.g. INR, USD")

    highlights: list[str] = Field(default_factory=list)
    specifications: dict[str, Any] = Field(default_factory=dict)
    image_links: list[str] = Field(default_factory=list)
    image_notes: list[str] = Field(default_factory=list, description="Notes about missing image URLs")
    tags: list[str] = Field(default_factory=list)

    # Nested social/email objects
    social_copy: SocialCopy | None = None
    email_copy: EmailCopy | None = None

    # Flat email fields the GPT sometimes sends instead of nested email_copy
    email_subject: str | None = Field(None, description="Flat email subject (merged into email_copy)")
    email_body: str | None = Field(None, description="Flat email body (merged into email_copy)")

    source_notes: list[str] = Field(default_factory=list)

    def resolved_sales_title(self) -> str:
        """Return whichever title field is populated, with a safe fallback."""
        return self.sales_title or self.short_title or self.product_name

    def resolved_price(self) -> str | None:
        """Return price as a clean string, combining currency if separate."""
        if self.price is None:
            return None
        price_str = str(self.price)
        if self.currency and not any(c in price_str for c in ["₹", "$", "€", "£", self.currency]):
            return f"{price_str} {self.currency}"
        return price_str

    def resolved_email_copy(self) -> dict | None:
        """Merge flat subject/body into email_copy structure."""
        subj = (self.email_copy and self.email_copy.subject) or self.email_subject
        body = (self.email_copy and self.email_copy.body) or self.email_body
        if subj or body:
            return {"subject": subj, "body": body}
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_key(x_action_key: str) -> None:
    """Raise 401 if the key is missing or wrong."""
    if not ACTION_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfiguration: ACTION_API_KEY not set.",
        )
    if x_action_key != ACTION_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Action-Key.",
        )


def _slug(text: str, max_len: int = 50) -> str:
    """Turn arbitrary text into a URL-safe lowercase slug."""
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug[:max_len]


def _github_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _commit_file(path: str, content_bytes: bytes, message: str) -> dict[str, str]:
    """
    Create or update a file in the GitHub repository.
    Returns {"sha": ..., "html_url": ..., "download_url": ...}
    """
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{path}"
    encoded = base64.b64encode(content_bytes).decode()

    # Check if file already exists (to get its SHA for updates)
    existing_sha: str | None = None
    with httpx.Client(timeout=15) as client:
        r = client.get(url, headers=_github_headers(), params={"ref": GITHUB_BRANCH})
        if r.status_code == 200:
            existing_sha = r.json().get("sha")

        payload: dict[str, Any] = {
            "message": message,
            "content": encoded,
            "branch": GITHUB_BRANCH,
        }
        if existing_sha:
            payload["sha"] = existing_sha

        r2 = client.put(url, headers=_github_headers(), json=payload)
        if r2.status_code not in (200, 201):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"GitHub API error {r2.status_code}: {r2.text[:400]}",
            )
        data = r2.json()
        file_info = data.get("content", {})
        return {
            "sha": file_info.get("sha", ""),
            "html_url": file_info.get("html_url", ""),
            "download_url": file_info.get("download_url", ""),
        }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def health() -> JSONResponse:
    """Health-check / root endpoint."""
    return JSONResponse({"status": "ok", "service": "product-gig-github-api"})


@app.post("/products", status_code=status.HTTP_201_CREATED)
async def save_product_gig(
    payload: ProductGigPayload,
    x_action_key: str = Header(..., alias="X-Action-Key"),
) -> JSONResponse:
    """
    Validate the action key, build a unique filename, and commit the gig JSON
    to GitHub under <GITHUB_DIR>/YYYY/MM/<slug>-<uuid-short>.json.
    """
    _check_key(x_action_key)

    if not GITHUB_TOKEN or not GITHUB_REPO:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfiguration: GITHUB_TOKEN or GITHUB_REPO not set.",
        )

    now = datetime.now(tz=timezone.utc)
    year = now.strftime("%Y")
    month = now.strftime("%m")
    uid_short = str(uuid.uuid4())[:8]
    name_slug = _slug(payload.product_name)
    filename = f"{name_slug}-{uid_short}.json"
    file_path = f"{GITHUB_DIR}/{year}/{month}/{filename}"

    # Build a clean, normalised document regardless of which field variants GPT sent
    document: dict[str, Any] = {
        "saved_at": now.isoformat(),
        "file_path": file_path,
        "product_name": payload.product_name,
        "category": payload.category,
        "sales_title": payload.resolved_sales_title(),
        "description": payload.description,
        "price": payload.resolved_price(),
        "highlights": payload.highlights,
        "specifications": payload.specifications,
        "image_links": payload.image_links,
        "image_notes": payload.image_notes,
        "tags": payload.tags,
        "social_copy": payload.social_copy.model_dump(exclude_none=True) if payload.social_copy else None,
        "email_copy": payload.resolved_email_copy(),
        "source_notes": payload.source_notes,
    }
    # Strip None values for a clean file
    document = {k: v for k, v in document.items() if v is not None and v != [] and v != {}}

    content_bytes = json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8")

    commit_msg = f"gig: add {name_slug} [{uid_short}]"
    result = _commit_file(file_path, content_bytes, commit_msg)

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "status": "saved",
            "file_path": file_path,
            "github_url": result["html_url"],
            "download_url": result["download_url"],
        },
    )

