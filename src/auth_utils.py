"""
Authentication helpers for HuggingFace Hub and the GitHub API.

Tokens are read, in order of precedence, from:
    1. An explicit ``token`` argument.
    2. Environment variables (see ``HF_TOKEN_VARS`` / ``GITHUB_TOKEN_VARS``).
    3. A local ``.env`` file at the repository root (optional, no dependency
       on ``python-dotenv``).

Never hard-code tokens in source. Add ``.env`` to ``.gitignore``.

Typical usage:
    from src.auth_utils import login_huggingface, GitHubClient

    login_huggingface()                       # auth for gated/OLMo models
    gh = GitHubClient()                        # reads GITHUB_TOKEN
    gh.upload_file("results/layer_a/results.json", "owner/repo")
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Environment variable names checked, in order, for each service.
HF_TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")
GITHUB_TOKEN_VARS = ("GITHUB_TOKEN", "GH_TOKEN")

_GITHUB_API = "https://api.github.com"


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

def load_dotenv(dotenv_path: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """
    Load ``KEY=VALUE`` pairs from a ``.env`` file into ``os.environ``.

    Existing environment variables are *not* overwritten. Lines that are blank
    or start with ``#`` are ignored. Values may be optionally quoted.

    Args:
        dotenv_path: Path to the .env file. Defaults to ``<repo_root>/.env``.

    Returns:
        Dict of the keys that were newly set from the file.
    """
    if dotenv_path is None:
        dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    path = Path(dotenv_path)

    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value

    if loaded:
        logger.info("Loaded %d variable(s) from %s", len(loaded), path)
    return loaded


def _resolve_token(
    token: str | None,
    env_vars: tuple[str, ...],
    *,
    use_dotenv: bool,
) -> str | None:
    """Resolve a token from arg → environment → optional .env file."""
    if token:
        return token
    if use_dotenv:
        load_dotenv()
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            return value
    return None


# ---------------------------------------------------------------------------
# HuggingFace
# ---------------------------------------------------------------------------

def get_hf_token(
    token: str | None = None, *, use_dotenv: bool = True
) -> str | None:
    """
    Resolve the HuggingFace token without performing a login.

    Args:
        token: Explicit token; takes precedence over all other sources.
        use_dotenv: Whether to consult a ``.env`` file as a fallback.

    Returns:
        The token string, or ``None`` if none was found.
    """
    return _resolve_token(token, HF_TOKEN_VARS, use_dotenv=use_dotenv)


def login_huggingface(
    token: str | None = None,
    *,
    use_dotenv: bool = True,
    required: bool = False,
) -> str | None:
    """
    Authenticate with the HuggingFace Hub for gated and checkpoint models.

    This is needed for gated repos (LLaMA-2, LLaMA-3.1) and avoids rate limits
    when listing OLMo-2 checkpoint revisions.

    Args:
        token: Explicit token; falls back to env vars / ``.env``.
        use_dotenv: Whether to consult a ``.env`` file as a fallback.
        required: If True, raise ``RuntimeError`` when no token is found.

    Returns:
        The token used, or ``None`` if no token was available and
        ``required`` is False.

    Raises:
        RuntimeError: If ``required`` is True and no token is found.
    """
    resolved = get_hf_token(token, use_dotenv=use_dotenv)
    if not resolved:
        msg = (
            "No HuggingFace token found. Set one of "
            f"{', '.join(HF_TOKEN_VARS)} or run `huggingface-cli login`."
        )
        if required:
            raise RuntimeError(msg)
        logger.warning("%s Continuing unauthenticated (public models only).", msg)
        return None

    try:
        from huggingface_hub import login  # type: ignore

        login(token=resolved, add_to_git_credential=False)
        # Export so downstream transformers / hub calls pick it up too.
        os.environ.setdefault("HF_TOKEN", resolved)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", resolved)
        logger.info("Authenticated with HuggingFace Hub.")
    except Exception as exc:  # pragma: no cover - network/credential errors
        logger.error("HuggingFace login failed: %s", exc)
        if required:
            raise
        return None
    return resolved


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------

def get_github_token(
    token: str | None = None, *, use_dotenv: bool = True
) -> str | None:
    """
    Resolve the GitHub API token.

    Args:
        token: Explicit token; takes precedence over all other sources.
        use_dotenv: Whether to consult a ``.env`` file as a fallback.

    Returns:
        The token string, or ``None`` if none was found.
    """
    return _resolve_token(token, GITHUB_TOKEN_VARS, use_dotenv=use_dotenv)


class GitHubClient:
    """
    Minimal authenticated GitHub REST API client (stdlib only).

    Used for pushing analysis artifacts (results JSON, figures) to a repo or
    release, and for read access to repository metadata. Relies only on
    ``urllib`` so it adds no dependency to ``requirements.txt``.
    """

    def __init__(
        self,
        token: str | None = None,
        *,
        use_dotenv: bool = True,
        required: bool = True,
        api_base: str = _GITHUB_API,
    ) -> None:
        """
        Args:
            token: Explicit token; falls back to env vars / ``.env``.
            use_dotenv: Whether to consult a ``.env`` file as a fallback.
            required: If True, raise ``RuntimeError`` when no token is found.
            api_base: GitHub API base URL (override for Enterprise).

        Raises:
            RuntimeError: If ``required`` is True and no token is found.
        """
        self.token = get_github_token(token, use_dotenv=use_dotenv)
        self.api_base = api_base.rstrip("/")
        if not self.token and required:
            raise RuntimeError(
                "No GitHub token found. Set one of "
                f"{', '.join(GITHUB_TOKEN_VARS)} or pass token=..."
            )

    # -- low-level request ------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        """Perform an authenticated JSON request against the GitHub API."""
        url = path if path.startswith("http") else f"{self.api_base}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None

        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        request.add_header("User-Agent", "rope-retrieval-heads")
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        if data is not None:
            request.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(request) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            logger.error("GitHub API %s %s -> %s: %s", method, url, exc.code, detail)
            raise

    # -- convenience methods ---------------------------------------------

    def whoami(self) -> dict[str, Any]:
        """Return the authenticated user's profile (verifies the token)."""
        return self._request("GET", "/user")

    def get_repo(self, repo: str) -> dict[str, Any]:
        """
        Fetch repository metadata.

        Args:
            repo: Repository in ``owner/name`` form.
        """
        return self._request("GET", f"/repos/{repo}")

    def upload_file(
        self,
        local_path: str | os.PathLike[str],
        repo: str,
        *,
        dest_path: str | None = None,
        branch: str = "main",
        message: str | None = None,
    ) -> dict[str, Any]:
        """
        Create or update a file in a repository via the Contents API.

        Args:
            local_path: Path to the local file to upload.
            repo: Target repository in ``owner/name`` form.
            dest_path: Path within the repo; defaults to the file name.
            branch: Target branch.
            message: Commit message; auto-generated if omitted.

        Returns:
            The API response describing the commit.
        """
        import base64

        local = Path(local_path)
        content_b64 = base64.b64encode(local.read_bytes()).decode("ascii")
        dest = dest_path or local.name
        commit_message = message or f"Upload {dest}"

        # Look up an existing file's SHA so we can update rather than fail.
        sha: str | None = None
        try:
            existing = self._request(
                "GET", f"/repos/{repo}/contents/{dest}?ref={branch}"
            )
            if isinstance(existing, dict):
                sha = existing.get("sha")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise

        payload: dict[str, Any] = {
            "message": commit_message,
            "content": content_b64,
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha

        result = self._request("PUT", f"/repos/{repo}/contents/{dest}", payload)
        logger.info("Uploaded %s -> %s/%s@%s", local, repo, dest, branch)
        return result


__all__ = [
    "HF_TOKEN_VARS",
    "GITHUB_TOKEN_VARS",
    "load_dotenv",
    "get_hf_token",
    "login_huggingface",
    "get_github_token",
    "GitHubClient",
]
