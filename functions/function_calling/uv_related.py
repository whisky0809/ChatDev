"""Utility tool to manage Python environments via uv."""

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from utils.exceptions import WorkflowCancelledError

_SAFE_PACKAGE_RE = re.compile(r"^[A-Za-z0-9_.\-+=<>!\[\],@:/]+$")
_DEFAULT_TIMEOUT = float(os.getenv("LIB_INSTALL_TIMEOUT", "120"))
_OUTPUT_SNIPPET_LIMIT = 240
# Max streaming output buffer size per tool call (default 100KB)
_MAX_STREAM_OUTPUT_SIZE = int(os.getenv("UV_STREAM_OUTPUT_LIMIT", "102400"))


def _trim_output_preview(stdout: str, stderr: str) -> str | None:
    """Return a short preview from stdout or stderr for error messaging."""

    preview_source = stdout.strip() or stderr.strip()
    if not preview_source:
        return None
    if len(preview_source) <= _OUTPUT_SNIPPET_LIMIT:
        return preview_source
    return f"{preview_source[:_OUTPUT_SNIPPET_LIMIT].rstrip()}... [truncated]"


def _build_timeout_message(step: str | None, timeout_value: float, stdout: str, stderr: str) -> str:
    """Create a descriptive timeout error message with optional output preview."""

    label = "uv command"
    if step:
        label = f"{label} ({step})"
    message = f"{label} timed out after {timeout_value} seconds"
    preview = _trim_output_preview(stdout, stderr)
    if preview:
        return f"{message}. Last output: {preview}"
    return message


class WorkspaceCommandContext:
    """Resolve the workspace root from the injected runtime context."""

    def __init__(self, ctx: Dict[str, Any] | None):
        if ctx is None:
            raise ValueError("_context is required for uv tools")
        self.workspace_root = self._require_workspace(ctx.get("python_workspace_root"))
        self._raw_ctx = ctx

    @staticmethod
    def _require_workspace(raw_path: Any) -> Path:
        if raw_path is None:
            raise ValueError("python_workspace_root missing from _context")
        path = Path(raw_path).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolve_under_workspace(self, relative_path: str | Path) -> Path:
        candidate = Path(relative_path)
        absolute = candidate if candidate.is_absolute() else self.workspace_root / candidate
        absolute = absolute.expanduser().resolve()
        if self.workspace_root not in absolute.parents and absolute != self.workspace_root:
            raise ValueError("script path is outside workspace root")
        return absolute


class _OutputBuffer:
    """Buffer for streaming output with size limit and tail retention.

    Keeps the most recent output up to max_size. When limit is exceeded,
    truncates from the beginning and adds a truncation notice.
    """

    _TRUNCATION_NOTICE = "\n... [output truncated, showing last {size}] ...\n"
    _TRUNCATION_NOTICE_LENGTH = 50  # Approximate length of the notice

    def __init__(self, max_size: int = _MAX_STREAM_OUTPUT_SIZE):
        self.max_size = max_size
        self._buffer: List[str] = []
        self._total_length = 0
        self._is_truncated = False

    def append(self, chunk: str) -> None:
        """Add a chunk to the buffer, truncating from front if needed."""
        if not chunk:
            return

        # If adding this chunk would exceed limit, truncate
        if self._total_length + len(chunk) > self.max_size:
            self._truncate_to_fit(chunk)

        self._buffer.append(chunk)
        self._total_length += len(chunk)

    def _truncate_to_fit(self, new_chunk: str) -> None:
        """Remove oldest chunks to make room, adding truncation notice."""
        target_size = self.max_size - len(new_chunk) - self._TRUNCATION_NOTICE_LENGTH

        # Remove chunks from the beginning until we have enough space
        while self._buffer and self._total_length > target_size:
            removed = self._buffer.pop(0)
            self._total_length -= len(removed)

        if not self._is_truncated:
            # Add truncation notice at the beginning
            notice = self._TRUNCATION_NOTICE.format(size=f"{self.max_size // 1024}KB")
            self._buffer.insert(0, notice)
            self._total_length += len(notice)
            self._is_truncated = True

    def get_content(self) -> str:
        """Get the full buffered content."""
        return "".join(self._buffer)

    def __len__(self) -> int:
        return self._total_length


def _validate_packages(packages: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for pkg in packages:
        if not isinstance(pkg, str):
            raise ValueError("package entries must be strings")
        stripped = pkg.strip()
        if not stripped:
            raise ValueError("package names cannot be empty")
        if not _SAFE_PACKAGE_RE.match(stripped):
            raise ValueError(f"unsafe characters detected in package spec {pkg}")
        if stripped.startswith("-"):
            raise ValueError(f"flags are not allowed in packages list: {pkg}")
        normalized.append(stripped)
    if not normalized:
        raise ValueError("at least one package is required")
    return normalized


def _coerce_timeout_seconds(timeout_seconds: Any) -> float | None:
    if timeout_seconds is None:
        return None
    if isinstance(timeout_seconds, bool):
        raise ValueError("timeout_seconds must be a number")
    if isinstance(timeout_seconds, (int, float)):
        value = float(timeout_seconds)
    elif isinstance(timeout_seconds, str):
        raw = timeout_seconds.strip()
        if not raw:
            raise ValueError("timeout_seconds cannot be empty")
        try:
            if re.fullmatch(r"[+-]?\d+", raw):
                value = float(int(raw))
            else:
                value = float(raw)
        except ValueError as exc:
            raise ValueError("timeout_seconds must be a number") from exc
    else:
        raise ValueError("timeout_seconds must be a number")

    if value <= 0:
        raise ValueError("timeout_seconds must be positive")
    return value


def _validate_flag_args(args: Sequence[str] | None) -> List[str]:
    normalized: List[str] = []
    if not args:
        return normalized
    for arg in args:
        if not isinstance(arg, str):
            raise ValueError("extra args must be strings")
        stripped = arg.strip()
        if not stripped:
            raise ValueError("extra args cannot be empty")
        if not stripped.startswith("-"):
            raise ValueError(f"extra args must be flags, got {arg}")
        normalized.append(stripped)
    return normalized


def _validate_args(args: Sequence[str] | None) -> List[str]:
    normalized: List[str] = []
    if not args:
        return normalized
    for arg in args:
        if not isinstance(arg, str):
            raise ValueError("args entries must be strings")
        stripped = arg.strip()
        if not stripped:
            raise ValueError("args entries cannot be empty")
        normalized.append(stripped)
    return normalized


def _validate_env(env: Mapping[str, str] | None) -> Dict[str, str]:
    if env is None:
        return {}
    result: Dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str) or not key:
            raise ValueError("environment variable keys must be non-empty strings")
        if not isinstance(value, str):
            raise ValueError("environment variable values must be strings")
        result[key] = value
    return result


def _kill_process_tree(pid: int) -> None:
    """Kill a process and its children (Windows-specific)."""
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True,
                check=False,
            )
        except Exception:
            pass
    else:
        try:
            import signal

            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass


def _send_stream_output(
    websocket_manager: Any,
    session_id: str,
    node_id: str,
    tool_name: str,
    chunk: str,
    is_stderr: bool = False,
) -> None:
    """Send streaming output to the frontend via websocket."""
    if websocket_manager is None or session_id is None or node_id is None:
        return
    try:
        # Use sync version if available, otherwise fall back to async
        if hasattr(websocket_manager, "send_tool_stream_output_sync"):
            websocket_manager.send_tool_stream_output_sync(
                session_id, node_id, tool_name, chunk, is_stderr
            )
        elif hasattr(websocket_manager, "send_tool_stream_output"):
            import asyncio

            asyncio.create_task(
                websocket_manager.send_tool_stream_output(
                    session_id, node_id, tool_name, chunk, is_stderr
                )
            )
    except Exception:
        # Silently ignore streaming errors to not break tool execution
        pass


def _run_uv_command(
    cmd: List[str],
    workspace_root: Path,
    *,
    step: str | None = None,
    env: Dict[str, str] | None = None,
    timeout: float | None = None,
    _context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run a uv command with optional streaming output support.

    If _context contains websocket_manager, session_id, and node_id,
    output will be streamed to the frontend in real-time.
    """
    timeout_value = _DEFAULT_TIMEOUT if timeout is None else timeout
    env_vars = None if env is None else {**os.environ, **env}

    # Extract streaming context if available
    websocket_manager = _context.get("websocket_manager") if _context else None
    session_id = _context.get("session_id") if _context else None
    node_id = _context.get("node_id") if _context else None
    cancel_event: Optional[threading.Event] = (
        _context.get("cancel_event") if _context else None
    )
    tool_name = step or "uv_run"

    # Check if streaming is available
    can_stream = websocket_manager is not None and session_id is not None and node_id is not None

    try:
        if can_stream:
            return _run_uv_command_streaming(
                cmd,
                workspace_root,
                step=step,
                env=env_vars,
                timeout=timeout_value,
                websocket_manager=websocket_manager,
                session_id=session_id,
                node_id=node_id,
                cancel_event=cancel_event,
                tool_name=tool_name,
            )
        else:
            # Fall back to non-streaming execution
            completed = subprocess.run(
                cmd,
                cwd=str(workspace_root),
                capture_output=True,
                text=True,
                timeout=timeout_value,
                check=False,
                env=env_vars,
            )
            return {
                "command": cmd,
                "stdout": completed.stdout or "",
                "stderr": completed.stderr or "",
                "returncode": completed.returncode,
                "step": step,
            }
    except FileNotFoundError as exc:
        raise RuntimeError("uv command not found in PATH") from exc
    except subprocess.TimeoutExpired as exc:
        stdout_text = exc.stdout
        if stdout_text is None:
            stdout_text = getattr(exc, "output", "") or ""
        stderr_text = exc.stderr or ""
        message = _build_timeout_message(step, timeout_value, stdout_text, stderr_text)
        return {
            "command": cmd,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "returncode": None,
            "step": step,
            "timed_out": True,
            "timeout": timeout_value,
            "error": message,
        }


def _run_uv_command_streaming(
    cmd: List[str],
    workspace_root: Path,
    *,
    step: str | None = None,
    env: Dict[str, str] | None = None,
    timeout: float | None = None,
    websocket_manager: Any,
    session_id: str,
    node_id: str,
    cancel_event: Optional[threading.Event],
    tool_name: str,
) -> Dict[str, Any]:
    """Run uv command with streaming output to websocket.

    Uses Popen to read stdout/stderr line by line and stream to frontend.
    Checks for cancellation periodically.
    Output is buffered with size limit, keeping the most recent content.
    """
    timeout_value = timeout or _DEFAULT_TIMEOUT
    stdout_buffer = _OutputBuffer(max_size=_MAX_STREAM_OUTPUT_SIZE)
    stderr_buffer = _OutputBuffer(max_size=_MAX_STREAM_OUTPUT_SIZE)

    proc = subprocess.Popen(
        cmd,
        cwd=str(workspace_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )

    start_time = time.time()

    # Use select for non-blocking reads on Unix, but on Windows we need threads
    if sys.platform == "win32":
        # Windows: use threads to read stdout/stderr concurrently
        def read_stream(pipe, buffer: _OutputBuffer, is_stderr):
            try:
                for line in iter(pipe.readline, ""):
                    if not line:
                        break
                    buffer.append(line)
                    _send_stream_output(
                        websocket_manager,
                        session_id,
                        node_id,
                        tool_name,
                        line,
                        is_stderr,
                    )
            except Exception:
                pass
            finally:
                pipe.close()

        stdout_thread = threading.Thread(
            target=read_stream, args=(proc.stdout, stdout_buffer, False)
        )
        stderr_thread = threading.Thread(
            target=read_stream, args=(proc.stderr, stderr_buffer, True)
        )
        stdout_thread.start()
        stderr_thread.start()

        # Poll for completion and cancellation
        try:
            while proc.poll() is None:
                # Check for cancellation
                if cancel_event is not None and cancel_event.is_set():
                    _kill_process_tree(proc.pid)
                    raise WorkflowCancelledError("Tool execution cancelled")

                # Check for timeout
                if time.time() - start_time > timeout_value:
                    _kill_process_tree(proc.pid)
                    stdout_text = stdout_buffer.get_content()
                    stderr_text = stderr_buffer.get_content()
                    message = _build_timeout_message(step, timeout_value, stdout_text, stderr_text)
                    return {
                        "command": cmd,
                        "stdout": stdout_text,
                        "stderr": stderr_text,
                        "returncode": None,
                        "step": step,
                        "timed_out": True,
                        "timeout": timeout_value,
                        "error": message,
                    }

                time.sleep(0.05)

            # Wait for threads to finish
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)

        except WorkflowCancelledError:
            _kill_process_tree(proc.pid)
            raise
    else:
        # Unix: use select for non-blocking I/O
        import select

        stdout_fd = proc.stdout.fileno() if proc.stdout else None
        stderr_fd = proc.stderr.fileno() if proc.stderr else None

        try:
            while proc.poll() is None:
                # Check for cancellation
                if cancel_event is not None and cancel_event.is_set():
                    _kill_process_tree(proc.pid)
                    raise WorkflowCancelledError("Tool execution cancelled")

                # Check for timeout
                if time.time() - start_time > timeout_value:
                    _kill_process_tree(proc.pid)
                    stdout_text = stdout_buffer.get_content()
                    stderr_text = stderr_buffer.get_content()
                    message = _build_timeout_message(step, timeout_value, stdout_text, stderr_text)
                    return {
                        "command": cmd,
                        "stdout": stdout_text,
                        "stderr": stderr_text,
                        "returncode": None,
                        "step": step,
                        "timed_out": True,
                        "timeout": timeout_value,
                        "error": message,
                    }

                # Read available output
                readable = []
                fds = [fd for fd in [stdout_fd, stderr_fd] if fd is not None]
                if fds:
                    readable, _, _ = select.select(fds, [], [], 0.05)

                for fd in readable:
                    if fd == stdout_fd:
                        line = proc.stdout.readline()
                        if line:
                            stdout_buffer.append(line)
                            _send_stream_output(
                                websocket_manager,
                                session_id,
                                node_id,
                                tool_name,
                                line,
                                False,
                            )
                    elif fd == stderr_fd:
                        line = proc.stderr.readline()
                        if line:
                            stderr_buffer.append(line)
                            _send_stream_output(
                                websocket_manager,
                                session_id,
                                node_id,
                                tool_name,
                                line,
                                True,
                            )

            # Read any remaining output
            if proc.stdout:
                remaining = proc.stdout.read()
                if remaining:
                    stdout_buffer.append(remaining)
                    _send_stream_output(
                        websocket_manager, session_id, node_id, tool_name, remaining, False
                    )
            if proc.stderr:
                remaining = proc.stderr.read()
                if remaining:
                    stderr_buffer.append(remaining)
                    _send_stream_output(
                        websocket_manager, session_id, node_id, tool_name, remaining, True
                    )

        except WorkflowCancelledError:
            _kill_process_tree(proc.pid)
            raise

    return {
        "command": cmd,
        "stdout": stdout_buffer.get_content(),
        "stderr": stderr_buffer.get_content(),
        "returncode": proc.returncode,
        "step": step,
    }


def install_python_packages(
    packages: Sequence[str],
    *,
    upgrade: bool = False,
    # extra_args: Sequence[str] | None = None,
    _context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Install Python packages inside the workspace using uv add."""

    ctx = WorkspaceCommandContext(_context)
    safe_packages = _validate_packages(packages)
    cmd: List[str] = ["uv", "add"]
    if upgrade:
        cmd.append("--upgrade")

    # if extra_args:
    #     flags = _validate_flag_args(extra_args)
    #     cmd.extend(flags)

    cmd.extend(safe_packages)
    result = _run_uv_command(cmd, ctx.workspace_root, step="uv add", _context=_context)
    # result["workspace_root"] = str(ctx.workspace_root)
    return result


def init_python_env(
    *,
    # recreate: bool = False,
    python_version: str | None = None,
    # lock_args: Sequence[str] | None = None,
    # venv_args: Sequence[str] | None = None,
    _context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run uv lock and uv venv inside the workspace."""

    ctx = WorkspaceCommandContext(_context)
    steps: List[Dict[str, Any]] = []

    lock_cmd: List[str] = ["uv", "lock"]
    # lock_cmd.extend(_validate_flag_args(lock_args))
    lock_result = _run_uv_command(lock_cmd, ctx.workspace_root, step="uv lock", _context=_context)
    steps.append(lock_result)
    if lock_result["returncode"] != 0:
        return {
            "workspace_root": str(ctx.workspace_root),
            "steps": steps,
        }

    venv_cmd: List[str] = ["uv", "venv"]
    # if recreate:
    #     venv_cmd.append("--recreate")
    # venv_cmd.extend(_validate_flag_args(venv_args))
    if python_version is not None:
        python_spec = python_version.strip()
        if not python_spec:
            raise ValueError("python argument cannot be empty")
        venv_cmd.extend(["--python", python_spec])

    venv_result = _run_uv_command(venv_cmd, ctx.workspace_root, step="uv venv", _context=_context)
    steps.append(venv_result)

    init_cmd: List[str] = ["uv", "init", "--bare", "--no-workspace"]
    init_result = _run_uv_command(init_cmd, ctx.workspace_root, step="uv init", _context=_context)
    steps.append(init_result)

    return {
        "workspace_root": str(ctx.workspace_root),
        "steps": steps,
    }


def uv_run(
    *,
    module: str | None = None,
    script: str | None = None,
    args: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
    _context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Execute uv run for a module or script inside the workspace root."""

    ctx = WorkspaceCommandContext(_context)
    timeout_seconds = _coerce_timeout_seconds(timeout_seconds)

    has_module = module is not None
    has_script = script is not None
    if has_module == has_script:
        raise ValueError("Provide exactly one of module or script")

    cmd: List[str] = ["uv", "run"]
    if has_module:
        module_name = module.strip()
        if not module_name:
            raise ValueError("module cannot be empty")
        cmd.extend(["python", "-m", module_name])
    else:
        script_value = script.strip() if isinstance(script, str) else script
        if not script_value:
            raise ValueError("script cannot be empty")
        script_path = ctx.resolve_under_workspace(script_value)
        cmd.append(str(script_path))

    cmd.extend(_validate_args(args))
    env_overrides = _validate_env(env)
    result = _run_uv_command(
        cmd,
        ctx.workspace_root,
        step="uv run",
        env=env_overrides,
        timeout=timeout_seconds,
        _context=_context,
    )
    result["workspace_root"] = str(ctx.workspace_root)
    return result
