"""API routes for workspace management."""

from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter

from server.settings import WARE_HOUSE_DIR


router = APIRouter()


def _is_project_directory(path: Path) -> bool:
    """Check if a directory looks like a project workspace.

    Checks for workflow output files that indicate this is a
    workspace created by previous workflow runs.
    """
    return any(
        (path / name).exists()
        for name in ["node_outputs.yaml", "workflow_summary.yaml"]
    )


def _is_valid_workspace(path: Path) -> bool:
    """Check if directory is a valid workspace."""
    code_workspace = path / "code_workspace"
    if code_workspace.exists() and code_workspace.is_dir():
        return True
    return _is_project_directory(path)


@router.get("/api/workspaces")
async def list_workspaces() -> Dict[str, List[str]]:
    """List existing workspace directories in the WareHouse folder.

    Returns directories that contain a code_workspace subfolder or have
    workflow output files, indicating they are valid workspaces.
    """
    if not WARE_HOUSE_DIR.exists():
        return {"workspaces": []}

    workspaces = [
        entry.name
        for entry in sorted(WARE_HOUSE_DIR.iterdir())
        if entry.is_dir()
        and not entry.name.startswith("session_")
        and _is_valid_workspace(entry)
    ]

    return {"workspaces": workspaces}
