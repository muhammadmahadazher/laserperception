"""Provider-neutral worker metadata and verified artifact tooling. No runtime discovery."""

from .artifacts import VerifiedArtifact, file_identity, verify_artifact
from .manifests import AuthorizationReference, RuntimeDescriptor, TaskManifest, TaskType
from .planning import WorkerPlan, plan_worker
from .transfer import ArtifactTransfer, RemoteObject

__all__ = [
    "ArtifactTransfer",
    "AuthorizationReference",
    "RemoteObject",
    "RuntimeDescriptor",
    "TaskManifest",
    "TaskType",
    "VerifiedArtifact",
    "WorkerPlan",
    "file_identity",
    "plan_worker",
    "verify_artifact",
]
