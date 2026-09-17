"""Non-executing M8 plans and externally reported qualification evidence."""

import re
from dataclasses import dataclass
from typing import Literal

from laserperception.perception import registry
from laserperception.perception.serialization import JsonRecord

from .artifacts import VerifiedArtifact
from .bootstrap import QualificationRecord
from .manifests import AuthorizationReference, TaskType
from .paths import validate_relative_path

CANDIDATE_SHA256 = "aa456e0386e46e9d089a957b1f1a8a4f74ceae70435c7ad8e6ca5e67bb90f4e7"
CONFIG_SHA256 = "b0832e03ad11d4e0b61f0fb07d977e687763caae472a4f87ed750bdc2d13be0f"
CHECKPOINT_SHA256 = "a675149d095eef8ddc0c137ae46eeac075ccc504c7608162c71e7adf318793fb"
INPUT_SHA256 = "76bd5f7adac3d892ad2fb00cb9cf5f4f73dd475682ee011b7ea9524060c46c95"
UPSTREAM_COMMIT = "8cfc2a6f23eed0b10aabcdc4768c60b184357061"
RETIRED_POLICY_SHA256 = "703e453a8bca0e6e2e4b1c4b976deaa5bc4ed27b3a4847144204193baab77563"
RETIRED_AUTHORIZATIONS = (
    "3c16a0c0ff9680b6418a53b20a5a51dd8f2a40d864a75c380397f2454ce06b9c",
    "0f67b939dd57fd782ca55a525f609c58d0349b216f2ed89a697e328995cb4ddd",
)


def _commit(value: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("repository identity requires full Git SHA")


def _digest(value: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("evidence requires full SHA256")


def required_m8_artifacts() -> tuple[VerifiedArtifact, ...]:
    """Frozen byte identities, with portable future worker paths; no local inspection."""
    rows = (
        ("candidate", "configs/m8/dsvt_nuscenes_pillar.json", 4915, CANDIDATE_SHA256),
        ("config", "inputs/dsvt_plain_1f_onestage_nusences.yaml", 5048, CONFIG_SHA256),
        ("checkpoint", "inputs/DSVT_Nuscenes_val.pth", 28665215, CHECKPOINT_SHA256),
        (
            "protocol-json",
            "benchmarks/m8/preregistration/m8_s1_protocol.json",
            15956,
            "c132f60257c6a39debb548461c79bd59c98325484d233db6095b441c638d8e88",
        ),
        (
            "protocol-markdown",
            "docs/m8/M8_S1_PROTOCOL.md",
            23802,
            "1ad58ebbdd04897558ef9802fee6288b806c5e633d393f1bed957ecc6d6f6b10",
        ),
        (
            "input-ledger",
            "benchmarks/m8/diagnostics/m8_input_projection_ledger.json",
            669345,
            "474e87e34c64d669750d4b6f7a64ac46fc9c5c462693fad79ff7c9547a7f1f7c",
        ),
        (
            "input-revalidation",
            "benchmarks/m8/diagnostics/m8_input_projection_revalidation.json",
            966,
            "71ac9418c29da5efd64f9eaeb03e859f85d6b1c56dc2fe47cef6563a9f960341",
        ),
        (
            "full-transform-ledger",
            "inputs/pre_inference_input_ledger_full.json",
            5837452,
            "e25b3d62113cc7e8c1fcf736caa68b1ab698f965f007c758ff91d3e498ca6caa",
        ),
    )
    return tuple(
        VerifiedArtifact(
            name,
            path,
            size,
            sha,
            name,
            "frozen M8 S1 identity; historical availability does not grant execution",
        )
        for name, path, size, sha in rows
    )


@dataclass(frozen=True)
class M8ArtifactAvailability(JsonRecord):
    name: str
    status: Literal["READY", "MISSING", "RECONSTRUCTIBLE", "REDOWNLOAD_REQUIRED", "HISTORICAL_ONLY"]
    identity: VerifiedArtifact | None
    source: str
    verification: str


def m8_readiness_inventory() -> tuple[M8ArtifactAvailability, ...]:
    """Bounded 2026-09-17 availability snapshot; never scan or hydrate private state."""
    records = []
    for artifact in required_m8_artifacts():
        source = "Git canonical bytes: " + artifact.relative_path
        if artifact.name in ("config", "checkpoint"):
            source = (
                "private Drive retirement/04_MODELS_AND_DEPLOYMENT/"
                + artifact.relative_path.split("/")[-1]
            )
        elif artifact.name == "full-transform-ledger":
            source = "private Drive .local/m6b-r2/evidence/pre_inference_input_ledger_full.json"
        verification = (
            "2026-09-17 bounded exact-path hashing matched size/SHA256; reverify on worker"
        )
        if artifact.name == "protocol-markdown":
            verification = (
                "Git LF identity matched; Windows working tree is CRLF; "
                "external checkout must materialize LF"
            )
        records.append(
            M8ArtifactAvailability(artifact.name, "READY", artifact, source, verification)
        )
    rows = (
        (
            "KITTI source drives",
            "READY",
            "private Drive .local/assets/m6b/data/2011_09_26/",
            "Structure only: 108/340 contiguous Velodyne+OXTS "
            "rows/calibrations/timestamps present; payload hashes unverified",
        ),
        (
            "DSVT upstream checkout",
            "RECONSTRUCTIBLE",
            "official Haiyang-W/DSVT@" + UPSTREAM_COMMIT,
            "Exact clean checkout must be reconstructed and verified; audit "
            "OpenPCDet checkout is not the runtime",
        ),
        (
            "isolated scientific environment",
            "RECONSTRUCTIBLE",
            "frozen candidate runtime + private retirement/09_ENVIRONMENT_AND_PACKAGES",
            "Reconstruct on external worker; all live versions/capacity remain unverified",
        ),
        (
            "optional nuScenes source-domain smoke input",
            "REDOWNLOAD_REQUIRED",
            "official nuScenes source under its terms",
            "Optional separately scoped smoke; not required by frozen KITTI "
            "input source; do not download during readiness",
        ),
        (
            "fresh qualification owner scope",
            "MISSING",
            "new selected runtime",
            "Provider/runtime not selected; no fresh qualification authorization",
        ),
        (
            "fresh machine policy",
            "MISSING",
            "new selected runtime",
            "Requires verified environment/capacity and owner review",
        ),
        (
            "fresh Stage R authorization/raw review",
            "MISSING",
            "new selected runtime",
            "Fresh Stage-R-only permission and repeated raw owner review/freeze required",
        ),
        (
            "fresh primary authorization",
            "MISSING",
            "new selected runtime",
            "Requires fresh reviewed Stage R; primary calls remain 0",
        ),
        (
            "retired Stage R/policy/primary authorization",
            "HISTORICAL_ONLY",
            "frozen retired-machine Git/Drive evidence",
            "Non-portable; historical 140 Stage R calls and zero primary "
            "calls; cannot satisfy fresh gates",
        ),
        (
            "historical PointPillars/ONNX/TensorRT artifacts",
            "HISTORICAL_ONLY",
            "private Drive retirement deployment history",
            "Preserved historical artifacts; not current M8 primary permission or parity",
        ),
    )
    records.extend(
        M8ArtifactAvailability.from_dict(
            {
                "name": name,
                "status": status,
                "identity": None,
                "source": source,
                "verification": verification,
            }
        )
        for name, status, source, verification in rows
    )
    return tuple(records)


@dataclass(frozen=True)
class M8ReadinessRequest(JsonRecord):
    execution_commit: str
    task_id: str
    task_type: TaskType = TaskType.QUALIFICATION
    candidate_sha256: str = CANDIDATE_SHA256
    config_sha256: str = CONFIG_SHA256
    checkpoint_sha256: str = CHECKPOINT_SHA256
    upstream_commit: str = UPSTREAM_COMMIT
    ordered_input_sha256: str = INPUT_SHA256

    def __post_init__(self) -> None:
        super().__post_init__()
        _commit(self.execution_commit)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.task_id):
            raise ValueError("task_id must be one portable identifier")
        if self.task_type != TaskType.QUALIFICATION:
            raise ValueError("M8 readiness plans qualification only")
        expected = (
            CANDIDATE_SHA256,
            CONFIG_SHA256,
            CHECKPOINT_SHA256,
            UPSTREAM_COMMIT,
            INPUT_SHA256,
        )
        actual = (
            self.candidate_sha256,
            self.config_sha256,
            self.checkpoint_sha256,
            self.upstream_commit,
            self.ordered_input_sha256,
        )
        if actual != expected:
            raise ValueError("frozen candidate/config/checkpoint/upstream/input identity mismatch")


@dataclass(frozen=True)
class M8ReadinessPlan(JsonRecord):
    schema_version: Literal["laserperception.m8.readiness-plan.v1"]
    request: M8ReadinessRequest
    model_id: Literal["dsvt-pillar-transfusion-m8"]
    artifacts: tuple[VerifiedArtifact, ...]
    availability: tuple[M8ArtifactAvailability, ...]
    raw_input_requirements: tuple[str, ...]
    runtime_requirements: tuple[str, ...]
    capsule_target: str
    minimum_vram_gib: Literal[16]
    preferred_vram_gib: Literal[24]
    missing_gates: tuple[str, ...]
    authorization_state: Literal["missing_fresh_runtime_scoped_authorizations"]
    provider_selected: Literal[False] = False
    hardware_probed: Literal[False] = False
    executes: Literal[False] = False
    scientific_calls: Literal[0] = 0
    source_payload_verified: Literal[False] = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.artifacts != required_m8_artifacts():
            raise ValueError("readiness plan frozen artifact identities changed")
        if self.availability != m8_readiness_inventory():
            raise ValueError("readiness plan availability snapshot changed")
        if self.capsule_target != f"lpdrive:_CLOUD_WORK/{self.request.task_id}":
            raise ValueError("readiness plan capsule differs from task identity")


def plan_m8_qualification(
    request: M8ReadinessRequest, *, expected_repository_commit: str
) -> M8ReadinessPlan:
    """Describe work before runtime selection; do not fabricate a scoped TaskManifest."""
    _commit(expected_repository_commit)
    if request.execution_commit != expected_repository_commit:
        raise ValueError("repository SHA differs from authoritative expected SHA")
    model = registry.get("dsvt-pillar-transfusion-m8")
    return M8ReadinessPlan(
        "laserperception.m8.readiness-plan.v1",
        request,
        "dsvt-pillar-transfusion-m8",
        required_m8_artifacts(),
        m8_readiness_inventory(),
        (
            "KITTI Raw 2011_09_26 drive0001: contiguous frames 0..107, Velodyne/OXTS/timestamps",
            "KITTI Raw 2011_09_26 drive0091: contiguous frames 0..339, Velodyne/OXTS/timestamps",
            "All three date calibration files; "
            "verify exact source bytes/transforms against full ledger",
            "428 ordered frames; H10/H5 gives 856 conditions; "
            "exclude GT/tracklet access during qualification",
            "float32 contiguous [x,y,z,intensity,time_lag]; "
            "preserve frozen source rows; no shuffle",
            "Canonical LF checkout required for exact frozen Markdown byte identity",
        ),
        model.runtime_requirements
        + (
            "Isolated Python 3.10 / Torch 2.1.0+cu118 / CUDA 11.8 / spconv 2.3.8 / "
            "torch-scatter 2.1.2+pt21cu118 / NumPy 1.23.5",
            "Do not upgrade frozen scientific NumPy to the core wheel's NumPy>=1.24; "
            "use isolated source runtime",
            "16 GiB practical lower bound, 24 GiB preferred; "
            "capacity alone does not qualify a runtime",
        ),
        f"lpdrive:_CLOUD_WORK/{request.task_id}",
        16,
        24,
        (
            "Owner runtime/provider selection and fresh qualification-only authorization",
            "Exact Git/artifact/source verification and externally reported environment",
            "Separately scoped GT-blind sizing/capacity authorization and evidence",
            "Machine-specific policy and owner review",
            "Fresh Stage-R-only authorization, fresh Stage R, persisted raw owner review/freeze",
            "Fresh primary authorization for three uninterrupted primary processes",
        ),
        "missing_fresh_runtime_scoped_authorizations",
    )


@dataclass(frozen=True)
class M8CapacityRecord(JsonRecord):
    result: Literal["pending", "passed", "failed"]
    evidence_sha256: str | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    method: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.evidence_sha256 is not None:
            _digest(self.evidence_sha256)
        if self.result != "pending" and self.evidence_sha256 is None:
            raise ValueError("completed capacity result requires evidence identity")
        if self.result == "passed" and (
            self.peak_allocated_bytes is None or self.peak_reserved_bytes is None
        ):
            raise ValueError("passed capacity requires allocated and reserved memory evidence")
        if any(
            v is not None and v < 0 for v in (self.peak_allocated_bytes, self.peak_reserved_bytes)
        ):
            raise ValueError("capacity memory values must be non-negative")


@dataclass(frozen=True)
class M8QualificationRecord(JsonRecord):
    """Externally reported data only; this type performs no collection or execution."""

    request: M8ReadinessRequest
    environment: QualificationRecord
    python: str
    operating_system: str
    upstream_commit: str
    candidate_sha256: str
    config_sha256: str
    checkpoint_sha256: str
    ordered_input_sha256: str
    capacity: M8CapacityRecord
    machine_policy_sha256: str | None
    status: Literal["external_report_pending_owner_review_not_permission"] = (
        "external_report_pending_owner_review_not_permission"
    )

    schema_version: Literal["laserperception.m8.qualification-record.v1"] = (
        "laserperception.m8.qualification-record.v1"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.environment.execution_commit != self.request.execution_commit
            or self.environment.task_id != self.request.task_id
        ):
            raise ValueError("qualification environment repository/task mismatch")
        if (
            self.upstream_commit,
            self.candidate_sha256,
            self.config_sha256,
            self.checkpoint_sha256,
            self.ordered_input_sha256,
        ) != (UPSTREAM_COMMIT, CANDIDATE_SHA256, CONFIG_SHA256, CHECKPOINT_SHA256, INPUT_SHA256):
            raise ValueError("qualification frozen identities mismatch")
        if not self.environment.gpus:
            raise ValueError("external qualification record requires reported GPU data")
        if self.environment.verified_inputs != required_m8_artifacts():
            raise ValueError("qualification verified artifact identities incomplete or changed")
        if self.capacity.result == "passed":
            assert self.capacity.peak_allocated_bytes is not None
            assert self.capacity.peak_reserved_bytes is not None
            if (
                not self.capacity.peak_allocated_bytes
                <= self.capacity.peak_reserved_bytes
                <= self.environment.gpus[0].vram_mib * 1024**2
            ):
                raise ValueError("reported capacity memory exceeds GPU or reserved memory")
        if self.machine_policy_sha256 is not None:
            _digest(self.machine_policy_sha256)
            if self.machine_policy_sha256 == RETIRED_POLICY_SHA256:
                raise ValueError("retired policy is historical only, not portable")


GATES = (
    "planned",
    "artifacts_verified",
    "environment_recorded",
    "capacity_reviewed",
    "policy_bound",
    "owner_reviewed",
    "stage_r_authorized",
    "stage_r_reviewed",
    "primary_authorized",
)


@dataclass(frozen=True)
class M8GateEvidence(JsonRecord):
    runtime_id: str
    execution_commit: str
    evidence_sha256: str
    persisted_path: str
    owner_reviewed: bool = False
    authorization: AuthorizationReference | None = None
    machine_policy_sha256: str | None = None
    accepted_stage_r_calls: int | None = None
    qualification: M8QualificationRecord | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _commit(self.execution_commit)
        _digest(self.evidence_sha256)
        validate_relative_path(self.persisted_path)
        if (
            self.evidence_sha256 in RETIRED_AUTHORIZATIONS
            or self.evidence_sha256 == RETIRED_POLICY_SHA256
        ):
            raise ValueError("historical evidence cannot substitute for a fresh runtime gate")
        if self.machine_policy_sha256 is not None:
            _digest(self.machine_policy_sha256)
            if self.machine_policy_sha256 == RETIRED_POLICY_SHA256:
                raise ValueError("historical runtime policy is not portable")
        if self.authorization is not None:
            if self.authorization.sha256 in RETIRED_AUTHORIZATIONS:
                raise ValueError("historical authorization is not portable")
            if (
                self.authorization.runtime_id != self.runtime_id
                or self.authorization.execution_commit != self.execution_commit
            ):
                raise ValueError("gate authorization runtime/repository mismatch")


@dataclass(frozen=True)
class M8QualificationProgress(JsonRecord):
    """Evidence index, never execution permission; frozen runner verification remains required."""

    request: M8ReadinessRequest
    runtime_id: str
    gates: tuple[M8GateEvidence, ...] = ()
    state: str = "planned"
    executes: Literal[False] = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.state not in GATES or len(self.gates) != GATES.index(self.state):
            raise ValueError("qualification state/gate count mismatch")
        policy = None
        environment_identity = None
        for gate_name, evidence in zip(GATES[1:], self.gates, strict=False):
            if (
                evidence.runtime_id != self.runtime_id
                or evidence.execution_commit != self.request.execution_commit
            ):
                raise ValueError("qualification gate runtime/repository mismatch")
            if gate_name in ("environment_recorded", "capacity_reviewed"):
                report = evidence.qualification
                if (
                    report is None
                    or report.request != self.request
                    or report.environment.runtime_id != self.runtime_id
                ):
                    raise ValueError("matching externally reported qualification record required")
                stable_identity = (
                    tuple((gpu.model, gpu.vram_mib, gpu.driver) for gpu in report.environment.gpus),
                    report.environment.cuda,
                    report.environment.pytorch,
                    report.environment.frameworks,
                    report.python,
                    report.operating_system,
                )
                if gate_name == "environment_recorded":
                    environment_identity = stable_identity
                elif stable_identity != environment_identity:
                    raise ValueError("reported hardware/software identity changed between gates")
                if gate_name == "capacity_reviewed" and report.capacity.result != "passed":
                    raise ValueError("passed externally reviewed capacity evidence required")
            if gate_name == "policy_bound":
                if evidence.machine_policy_sha256 is None:
                    raise ValueError("policy binding identity required")
                policy = evidence.machine_policy_sha256
            if policy is not None and evidence.machine_policy_sha256 != policy:
                raise ValueError("machine policy changed between gates")
            if gate_name in ("owner_reviewed", "stage_r_reviewed") and not evidence.owner_reviewed:
                raise ValueError("explicit owner evidence review required")
            if gate_name in ("stage_r_authorized", "primary_authorized"):
                role = TaskType.STAGE_R if gate_name == "stage_r_authorized" else TaskType.PRIMARY
                if evidence.authorization is None or evidence.authorization.role != role:
                    raise ValueError(f"fresh {role.value} authorization required")
            if gate_name == "stage_r_reviewed" and evidence.accepted_stage_r_calls != 140:
                raise ValueError(
                    "fresh Stage R must record 140 accepted calls and owner raw review"
                )

    def advance(self, state: str, evidence: M8GateEvidence) -> "M8QualificationProgress":
        index = GATES.index(self.state)
        if index + 1 >= len(GATES) or GATES[index + 1] != state:
            raise ValueError("qualification gates cannot be skipped or reordered")
        return M8QualificationProgress(
            self.request, self.runtime_id, self.gates + (evidence,), state
        )
