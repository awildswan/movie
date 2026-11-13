"""Structural properties of the three workflows that build and ship production.

Nothing in this repository executes ``.github/workflows/ci.yml``'s
``publish-images`` job, ``deploy-production.yml`` or ``production-canary.yml`` —
GitHub does, against a Hetzner box — so the properties whose violation is
invisible until a release goes wrong are the ones worth pinning here.

Most of them are couplings across files that no single file can protect:

* the required-check list against the jobs ``ci.yml`` actually declares, so a
  renamed, deleted or newly added CI job cannot leave the deploy gate quietly
  asserting nothing;
* the published image names against the services ``docker-compose.prod.yml``
  builds, because an image the publish job forgets is an image the host cannot
  pull and would have to build for itself — which is the one thing the host is
  never asked to do;
* the image namespace against the default the same Compose file resolves when
  ``IMAGE_REPOSITORY`` is unset (``infra/deploy/production.env.example`` carries
  the same string), because a publish job and a host that disagree about the
  namespace is a deploy that pulls a tag nobody pushed;
* the sentinels against ``infra/deploy/deploy.sh`` and ``src/release``, which is
  where the scripts that print them define the literals;
* the k6 pin against ``infra/ci/k6-version``, since a Dockerfile cannot read a
  file to build its own ``FROM`` line and ADR 0010 makes that version the thing
  that stops local and CI measurements drifting apart;
* the two plan-named gates ``api-contract-check`` and ``web-api-types-check``,
  which are *steps* rather than jobs. Requiring the ``lint`` and ``frontend``
  jobs is what requiring those two checks means, and that only stays true while
  the steps are still in those jobs.

The sentinel match is exercised with the real ``grep`` rather than a
reimplementation of its pattern, because the property that matters —
``VERIFY-OK`` must not be satisfied by a run that printed ``VERIFY-SUBSET-OK``
— is a property of that regular expression under that tool.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.release import VERIFY_SENTINEL, VERIFY_SUBSET_SENTINEL

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
CI_WORKFLOW = WORKFLOW_DIR / "ci.yml"
DEPLOY_WORKFLOW = WORKFLOW_DIR / "deploy-production.yml"
CANARY_WORKFLOW = WORKFLOW_DIR / "production-canary.yml"
WORKFLOWS = (CI_WORKFLOW, DEPLOY_WORKFLOW, CANARY_WORKFLOW)

DEPLOY_SCRIPT = REPO_ROOT / "infra" / "deploy" / "deploy.sh"
PROD_COMPOSE = REPO_ROOT / "docker-compose.prod.yml"

# The GHCR namespace every published image lives under. Lowercase because GHCR
# namespaces are, while the repository is `MovieLens-RecSys` — which is why the
# publish job cannot interpolate `github.repository` and carries the literal.
IMAGE_REPOSITORY = "ghcr.io/awildswan/movie"

# The one lane to the box. deploy-production.yml and production-canary.yml share
# it so a canary cannot read a stack that is mid-release.
CONCURRENCY_GROUP = "production-hetzner"

# infra/deploy/deploy.sh's last word, and the only thing the deploy workflow
# treats as proof that the release landed.
DEPLOY_SENTINEL = "DEPLOY-OK"
ROLLBACK_SENTINEL = "ROLLBACK-OK"

# The host layout the deploy workflow and deploy.sh both assume.
HOST_CHECKOUT = "/opt/movielens"


def load_workflow(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    # PyYAML resolves the bare key `on` to the boolean True under YAML 1.1,
    # which is the one place a workflow file and a YAML parser disagree.
    return workflow.get("on", workflow.get(True))  # type: ignore[call-overload]


def steps(workflow: dict[str, Any], job: str) -> list[dict[str, Any]]:
    return list(workflow["jobs"][job]["steps"])


def step_by_name(workflow: dict[str, Any], job: str, name: str) -> dict[str, Any]:
    for step in steps(workflow, job):
        if step.get("name") == name:
            return step
    raise AssertionError(f"{job} has no step named {name!r}")


def published_images() -> list[tuple[str, str, str]]:
    """The publish job's image table: (name, build context, Dockerfile)."""
    body = CI_WORKFLOW.read_text(encoding="utf-8")
    match = re.search(r"<<'IMAGES'\n(.*?)\n *IMAGES\n", body, re.DOTALL)
    assert match is not None, "publish-images no longer carries an IMAGES heredoc"
    rows = []
    for line in match.group(1).splitlines():
        fields = line.split()
        if not fields:
            continue
        assert len(fields) == 3, f"malformed image row: {line!r}"
        rows.append((fields[0], fields[1], fields[2]))
    return rows


def compose_built_image_names() -> set[str]:
    """The distinct image names docker-compose.prod.yml builds rather than pulls."""
    document = yaml.safe_load(PROD_COMPOSE.read_text(encoding="utf-8"))
    names = set()
    for service in document["services"].values():
        if not isinstance(service, dict) or "build" not in service:
            continue
        reference = service["image"]
        # `${IMAGE_REPOSITORY:-...}/<name>:${IMAGE_TAG:-main}`
        match = re.search(r"\}/(?P<name>[a-z0-9-]+):", reference)
        assert match is not None, f"unexpected image reference {reference!r}"
        names.add(match.group("name"))
    return names


@pytest.fixture(scope="module")
def ci() -> dict[str, Any]:
    return load_workflow(CI_WORKFLOW)


@pytest.fixture(scope="module")
def deploy() -> dict[str, Any]:
    return load_workflow(DEPLOY_WORKFLOW)


@pytest.fixture(scope="module")
def canary() -> dict[str, Any]:
    return load_workflow(CANARY_WORKFLOW)


# --- CI: the images the host pulls ------------------------------------------


def test_images_are_published_only_from_a_push_to_main(ci: dict[str, Any]) -> None:
    condition = ci["jobs"]["publish-images"]["if"]
    assert "github.event_name == 'push'" in condition
    assert "github.ref == 'refs/heads/main'" in condition


def test_publishing_waits_for_every_other_ci_job(ci: dict[str, Any]) -> None:
    # "Published" has to mean "passed everything", or the deploy gate's whole
    # premise — that only a green commit has images — is a coincidence of
    # ordering rather than a property.
    required = set(ci["jobs"]["publish-images"]["needs"])
    others = set(ci["jobs"]) - {"publish-images"}
    assert required == others, (
        "publish-images and ci.yml have drifted: "
        f"jobs it does not wait for {sorted(others - required)}, "
        f"jobs it waits for that no longer exist {sorted(required - others)}"
    )


def test_publishing_writes_packages_with_the_run_s_own_token(ci: dict[str, Any]) -> None:
    job = ci["jobs"]["publish-images"]
    assert job["permissions"] == {"contents": "read", "packages": "write"}
    login = step_by_name(ci, "publish-images", "Log in to GHCR")
    assert login["env"]["GHCR_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"
    # No long-lived registry credential exists anywhere: the token GitHub mints
    # for the run is scoped to this repository's packages and expires with it.
    body = CI_WORKFLOW.read_text(encoding="utf-8")
    for reference in re.findall(r"\$\{\{\s*secrets\.([A-Za-z0-9_]+)\s*\}\}", body):
        assert reference == "GITHUB_TOKEN"


def test_every_image_the_production_stack_builds_is_published() -> None:
    # An image missing from the publish table is one the box cannot pull, and
    # the box has no toolchain, no source tree at build time and no intention of
    # building anything.
    assert {name for name, _, _ in published_images()} == compose_built_image_names()


def test_every_published_image_names_a_context_and_a_dockerfile_that_exist() -> None:
    for name, context, dockerfile in published_images():
        assert (REPO_ROOT / context).is_dir(), f"{name}: build context {context} is missing"
        assert (REPO_ROOT / dockerfile).is_file(), f"{name}: {dockerfile} is missing"


def test_images_are_published_where_the_host_pulls_them_from(ci: dict[str, Any]) -> None:
    assert ci["jobs"]["publish-images"]["env"]["IMAGE_REPOSITORY"] == IMAGE_REPOSITORY
    # The same string the box resolves when .env.prod leaves IMAGE_REPOSITORY
    # unset. Two namespaces is a deploy that pulls a tag nobody pushed.
    compose = PROD_COMPOSE.read_text(encoding="utf-8")
    assert f"${{IMAGE_REPOSITORY:-{IMAGE_REPOSITORY}}}" in compose
    # And the value the operator's .env.prod is seeded from, which is the one
    # they would notice being wrong last.
    env_example = (REPO_ROOT / "infra" / "deploy" / "production.env.example").read_text(
        encoding="utf-8"
    )
    assert f"\nIMAGE_REPOSITORY={IMAGE_REPOSITORY}\n" in env_example


def test_images_are_built_for_the_architecture_the_box_runs(ci: dict[str, Any]) -> None:
    build = step_by_name(
        ci, "publish-images", "Build and push every image the production stack runs"
    )
    script = build["run"]
    # The CX22 is x86-64 and this repository is usually developed on arm64. An
    # image built for the wrong architecture fails on the box, not here.
    assert "--platform linux/amd64" in script
    assert "--push" in script


def test_each_image_is_tagged_with_the_commit_and_with_main(ci: dict[str, Any]) -> None:
    build = step_by_name(
        ci, "publish-images", "Build and push every image the production stack runs"
    )
    assert build["env"]["IMAGE_TAG"] == "${{ github.sha }}"
    script = build["run"]
    # The SHA tag is what a deploy pins and what a rollback returns to; `main`
    # is the moving tag docker-compose.prod.yml defaults IMAGE_TAG to.
    assert '--tag "${IMAGE_REPOSITORY}/${name}:${IMAGE_TAG}"' in script
    assert '--tag "${IMAGE_REPOSITORY}/${name}:main"' in script


def test_the_published_k6_image_carries_the_repository_wide_k6_version(
    ci: dict[str, Any],
) -> None:
    # Compose passes `K6_VERSION` as a build arg and
    # tests/unit/test_prod_compose.py holds that path to the pin. This job does
    # not: the build loop is uniform and passes no --build-arg at all, so for
    # the image the box actually pulls, infra/k6/Dockerfile's ARG default *is*
    # the version that ships. A Dockerfile cannot read infra/ci/k6-version to
    # build its own FROM line, so the duplicate is asserted rather than trusted
    # — a canary running a different k6 than the pinned gate quietly undoes
    # ADR 0010.
    build = step_by_name(
        ci, "publish-images", "Build and push every image the production stack runs"
    )
    assert "--build-arg" not in build["run"]

    pinned = (REPO_ROOT / "infra" / "ci" / "k6-version").read_text(encoding="utf-8").strip()
    dockerfile = (REPO_ROOT / "infra" / "k6" / "Dockerfile").read_text(encoding="utf-8")
    default = re.search(r"^ARG K6_VERSION=(\S+)$", dockerfile, flags=re.MULTILINE)
    assert default is not None, "infra/k6/Dockerfile must pin k6 through ARG K6_VERSION"
    assert default.group(1) == pinned
    assert "FROM grafana/k6:${K6_VERSION}" in dockerfile


# --- Deploy: what may reach the box -----------------------------------------


def test_the_deploy_waits_for_ci_to_finish_rather_than_racing_the_push(
    deploy: dict[str, Any],
) -> None:
    on = triggers(deploy)
    # A workflow that fired on the push would reach the box before
    # publish-images had pushed the tag it is about to pull.
    assert on["workflow_run"]["workflows"] == ["CI"]
    assert on["workflow_run"]["types"] == ["completed"]
    assert on["workflow_run"]["branches"] == ["main"]
    assert "push" not in on


def test_the_deploy_refuses_a_ci_run_that_did_not_succeed(deploy: dict[str, Any]) -> None:
    # `types: [completed]` fires for a failed run too.
    condition = deploy["jobs"]["gate"]["if"]
    assert "github.event.workflow_run.conclusion == 'success'" in condition
    assert "github.event.workflow_run.head_branch == 'main'" in condition
    assert "github.event_name == 'workflow_dispatch'" in condition


def test_a_dispatch_can_name_a_commit_or_ask_for_a_rollback(deploy: dict[str, Any]) -> None:
    inputs = triggers(deploy)["workflow_dispatch"]["inputs"]
    assert inputs["sha"]["type"] == "string"
    assert inputs["rollback"]["type"] == "boolean"
    assert inputs["rollback"]["default"] is False


def test_the_ci_gate_runs_before_the_reviewer_is_asked(deploy: dict[str, Any]) -> None:
    # The `production` environment carries a required reviewer. Proving CI green
    # in a job that does not declare the environment means the approval prompt
    # only ever appears for a commit that has already passed.
    assert "environment" not in deploy["jobs"]["gate"]
    assert deploy["jobs"]["deploy"]["environment"] == "production"
    assert deploy["jobs"]["deploy"]["needs"] == "gate"


def test_the_gate_names_every_ci_job(deploy: dict[str, Any], ci: dict[str, Any]) -> None:
    gated = set(deploy["env"]["REQUIRED_CHECKS"].split())
    gated |= set(deploy["env"]["CONDITIONAL_CHECKS"].split())
    declared = set(ci["jobs"])
    assert gated == declared, (
        "the deploy gate and ci.yml have drifted: "
        f"missing from the gate {sorted(declared - gated)}, "
        f"named by the gate but not declared by ci.yml {sorted(gated - declared)}"
    )


def test_only_the_path_filtered_ci_jobs_may_be_skipped(
    deploy: dict[str, Any], ci: dict[str, Any]
) -> None:
    # A `skipped` conclusion is accepted for exactly the jobs ci.yml gates on a
    # path filter, and for no others: everywhere else a skip is what a deleted or
    # mis-conditioned job looks like, and treating that as a pass is the whole
    # failure this gate exists to prevent.
    #
    # `publish-images` carries an `if` too, and is deliberately *not* skippable:
    # its condition is an event guard rather than a path filter, and a skipped
    # publish means this commit has no images in GHCR at all.
    conditional = set(deploy["env"]["CONDITIONAL_CHECKS"].split())
    path_filtered = {
        name
        for name, job in ci["jobs"].items()
        if "needs.changed-paths.outputs" in str(job.get("if", ""))
    }
    assert conditional == path_filtered, (
        "ci.yml's path-filtered jobs and the gate's skippable list have drifted: "
        f"path-filtered but not skippable {sorted(path_filtered - conditional)}, "
        f"skippable but not path-filtered {sorted(conditional - path_filtered)}"
    )
    required = set(deploy["env"]["REQUIRED_CHECKS"].split())
    assert not (conditional & required)
    assert "publish-images" in required


def test_the_two_step_level_gates_are_still_inside_the_jobs_the_gate_requires(
    deploy: dict[str, Any], ci: dict[str, Any]
) -> None:
    # `api-contract-check` and `web-api-types-check` are named as required gates
    # by the deployment plan but exist as steps, not jobs. Requiring `lint` and
    # `frontend` is equivalent only while these two commands live there.
    required = set(deploy["env"]["REQUIRED_CHECKS"].split())
    assert {"lint", "frontend"} <= required

    lint_commands = " ".join(step.get("run", "") for step in ci["jobs"]["lint"]["steps"])
    assert "scripts.generate_openapi --check" in lint_commands

    frontend_commands = " ".join(step.get("run", "") for step in ci["jobs"]["frontend"]["steps"])
    assert "api:types:check" in frontend_commands


def test_the_gate_refuses_a_run_that_is_not_completed_and_successful(
    deploy: dict[str, Any],
) -> None:
    script = step_by_name(deploy, "gate", "Require every CI check to be green on this commit")[
        "run"
    ]
    # Only `completed` plus `success` may pass, and each required check is then
    # confirmed in its own right, because a skipped job does not fail its run.
    assert '"$status" != "completed"' in script
    assert '"$conclusion" != "success"' in script
    assert '"$result" != "success"' in script
    assert "Refusing to deploy" in script


def test_a_rollback_is_not_blocked_on_a_ci_query(deploy: dict[str, Any]) -> None:
    # The release a rollback returns to passed this same gate when it was
    # deployed. Re-asking GitHub about a commit that is not being shipped is how
    # an incident gets longer.
    check = step_by_name(deploy, "gate", "Require every CI check to be green on this commit")
    assert check["if"] == "steps.resolve.outputs.rollback != 'true'"


def test_only_a_forty_character_sha_reaches_the_host(deploy: dict[str, Any]) -> None:
    # The resolved value is interpolated into a remote command line, so its
    # shape is validated where it is decided rather than where it is used.
    script = step_by_name(deploy, "gate", "Resolve the commit this run acts on")["run"]
    assert "'^[0-9a-f]{40}$'" in script
    assert "Refusing to deploy" in script


def test_the_release_command_is_the_one_the_host_expects(deploy: dict[str, Any]) -> None:
    script = step_by_name(deploy, "deploy", "Run the release on the host")["run"]
    # The checkout has to match the images: deploy.sh reads the compose file,
    # the Makefile targets and the release job out of the working tree, so a
    # tree at a different commit than IMAGE_TAG is a stack described by one
    # release and running another.
    assert (
        f"cd {HOST_CHECKOUT} && git fetch --tags origin && "
        "git checkout --detach ${DEPLOY_SHA} && infra/deploy/deploy.sh ${DEPLOY_SHA}"
    ) in script
    assert f"cd {HOST_CHECKOUT} && infra/deploy/deploy.sh --rollback" in script
    # Nothing is built or copied here; the payload is one command naming a commit.
    assert "actions/checkout" not in DEPLOY_WORKFLOW.read_text(encoding="utf-8")


def test_the_release_is_believed_only_when_the_host_prints_its_sentinel(
    deploy: dict[str, Any],
) -> None:
    script = step_by_name(deploy, "deploy", "Run the release on the host")["run"]
    assert f'sentinel="{DEPLOY_SENTINEL}"' in script
    assert f'sentinel="{ROLLBACK_SENTINEL}"' in script
    # A remote command that dies with its connection can still exit 0 through
    # some shells, so the sentinel decides and the exit status is a second
    # opinion — both have to agree.
    assert 'grep -Eq "(^|[[:space:]])${sentinel}([[:space:]]|$)"' in script
    assert "never printed" in script
    assert '"${ssh_status}" -ne 0' in script


def test_an_automatic_rollback_still_fails_the_run(deploy: dict[str, Any]) -> None:
    # deploy.sh rolls back by itself when verification fails, and the box ends
    # up serving. Nobody should read a green tick and believe the commit
    # shipped.
    script = step_by_name(deploy, "deploy", "Run the release on the host")["run"]
    assert f'[ "${{sentinel}}" = "{DEPLOY_SENTINEL}" ]' in script
    assert "failed verification and the host rolled back" in script


def test_the_rollback_target_is_published_as_an_artifact(deploy: dict[str, Any]) -> None:
    read_back = step_by_name(deploy, "deploy", "Upload the recorded rollback target")
    upload = step_by_name(deploy, "deploy", "Publish the rollback target")
    # Both uploaded even when the release failed: that is the run whose rollback
    # target somebody actually needs. `always()` still yields to the host guard,
    # because a dormant run never opened an SSH agent and has nothing to read.
    assert read_back["if"] == "always() && steps.configured.outputs.configured == 'true'"
    assert upload["if"] == "always() && steps.configured.outputs.configured == 'true'"
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert upload["with"]["name"] == "rollback-target"
    # deploy.sh records current→previous before it pulls anything, so `previous`
    # names whatever was serving when this run started.
    assert f"{HOST_CHECKOUT}/.release" in read_back["run"]
    assert "current=" in read_back["run"]
    assert "previous=" in read_back["run"]


def test_the_deploy_script_prints_the_sentinels_the_workflow_greps_for() -> None:
    # The one coupling neither file can protect on its own: the workflow's
    # verdict is a grep for a literal that lives in a shell script.
    assert DEPLOY_SCRIPT.is_file(), "infra/deploy/deploy.sh is what the deploy workflow runs"
    script = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    assert DEPLOY_SENTINEL in script
    assert ROLLBACK_SENTINEL in script


# --- Canary: what watches the box -------------------------------------------


def test_the_canary_runs_monthly_and_on_demand(canary: dict[str, Any]) -> None:
    on = triggers(canary)
    assert on["schedule"] == [{"cron": "0 0 1 * *"}]
    assert "workflow_dispatch" in on


def test_the_canary_shares_the_deploy_workflow_s_concurrency_lane(
    deploy: dict[str, Any], canary: dict[str, Any]
) -> None:
    # A canary that runs the verify matrix in the middle of a release reports on
    # a system that is mid-change, against a half-restarted stack. Cancelling is
    # worse than queueing: a release cancelled between `pull` and `up -d` leaves
    # the box in neither state.
    assert deploy["concurrency"]["group"] == CONCURRENCY_GROUP
    assert canary["concurrency"]["group"] == CONCURRENCY_GROUP
    assert deploy["concurrency"]["cancel-in-progress"] is False
    assert canary["concurrency"]["cancel-in-progress"] is False


def test_the_canary_gates_nothing_and_needs_its_own_environment(canary: dict[str, Any]) -> None:
    # A required reviewer on a job that fires every thirty minutes is no canary
    # at all: the runs queue for approval and are superseded before anyone reads
    # them. Acceptable only because the canary changes nothing.
    assert canary["jobs"]["canary"]["environment"] == "production-canary"


def test_the_canary_never_ships_code(canary: dict[str, Any]) -> None:
    body = CANARY_WORKFLOW.read_text(encoding="utf-8")
    for forbidden in ("actions/checkout", "docker compose pull", "IMAGE_TAG", "deploy.sh"):
        assert forbidden not in body, f"the canary must not {forbidden!r}"


def test_the_canary_s_verdict_comes_from_prod_verify(canary: dict[str, Any]) -> None:
    verify = step_by_name(canary, "canary", "Run the verification matrix")["run"]
    assert f"cd {HOST_CHECKOUT} && make prod-verify" in verify
    assert VERIFY_SENTINEL in verify
    assert "exit 1" in verify
    # A partial matrix prints VERIFY-SUBSET-OK, and a canary that accepted it
    # would be reporting on checks it never ran.
    assert VERIFY_SUBSET_SENTINEL not in verify


def test_the_loadcheck_never_runs_on_the_schedule(canary: dict[str, Any]) -> None:
    # ADR 0014's shipped defaults refuse 37.9% of one keep-alive subject's
    # requests at 5/s, so a scheduled loadcheck would report red every half hour
    # for a limiter behaving exactly as configured — and teach everyone to
    # ignore the canary.
    inputs = triggers(canary)["workflow_dispatch"]["inputs"]
    assert inputs["run_loadcheck"]["default"] is False
    condition = step_by_name(canary, "canary", "Run the advisory loadcheck")["if"]
    assert "github.event_name == 'workflow_dispatch'" in condition
    assert "inputs.run_loadcheck" in condition


def test_the_loadcheck_records_latency_and_never_decides_the_run(canary: dict[str, Any]) -> None:
    loadcheck = step_by_name(canary, "canary", "Run the advisory loadcheck")["run"]
    assert f"cd {HOST_CHECKOUT} && make prod-load" in loadcheck
    # V-10 records latency with no verdict: the SLO's only authority is the
    # pinned k6 gate in CI (ADR 0010).
    assert "exit 1" not in loadcheck
    assert "::warning::" in loadcheck


# --- Both: how the box is reached -------------------------------------------


@pytest.mark.parametrize("workflow,job", [(DEPLOY_WORKFLOW, "deploy"), (CANARY_WORKFLOW, "canary")])
def test_the_host_key_is_pinned_from_the_secret(workflow: Path, job: str) -> None:
    document = load_workflow(workflow)
    agent = step_by_name(document, job, "Open an SSH agent holding the deploy key")
    assert agent["env"]["DEPLOY_KNOWN_HOSTS"] == "${{ secrets.DEPLOY_KNOWN_HOSTS }}"
    assert agent["env"]["DEPLOY_SSH_KEY"] == "${{ secrets.DEPLOY_SSH_KEY }}"
    assert (
        'printf \'%s\\n\' "${DEPLOY_KNOWN_HOSTS}" > "${RUNNER_TEMP}/ssh/known_hosts"'
        in agent["run"]
    )
    # The key is held by an agent for the life of the job and never written to
    # disk.
    assert "ssh-add -" in agent["run"]

    body = workflow.read_text(encoding="utf-8")
    for ssh_call in re.findall(r"ssh -o [^\n]*", body):
        assert "StrictHostKeyChecking=yes" in ssh_call
    assert body.count("UserKnownHostsFile=") == body.count("StrictHostKeyChecking=")


@pytest.mark.parametrize("path", WORKFLOWS)
def test_no_workflow_accepts_whatever_key_answers_on_port_22(path: Path) -> None:
    # An SSH session that trusts an unverified host key authenticates a
    # Docker-capable account to an unknown server. There is no situation in this
    # repository where that is the right tradeoff, including the first deploy —
    # `ssh-keyscan` into the DEPLOY_KNOWN_HOSTS secret is the setup step.
    assert "StrictHostKeyChecking=no" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("workflow,job", [(DEPLOY_WORKFLOW, "deploy"), (CANARY_WORKFLOW, "canary")])
def test_the_host_and_user_are_configuration_rather_than_secrets(workflow: Path, job: str) -> None:
    env = load_workflow(workflow)["jobs"][job]["env"]
    # A wrong hostname should be visible in the log rather than masked into
    # three asterisks.
    assert env["DEPLOY_HOST"] == "${{ vars.DEPLOY_HOST }}"
    assert env["DEPLOY_USER"] == "${{ vars.DEPLOY_USER }}"


@pytest.mark.parametrize("path", [DEPLOY_WORKFLOW, CANARY_WORKFLOW])
def test_neither_workflow_carries_a_credential_beyond_the_deploy_key(path: Path) -> None:
    body = path.read_text(encoding="utf-8")
    for reference in re.findall(r"\$\{\{\s*secrets\.([A-Za-z0-9_]+)\s*\}\}", body):
        assert reference in {"DEPLOY_SSH_KEY", "DEPLOY_KNOWN_HOSTS"}


@pytest.mark.parametrize("path", WORKFLOWS)
def test_no_workflow_still_talks_to_railway(path: Path) -> None:
    assert "railway" not in path.read_text(encoding="utf-8").lower()


@pytest.mark.parametrize("path", WORKFLOWS)
def test_every_embedded_script_parses(path: Path) -> None:
    bash = shutil.which("bash")
    if bash is None:  # pragma: no cover - bash is present on every CI runner
        pytest.skip("bash is not available")

    document = load_workflow(path)
    for job_name, job in document["jobs"].items():
        for step in job["steps"]:
            script = step.get("run")
            if script is None:
                continue
            # GitHub substitutes expressions before the shell ever sees them.
            script = re.sub(r"\$\{\{[^}]*\}\}", "expression", script)
            result = subprocess.run(
                [bash, "-n"], input=script, capture_output=True, text=True, check=False
            )
            assert (
                result.returncode == 0
            ), f"{path.name}:{job_name}:{step.get('name', step.get('uses'))}\n{result.stderr}"


def test_the_sentinel_match_rejects_a_partial_verify_run(canary: dict[str, Any]) -> None:
    grep = shutil.which("grep")
    if grep is None:  # pragma: no cover - grep is present on every CI runner
        pytest.skip("grep is not available")

    script = step_by_name(canary, "canary", "Run the verification matrix")["run"]
    match = re.search(r'grep -Eq "(?P<pattern>[^"]+)"', script)
    assert match is not None, "the canary no longer greps for the verify sentinel"
    pattern = match.group("pattern")

    def matches(line: str) -> bool:
        return (
            subprocess.run(
                [grep, "-Eq", pattern], input=line, capture_output=True, text=True, check=False
            ).returncode
            == 0
        )

    assert matches(VERIFY_SENTINEL)
    assert matches(f"{VERIFY_SENTINEL} in 41s")
    # The one that matters: a run that skipped rows of the matrix prints
    # VERIFY-SUBSET-OK, and must not be read as a full pass.
    assert not matches(VERIFY_SUBSET_SENTINEL)
    assert not matches("not-VERIFY-OK-either")


@pytest.mark.parametrize("workflow,job", [(DEPLOY_WORKFLOW, "deploy"), (CANARY_WORKFLOW, "canary")])
def test_the_box_is_reached_only_once_something_is_configured(workflow: Path, job: str) -> None:
    # Both workflows start firing the moment they land on main — the canary on
    # its schedule, the deploy on every green CI run — and that is well before a
    # host exists to reach. A run that goes red on every merge for a deployment
    # nobody has created yet is how a red deploy stops being read at all, so an
    # environment carrying none of the four values makes the run a green no-op
    # with a notice. Anything *partially* configured must still fail loudly, or
    # a deleted secret would turn releases into silence rather than an error.
    job_steps = steps(load_workflow(workflow), job)
    guard = job_steps[0]
    assert guard["id"] == "configured"
    assert guard["name"] == "Decide whether a production host exists yet"

    script = guard["run"]
    for name in ("DEPLOY_HOST", "DEPLOY_USER", "DEPLOY_SSH_KEY", "DEPLOY_KNOWN_HOSTS"):
        assert f'-z "${{{name}}}"' in script
    # The two secrets have to be read into this step's own env: a secret is not
    # legible from the `if` expressions the guard's output then feeds.
    assert guard["env"]["DEPLOY_SSH_KEY"] == "${{ secrets.DEPLOY_SSH_KEY }}"
    assert guard["env"]["DEPLOY_KNOWN_HOSTS"] == "${{ secrets.DEPLOY_KNOWN_HOSTS }}"
    # Dormancy is the conjunction of all four being empty, never a disjunction.
    assert script.count("&&") >= 3 and "||" not in script.split("then")[0]
    assert "configured=false" in script and "configured=true" in script
    # The notice has to say what to do about it, not just that nothing happened.
    assert "docs/deployment-runbook.md section 6" in script

    # Every later step, with no exceptions: one that ran anyway would be the
    # failure the guard exists to prevent. The `always()` and `!cancelled()`
    # steps carry the guard as a conjunct rather than instead of it — `always()`
    # alone would run them on a dormant run, against an SSH agent that was never
    # opened.
    for step in job_steps[1:]:
        condition = str(step.get("if", ""))
        assert (
            "steps.configured.outputs.configured == 'true'" in condition
        ), f"{workflow.name}:{job}: step {step.get('name')!r} runs without the host guard"
        if "always()" in condition or "cancelled()" in condition:
            assert "&&" in condition
