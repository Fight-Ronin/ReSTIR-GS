# Documentation Index

Use this directory as a project log plus handoff reference. The current docs are
the source of truth for the active renderer; the phase docs preserve context for
decisions that led there.

## Start Here

- `active_workflow.md`: active commands for downloading assets, running the
  baseline, and inspecting results.
- `active_baseline_handoff.md`: current baseline definition, outputs, known
  limits, and future-work guidance.
- `current_architecture.md`: active modules, package boundaries, and removed
  legacy surfaces.
- `current_milestone_snapshot.md`: compact status snapshot for the current
  handoff point.

## Cleanup And Wrap-Up

- `phase55_project_wrap_up.md`: final selected-fast visibility conclusion and
  retained policy.
- `phase56_codebase_cleanup_refactor_plan.md`: cleanup boundaries, completed
  splits, and stop criteria.

## Historical Phase Notes

The `phase*.md` files are chronological implementation notes. Treat them as
decision history, not as the current runbook, unless a current handoff doc links
to a specific phase.

Useful clusters:

- `phase25`, `phase48`, `phase49`: interactive viewer and local asset
  generation context.
- `phase27` through `phase32`: aligned assets, lighting, active renderer, and
  temporal compatibility.
- `phase37` through `phase46`: visibility target, temporal filtering, and demo
  snapshot work.
- `phase50` through `phase54`: selected-fast visibility profiling and quality
  validation.

## Documentation Rules

- Keep active commands in `README.md`, `active_workflow.md`, and
  `scripts/README.md` aligned.
- Put new architectural decisions in `current_architecture.md` only when they
  reflect maintained behavior.
- Keep experimental or one-off measurements in phase notes so they do not look
  like active policy.
