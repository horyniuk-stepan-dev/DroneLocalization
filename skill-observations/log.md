# Skill Observation Log

Observations captured during task-oriented work. Each entry identifies a
potential skill improvement or new skill opportunity.

**Status key:** OPEN = not yet actioned | ACTIONED = skill updated/created |
DECLINED = user decided not to pursue

---

## 2026-09-18 — Drone localization implementation session

### Observation 1: Task-observer instructions are too large for reliable activation

**Status:** OPEN
**Date:** 2026-09-18
**Session context:** A substantive coding session required loading task-observer before implementation.
**Skill:** task-observer
**Type:** open-source
**Phase/Area:** Lean Content and Session Start Protocol

**Issue:** The live SKILL.md is 1,524 lines and could not be read in one tool response without truncation. The agent had to split it into multiple reads before beginning the user's work. Much of the loaded material concerns skill authoring, licensing, scheduled reviews, delivery, and archival rather than the immediate observation workflow. This makes the mandatory start-of-session activation slow and increases the chance that the operational rules are only partially loaded.

**Suggested improvement:** Apply the skill's own Lean Content rule to the Task Observer. Keep a short executable SKILL.md containing activation, session-start checks, the observation trigger/format, safe logging, and the end-of-session protocol. Move taxonomy, licensing templates, comprehensive-review mechanics, environment-specific delivery, and extended rationale into directly linked reference files that are loaded only when their trigger fires. Add a small pre-flight check listing which references are required for observation-only sessions versus review or skill-authoring sessions.

**Principle:** A skill that must load on every substantive task should keep its always-loaded instructions minimal and route infrequent workflows to triggered references; otherwise the enforcement mechanism becomes the main source of task latency and partial compliance.
