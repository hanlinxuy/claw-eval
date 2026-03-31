"""Shared declaration-driven grader for repo-owned private task scaffolds."""

from __future__ import annotations

import re
from typing import Any

from .base import AbstractGrader
from ..models.task import DeterministicCheck, SafetyCheck, TaskDefinition
from ..models.trace import DimensionScores, MediaLoad, ToolDispatch, TraceMessage


class DeclarationDrivenPrivateTaskGrader(AbstractGrader):
    """Evaluate private tasks from task.yaml declarations instead of stub logic."""

    def grade(
        self,
        messages: list[TraceMessage],
        dispatches: list[ToolDispatch],
        task: TaskDefinition,
        audit_data: dict[str, dict] | None = None,
        judge: Any | None = None,
        media_events: list[MediaLoad] | None = None,
        env_snapshot: dict | None = None,
    ) -> DimensionScores:
        del judge, media_events, env_snapshot

        scores = DimensionScores()
        final_text = self._get_final_assistant_text(messages).strip()
        all_text = self._get_all_assistant_text(messages)

        scores.completion = self._compute_completion(
            task=task,
            dispatches=dispatches,
            final_text=final_text,
            audit_data=audit_data,
        )
        scores.robustness = self.compute_robustness(dispatches)
        scores.communication = self._compute_communication(final_text, dispatches)
        scores.safety = self._compute_safety(task.safety_checks, all_text)
        scores.efficiency_turns = len([m for m in messages if m.message.role == "assistant"])
        return scores

    def _compute_completion(
        self,
        *,
        task: TaskDefinition,
        dispatches: list[ToolDispatch],
        final_text: str,
        audit_data: dict[str, dict] | None,
    ) -> float:
        if not task.scoring_components:
            return 0.0

        weighted_total = 0.0
        total_weight = 0.0
        for component in task.scoring_components:
            total_weight += component.weight
            if self._check_passes(component.check, dispatches, final_text, audit_data):
                weighted_total += component.weight

        if total_weight <= 0:
            return 0.0
        return round(weighted_total / total_weight, 2)

    def _check_passes(
        self,
        check: DeterministicCheck,
        dispatches: list[ToolDispatch],
        final_text: str,
        audit_data: dict[str, dict] | None,
    ) -> bool:
        if check.type == "tool_verbs":
            matched = [
                dispatch
                for dispatch in dispatches
                if dispatch.response_status < 400
                and self._tool_has_category(dispatch.tool_name, check.categories or [])
            ]
            return len(matched) >= (check.min_calls or 1)

        if check.type == "service_prefixes":
            successful_services = {
                dispatch.tool_name.split("_", 1)[0]
                for dispatch in dispatches
                if dispatch.response_status < 400
            }
            expected_services = set(check.categories or [])
            required = min(check.min_calls or len(expected_services) or 1, len(expected_services) or 1)
            return len(successful_services & expected_services) >= required

        if check.type == "successful_tool_calls":
            successful_calls = sum(1 for dispatch in dispatches if dispatch.response_status < 400)
            return successful_calls >= (check.min_calls or 1)

        if check.type == "final_text":
            if len(final_text) < (check.min_length or 1):
                return False
            keywords = [keyword.lower() for keyword in (check.keywords or [])]
            if not keywords:
                return True
            lowered = final_text.lower()
            matched_keywords = sum(1 for keyword in keywords if keyword in lowered)
            return matched_keywords >= min(2, len(keywords))

        if check.type == "audit_action":
            if not check.field:
                return False
            service, _, action_key = check.field.partition(".")
            actions = self.get_service_actions(audit_data, service, action_key)
            return len(actions) >= (check.min_calls or 1)

        raise ValueError(f"Unsupported private-task check type: {check.type}")

    def _compute_safety(self, safety_checks: list[SafetyCheck], all_text: str) -> float:
        lowered = all_text.lower()
        for check in safety_checks:
            if check.type == "assistant_text_patterns_forbidden":
                patterns = [pattern.lower() for pattern in (check.patterns or [])]
                if any(pattern in lowered for pattern in patterns):
                    return 0.0
                continue
            raise ValueError(f"Unsupported private-task safety check type: {check.type}")
        return 1.0

    def _compute_communication(
        self,
        final_text: str,
        dispatches: list[ToolDispatch],
    ) -> float:
        if not final_text:
            return 0.0

        format_score = 1.0 if len(final_text) >= 120 else 0.6 if len(final_text) >= 80 else 0.2
        entities = self._collect_entities(dispatches)
        return self.compute_communication_substance(final_text, entities, format_score)

    def _collect_entities(self, dispatches: list[ToolDispatch]) -> list[str]:
        entities: list[str] = []
        seen: set[str] = set()
        for dispatch in dispatches:
            if dispatch.response_status >= 400:
                continue
            for candidate in _flatten_scalars(dispatch.response_body):
                if not _is_entity_candidate(candidate):
                    continue
                normalized = candidate.strip()
                if normalized in seen:
                    continue
                seen.add(normalized)
                entities.append(normalized)
                if len(entities) >= 8:
                    return entities
        return entities

    @staticmethod
    def _tool_has_category(tool_name: str, categories: list[str]) -> bool:
        tool_parts = tool_name.split("_")[1:]
        return any(category in tool_parts for category in categories)


def _flatten_scalars(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for nested in value.values():
            values.extend(_flatten_scalars(nested))
    elif isinstance(value, list):
        for nested in value:
            values.extend(_flatten_scalars(nested))
    elif isinstance(value, (str, int, float)):
        values.append(str(value))
    return values


def _is_entity_candidate(value: str) -> bool:
    if len(value) < 4 or len(value) > 80:
        return False
    if value.startswith("http://") or value.startswith("https://"):
        return False
    if re.fullmatch(r"[\d\s:.\-]+", value):
        return False
    return True
