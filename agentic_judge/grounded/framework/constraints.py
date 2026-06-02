"""Typed hard-constraint predicates over the agent's summary JSON.

A HardConstraint is a deterministic check that returns (passed: bool,
detail: str).  Constraints take the parsed summary JSON dict the agent
emitted, plus optional task context, and return a verdict.  They never
call an LLM.

The framework runs every defined constraint and reports pass-rate + the
list of specific failures, with quoted evidence ('option #3 price=$82 > 75').
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class HardConstraintResult:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


class HardConstraint:
    """Base class.  Subclasses override .check(summary, ctx)."""
    name: str = "unnamed"

    def check(self, summary: dict, ctx: dict | None = None) -> HardConstraintResult:
        raise NotImplementedError


def _get(summary: dict, dotted_path: str, default=None):
    """Navigate a.b.c through a dict; return default on miss."""
    cur: Any = summary
    for part in dotted_path.split("."):
        if cur is None:
            return default
        if isinstance(cur, list) and part.isdigit():
            i = int(part)
            cur = cur[i] if 0 <= i < len(cur) else None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return default
    return cur if cur is not None else default


# ---------------------------------------------------------------------------
# Concrete predicates
# ---------------------------------------------------------------------------

class FieldPresent(HardConstraint):
    """The summary JSON must have a non-empty value at `path`."""
    def __init__(self, path: str, name: str | None = None):
        self.path = path
        self.name = name or f"field_present:{path}"

    def check(self, summary, ctx=None):
        v = _get(summary, self.path)
        ok = v not in (None, "", [], {})
        return HardConstraintResult(self.name, ok,
            f"{self.path}={v!r}" if ok else f"missing or empty: {self.path}")


class ListLengthInRange(HardConstraint):
    def __init__(self, path: str, low: int, high: int, name: str | None = None):
        self.path, self.low, self.high = path, low, high
        self.name = name or f"list_len:{path} in [{low},{high}]"

    def check(self, summary, ctx=None):
        v = _get(summary, self.path, [])
        if not isinstance(v, list):
            return HardConstraintResult(self.name, False, f"{self.path} is not a list")
        ok = self.low <= len(v) <= self.high
        return HardConstraintResult(self.name, ok,
            f"len({self.path})={len(v)}, expected [{self.low},{self.high}]")


class AllValuesLE(HardConstraint):
    """Every numeric value at `list_path[*].sub_path` is <= ceiling."""
    def __init__(self, list_path: str, sub_path: str, ceiling: float,
                 name: str | None = None):
        self.list_path, self.sub_path, self.ceiling = list_path, sub_path, ceiling
        self.name = name or f"all({list_path}.*.{sub_path} <= {ceiling})"

    def check(self, summary, ctx=None):
        items = _get(summary, self.list_path, []) or []
        if not isinstance(items, list):
            return HardConstraintResult(self.name, False, f"{self.list_path} is not a list")
        violations = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            v = _get(item, self.sub_path) if "." in self.sub_path else item.get(self.sub_path)
            if v is None:
                violations.append(f"item[{i}].{self.sub_path} missing")
                continue
            try:
                vn = float(v)
            except (TypeError, ValueError):
                violations.append(f"item[{i}].{self.sub_path}={v!r} not numeric")
                continue
            if vn > self.ceiling:
                violations.append(f"item[{i}].{self.sub_path}={vn} > {self.ceiling}")
        ok = not violations
        return HardConstraintResult(self.name, ok,
            "all within ceiling" if ok else f"violations: {violations[:5]}")


class AllValuesBetween(HardConstraint):
    """Every numeric `list_path[*].sub_path` is in [low, high]."""
    def __init__(self, list_path: str, sub_path: str, low: float, high: float,
                 name: str | None = None):
        self.list_path, self.sub_path, self.low, self.high = list_path, sub_path, low, high
        self.name = name or f"all({list_path}.*.{sub_path} in [{low},{high}])"

    def check(self, summary, ctx=None):
        items = _get(summary, self.list_path, []) or []
        if not isinstance(items, list):
            return HardConstraintResult(self.name, False, f"{self.list_path} is not a list")
        violations = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            v = item.get(self.sub_path)
            if v is None:
                violations.append(f"item[{i}].{self.sub_path} missing")
                continue
            try:
                vn = float(v)
            except (TypeError, ValueError):
                violations.append(f"item[{i}].{self.sub_path}={v!r} not numeric")
                continue
            if not (self.low <= vn <= self.high):
                violations.append(f"item[{i}].{self.sub_path}={vn} outside [{self.low},{self.high}]")
        ok = not violations
        return HardConstraintResult(self.name, ok,
            "all within range" if ok else f"violations: {violations[:5]}")


class EqualsValue(HardConstraint):
    def __init__(self, path: str, expected: Any, name: str | None = None,
                 case_insensitive: bool = False):
        self.path, self.expected, self.ci = path, expected, case_insensitive
        self.name = name or f"equals:{path} == {expected!r}"

    def check(self, summary, ctx=None):
        v = _get(summary, self.path)
        if self.ci and isinstance(v, str) and isinstance(self.expected, str):
            ok = v.strip().lower() == self.expected.strip().lower()
        else:
            ok = v == self.expected
        return HardConstraintResult(self.name, ok,
            f"{self.path}={v!r}" + ("" if ok else f", expected {self.expected!r}"))


class ContainsAllSubstrings(HardConstraint):
    """The value at `path` (str or list[str]) contains every required substring,
    case-insensitively."""
    def __init__(self, path: str, required: list[str], name: str | None = None):
        self.path, self.required = path, required
        self.name = name or f"contains_all({path}, {required})"

    def check(self, summary, ctx=None):
        v = _get(summary, self.path)
        if isinstance(v, list):
            haystack = " ".join(str(x) for x in v)
        elif isinstance(v, dict):
            haystack = " ".join(str(x) for x in v.values())
        else:
            haystack = str(v or "")
        h = haystack.lower()
        missing = [r for r in self.required if r.lower() not in h]
        ok = not missing
        return HardConstraintResult(self.name, ok,
            f"all present in {self.path}" if ok else f"missing: {missing}")


class AllURLsMatch(HardConstraint):
    """Every item at `list_path[*].url_field` matches the given URL regex."""
    def __init__(self, list_path: str, url_field: str, url_pattern: str,
                 name: str | None = None):
        self.list_path, self.url_field, self.pat = list_path, url_field, re.compile(url_pattern, re.I)
        self.name = name or f"all_urls:{list_path}.*.{url_field} matches {url_pattern!r}"

    def check(self, summary, ctx=None):
        items = _get(summary, self.list_path, []) or []
        if not isinstance(items, list):
            return HardConstraintResult(self.name, False, f"{self.list_path} not a list")
        bad = []
        for i, item in enumerate(items):
            u = (item or {}).get(self.url_field)
            if not u or not self.pat.search(u):
                bad.append(f"item[{i}].{self.url_field}={u!r}")
        ok = not bad
        return HardConstraintResult(self.name, ok,
            "all URLs match" if ok else f"non-matching: {bad[:3]}")


class AtLeastOneOf(HardConstraint):
    """Any one of the inner constraints passing makes this pass."""
    def __init__(self, *inner: HardConstraint, name: str | None = None):
        self.inner = inner
        self.name = name or "at_least_one_of(" + " | ".join(c.name for c in inner) + ")"

    def check(self, summary, ctx=None):
        sub = [c.check(summary, ctx) for c in self.inner]
        ok = any(s.passed for s in sub)
        return HardConstraintResult(self.name, ok,
            "; ".join(f"{s.name}={'P' if s.passed else 'F'}({s.detail[:60]})" for s in sub))


class JSONSchemaConforms(HardConstraint):
    """The summary JSON has all REQUIRED fields from the SummarySchema.

    Supports two path forms:
      'foo.bar'        — plain dotted lookup.
      'foo[].bar.baz'  — `foo` is a list; every element must have `bar.baz`.
                         An empty list also fails.
    """
    def __init__(self, required_paths: list[str], name: str = "summary_schema_conforms"):
        self.required = required_paths
        self.name = name

    def _check_path(self, summary, path: str) -> bool:
        # List-element path: split at '[].'
        if "[]" in path:
            head, _, tail = path.partition("[].")
            container = _get(summary, head)
            if not isinstance(container, list) or not container:
                return False
            for item in container:
                if not isinstance(item, dict):
                    return False
                v = _get(item, tail) if "." in tail else item.get(tail)
                if v in (None, "", [], {}):
                    return False
            return True
        # Plain dotted path
        v = _get(summary, path)
        return v not in (None, "", [], {})

    def check(self, summary, ctx=None):
        missing = [p for p in self.required if not self._check_path(summary, p)]
        ok = not missing
        return HardConstraintResult(self.name, ok,
            "all required fields present" if ok else f"missing: {missing}")


# Custom predicate via callable, for the long tail
class Custom(HardConstraint):
    def __init__(self, name: str, fn: Callable[[dict, dict | None], tuple[bool, str]]):
        self.name = name
        self.fn = fn

    def check(self, summary, ctx=None):
        ok, detail = self.fn(summary, ctx)
        return HardConstraintResult(self.name, ok, detail)
