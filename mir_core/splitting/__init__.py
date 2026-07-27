"""Task-independent, deterministic dataset splitting."""

from .plan import (
    SPLIT_ALGORITHM_VERSION,
    SPLIT_PLAN_SCHEMA_VERSION,
    SPLIT_ROLES,
    DatasetSplitPlan,
    FoldAssignment,
    SplitPlan,
    SplitRecord,
    SplitRole,
    build_split_plan,
    load_split_plan,
    write_split_plan,
)

__all__ = [
    "SPLIT_ALGORITHM_VERSION",
    "SPLIT_PLAN_SCHEMA_VERSION",
    "SPLIT_ROLES",
    "DatasetSplitPlan",
    "FoldAssignment",
    "SplitPlan",
    "SplitRecord",
    "SplitRole",
    "build_split_plan",
    "load_split_plan",
    "write_split_plan",
]
