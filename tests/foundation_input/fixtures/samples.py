from __future__ import annotations

from datetime import datetime, timezone


def sample_mapping(dataset_name: str) -> dict:
    return {
        "sample_id": f"{dataset_name}:sample-001",
        "dataset_name": dataset_name,
        "split": "train",
        "user_id": "user-001",
        "cutoff_time": datetime(2025, 9, 19, 2, 37, 25, tzinfo=timezone.utc),
        "cutoff_inclusive": True,
        "history_order_known": True,
        "history_truncated": False,
        "history_group_truncated": False,
        "profile": {
            "active_days_30d": 5,
            "event_count_30d": 2,
        },
        "events": [
            {
                "event_id": "user-001:0:0",
                "source_record_id": "user-001:0",
                "source_record_index": 0,
                "action_index": 0,
                "source_action_count": 1,
                "event_time": datetime(2025, 9, 18, tzinfo=timezone.utc),
                "time_precision": "day",
                "eligibility_time": datetime(2025, 9, 19, tzinfo=timezone.utc),
                "eligibility_inclusive": False,
                "scenario": "E-commerce",
                "event_type": "click",
                "context": {"product_id": "product-1"},
                "action": {"type": "click", "click_cnt": 1},
            }
        ],
    }
