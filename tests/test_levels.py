from wishrec.levels import build_levels, level_labels, render_l0_minus

FIELDS = {
    "intent": "Shift the digits of an integer right by a given amount.",
    "signature_types": "def circular_shift(x: int, shift: int) -> str:",
    "signature_untyped": "def circular_shift(x, shift):",
    "examples": ">>> circular_shift(12, 1)\n'21'",
    "constraints": "- If shift exceeds the number of digits, reverse the digits.",
    "error_behaviour": "- Raises ValueError for a negative shift.",
}


def test_l0_contains_all_fields():
    built = build_levels(FIELDS, signature_is_code=True)
    assert built["levels"]["L0"].startswith("Intent:")
    assert "Signature and types:" in built["levels"]["L0"]
    assert "Examples:" in built["levels"]["L0"]
    assert "Constraints and edge cases:" in built["levels"]["L0"]
    assert "Error behaviour:" in built["levels"]["L0"]
    assert "def circular_shift(x: int, shift: int) -> str:" in built["levels"]["L0"]


def test_l1_drops_examples_and_type_detail():
    built = build_levels(FIELDS, signature_is_code=True)
    level = built["levels"]["L1"]
    assert "Intent:" in level
    assert "def circular_shift(x, shift):" in level
    assert "Examples:" not in level
    assert "Constraints and edge cases:" in level
    assert "Error behaviour:" in level
    assert built["deleted_fields"]["L1"] == ["examples", "signature_types.annotations"]


def test_l2_keeps_intent_and_untyped_signature_only():
    built = build_levels(FIELDS, signature_is_code=True)
    level = built["levels"]["L2"]
    assert level.startswith("Intent:")
    assert "Signature:" in level
    assert "Constraints and edge cases:" not in level
    assert "Error behaviour:" not in level
    assert "Examples:" not in level


def test_l3_is_exactly_the_intent():
    built = build_levels(FIELDS, signature_is_code=True)
    assert built["levels"]["L3"] == FIELDS["intent"]


def test_l2_c_removes_exactly_one_field():
    built = build_levels(FIELDS, signature_is_code=True)
    built["levels"]["L0"]
    labels = {
        "intent": "Intent:",
        "signature_types": "Signature and types:",
        "examples": "Examples:",
        "constraints": "Constraints and edge cases:",
        "error_behaviour": "Error behaviour:",
    }
    for category, label in labels.items():
        drop = built["L2_c"][category]
        assert drop["field_present"] is True
        assert label not in drop["text"], category
        for other, other_label in labels.items():
            if other != category:
                assert other_label in drop["text"], (category, other)
    assert built["L2_c"]["examples"]["text"] == render_l0_minus(
        FIELDS, "examples", signature_is_code=True
    )


def test_missing_field_is_marked_absent_and_l2_c_equals_l0():
    fields = dict(FIELDS)
    fields["error_behaviour"] = ""
    built = build_levels(fields, signature_is_code=True)
    assert built["field_presence"]["error_behaviour"] is False
    assert built["L2_c"]["error_behaviour"]["field_present"] is False
    assert built["L2_c"]["error_behaviour"]["text"] == built["levels"]["L0"]


def test_labels_mark_present_fields_only():
    fields = dict(FIELDS, error_behaviour="", examples="")
    assert level_labels(fields, "L0") == {
        "intent": 0,
        "signature_types": 0,
        "examples": 0,
        "constraints": 0,
        "error_behaviour": 0,
    }
    assert level_labels(fields, "L1")["signature_types"] == 1
    assert level_labels(fields, "L1")["examples"] == 0
    assert level_labels(fields, "L2")["constraints"] == 1
    assert level_labels(fields, "L3")["error_behaviour"] == 0


def test_bigcodebench_signature_is_not_wrapped_in_python_code_block():
    fields = {
        "intent": "Add numbers.",
        "signature_types": "Args:\n- numbers (list): The numbers.",
        "signature_untyped": "def task_func(numbers):",
        "examples": "",
        "constraints": "",
        "error_behaviour": "",
    }
    built = build_levels(fields, signature_is_code=False)
    assert "Signature and types:\nArgs:" in built["levels"]["L0"]
    assert "```python" not in built["levels"]["L0"]
    assert "```python" in built["levels"]["L1"]
