# Why loop units ended without a solution

2700 completed pilot units. A NoSolution unit never wrote `solution.py`; the cause is how its last generator turn ended. Harness-terminated causes are `truncated_output`, `unparseable_tool_json`, `prose_reply`, `empty_reply`.

| condition | failures | NoSolution | truncated_output | explicit_finish | unparseable_tool_json | prose_reply | NoSolution after stub passed visible tests |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 113 | 10 | 1 | 0 | 9 | 0 | 0 |
| H0 | 144 | 9 | 1 | 0 | 8 | 0 | 0 |
| H1 | 214 | 146 | 8 | 72 | 21 | 45 | 76 |
| H2 | 167 | 54 | 1 | 33 | 8 | 12 | 2 |
| H3 | 139 | 1 | 0 | 0 | 1 | 0 | 0 |
| H4 | 145 | 6 | 2 | 0 | 4 | 0 | 0 |
| H3⁻ | 172 | 55 | 1 | 47 | 6 | 1 | 47 |
| **total** | | 281 | 14 | 152 | 57 | 58 | 125 |

JSON errors among `unparseable_tool_json`: Expecting ',' delimiter (44), Invalid \escape (10), Invalid control character at (2), Unterminated string starting at (1).

## pass@1 sensitivity (descriptive, not a corrected result)

| condition | pass@1 as recorded | excluded units | pass@1 excluding them |
| --- | --- | --- | --- |
| B1 | 0.623 | 10 | 0.645 |
| H0 | 0.520 | 9 | 0.536 |
| H1 | 0.287 | 74 | 0.381 |
| H2 | 0.443 | 21 | 0.477 |
| H3 | 0.537 | 1 | 0.538 |
| H4 | 0.517 | 6 | 0.527 |
| B2 | 0.513 | 0 | 0.513 |
| B3 | 0.310 | 0 | 0.310 |
| H3⁻ | 0.427 | 8 | 0.438 |

Excluding units changes the denominator and assumes nothing about how they would have scored; it bounds how much the termination rule could move each condition.
