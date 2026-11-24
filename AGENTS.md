# Project working rules

- Only the user stages, commits, or pushes changes.
- Add focused unit tests for new or changed behavior. Run `pytest` from the repository root. Tests should be deterministic and independent of network access and private datasets.
- Prioritize likely regressions, including input validation, boundaries, failure cases, and data alignment. Verify metric calculations with hand-calculated examples or a trusted independent reference. Expected results should come from outside the implementation under test.
- Use Pydantic for structured records where validation is useful. Give each Pydantic model its own Python file under `models/`.
