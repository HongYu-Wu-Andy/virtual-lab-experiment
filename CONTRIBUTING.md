# Contributing

Thank you for helping improve Virtual Lab Experiment. Contributions are welcome for scientific validation, provider support, documentation, testing, and carefully scoped extensions.

## Before opening a change

1. Search existing issues and pull requests.
2. Open an issue for a substantial scientific or architectural change.
3. Never include an API key, private dataset, personal path, or generated report containing sensitive information.
4. Keep model predictions labeled as proposals requiring real experimental confirmation.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/check_secrets.py
```

## Pull requests

- Keep each pull request focused.
- Add or update tests for behavioral changes.
- Update the README, schema, output contract, privacy policy, and changelog when applicable.
- Preserve provider-key redaction and the complete artifact trail.
- Explain scientific assumptions, validation choices, and unsupported scope.
- Do not prescribe a named multi-objective decision method unless the user explicitly asks for one.

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
