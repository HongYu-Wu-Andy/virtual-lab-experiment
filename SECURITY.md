# Security

## Supported version

Security fixes are applied to the latest release on `main` during the initial private-development phase.

## Reporting a vulnerability

Do not open a public issue containing a secret, private dataset, exploit, or sensitive path. Contact the repository owner privately through GitHub and include a minimal reproduction with sensitive values removed.

## Safe operation

- Inspect datasets and specifications before running them.
- Keep API keys in environment variables and out of files and Git history.
- Use a dedicated Python environment for dependencies.
- Run untrusted forks or modified pipelines in a sandbox.
- Review the generated pipeline and output paths before using results operationally.
- Treat all selected parameter sets as computational proposals requiring real experimental confirmation.
- Do not use model output as a substitute for laboratory safety review, regulatory approval, or domain expert sign-off.
