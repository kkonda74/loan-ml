# AGENTS.md

# General Instructions

- Read the existing code before making changes.
- Make the smallest change necessary to complete the task.
- Do not modify unrelated files.
- Preserve existing functionality unless explicitly requested.
- Prefer simple and readable solutions over complicated ones.

# Python Rules

- Use Python 3.11.9 or newer.
- Follow PEP 8 coding conventions.
- Use descriptive variable and function names.
- Add type hints to new functions.
- Use pathlib instead of os.path when practical.
- Avoid global variables.
- Do not use wildcard imports.

# Code Structure

- Keep functions small and focused.
- Avoid duplicate code.
- Create reusable helper functions when appropriate.
- Do not introduce new libraries unless necessary.
- Preserve the existing project architecture.

# Error Handling

- Validate external/user input.
- Handle expected exceptions explicitly.
- Do not use:

  except Exception:
      pass

- Provide meaningful error messages.

# Testing

After modifying code:

1. Run existing unit tests.
2. Run linting if configured.
3. Run type checking if configured.
4. Fix failures caused by your changes.

Typical commands:

pytest
ruff check .
mypy .

# File Changes

Before creating a new file:

- Check whether an appropriate file already exists.
- Prefer modifying the existing implementation.

Do not modify:

- .env
- credentials files
- API keys
- generated files
- virtual environments

unless explicitly requested.

# Dependencies

Before adding a dependency:

- Check whether the standard library can solve the problem.
- Check whether an existing dependency already provides the functionality.
- Explain why a new dependency is necessary.

# Documentation

For new public functions:

- Add a short docstring.
- Explain important parameters.
- Explain the return value when it is not obvious.

# Git
- use only local files.
- Do not rewrite existing Git history.
- Do not delete unrelated code.
- Review `diff` before completing the task.

# Final Response

At completion:

- Briefly summarize what changed.
- List the files modified.
- Report tests that were run.
- Mention any tests that failed or could not be run.