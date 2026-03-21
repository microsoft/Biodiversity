# Development Guide for PyTorch Wildlife

This guide covers setting up your development environment and following our code quality standards.

## Setup

### Prerequisites
- Python >= 3.8
- Git

### Virtual Environment

We recommend using a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
# Install package in development mode with all dependencies
pip install -e ".[dev]"

# Or install core dependencies + development tools
pip install -r requirements.txt
pip install pre-commit black flake8 isort
```

## Code Quality

We use automated tools to maintain consistent code quality across the project.

### Pre-commit Hooks

Pre-commit hooks automatically check your code before each commit, catching common issues early.

**Setup (one-time):**

```bash
pip install pre-commit
pre-commit install
```

Now every time you commit, hooks will run automatically. If any issues are found, the commit will be blocked until you fix them.

**Running Manually:**

```bash
# Check all files
pre-commit run --all-files

# Check specific file
pre-commit run --files path/to/file.py

# Skip a specific hook
SKIP=flake8 pre-commit run --all-files

# Update hook versions
pre-commit autoupdate
```

### What Each Hook Does

1. **black** - Automatic code formatting (PEP 8 compliant)
   - Enforces consistent style across the codebase
   - Max line length: 100 characters

2. **isort** - Import organization
   - Sorts imports alphabetically
   - Groups: stdlib → third-party → local
   - Compatible with black

3. **flake8** - Linting
   - Checks for common style issues
   - Detects unused variables/imports
   - Catches potential bugs

4. **trailing-whitespace** - Removes trailing spaces

5. **end-of-file-fixer** - Ensures files end with a newline

6. **check-yaml** - Validates YAML syntax

7. **check-merge-conflict** - Prevents committed merge conflict markers

### Fixing Code Issues

If pre-commit finds issues, you can fix them automatically:

```bash
# Black and isort will auto-fix formatting/imports
black .
isort .

# Flake8 requires manual fixes (shows issues, doesn't auto-fix)
flake8 .
```

After fixing, commit again:

```bash
git add .
git commit -m "feat: Your message here"
```

### Type Checking (Optional)

For advanced type checking (optional, not enforced on commit):

```bash
pre-commit run mypy --all-files
```

## Testing

Run the test suite before submitting PRs:

```bash
pytest tests/
```

## Committing Code

Follow these guidelines:

1. **Create a feature branch:**
   ```bash
   git checkout -b issue-600-add-precommit-hooks
   ```

2. **Make changes and test:**
   ```bash
   # Edit files...
   pre-commit run --all-files  # Run checks
   pytest tests/               # Run tests
   ```

3. **Commit with clear messages:**
   ```bash
   git commit -m "feat: Add pre-commit hooks for code quality"
   ```

4. **Push and create PR:**
   ```bash
   git push origin issue-600-add-precommit-hooks
   ```

## Commit Message Format

We follow conventional commits for clear, organized history:

```
type: short description

Optional longer description explaining the change.

Fixes #123
```

### Types
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `style:` - Code style (formatting, etc.)
- `refactor:` - Code refactoring without functionality change
- `test:` - Test additions/changes
- `chore:` - Maintenance tasks

Example:
```
feat: Add pre-commit hooks for code quality

Implements black, flake8, and isort for consistent code formatting.
Developers can now use pre-commit hooks to catch issues before commit.

Fixes #600
```

## Troubleshooting

**Pre-commit is slow:**
- Large repositories can be slow on first run. Subsequent runs are faster.
- You can run specific hooks: `pre-commit run black --all-files`

**Hook keeps failing on the same issue:**
- Some hooks (black, isort) auto-fix. Run them to fix automatically.
- For flake8: read the output and fix manually.

**Need to bypass hooks temporarily:**
- For emergencies only: `git commit --no-verify`
- But please fix issues before pushing!

**Confused about a hook error:**
- Run the tool directly: `black path/to/file.py`
- Check the hook documentation in `.pre-commit-config.yaml`

## Resources

- [Pre-commit Documentation](https://pre-commit.com/)
- [Black Documentation](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [isort Documentation](https://pycqa.github.io/isort/)

## Questions?

If you have questions about the development setup, open an issue or reach out on Discord: https://discord.gg/TeEVxzaYtm
