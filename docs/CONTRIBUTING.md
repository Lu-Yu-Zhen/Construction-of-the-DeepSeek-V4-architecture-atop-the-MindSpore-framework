# Contributing to DeepSeek-V4 MindSpore

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub:

1. Check if the issue already exists
2. Provide a clear title and description
3. Include steps to reproduce (for bugs)
4. Mention your environment (OS, MindSpore version, Python version)

### Submitting Code

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** following the coding guidelines
4. **Write tests** for your changes
5. **Run the tests**: `python -m pytest tests/ -v`
6. **Commit** with a clear message following conventional commit format
7. **Push** to your fork
8. **Open a Pull Request**

### Commit Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Build process or tool changes

Examples:
```
feat(attention): add variable compression rate support
fix(moe): correct expert routing normalization
docs(architecture): update mHC explanation
```

## Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use type hints for function signatures
- Write docstrings for public methods (NumPy style)
- Keep functions focused and modular
- Maximum line length: 100 characters

### Testing

- Write unit tests for all new components
- Ensure existing tests pass before submitting
- Test with both Flash and Pro configurations
- Include edge cases (empty inputs, extreme values)

```bash
# Run tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

### Documentation

- Update relevant documentation when changing behavior
- Use clear, concise language
- Include code examples where appropriate
- Document both English and Chinese (`README_zh.md`)

## Project Structure

```
DeepSeek-V4/
├── src/              # Core model implementation
├── configs/          # YAML configuration files
├── scripts/          # Training and inference scripts
├── tests/            # Unit tests
├── docs/             # Documentation
├── data/             # Data directory
└── checkpoints/      # Model checkpoints
```

## Getting Help

- Open an issue on GitHub
- Check the [MindSpore Documentation](https://www.mindspore.cn/docs)
- Review the [DeepSeek-V4 paper](https://arxiv.org/abs/2504.09286)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.