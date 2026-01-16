# Contributing to algofft

We welcome contributions to the algofft project! This document provides guidelines for contributing.

## Code of Conduct

Be respectful and constructive in all interactions. We're committed to providing a welcoming and inclusive environment for all contributors.

## Getting Started

1. **Fork and clone** the repository
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Set up development environment**:

   ```bash
   go mod download
   just build
   just test
   ```

## Development Workflow

### Building and Testing

```bash
just build      # Compile the library
just test       # Run all tests with race detector
just bench      # Run benchmarks
just lint       # Check code quality
just fmt        # Format code
just cover      # Generate coverage report
```

### Code Style

- Follow standard Go conventions (gofmt, golangci-lint)
- Write clear, descriptive variable and function names
- Add documentation comments for all exported symbols
- Keep functions focused and reasonably sized

### Testing

- Write tests for all new functionality
- Ensure all tests pass: `just test`
- Add property-based tests where applicable
- Include edge case testing
- Aim for >80% code coverage

### Commit Messages

Write clear, descriptive commit messages:

```
Short summary (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.

- Use bullet points for multiple changes
- Reference issues with #123 if applicable
```

## Submitting Changes

1. **Push to your fork**: `git push origin feature/my-feature`
2. **Create a Pull Request** with:
   - Clear description of changes
   - Reference to related issues
   - Test results showing all tests pass
   - Any relevant benchmark results

3. **Code Review**: Address feedback from reviewers

## Areas for Contribution

### Good First Issues

- Documentation improvements
- Adding more test cases
- Improving error messages
- Performance benchmarking

### Feature Development

- Additional FFT variants (DCT, Hilbert transform)
- Platform-specific optimizations
- New SIMD implementations
- Example applications

### Testing & Quality

- Cross-platform testing (Windows, macOS, Linux, ARM)
- Numerical precision improvements
- Performance optimization
- Memory profiling and optimization

## Architecture Overview

The library is organized as follows:

- `/` - Main package with public API
- `/internal/` - Internal implementation details
- `/testdata/` - Test fixtures and reference data
- `/examples/` - Usage examples and documentation

## Key Design Principles

1. **Zero-allocation transforms** - Pre-allocate buffers in Plan
2. **Correctness first** - Extensive testing and validation
3. **Performance** - SIMD optimization across platforms
4. **API stability** - Avoid breaking changes

## Performance Considerations

When implementing FFT algorithms:

- Minimize allocations in hot paths
- Use efficient memory access patterns
- Consider cache locality
- Profile before and after optimizations
- Add benchmarks for new features

## Documentation

- Add GoDoc comments to all exported items
- Include usage examples in docstrings
- Update README if adding major features
- Document any algorithmic choices

## Questions?

- Open an issue for bug reports
- Start a discussion for feature requests
- Reference existing documentation when possible

Thank you for contributing to algofft!
