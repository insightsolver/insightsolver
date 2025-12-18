# InsightSolver Unit Tests

This directory contains the automated unit tests for the `insightsolver` library. These tests are designed to be run locally by developers and automatically via CI/CD pipelines (GitHub Actions).

## Running Tests

To run all tests, simply execute:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_client_mock.py
```

## Test Coverage

### 1. Client Logic (`test_client_mock.py`)

These tests verify the behavior of the main `InsightSolver` class. They use **mocking** to simulate API responses, ensuring tests run fast, offline, and without requiring a valid service key.

*   **`test_insightsolver_initialization`**:
    *   Verifies that the `InsightSolver` object initializes correctly with a DataFrame.
    *   Checks that target name and goal are correctly stored.
    *   Ensures default column types are inferred correctly.

*   **`test_insightsolver_initialization_custom_types`**:
    *   Verifies that users can manually override column types (e.g., setting a column to `'ignore'`).
    *   Ensures manual configuration takes precedence over automatic inference.

*   **`test_fit_mocked`**:
    *   Simulates a complete API response (including rules, benchmarks, and metadata).
    *   Executes the `.fit()` method.
    *   Verifies that the solver state updates to `_is_fitted = True`.
    *   Checks that API results are correctly stored in `rule_mining_results`.
    *   Ensures column types are updated based on server feedback.
    *   Validates that the API was called with the correct parameters.

*   **`test_not_fitted_error`**:
    *   Verifies that the solver correctly reports its status as not fitted (`is_fitted() == False`) immediately after initialization.

### 2. API Utilities (`test_api_utilities.py`)

These tests validate the low-level utility functions responsible for data processing and secure communication.

*   **`test_hash_string`**:
    *   Ensures the hashing function returns consistent SHA-256 hashes for data integrity.

*   **`test_base64_conversion_roundtrip`**:
    *   Verifies that binary data survives a round-trip conversion (`bytes -> base64 -> bytes`), essential for file encoding.

*   **`test_compression_roundtrip`**:
    *   Verifies that string compression and decompression (gzip) work correctly and losslessly.

*   **`test_compression_empty_string`**:
    *   Ensures the compression logic handles edge cases like empty strings without errors.
