import pytest
from insightsolver.api_utilities import (
    hash_string,
    convert_bytes_to_base64_string,
    convert_base64_string_to_bytes,
    compress_string,
    decompress_string
)

def test_hash_string():
    """Test that hash_string returns a consistent SHA256 hash."""
    s = "test string"
    # SHA256 of "test string"
    expected_hash = "d5579c46dfcc7f18207013e65b44e4cb4e2c2298f4ac457ba8f82743f31e930b"
    assert hash_string(s) == expected_hash

def test_base64_conversion_roundtrip():
    """Test that bytes -> base64 -> bytes returns the original data."""
    original_bytes = b"Hello World"
    base64_str = convert_bytes_to_base64_string(original_bytes)
    
    assert isinstance(base64_str, str)
    assert convert_base64_string_to_bytes(base64_str) == original_bytes

def test_compression_roundtrip():
    """Test that string -> compress -> decompress returns the original string."""
    original_str = "This is a test string that should be compressed." * 10
    compressed_str = compress_string(original_str)
    
    assert isinstance(compressed_str, str)
    assert compressed_str != original_str
    assert decompress_string(compressed_str) == original_str

def test_compression_empty_string():
    """Test compression of an empty string."""
    original_str = ""
    compressed_str = compress_string(original_str)
    assert decompress_string(compressed_str) == original_str
