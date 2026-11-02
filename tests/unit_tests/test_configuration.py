import os

from react_agent.context import Context


def test_context_init() -> None:
    context = Context(model="openai/gpt-4o-mini")
    assert context.model == "openai/gpt-4o-mini"


def test_context_init_with_env_vars() -> None:
    os.environ["MODEL"] = "openai/gpt-4o-mini"
    context = Context()
    assert context.model == "openai/gpt-4o-mini"


def test_context_init_with_env_vars_and_passed_values() -> None:
    os.environ["MODEL"] = "openai/gpt-4o-mini"
    context = Context(model="openai/gpt-5o-mini")
    assert context.model == "openai/gpt-5o-mini"


def test_context_int_type_conversion() -> None:
    """Test that integer environment variables are properly converted."""
    # Clean up environment
    os.environ.pop("MAX_SEARCH_RESULTS", None)

    # Test int conversion
    os.environ["MAX_SEARCH_RESULTS"] = "20"
    context = Context()
    assert context.max_search_results == 20
    assert isinstance(context.max_search_results, int)

    # Clean up
    os.environ.pop("MAX_SEARCH_RESULTS", None)


def test_context_int_type_conversion_invalid() -> None:
    """Test that invalid integer environment variables keep default value."""
    # Clean up environment
    os.environ.pop("MAX_SEARCH_RESULTS", None)

    # Test invalid int conversion - should keep default value
    os.environ["MAX_SEARCH_RESULTS"] = "not_a_number"
    context = Context()
    # Should keep default value when int conversion fails
    assert context.max_search_results == 10  # default value
    assert isinstance(context.max_search_results, int)

    # Clean up
    os.environ.pop("MAX_SEARCH_RESULTS", None)


def test_context_string_type_conversion() -> None:
    """Test that string environment variables work correctly."""
    # Clean up environment
    os.environ.pop("MODEL", None)

    # Test string conversion (no conversion needed)
    os.environ["MODEL"] = "test/model-name"
    context = Context()
    assert context.model == "test/model-name"
    assert isinstance(context.model, str)

    # Clean up
    os.environ.pop("MODEL", None)


def test_context_env_vars_only_used_for_defaults() -> None:
    """Test that environment variables are only used when field has default value."""
    # Clean up environment
    os.environ.pop("MAX_SEARCH_RESULTS", None)
    os.environ.pop("MODEL", None)

    # Set environment variables
    os.environ["MAX_SEARCH_RESULTS"] = "99"
    os.environ["MODEL"] = "env/model"

    # Pass explicit values - should override env vars
    context = Context(max_search_results=5, model="explicit/model")
    assert context.max_search_results == 5
    assert context.model == "explicit/model"

    # Clean up
    os.environ.pop("MAX_SEARCH_RESULTS", None)
    os.environ.pop("MODEL", None)


def test_context_float_type_conversion() -> None:
    """Test that float environment variables are properly converted."""
    # Clean up environment
    os.environ.pop("TEMPERATURE", None)

    # Test float conversion
    os.environ["TEMPERATURE"] = "0.5"
    context = Context()
    assert context.temperature == 0.5
    assert isinstance(context.temperature, float)

    # Clean up
    os.environ.pop("TEMPERATURE", None)


def test_context_float_type_conversion_invalid() -> None:
    """Test that invalid float environment variables keep default value."""
    # Clean up environment
    os.environ.pop("TEMPERATURE", None)

    # Test invalid float conversion - should keep default value
    os.environ["TEMPERATURE"] = "not_a_float"
    context = Context()
    # Should keep default value when float conversion fails
    assert context.temperature == 0.1  # default value
    assert isinstance(context.temperature, float)

    # Clean up
    os.environ.pop("TEMPERATURE", None)


def test_context_bool_type_conversion() -> None:
    """Test that boolean environment variables are properly converted."""
    # Clean up environment
    os.environ.pop("ENABLE_DEBUG", None)

    # Test various true values
    for true_value in ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON"]:
        os.environ["ENABLE_DEBUG"] = true_value
        context = Context()
        assert context.enable_debug is True
        assert isinstance(context.enable_debug, bool)

    # Test various false values
    for false_value in [
        "false",
        "False",
        "FALSE",
        "0",
        "no",
        "NO",
        "off",
        "OFF",
        "anything_else",
    ]:
        os.environ["ENABLE_DEBUG"] = false_value
        context = Context()
        assert context.enable_debug is False
        assert isinstance(context.enable_debug, bool)

    # Clean up
    os.environ.pop("ENABLE_DEBUG", None)


def test_context_multiple_type_conversions() -> None:
    """Test multiple type conversions at once."""
    # Clean up environment
    os.environ.pop("MAX_SEARCH_RESULTS", None)
    os.environ.pop("TEMPERATURE", None)
    os.environ.pop("ENABLE_DEBUG", None)
    os.environ.pop("MODEL", None)

    # Set multiple environment variables
    os.environ["MAX_SEARCH_RESULTS"] = "25"
    os.environ["TEMPERATURE"] = "0.8"
    os.environ["ENABLE_DEBUG"] = "true"
    os.environ["MODEL"] = "test/model"

    context = Context()
    assert context.max_search_results == 25
    assert isinstance(context.max_search_results, int)
    assert context.temperature == 0.8
    assert isinstance(context.temperature, float)
    assert context.enable_debug is True
    assert isinstance(context.enable_debug, bool)
    assert context.model == "test/model"
    assert isinstance(context.model, str)

    # Clean up
    os.environ.pop("MAX_SEARCH_RESULTS", None)
    os.environ.pop("TEMPERATURE", None)
    os.environ.pop("ENABLE_DEBUG", None)
    os.environ.pop("MODEL", None)
