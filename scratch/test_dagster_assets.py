import pytest
from dagster import materialize_to_memory, build_op_context

# We import the assets we just created
from data.orchestration.assets import raw_market_macro, validated_market_macro

def test_asset_execution():
    """
    Tests the Dagster asset graph in-memory to ensure dependencies resolve
    and execution succeeds without requiring a persistent Dagster instance.
    """
    # Materialize the assets sequentially to memory.
    # The framework automatically handles the dependency of validated_market_macro on raw_market_macro
    result = materialize_to_memory([raw_market_macro, validated_market_macro])
    
    assert result.success

    # Verify that raw_market_macro produced a DataPackage
    raw_output = result.output_for_node("raw_market_macro")
    assert raw_output is not None
    assert raw_output.dataset_name == "market_macro"

    # Verify that validated_market_macro successfully received raw_market_macro,
    # validated it, and produced the final Output DataPackage
    validated_output = result.output_for_node("validated_market_macro")
    assert validated_output is not None
    
    # Check publication state
    from data.contracts import PublicationState
    assert validated_output.publication_state == PublicationState.PUBLISHED

if __name__ == "__main__":
    pytest.main([__file__])
