def requires_xgboost():
    """Check if xgboost is available."""
    try:
        import xgboost

        return True
    except ImportError:
        return False


XGBOOST_AVAILABLE = requires_xgboost()
