# This makes the src directory into a proper Python package

# Import key components for easy access
try:
    from .recursive_engine import UnifiedRecursiveSystem
    from .recursive_integration import EnhancedRecursiveSystem
    from .structured_expansion import StructuredExpansionSystem
except ImportError as e:
    print(f"Error importing components in __init__.py: {e}")