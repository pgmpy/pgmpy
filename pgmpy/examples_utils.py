"""
Utility classes for pgmpy examples and notebooks.

Provides helpers for:
- Visualization with fallback
- File output management
- Error handling and logging
- Model inspection
"""

import logging
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Any


logger = logging.getLogger(__name__)


class VisualizationHelper:
    """Handles visualization with graceful degradation."""

    @staticmethod
    def visualize_graph(model, output_file: Optional[str] = None, prog: str = 'dot') -> Optional[str]:
        """
        Visualize a Bayesian Network with fallback.
        
        Parameters
        ----------
        model : DiscreteBayesianNetwork
            The model to visualize
        output_file : str, optional
            Output file path. If None, uses temp directory.
        prog : str, default='dot'
            Graphviz layout program
            
        Returns
        -------
        str or None
            Path to output file if successful, else None
        """
        try:
            import pygraphviz
            import shutil
            
            if shutil.which("dot") is None:
                logger.warning("⚠ Graphviz 'dot' not found. Visualization skipped.")
                return None
                
            # Generate visualization
            viz = model.to_graphviz()
            
            # Determine output path
            if output_file is None:
                output_file = os.path.join(tempfile.gettempdir(), 'pgmpy_model.png')
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(os.path.abspath(output_file)) or '.', exist_ok=True)
            
            # Save visualization
            viz.draw(output_file, prog=prog)
            logger.info(f"✓ Graph saved to: {output_file}")
            return output_file
            
        except ImportError:
            logger.warning("⚠ pygraphviz not installed. Install with: pip install pygraphviz")
            return None
        except Exception as e:
            logger.error(f"✗ Visualization failed: {e}")
            return None

    @staticmethod
    def display_image(file_path: str) -> None:
        """
        Display image with fallback for different environments.
        
        Parameters
        ----------
        file_path : str
            Path to image file
        """
        if not os.path.exists(file_path):
            logger.warning(f"⚠ Image file not found: {file_path}")
            return
        
        try:
            from IPython.display import Image, display
            display(Image(file_path))
            logger.info(f"✓ Displayed: {file_path}")
        except ImportError:
            logger.info(f"Image available at: {file_path}")


class ModelInspector:
    """Inspect and summarize Bayesian Network properties."""

    @staticmethod
    def inspect_model(model) -> Dict[str, Any]:
        """
        Get comprehensive model summary.
        
        Parameters
        ----------
        model : DiscreteBayesianNetwork
            The model to inspect
            
        Returns
        -------
        dict
            Model properties including nodes, edges, structure, CPDs
        """
        try:
            summary = {
                'nodes': list(model.nodes()),
                'n_nodes': model.number_of_nodes(),
                'edges': list(model.edges()),
                'n_edges': model.number_of_edges(),
                'roots': model.get_roots() if hasattr(model, 'get_roots') else [],
                'leaves': model.get_leaves() if hasattr(model, 'get_leaves') else [],
                'has_cpds': len(model.get_cpds()) > 0 if hasattr(model, 'get_cpds') else False,
                'is_valid': model.check_model() if hasattr(model, 'check_model') else None,
            }
            return summary
        except Exception as e:
            logger.error(f"Inspection failed: {e}")
            return {}

    @staticmethod
    def print_summary(model) -> None:
        """
        Print formatted model summary.
        
        Parameters
        ----------
        model : DiscreteBayesianNetwork
            The model to summarize
        """
        summary = ModelInspector.inspect_model(model)
        
        print("\n" + "="*60)
        print("MODEL SUMMARY")
        print("="*60)
        print(f"Nodes: {summary.get('n_nodes', 'N/A')} — {summary.get('nodes', [])}")
        print(f"Edges: {summary.get('n_edges', 'N/A')}")
        print(f"Roots: {summary.get('roots', [])}")
        print(f"Leaves: {summary.get('leaves', [])}")
        print(f"Has CPDs: {summary.get('has_cpds', False)}")
        print(f"Valid Model: {summary.get('is_valid', 'Unknown')}")
        print("="*60 + "\n")


class InferenceWrapper:
    """Wrapper for inference with error handling and logging."""

    def __init__(self, model, inference_class='VariableElimination'):
        """
        Initialize inference wrapper.
        
        Parameters
        ----------
        model : DiscreteBayesianNetwork
            The model for inference
        inference_class : str, default='VariableElimination'
            Type of inference to use
        """
        self.model = model
        self.inference_type = inference_class
        self._init_inference()

    def _init_inference(self) -> None:
        """Initialize the inference engine."""
        try:
            if self.inference_type == 'VariableElimination':
                from pgmpy.inference import VariableElimination
                self.infer = VariableElimination(self.model)
            elif self.inference_type == 'BeliefPropagation':
                from pgmpy.inference import BeliefPropagation
                self.infer = BeliefPropagation(self.model)
            else:
                raise ValueError(f"Unknown inference type: {self.inference_type}")
            logger.info(f"✓ Initialized {self.inference_type}")
        except Exception as e:
            logger.error(f"Inference initialization failed: {e}")
            raise

    def query(self, variables: List[str], evidence: Optional[Dict] = None, **kwargs):
        """
        Run inference query.
        
        Parameters
        ----------
        variables : list
            Variables to query
        evidence : dict, optional
            Evidence values
        **kwargs
            Additional arguments for inference
            
        Returns
        -------
        FactorTable or None
            Query result
        """
        try:
            result = self.infer.query(variables=variables, evidence=evidence or {}, **kwargs)
            logger.info(f"✓ Query successful for {variables}")
            return result
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return None

    def predict(self, test_data, variables: Optional[List[str]] = None):
        """
        Make predictions for test data.
        
        Parameters
        ----------
        test_data : DataFrame
            Test data with evidence
        variables : list, optional
            Variables to predict
            
        Returns
        -------
        dict
            Predictions
        """
        try:
            predictions = {}
            for idx, row in test_data.iterrows():
                evidence = row.to_dict()
                result = self.infer.query(variables=variables, evidence=evidence)
                predictions[idx] = result
            logger.info(f"✓ Predicted for {len(test_data)} samples")
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {}


class CausalInferenceWrapper:
    """Wrapper for causal inference with utilities."""

    def __init__(self, model):
        """
        Initialize causal inference wrapper.
        
        Parameters
        ----------
        model : DiscreteBayesianNetwork
            The causal model
        """
        self.model = model
        self._init_causal_inference()

    def _init_causal_inference(self) -> None:
        """Initialize causal inference."""
        try:
            from pgmpy.inference import CausalInference
            self.infer = CausalInference(self.model)
            logger.info("✓ Initialized CausalInference")
        except Exception as e:
            logger.error(f"Causal inference initialization failed: {e}")
            raise

    def do_query(self, variables: List[str], do_values: Dict[str, Any], **kwargs):
        """
        Run do-calculus query.
        
        Parameters
        ----------
        variables : list
            Variables to query
        do_values : dict
            Do-operation values
        **kwargs
            Additional arguments
            
        Returns
        -------
        FactorTable or None
            Query result
        """
        try:
            result = self.infer.query(variables=variables, do=do_values, **kwargs)
            logger.info(f"✓ Do-query successful: do({do_values}) -> {variables}")
            return result
        except Exception as e:
            logger.error(f"Do-query failed: {e}")
            return None


class ExampleRunner:
    """Base class for running examples with standardized structure."""

    def __init__(self, name: str, verbose: int = 1):
        """
        Initialize example runner.
        
        Parameters
        ----------
        name : str
            Name of the example
        verbose : int, default=1
            Verbosity level (0=silent, 1=info, 2=debug)
        """
        self.name = name
        self.verbose = verbose
        self.results = {}
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging for the example."""
        if self.verbose == 0:
            logging.getLogger('pgmpy').setLevel(logging.WARNING)
        elif self.verbose == 1:
            logging.getLogger('pgmpy').setLevel(logging.INFO)
        else:
            logging.getLogger('pgmpy').setLevel(logging.DEBUG)

    def run(self) -> Dict[str, Any]:
        """
        Run the example. Should be overridden by subclasses.
        
        Returns
        -------
        dict
            Results of the example
        """
        raise NotImplementedError("Subclasses must implement run()")

    def summary(self) -> str:
        """
        Get results summary.
        
        Returns
        -------
        str
            Formatted summary
        """
        lines = [f"\n{'='*60}", f"EXAMPLE: {self.name}", '='*60]
        for key, value in self.results.items():
            lines.append(f"{key}: {value}")
        lines.append('='*60)
        return '\n'.join(lines)

    def print_summary(self) -> None:
        """Print results summary."""
        print(self.summary())


# Setup module logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
