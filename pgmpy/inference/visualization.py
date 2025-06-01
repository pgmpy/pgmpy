#!/usr/bin/env python

import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, Any, Union, Hashable


def plot_causal_graph(
    model,
    node_pos: Optional[Union[str, Dict[Hashable, Tuple[float, float]]]] = "circular",
    node_color: str = "lightblue",
    node_size: int = 500,
    font_size: int = 10,
    font_weight: str = "bold",
    edge_color: str = "black",
    arrowsize: int = 20,
    arrowstyle: str = "->",
    width: float = 1.0,
    with_labels: bool = True,
    ax: Optional[plt.Axes] = None,
    output_file: Optional[str] = None,
    show: bool = False,
    **kwargs,
):
    """
    Draws a causal graph for the model using NetworkX and matplotlib.

    This function provides a simple way to visualize causal graphs with customizable
    appearance options. It can either display the graph or save it to a file.

    Parameters
    ----------
    model : DAG, DiscreteBayesianNetwork, or CausalInference
        The model containing the causal graph to be visualized.

    node_pos : str or dict, optional (default="circular")
        Positioning of nodes. If str, must be one of the following:
        'circular', 'kamada_kawai', 'planar', 'random', 'shell', 'spring',
        'spectral', or 'spiral'. If dict, must be of the form
        {node: (x_coord, y_coord)} specifying the position of each node.

    node_color : str, optional (default="lightblue")
        Color of the nodes.

    node_size : int, optional (default=500)
        Size of the nodes.

    font_size : int, optional (default=10)
        Font size for node labels.

    font_weight : str, optional (default="bold")
        Font weight for node labels.

    edge_color : str, optional (default="black")
        Color of the edges.

    arrowsize : int, optional (default=20)
        Size of the arrows.

    arrowstyle : str, optional (default="->")
        Style of the arrows.

    width : float, optional (default=1.0)
        Width of the edges.

    with_labels : bool, optional (default=True)
        Whether to display node labels.

    ax : matplotlib.axes.Axes, optional (default=None)
        Axes to draw on. If None, a new figure and axes are created.

    output_file : str, optional (default=None)
        Path to save the graph image. If None, the graph is not saved.
        Supported formats include PNG, PDF, SVG, etc. based on matplotlib's
        supported formats.

    show : bool, optional (default=False)
        Whether to display the plot. Set to True to show the plot immediately.

    **kwargs :
        Additional keyword arguments to pass to networkx.draw_networkx().

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib Axes object with the drawn graph.

    Examples
    --------
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.inference import CausalInference
    >>> from pgmpy.inference.visualization import plot_causal_graph
    >>> # Create a simple causal model
    >>> model = DiscreteBayesianNetwork([("X", "Y"), ("Z", "X"), ("Z", "Y")])
    >>> # Visualize the model
    >>> plot_causal_graph(model, output_file="causal_model.png", show=True)

    >>> # Using with CausalInference
    >>> inference = CausalInference(model)
    >>> plot_causal_graph(inference, node_color="lightgreen", show=True)

    >>> # Custom node positions
    >>> pos = {"X": (0, 0), "Y": (1, 0), "Z": (0.5, 0.5)}
    >>> plot_causal_graph(model, node_pos=pos, show=True)
    """
    # Extract the graph structure based on the input model type
    if hasattr(model, "dag"):
        # For CausalInference objects
        graph = model.dag
    elif hasattr(model, "to_directed"):
        # For DAG or BayesianNetwork objects
        graph = model.to_directed()
    else:
        raise TypeError(
            "Model must be a DAG, DiscreteBayesianNetwork, or CausalInference object"
        )

    # Create a new figure if no axes is provided
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # Determine node positions
    if isinstance(node_pos, str):
        supported_layouts = {
            "circular": nx.circular_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            # "planar": nx.planar_layout,  # Removing planar layout as it can cause StopIteration errors
            "random": nx.random_layout,
            "shell": nx.shell_layout,
            # "spring": nx.spring_layout,  # Removing spring layout as it can cause StopIteration errors
            "spectral": nx.spectral_layout,
            "spiral": nx.spiral_layout,
        }

        if node_pos not in supported_layouts:
            raise ValueError(
                f"Unknown layout: {node_pos}. Must be one of: {', '.join(supported_layouts.keys())}"
            )

        pos = supported_layouts[node_pos](graph)
    else:
        # Assume node_pos is a dictionary of positions
        pos = node_pos

    # Draw the graph
    nx.draw_networkx(
        graph,
        pos=pos,
        node_color=node_color,
        node_size=node_size,
        font_size=font_size,
        font_weight=font_weight,
        edge_color=edge_color,
        arrowsize=arrowsize,
        arrowstyle=arrowstyle,
        width=width,
        with_labels=with_labels,
        ax=ax,
        **kwargs,
    )

    # Remove axis ticks and labels for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Causal Graph")

    # Save the figure if output_file is specified
    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
        print(f"Causal graph saved to {output_file}")

    # Show the plot if requested
    if show:
        plt.show()

    return ax
