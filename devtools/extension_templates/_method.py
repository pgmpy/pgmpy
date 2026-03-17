# This extension template provides instructions to add new methods to pgmpy.
#
# Please follow the instructions below:
# 1. Copy the `template_method` below and add it to the file you are working on.
# 2. Implement the method's core logic and write pytest-compatible unit tests.
# 3. Provide type hints for the input parameters and the return value.
# 4. Ensure the docstring follows the format provided below.
#    Note: The documentation follows the numpydoc style. (https://numpydoc.readthedocs.io/en/latest/format.html#)


class TemplateClass:
    def template_method(
        self,
    ):  # TODO: Provide type hints for the input parameters and the return value.
        """
        [One line description of the method.]

        [Detailed description of the method.]

        Parameters
        ----------
        hyperparam1: type, optional
            Description of hyperparam1.

        hyperparam2: type, optional
            Description of hyperparam2.

        Attributes
        ----------
        attribute1: []
            Description of attribute1.

        attribute2: []
            Description of attribute2.

        Returns
        -------
        return's type: []
            Description of return value.

        See Also
        --------
        [List of related methods for user reference.]

        Notes
        -----
        User-facing comments (optional)

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> ...

        References
        ----------
        .. [1] Citation1
        .. [2] Citation2
        """
        # pgmpy-engineer-facing comments (optional)
        # NOTE: Summarize of implementation background for pgmpy-engineer.
        # TODO: Summarize of need to work for pgmpy-engineer.
        # FIXME: Summarize the bug to fix for pgmpy-engineer.
        # If you plan to work on it yourself or if there is a related issue or pull request, you can write it as follows
        # NOTE(@your-github-username): [#issue_number] Summarize of implementation background.
        # TODO(@your-github-username): [#issue_number] Summarize of need to work.
        # FIXME(@your-github-username): [#issue_number] Summarize the bug to fix.

        # TODO: Implement the method's core logic and write pytest-compatible unit tests.
        ...
