from scipy.stats import rv_continuous

from pgmpy.factors import Factor


class ContinuousNode(rv_continuous):
	"""
	Base class for continuous node representation.
	It is a subclass of scipy.stats.rv_continuous.
	It has an extra method to discretize the continuous node into
	a discrete factor.
	"""
	def __init__(self, pdf, lb=None, ub=None, name=None):
		"""
		Parameters
		----------
		pdf : function
		The user defined probability density function.

		lb : float, optional
		Lower bound of the support of the distribution , default is minus
		infinity.

		ub : float, optional
		Upper bound of the support of the distribution , default is plus
		infinity.

		name : str, optional
		The name of the instance . This string is used to construct the default
		example for distributions.

		Examples
		--------
		>>> from pgmpy.factors import ContinuousNode
		>>> custom_pdf = lambda x : 0.5 if x > -1 and x < 1 else 0
		>>> node = ContinuousNode (custom_pdf, -3, 3)
		"""
		self.pdf = pdf
		super(ContinuousNode, self).__init__(momtype=0, a=lb, b=ub, name=name)

	def _pdf(self, *args):
		"""
		Defines the probability density function of the given continuous variable.
		"""
		return self.pdf(*args)

	def discretize(self, frm, to, step, method_type='rounding', *args):
		pass