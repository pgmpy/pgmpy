pgmpy [![Build Status](https://travis-ci.org/pgmpy/pgmpy.png)](https://travis-ci.org/pgmpy/pgmpy) [![Coverage Status](https://coveralls.io/repos/pgmpy/pgmpy/badge.png?branch=dev)](https://coveralls.io/r/pgmpy/pgmpy?branch=dev)
=====

Python Library for Probabilistic Graphical Models  
Documentation: [pgmpy](http://pgmpy.org/)  
Mailing List: pgmpy@googlegroups.com  
irc: #pgmpy on freenode.net

Dependencies:
=============
- Python 3.3
- NetworkX 1.8.1
- Scipy 0.12.1
- Numpy 1.7.1
- Cython 0.21
- Pandas 0.15.1

To install all the depedencies 

- Either using <code>pip</code>, use
<code><pre>
pip install -r requirements.txt
</pre></code>

- Else using <code>conda</code>, use
<code><pre>
conda install --file requirements.txt
</pre></code>

Installation:
=============
pgmpy is installed using <code>distutils</code>. If you have the tools installed
to build a python extension module:

<code>sudo python3 setup.py install</code>

Example:
========
```python3
from pgmpy.models import BayesianModel
from pgmpy.factors import TabularCPD
student = bm.BayesianModel()
# instantiates a new Bayesian Model called 'student'

student.add_nodes_from(['diff', 'intel', 'grade'])
# adds nodes labelled 'diff', 'intel', 'grade' to student

student.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
# adds directed edges from 'diff' to 'grade' and 'intel' to 'grade'

"""
diff cpd:

+-------+--------+
|diff:  |        |
+-------+--------+
|easy	|	0.2	 |
+-------+--------+
|hard	|	0.8	 |
+-------+--------+
"""
diff_cpd = TabularCPD('diff', 2, [[0.2], [0.8]])

"""
intel cpd:

+-------+--------+
|intel: |        |
+-------+--------+
|dumb	|	0.5	 |
+-------+--------+
|avg	|	0.3	 |
+-------+--------+
|smart	|	0.2	 |
+-------+--------+
"""
intel_cpd = TabularCPD('intel', 3, [[0.5], [0.3], [0.2]])

"""
grade cpd:

+------+-----------------------+---------------------+
|diff: |          easy         |         hard        |
+------+------+------+---------+------+------+-------+
|intel:| dumb |  avg |  smart  | dumb | avg  | smart |
+------+------+------+---------+------+------+-------+
|gradeA| 0.1  | 0.1  |   0.1   |  0.1 |  0.1 |   0.1 |
+------+------+------+---------+------+------+-------+
|gradeB| 0.1  | 0.1  |   0.1   |  0.1 |  0.1 |   0.1 |
+------+------+------+---------+------+------+-------+
|gradeC| 0.8  | 0.8  |   0.8   |  0.8 |  0.8 |   0.8 |
+------+------+------+---------+------+------+-------+
"""
grade_cpd = TabularCPD('grade', 3,
					   [[0.1,0.1,0.1,0.1,0.1,0.1],
                        [0.1,0.1,0.1,0.1,0.1,0.1], 
                        [0.8,0.8,0.8,0.8,0.8,0.8]],
					   evidence=['intel', 'diff'],
					   evidence_card=[3, 2])

student.add_cpds(diff_cpd, intel_cpd, grade_cpd)

# Finding active trail
student.active_trail_nodes('diff')

# Finding active trail with observation
student.active_trail_nodes('diff', observed='grades')
```
