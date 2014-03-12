PgmPy [![Build Status](https://travis-ci.org/pgmpy/pgmpy.png)](https://travis-ci.org/pgmpy/pgmpy) [![Coverage Status](https://coveralls.io/repos/pgmpy/pgmpy/badge.png?branch=dev)](https://coveralls.io/r/pgmpy/pgmpy?branch=dev)[![Bitdeli Badge](https://d2weczhvl823v0.cloudfront.net/pgmpy/pgmpy/trend.png)](https://bitdeli.com/free "Bitdeli Badge")
=====

Python Library for Probabilistic Graphical Models  
Documentation: [pgmpy](http://pgmpy.readthedocs.org/en/latest/)  
Mailing List: pgmpy@googlegroups.com  
irc: #pgmpy on freenode.net

Dependencies:
=============
- Python 3.3
- NetworkX 1.8.1
- Scipy 0.12.1
- Numpy 1.7.1
- Cython 0.19 (optional)

Installation:
=============
pgmpy is installed using <code>distutils</code>. If you have the tools installed
to build a python extension module:

<code>sudo python3 setup.py install</code>

Example:
========
```python3
from pgmpy import BayesianModel as bm
student = bm.BayesianModel()
# instantiates a new Bayesian Model called 'student'

student.add_nodes_from(['diff', 'intel', 'grade'])
# adds nodes labelled 'diff', 'intel', 'grade' to student

student.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
# adds directed edges from 'diff' to 'grade' and 'intel' to 'grade'

student.set_states({'diff': ['hard', 'easy']})
student.set_rule_for_states('diff', ['easy', 'hard'])
student.set_cpd('diff', [[0.2],[0.8]])
#easy=0.2
#hard=0.8

student.set_states({'intel': ['avg', 'dumb', 'smart']})
student.set_rule_for_states('intel', ['dumb', 'avg', 'smart'])
student.set_cpd('intel', [[0.5], [0.3], [0.2]]) 
#dumb=0.5
#avg=0.3
#smart=0.2

student.set_states({'grade': ['A','C','B']})
student.set_rule_for_parents('grade', ['diff', 'intel'])
student.set_rule_for_states('grade', ['A', 'B', 'C'])
student.set_cpd('grade',
                [[0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1], 
                [0.8,0.8,0.8,0.8,0.8,0.8]]
                )

#diff:       easy                 hard
#intel: dumb   avg   smart    dumb  avg   smart
#gradeA: 0.1    0.1    0.1     0.1  0.1    0.1  
#gradeB: 0.1    0.1    0.1     0.1  0.1    0.1
#gradeC: 0.8    0.8    0.8     0.8  0.8    0.8

student.set_observations({'intel': 'smart', 'diff': 'easy'})
# observed parameters are that intel of student is smart and
# difficulty is easy

student.reset_observations('intel')
# reset observations for intel

active_trail = student.is_active_trail('grade', 'intel')
# returns True if active trail exists between grade and intel

```
