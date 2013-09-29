PgmPy [![Build Status](https://travis-ci.org/pgmpy/pgmpy.png)](https://travis-ci.org/pgmpy/pgmpy)
=====

Python Library for Probabilistic Graphical Models

Dependencies:
=============
- Python 3.3
- NetworkX 1.8.1
- Scipy 0.12.1
- Numpy 1.7.1

Example:
========
```python3
student = BayesianModel()
# instantiates a new Bayseian Model called 'student'

student.add_nodes('diff', 'intel', 'grade')
# adds nodes labelled 'diff', 'intel', 'grade' to student

student.add_edges(('diff', 'intel'), 'grade')
# adds directed edges from 'diff' to 'grade' and 'intel' to 'grade'


student.add_states('diff', ('hard', 'easy'))
student.add_rule_for_states('diff', ('easy', 'hard'))
student.add_cpd('diff', [[0.2],[0.8]])
#easy=0.2
#hard=0.8

student.add_states('intel', ('avg', 'dumb', 'smart'))
student.add_rule_for_states('intel', ('dumb', 'avg', 'smart'))
student.add_cpd('intel', [[0.5], [0.3], [0.2]]) 
#dumb=0.5
#avg=0.3
#smart=0.2

student.add_states('grades', ('A','C','B'))
student.add_rule_for_parents('grades', ('diff', 'intel'))
student.add_rule_for_states('grades', ('A', 'B', 'C'))
student.add_cpd('grade',
                [[0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1], 
                [0.8,0.8,0.8,0.8,0.8,0.8]]
                )

#diff:       easy                 hard
#intel: dumb   avg   smart    dumb  avg   smart
#gradeA: 0.1    0.1    0.1     0.1  0.1    0.1  
#gradeB: 0.1    0.1    0.1     0.1  0.1    0.1
#gradeC: 0.8    0.8    0.8     0.8  0.8    0.8

student.add_observation(intel.smart, intel.avg, diff.easy)
# observed parameters are that intel of student is either smart or avg and 
# difficulty is easy

student.remove_observation(intel.smart)
# observed parameters are that intel of student is either smart or avg and 
# difficulty is easy


active_trail = student.is_active_trail(grade, intel)
# returns True if active trail exists between grade and intel

updated_cpd = student.observed_cpd('grade')
# returns 2D array of new cpd after last observance

student.reset()
# makes all parameters non-observed and resets model to initial state with 
# initial user-given CPDs
```
