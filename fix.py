with open('pgmpy/readwrite/BIF.py', 'r') as f:
    text = f.read()
text = text.replace('"""network', "'''network").replace('... """', "... '''")
with open('pgmpy/readwrite/BIF.py', 'w') as f:
    f.write(text)
