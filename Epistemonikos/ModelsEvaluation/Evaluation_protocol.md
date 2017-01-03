## Protocol for model parameters evaluation:

### Metrics:
- **Accuracy**
- **Precision mean**
- **Recall mean**
- **Relatedness**

### Parameters:
- **Embedding dimensionality** (600 | 900 | 1,200 | 1,500 | 1,800)
- **Learning rate** (1 | 0.1 | 0.01 | 0.001)
- **Vocabulary size** (30,000 | 50,000 | 70,000)
- **Window size** (1 | 2 | 3 | 4 | 5)

### Constants:
- **Number of steps** (30,000)
- **Batch size** (50)
- **Train size**  (400,000)
- **Test size** (30,000)

### Method:
For every parameter there's going to be some hipothesis, in regard of what will happen with it's change. Each value will be tested a substantial amount of times : (insert number of times). The other parameters will be set to a fixed value. In further research, a relation between parameters could be helpful.

### Results:
Results will be graphed on boxplots, one for every value of the parameter. The hipothesis will then be accepted/rejected with the corresponding arguments.

### Further work:
- **Study relations between parameters**

