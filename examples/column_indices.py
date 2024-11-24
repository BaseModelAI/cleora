import numpy as np
from pycleora import SparseMatrix

hyperedges = [
    'a\t1',
    'a\t2',
    'b\t5',
    'b\t2',
    'c\t8',
]

graph = SparseMatrix.from_iterator((e for e in hyperedges), "char num")

entity_ids = np.array(graph.entity_ids)
print(entity_ids)
print(graph.entity_degrees)

print(graph.get_entity_column_mask('char'))
print(graph.get_entity_column_mask('num'))

print(entity_ids[graph.get_entity_column_mask('char')])
print(entity_ids[graph.get_entity_column_mask('num')])