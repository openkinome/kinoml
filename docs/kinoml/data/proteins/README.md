## File description

### 4f8o.pdb

This protein was chosen for writing unit tests, since it contains protein and ligand residues as 
well as multiple chains and alternate locations.

### 4f8o_edit.pdb

The 4f8o.pdb structure was altered in the following fashion:
 - translated along x axis by 20 A --> superposition
 - selected alternate location A 
 - removed non protein atoms
 - deleted ASP82 --> loop modeling
 - deleted LYS135 --> detection of short protein segments
 - deleted sidechain of ASN2 --> sidechain perception and modeling
 - altered Chi1 dihedral of PHE4 to -1 radians --> detection of sidechain clashes

### kinoml_tests_4f8o_spruce.loop_db

This loop template database was created using the loopdb_builder app based on 4f8o.pdb. It is used 
for testing loop modeling.
