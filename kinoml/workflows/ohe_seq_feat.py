"""
This script tests the one hot encoding class.
The script assumes that you have a conda environment with requests and matplotlib installed.
It take the '3W2S' PDB file and uses KILFS Swagger and requests to obtain the binding site pocket.
Run the following command before executing the script:
$ export PYTHONPATH=$(dirname $(pwd))
# Check PYTHONPATH with:
$ echo $PYTHONPATH
"""

# Imports
from features.protein import OneHotEncodingAminoAcid, ALL_AMINOACIDS
import requests
import matplotlib.pyplot as plt

if __name__ == '__main__':

	print('All possible amino acid that compose the dictionary: ', ALL_AMINOACIDS)

	PDB_ID = '3W2S' # Pick a PDB ID
	url = f'https://klifs.vu-compmedchem.nl/api/kinase_information?kinase_ID={PDB_ID}&species=HUMAN'
	r = requests.get(url) # Obtain the binding site pocket with KLIFS Swagger
	seq = r.json()[0]['pocket']
	print(f'Binding pocket sequence for PDB {PDB_ID} : {seq} .')

	ohe_seq = OneHotEncodingAminoAcid(seq) # One hot encoding of that sequence
	oh = ohe_seq.from_seq2oh()
	print(oh)

	# Plot the associated matrix
	f = plt.figure()
	plt.imshow(oh)
	plt.show(block=False)
	plt.pause(3)
	plt.close()