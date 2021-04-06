from kinoml.features.protein import KLIFS_composition
from kinoml.features.ligand import MorganFingerprint
from kinoml.features.complex import KLFISInteractionFingerprint

# Create a PKIS2 dataset with simple protein, ligand, and complex features
from kinoml.tasks import HitClassification, pKd, pIC50


from kinoml.datasets.discoverx.tasks import DiscoverX_HitClassification, DiscoverX_pKd, DiscoverX_pIC50, DiscoverX_MultiClass
from kinoml.datasets.discoverx.datasets import PKIS2
from kinoml.datasets.filters import KinaseFilter, FractionFilter
dataset = PKIS2(protein_features=[KLIFS_composition],
                ligand_features=[MorganFingerprint(radius=3)],
                complex_features=[KLIFSInteractionFingerprint],
                task=DiscoverX_HitClassification,
                pre_transform=None,
                pre_filter=FractionFilter(fraction=0.2, deterministic=True),
                transform=None)


# Create a PKIS2 dataset with simple protein, ligand, and complex features
from kinoml.datasets.chembl.tasks import ChEMBL_HitClassification, ChEMBL_pIC50, ChEMBL_pKd
from kinoml.datasets.chembl.datasets import ChEMBL
from kinoml.datasets.filters import KinaseFilter, FractionFilter

# Also, look into chembl.target.filter
chembl_filter = ChEMBLFilter(
    chembl_id=chembl_ids,
    activities_standard_relation='=',
    assays_assay_type='B',
    activities_standard_type=['IC50', 'Ki'],
    assays_confidence_score='>0')

# Need ability to access specific snapshot of ChEMBL (e.g. ChEMBL 25 from Mar 2019) or live version

dataset = ChEMBL(protein_features=[KLIFS_composition],
                ligand_features=[MorganFingerprint(radius=3)],
                complex_features=[PLIFS],
                source='CHEMBL25', # or live online version, or other version
                query=chembl_filter,
                task=ChEMBL_Ki,
                pre_transform=None,
                pre_filter=FractionFilter(fraction=0.2, deterministic=True),
                transform=None)

# How will we be able to combine datasets in the future?
# Do the datasets have to be homogeneous?
# Can we build a model for one dataset and train for another dataset?

pytorch_dataset = dataset.to_pytorch()
tensorflow_dataset = dataset.to_tensorflow()

# How could we filter by a specific kinase or family? Would we use pre_filter?

# Create a pytorch_geometric dataset that only featurizes ligand into graph
from kinoml.features.graph.ligand import SmallMoleculeGraphFeaturizer

from kinoml.features.graph.ligand.atom_features import default_atom_features
from kinoml.features.graph.ligand.bond_features import default_bond_features
featurizer = SmallMoleculeGraphFeaturizer(
                explicit_hydrogens=True,
                atom_features=default_atom_features,
                edge_features=default_bond_features)

from kinoml.features.graph.ligand import atom_features as af
from kinoml.features.graph.ligand import bond_features as bf
featurizer = SmallMoleculeGraphFeaturizer(
                explicit_hydrogens=False,
                atom_features=[af.atomic_number, af.one_hot_elements, af.degree],
                edge_features=[bf.bond_order, bf.is_aromatic])

import pint
from kinoml.features.graph.complex import ComplexResidueGraphFeaturizer
featurizer = ComplexResidueGraphFeaturizer(
                residue_cutoff=6*angstroms,
                atom_features=default_atom_features,
                bond_features=default_bond_features,
                residue_features=default_residue_features,
)

from kinoml.datasets.graph import PKIS2
geometric_dataset = PKIS2(protein_features=[],
                ligand_features=[],
                complex_features=[KLIFSInteractionFingerprint],
                task=DiscoverX_HitClassification,
                pre_transform=None, pre_filter=None, transform=None)



# Split data into training, validation, and test set using best best practices
from kinoml.ml.cross_validation import ScaffoldSplit
splitter = ScaffoldSplit()
splitter.test_fraction = 0 # Build a model for production use by omitting test set (but not validation set)
training_dataset, validation_dataset, test_dataset = splitter.split_dataset(dataset)

# Create a stochastic minibatch view of the training dataset
from torch.utils.data import DataLoader
data_loader = DatasetLoader(training_dataset, batch_size=10)

# Model defined here
from kinoml.ml.models import DNN
model = DNN(nlayers=10, ...)

# Train the model
import torch.optim as optim

n_epochs = 100
for epoch in range(n_epochs):
    running_training_loss = 0.0
    running_validation_loss = 0.0
    for minibatch_index, minibatch in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        minibatch_inputs, minibatch_labels = minibatch

        # zero the parameter gradients
        model.optimizer.zero_grad()

        # forward + backward + optimize
        training_outputs = model.forward(training_inputs)
        training_loss = model.loss(training_outputs, training_labels)
        training_loss.backward()
        running_training_loss += training_loss.item()
        model.optimizer.step()

    # Compute validation set loss
    validation_inputs, validation_labels
    validation_outputs = model.forward(validation_inputs)
    validation_loss = model.loss(validation_outputs, validation_labels)
    running_validation_loss += validation_loss.item()

    # Report statistics
    n_minibatches = minibatch_index
    running_training_loss /= n_minibatches

    print('[%d] training loss: %.3f  validation loss: %.3f' %
              (epoch, running_training_loss, running_validation_loss ))

print('Finished Training')
