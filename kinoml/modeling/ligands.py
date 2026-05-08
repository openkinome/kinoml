"""
Modeling tools for small compounds
"""

import os
from openforcefield.topology import Molecule
from openforcefield.utils.toolkits import OpenEyeToolkitWrapper


def load_molecule(file_or_smiles):
    if os.path.isfile(file_or_smiles):
        return Molecule.from_file(file_or_smiles)
    return Molecule.from_smiles(file_or_smiles)


def generate_conformers(molecule):
    """
    Refactor this so it uses the Python API

    >>> def conf_gen(smi,maxconfs=2000):
    >>>     smi_prefix = os.path.splitext(os.path.basename(smi))[0]
    >>>     print('{0} -in {1}/{2} -out {1}/OMEGA/{3}_omega.sdf -prefix {1}/OMEGA/{3}_omega -warts true -maxconfs {4} -strict false'.format(OMEGA, os.getcwd(), smi, smi_prefix, maxconfs))
    >>>     os.system('{0} -in {1}/{2} -out {1}/OMEGA/{3}_omega.sdf -prefix {1}/OMEGA/{3}_omega -warts true -maxconfs {4} -strict false'.format(OMEGA, os.getcwd(), smi, smi_prefix, maxconfs))
    """
    # read here: https://docs.eyesopen.com/toolkits/python/omegatk/omegaexamples.html
    # or simply use the Openforcefield wrapper


def superpose(molecule, *targets):
    """
    Refactor this so it uses the Python API

    >>> def lig_alignment(conformer, template_database, rocs_maxconfs_output=100):
    >>>     sdf_prefix = os.path.basename(os.path.splitext(conformer)[0]).split('_')[0]
    >>>     for template in template_database:
    >>>         template_id = "_".join(os.path.basename(template).split("_")[0:3])
    >>>         print('{0} -dbase {1}/{2} -query {3} -prefix {4}_{5}_rocs -oformat sdf -maxconfs 30 -outputquery false -qconflabel title -outputdir {1}/ROCS/'.format(ROCS, os.getcwd(),conformer, template, sdf_prefix, template_id))
    >>>         os.system('{0} -dbase {1}/{2} -query {3} -prefix {4}_{5}_rocs -oformat sdf -maxconfs 30 -outputquery false -qconflabel title -outputdir {1}/ROCS/'.format(ROCS, os.getcwd(),conformer, template, sdf_prefix, template_id))
    """
    if not OpenEyeToolkitWrapper.is_available():
        raise RuntimeError("OpenEye Toolkit must be installed and licensed")
    from openeye import oeshape
    # read here: https://docs.eyesopen.com/toolkits/python/shapetk/shape_examples.html#rocs


def parameterize_for_rosetta(molecule):
    """
    Refactor this so it uses the Python API

    >>> def sdftoparams(mol2params, top_hits_sdf_path):
    >>>     for file in top_hits_sdf_path:
    >>>         out_put_file_name = os.path.splitext(os.path.basename(file))[0]
    >>>         print('{0} {1} -p sdf2params/{2}'.format(mol2params, file, out_put_file_name))
    >>>         os.system('{0} {1} -p sdf2params/{2}'.format(mol2params, file, out_put_file_name))
    """
    # import importlib.util
    # from distutils.spawn import find_executable
    # spec = importlib.util.spec_from_file_location("molfile_to_params", find_executable("molfile_to_params.py"))
    # molfile_to_params = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(molfile_to_params)

    # from molfile_to_params import main as generate_rosetta_params
    # args = []
    # generate_rosetta_params(args)
