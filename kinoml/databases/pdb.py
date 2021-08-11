from typing import Iterable


def smiles_from_pdb(ligand_ids: Iterable[str]) -> dict:
    """
    Retrieve SMILES of molecules defined by their PDB chemical identifier.

    Parameters
    ----------
    ligand_ids: iterable of str
        PDB chemical identifier.

    Returns
    -------
    ligands: dict
        Dictionary with PDB chemical identifier as keys and SMILES as values.
    """
    import json
    import requests
    import urllib

    ligands = {}
    base_url = "https://data.rcsb.org/graphql?query="
    query = '{chem_comps(comp_ids:[' + \
            ','.join(['"' + ligand_id + '"' for ligand_id in set(ligand_ids)]) + \
            ']){chem_comp{id}rcsb_chem_comp_descriptor{SMILES_stereo}}}'
    response = requests.get(base_url + urllib.parse.quote(query))
    for ligand in json.loads(response.text)["data"]["chem_comps"]:
        try:
            ligands[ligand["chem_comp"]["id"]] = ligand["rcsb_chem_comp_descriptor"]["SMILES_stereo"]
        except TypeError:
            # missing smiles entry
            pass

    return ligands
