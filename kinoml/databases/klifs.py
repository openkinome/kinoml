import pandas as pd


def klifs_kinase_from_uniprot_id(uniprot_id: str) -> pd.DataFrame:
    """
    Retrieve KLIFS kinase details about the kinase matching the given Uniprot ID.

    Parameters
    ----------
    uniprot_id: str
        Uniprot identifier.

    Returns
    -------
    kinase: pd.Series
        KLIFS kinase details.

    Raises
    ------
    ValueError:
        No KLIFS kinase found for UniProt ID.
    ValueError:
        Multiple KLIFS kinases found for UniProt ID.
    """
    from opencadd.databases.klifs import setup_remote

    remote = setup_remote()
    kinase_ids = remote.kinases.all_kinases()["kinase.klifs_id"]
    kinases = remote.kinases.by_kinase_klifs_id(list(kinase_ids))
    kinases = kinases[kinases["kinase.uniprot"] == uniprot_id]
    if len(kinases) == 0:
        raise ValueError("No KLIFS kinase found for UniProt ID.")
    elif len(kinases) > 1:
        raise ValueError("Multiple KLIFS kinases found for UniProt ID.")
    kinase = kinases.iloc[0]

    return kinase
