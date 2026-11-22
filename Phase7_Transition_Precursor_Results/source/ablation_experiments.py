"""Prespecified deterministic candidate ordering; no access to test scores."""
EXPERIMENT_GROUPS={'E3':'threshold','E4':'temporal','E5':'all','E6':'all','E7':'all'}
def rank(row):return (-row['J'],-row['macro_f1'],row.get('n_features',0),row.get('C',0),row.get('weighted',False),row['id'])
