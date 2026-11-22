"""Build both deliverable ZIPs and check CRC plus every member against current files."""
import hashlib,json,sys,zipfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent;AUDIT=ROOT.parent/'Phase7_Independent_Verification_Results'
sys.path.insert(0,str(AUDIT))
from audit_source.final_readiness_audit import write_reports

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def files(root):return sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts)
def inventory(root,name):
    target=root/'manifests'/name
    data=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in files(root) if p!=target]
    target.write_text(json.dumps({'self_excluded':True,'files':data},indent=2),encoding='utf8')
def build(root):
    target=root.parent/(root.name+'.zip')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in files(root):archive.write(p,root.name+'/'+p.relative_to(root).as_posix())
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for p in files(root):assert hashlib.sha256(archive.read(root.name+'/'+p.relative_to(root).as_posix())).hexdigest()==digest(p)
        assert len(archive.namelist())==len(files(root))
    return dict(archive=target.name,members=len(files(root)),bytes=target.stat().st_size,crc='PASS',member_bytes='PASS')

if __name__=='__main__':
    matrix_path=AUDIT/'results/final_verification_matrix.csv';m=pd.read_csv(matrix_path)
    # First make and independently open actual ZIPs before upgrading package status.
    write_reports(AUDIT,ROOT)
    for root,name in [(ROOT,'artifact_inventory.json'),(AUDIT,'audit_artifact_inventory.json')]:inventory(root,name);build(root)
    for name in [ROOT.name,AUDIT.name]:
        mask=m['check']==f'{name}: ZIP CRC';m.loc[mask,'status']='PASS';m.loc[mask,'detail']='CRC and every current member checked by package_results.py'
    m.to_csv(matrix_path,index=False)
    write_reports(AUDIT,ROOT)
    results=[]
    for root,name in [(ROOT,'artifact_inventory.json'),(AUDIT,'audit_artifact_inventory.json')]:inventory(root,name);results.append(build(root))
    print(json.dumps(results,indent=2));print(m.status.value_counts().to_dict())
