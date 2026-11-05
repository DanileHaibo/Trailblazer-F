"""Audit the public payload and write per-file sizes/hashes; no private source paths."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    forbidden=('__pycache__','.git','third_party','data','outputs')
    entries=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():continue
        rel=path.relative_to(ROOT)
        # Ignore generated local caches and Git metadata, never include them in inventory.
        if any(part in ('__pycache__', '.git', '.pytest_cache') or part.endswith('.egg-info') for part in rel.parts):continue
        if rel.as_posix()=='manifests/files.json':continue
        assert not any(part in forbidden for part in rel.parts),str(rel)
        assert path.suffix.lower() not in ('.pdf','.tex','.doc','.docx','.png','.jpg','.pkl','.csv','.jsonl'),str(rel)
        data=path.read_bytes()
        if path.suffix in ('.py','.json','.md','.toml','.txt'):
            text=data.decode()
            assert ('/'+'data2'+'/') not in text and ('/'+'home'+'/') not in text,str(rel)
        if path.suffix=='.py':ast.parse(data,filename=str(rel))
        entries.append({'path':rel.as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    manifest={'file_count_excluding_inventory':len(entries),'bytes_excluding_inventory':sum(r['bytes'] for r in entries),'files':entries}
    (ROOT/'manifests/files.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))

if __name__=='__main__':main()
