"""Restore the pinned notebook inputs without overwriting local experiment work."""
import hashlib
import json
from pathlib import Path
import zipfile

def restore(root):
    root=Path(root).resolve()
    manifest=root/'reproducibility/manifest.json'
    if not manifest.exists():
        return  # Existing local workspaces already contain these artifacts.
    record=json.loads(manifest.read_text())
    archive=manifest.parent/'replay-data.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=record['archive_sha256']:
        raise ValueError('Notebook replay archive checksum mismatch.')
    with zipfile.ZipFile(archive) as bundle:
        if set(bundle.namelist())!=set(record['files']):
            raise ValueError('Notebook replay manifest differs from the archive.')
        for name,expected in record['files'].items():
            destination=(root/name).resolve()
            if not destination.is_relative_to(root/'build'):
                raise ValueError('Replay files must stay in the notebook build directory.')
            if destination.exists():
                continue
            data=bundle.read(name)
            if hashlib.sha256(data).hexdigest()!=expected:
                raise ValueError(f'Replay input checksum mismatch: {name}')
            destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=destination.with_suffix(destination.suffix+'.restoring')
            temporary.write_bytes(data);temporary.replace(destination)
