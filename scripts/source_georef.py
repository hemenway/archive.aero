"""Read-only source georeferencing checks for the georef tool's work queue.

A cutline or CRS alone cannot locate pixels. Probe the same local sources the
slicer uses, including every face of a ZIP, without extracting or rasterizing
anything in rawtiffs. This checks metadata, not full raster readability or fit.
"""
import math
import os
import json
from pathlib import Path
import tempfile
import zipfile


class SourceGeorefIndex:
    def __init__(self, root, cache_path=None):
        self.root = Path(root)
        self.files = {}
        self.bundles = {}
        self.cache = {}
        self.zip_cache = {}
        self.directory_cache = {}
        self.cache_path = Path(cache_path) if cache_path else None
        if self.cache_path:
            try:
                data = json.loads(self.cache_path.read_text())
                if data.get('version') == 1:
                    for item in data['probes']:
                        key = (item['path'], item['src_crs'])
                        self.cache[key] = (tuple(tuple(s) for s in item['signature']),
                                           item['dependencies'], bool(item['valid']))
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                self.cache = {}

    def save_cache(self):
        """Keep verified metadata across restarts; signatures are rechecked on use."""
        if self.cache_path is None:
            return
        temp = None
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            probes = [{'path': path, 'src_crs': crs, 'signature': signature,
                       'dependencies': [str(p) for p in deps], 'valid': valid}
                      for (path, crs), (signature, deps, valid) in list(self.cache.items())]
            with tempfile.NamedTemporaryFile(mode='w', dir=self.cache_path.parent,
                                             prefix=self.cache_path.name + '.', delete=False) as output:
                temp = Path(output.name)
                json.dump({'version': 1, 'probes': probes}, output)
            temp.replace(self.cache_path)
        except OSError:
            pass  # a failed performance cache must not break the work queue
        finally:
            if temp is not None:
                try:
                    temp.unlink(missing_ok=True)
                except OSError:
                    pass

    def add_file(self, path):
        """Feed the tool's existing directory walk; no second rawtiffs walk."""
        path = Path(path)
        if path.suffix.lower() not in {'.tif', '.tiff', '.jpg', '.jpeg', '.pdf', '.zip'}:
            return
        self.files.setdefault(path.name, []).append(path)
        if path.suffix.lower() in {'.tif', '.tiff', '.pdf'}:
            for parent in path.parents:
                if parent == self.root:
                    break
                self.bundles.setdefault(parent.name, []).append(path)

    @staticmethod
    def _rasters(paths):
        # Slicer prefers materialized TIFFs; a PDF-only bundle still needs
        # its actual PDF georeferencing checked, never its filename/cutline.
        tifs = [p for p in paths if Path(p).suffix.lower() in {'.tif', '.tiff'}]
        return tifs or [p for p in paths if Path(p).suffix.lower() == '.pdf']

    @staticmethod
    def _signature(paths):
        signature = []
        for path in paths:
            try:
                stat = Path(path).stat()
                signature.append((str(path), stat.st_mtime_ns, stat.st_size))
            except OSError:
                signature.append((str(path), None, None))
        return tuple(signature)

    def sources(self, filename):
        if filename.lower().endswith('.zip'):
            extracted = self._rasters(self.bundles.get(Path(filename).stem, []))
            if extracted:
                return [(str(p), p) for p in extracted]
            sources = []
            for archive in self.files.get(filename, []):
                signature = self._signature([archive])
                cached = self.zip_cache.get(archive)
                if cached and cached[0] == signature:
                    members = cached[1]
                else:
                    try:
                        with zipfile.ZipFile(archive) as bundle:
                            members = self._rasters(bundle.namelist())
                    except (OSError, zipfile.BadZipFile):
                        members = []
                    self.zip_cache[archive] = (signature, members)
                if not members:
                    return []
                sources.extend((f'/vsizip/{archive}/{member}', archive) for member in members)
            return sources
        paths = self.files.get(filename, [])
        if not paths and Path(filename).suffix.lower() in {'.tif', '.tiff'}:
            # The slicer can materialize a missing TIFF row from its local
            # same-stem PDF. Inspect that PDF in place instead of writing it.
            paths = self.files.get(str(Path(filename).with_suffix('.pdf')), [])
        # A direct PDF row uses its already-materialized TIFF when present.
        sources = []
        for path in paths:
            if path.suffix.lower() == '.pdf' and path.with_suffix('.tif').is_file():
                path = path.with_suffix('.tif')
            sources.append((str(path), path))
        return sources

    def has_georef(self, filename, src_crs=''):
        sources = self.sources(filename)
        # A good north face cannot conceal an unreferenced south face.
        return bool(sources) and all(self._probe(path, physical, src_crs)
                                     for path, physical in sources)

    def _probe(self, path, physical, src_crs):
        key = (path, src_crs)
        cached = self.cache.get(key)
        if cached and cached[0] == self._signature(cached[1]):
            return cached[2]
        # Parent mtime catches newly added world files; GDAL's dependency list
        # catches edits/removals of existing world files and PAM metadata.
        dependencies = [physical, physical.parent]
        siblings = None
        if not path.startswith('/vsi'):
            parent_signature = self._signature([physical.parent])
            directory = self.directory_cache.get(physical.parent)
            if not directory or directory[0] != parent_signature:
                try:
                    directory = (parent_signature, os.listdir(physical.parent))
                    self.directory_cache[physical.parent] = directory
                except OSError:
                    directory = None
            if directory:
                # GDAL otherwise re-lists a source directory for each TIFF to
                # discover sidecars. That is expensive for thousands of FAA
                # files on SMB; share the listing, including world/PAM files.
                siblings = directory[1]
        valid = False
        from osgeo import gdal
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        dataset = transformer = None
        try:
            kwargs = {'sibling_files': siblings} if siblings is not None else {}
            dataset = gdal.OpenEx(path, gdal.OF_RASTER | gdal.OF_READONLY, **kwargs)
            if dataset is not None:
                dependencies.extend(p for p in dataset.GetFileList() or []
                                    if not p.startswith('/vsi') and Path(p) not in dependencies)
                transform = dataset.GetGeoTransform(can_return_null=True)
                affine = (transform is not None and all(math.isfinite(v) for v in transform)
                          and transform[1] * transform[5] - transform[2] * transform[4] != 0
                          and bool(src_crs or dataset.GetProjection()))
                gcps = dataset.GetGCPCount() > 0 and bool(src_crs or dataset.GetGCPProjection())
                if affine or gcps:
                    options = ['DST_SRS=EPSG:3857']
                    if src_crs:
                        options.append('SRC_SRS=' + src_crs)
                    transformer = gdal.Transformer(dataset, None, options)
                    if transformer is not None:
                        ok, xyz = transformer.TransformPoint(
                            False, dataset.RasterXSize / 2, dataset.RasterYSize / 2)
                        valid = bool(ok) and all(math.isfinite(v) for v in xyz)
        except (RuntimeError, OSError, ValueError):
            valid = False
        finally:
            transformer = dataset = None
            gdal.PopErrorHandler()
        self.cache[key] = (self._signature(dependencies), dependencies, valid)
        return valid
