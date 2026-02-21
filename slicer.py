#!/usr/bin/env python3
"""
FAA Chart Newslicer - CLI tool for processing sectional charts
Creates COGs for each date for 56 locations.
"""

import os
import sys
import csv
import re
import argparse
import zipfile
import sqlite3
import math
try:
    import requests
except ImportError:
    requests = None
import time
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict, deque
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

try:
    from osgeo import gdal, ogr, osr
    gdal.UseExceptions()
except ImportError:
    print("ERROR: GDAL is required but not installed.")
    print("Install with: pip install gdal --break-system-packages")
    sys.exit(1)

import multiprocessing
cpu_count = multiprocessing.cpu_count()
IS_APPLE_SILICON = (sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"})


def _get_memory_snapshot_mb(
    default_available_mb: int = 4096,
    default_total_mb: int = 8192
) -> Dict[str, int]:
    """Return a best-effort memory snapshot in MB."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        available_mb = int(vm.available // (1024 * 1024))
        total_mb = int(vm.total // (1024 * 1024))
        return {
            "available_mb": max(1, available_mb),
            "total_mb": max(max(1, available_mb), total_mb),
        }
    except Exception:
        fallback_total = max(default_total_mb, default_available_mb)
        return {
            "available_mb": max(1, default_available_mb),
            "total_mb": max(1, fallback_total),
        }


def _recommended_free_ram_floor_mb(total_ram_mb: int) -> int:
    """RAM headroom to preserve for OS + transient process spikes."""
    total_ram_mb = max(1024, int(total_ram_mb or 0))
    return max(768, min(2048, total_ram_mb // 10))


def _get_available_ram_mb(default_mb: int = 4096) -> int:
    """Return currently available RAM in MB, or a conservative fallback."""
    return _get_memory_snapshot_mb(default_available_mb=default_mb)["available_mb"]


def _auto_postprocess_settings(
    worker_count: int,
    requested_threads: Union[int, str, None] = "AUTO",
    requested_cache_mb: Optional[int] = None
) -> Tuple[int, int]:
    """Choose postprocess thread/cache settings with Apple Silicon-aware defaults."""
    worker_count = max(1, int(worker_count or 1))
    total_cpu = max(1, cpu_count)
    per_worker_cpu = max(1, total_cpu // worker_count)
    available_ram_mb = _get_available_ram_mb(default_mb=8192 if IS_APPLE_SILICON else 4096)

    req_threads = str(requested_threads).strip().upper() if requested_threads is not None else "AUTO"
    if req_threads == "AUTO":
        # Use all available per-worker CPU threads in AUTO mode.
        if IS_APPLE_SILICON:
            threads = max(2, per_worker_cpu)
        else:
            threads = max(1, per_worker_cpu)
    else:
        try:
            threads = int(req_threads)
        except ValueError:
            threads = per_worker_cpu
        threads = max(1, min(threads, total_cpu))

    cache_override = None
    if requested_cache_mb is not None:
        try:
            parsed_cache = int(requested_cache_mb)
            if parsed_cache > 0:
                cache_override = parsed_cache
        except (TypeError, ValueError):
            cache_override = None

    if cache_override is not None:
        cache_mb = max(256, cache_override)
    else:
        cache_cap = 3072 if worker_count <= 2 else 2048
        cache_mb = max(512, min(cache_cap, available_ram_mb // worker_count))

    return threads, cache_mb


def _overview_levels_for_mbtiles(mbtiles_path: Path) -> List[str]:
    """
    Compute overview levels so the smallest pyramid level is near 256 px.
    Avoids generating unnecessary deep overviews on small charts.
    """
    default_levels = ["2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"]
    try:
        ds = gdal.Open(str(mbtiles_path))
        if not ds:
            return default_levels
        min_dim = min(ds.RasterXSize, ds.RasterYSize)
        ds = None

        levels: List[str] = []
        level = 2
        while level <= 1024 and (min_dim // level) >= 256:
            levels.append(str(level))
            level *= 2

        return levels if levels else ["2"]
    except Exception:
        return default_levels

# Set minimal GDAL global defaults (defer thread decisions to phase-specific config)
available_ram_mb = _get_available_ram_mb(default_mb=4096)
cache_size_mb = max(512, min(8192, available_ram_mb * 3 // 4))  # 75% of RAM, max 8GB

# Only set cache at global level - thread settings will be phase-specific
gdal.SetConfigOption('GDAL_CACHEMAX', str(cache_size_mb))
gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK', 'YES')


def _configure_gdal_for_phase(phase: str, workers: int = 1, available_ram_mb: int = None):
    """
    Configure GDAL settings for specific processing phases.

    Args:
        phase: One of 'warp', 'geotiff'
        workers: Number of parallel workers
        available_ram_mb: Available RAM in MB (auto-detect if None)
    """
    if available_ram_mb is None:
        available_ram_mb = _get_available_ram_mb(default_mb=4096)

    if phase == 'warp':
        # Warp phase: single-threaded per worker to prevent oversubscription
        gdal.SetConfigOption('GDAL_NUM_THREADS', '1')
        cache_per_worker = max(256, available_ram_mb // workers)
        gdal.SetConfigOption('GDAL_CACHEMAX', str(cache_per_worker))

    elif phase == 'geotiff':
        # GeoTIFF phase: single-process translate with full CPU
        gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
        gdal.SetConfigOption('GDAL_CACHEMAX', str(max(512, min(8192, available_ram_mb * 3 // 4))))


def create_geotiff_worker(args):
    """Worker for parallel GeoTIFF creation."""
    try:
        input_vrt, output_tiff, date, compress, num_threads, clip_projwin, cache_size_mb = args
    except ValueError:
        input_vrt, output_tiff, date, compress, num_threads, clip_projwin = args
        cache_size_mb = None

    start_time = time.time()
    output_tiff = Path(output_tiff)

    try:
        from osgeo import gdal
        gdal.UseExceptions()
    except Exception as e:
        return {
            'success': False,
            'date': date,
            'output_tiff': str(output_tiff),
            'error': f"GDAL import failed: {e}"
        }

    try:
        if cache_size_mb:
            gdal.SetConfigOption('GDAL_CACHEMAX', str(cache_size_mb))
        if num_threads:
            gdal.SetConfigOption('GDAL_NUM_THREADS', str(num_threads))
    except Exception:
        pass

    output_tiff.parent.mkdir(parents=True, exist_ok=True)

    attempted_lzw = False
    current_compress = compress

    while True:
        try:
            creation_opts = [
                'TILED=YES',
                'BIGTIFF=YES',
                'BLOCKXSIZE=1024',
                'BLOCKYSIZE=1024',
                f'NUM_THREADS={num_threads}'
            ]

            if current_compress != 'NONE':
                creation_opts.extend([
                    f'COMPRESS={current_compress}',
                    'PREDICTOR=2'
                ])

            translate_options = gdal.TranslateOptions(
                format='GTiff',
                creationOptions=creation_opts,
                projWin=clip_projwin if clip_projwin else None
            )

            ds = gdal.Translate(str(output_tiff), str(input_vrt), options=translate_options)

            if ds:
                ds = None
                file_size_mb = output_tiff.stat().st_size / (1024 * 1024) if output_tiff.exists() else 0
                elapsed = time.time() - start_time
                rate = file_size_mb / elapsed if elapsed > 0 else 0
                return {
                    'success': True,
                    'date': date,
                    'output_tiff': str(output_tiff),
                    'elapsed': elapsed,
                    'file_size_mb': file_size_mb,
                    'rate': rate,
                    'compress': current_compress
                }

            return {
                'success': False,
                'date': date,
                'output_tiff': str(output_tiff),
                'error': 'gdal.Translate returned None'
            }

        except Exception as e:
            error_str = str(e).lower()
            if current_compress == 'ZSTD' and not attempted_lzw and (
                'zstd' in error_str or 'frame descriptor' in error_str or 'data corruption' in error_str
            ):
                attempted_lzw = True
                current_compress = 'LZW'
                try:
                    if output_tiff.exists():
                        output_tiff.unlink()
                except Exception:
                    pass
                continue

            import traceback
            return {
                'success': False,
                'date': date,
                'output_tiff': str(output_tiff),
                'error': str(e),
                'traceback': traceback.format_exc(),
                'compress': current_compress
            }


def postprocess_tif_worker(args):
    """Convert a mosaic raster (VRT/TIFF) to MBTiles/PMTiles and upload PMTiles via rclone."""
    if len(args) < 4:
        raise ValueError("postprocess_tif_worker expected at least 4 args")

    input_raster, output_dir, remote, rclone_flags = args[:4]
    delete_input = args[4] if len(args) > 4 else False
    mbtiles_quality = args[5] if len(args) > 5 else 80
    requested_threads = args[6] if len(args) > 6 else "AUTO"
    requested_cache_mb = args[7] if len(args) > 7 else None
    quiet_external_tools = bool(args[8]) if len(args) > 8 else True
    output_stem = args[9] if len(args) > 9 else None
    clip_projwin = args[10] if len(args) > 10 else None

    input_raster = Path(input_raster)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = str(output_stem).strip() if output_stem else input_raster.stem

    try:
        mbtiles_quality = int(mbtiles_quality)
    except Exception:
        mbtiles_quality = 80
    mbtiles_quality = max(1, min(100, mbtiles_quality))
    gdal_threads, gdal_cache_mb = _auto_postprocess_settings(
        worker_count=1,
        requested_threads=requested_threads,
        requested_cache_mb=requested_cache_mb
    )
    ogr_sqlite_cache_mb = max(512, min(4096, gdal_cache_mb // 2))

    mbtiles_path = output_dir / f"{output_stem}.mbtiles"
    pmtiles_path = output_dir / f"{output_stem}.pmtiles"
    partial_db_path = output_dir / f"{output_stem}.partial_tiles.db"
    stage_times: Dict[str, float] = {}

    def run(cmd):
        stage_start = time.time()
        if quiet_external_tools:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            tail = ""
            if result.returncode != 0 and result.stdout:
                lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
                tail = "\n".join(lines[-8:])
            return result.returncode == 0, result.returncode, tail, time.time() - stage_start
        result = subprocess.run(cmd)
        return result.returncode == 0, result.returncode, "", time.time() - stage_start

    input_mtime = input_raster.stat().st_mtime if input_raster.exists() else 0
    pmtiles_up_to_date = (
        pmtiles_path.exists()
        and pmtiles_path.stat().st_size > 0
        and pmtiles_path.stat().st_mtime >= input_mtime
    )

    if not pmtiles_up_to_date:
        if mbtiles_path.exists():
            try:
                mbtiles_path.unlink()
            except Exception:
                pass
        if pmtiles_path.exists():
            try:
                pmtiles_path.unlink()
            except Exception:
                pass
        # Clean stale partial tile sidecars from prior interrupted/failed attempts.
        for stale_path in (
            partial_db_path,
            Path(str(partial_db_path) + "-journal"),
            Path(str(partial_db_path) + "-wal"),
            Path(str(partial_db_path) + "-shm"),
            Path(str(mbtiles_path) + "-journal"),
            Path(str(mbtiles_path) + "-wal"),
            Path(str(mbtiles_path) + "-shm"),
        ):
            if stale_path.exists():
                try:
                    stale_path.unlink()
                except Exception:
                    pass
        translate_cmd = [
            "gdal_translate", "-q", "-of", "MBTILES",
            "--config", "GDAL_NUM_THREADS", str(gdal_threads),
            "--config", "GDAL_CACHEMAX", str(gdal_cache_mb),
            "-co", "TILE_FORMAT=WEBP",
            "-co", f"QUALITY={mbtiles_quality}",
        ]
        if clip_projwin:
            try:
                ulx, uly, lrx, lry = clip_projwin
                translate_cmd.extend(["-projwin", str(ulx), str(uly), str(lrx), str(lry)])
            except Exception:
                pass
        translate_cmd.extend([str(input_raster), str(mbtiles_path)])
        ok, rc, err_tail, elapsed_sec = run(translate_cmd)
        stage_times["gdal_translate"] = elapsed_sec
        if not ok:
            error_msg = f"gdal_translate failed (exit {rc})"
            if err_tail:
                error_msg += f"\n{err_tail}"
            return {
                'success': False,
                'input_raster': str(input_raster),
                'error': error_msg,
                'stage_times': stage_times
            }

        overview_levels = _overview_levels_for_mbtiles(mbtiles_path)
        ok, rc, err_tail, elapsed_sec = run([
            "gdaladdo", "-q",
            "--config", "GDAL_NUM_THREADS", str(gdal_threads),
            "--config", "OGR_SQLITE_JOURNAL", "WAL",
            "--config", "OGR_SQLITE_CACHE", str(ogr_sqlite_cache_mb),
            "--config", "GDAL_CACHEMAX", str(gdal_cache_mb),
            "-r", "bilinear",
            str(mbtiles_path),
            *overview_levels
        ])
        stage_times["gdaladdo"] = elapsed_sec
        if not ok:
            error_msg = f"gdaladdo failed (exit {rc})"
            if err_tail:
                error_msg += f"\n{err_tail}"
            return {
                'success': False,
                'input_raster': str(input_raster),
                'error': error_msg,
                'stage_times': stage_times
            }

        ok, rc, err_tail, elapsed_sec = run(["pmtiles", "convert", str(mbtiles_path), str(pmtiles_path)])
        stage_times["pmtiles_convert"] = elapsed_sec
        if not ok:
            error_msg = f"pmtiles convert failed (exit {rc})"
            if err_tail:
                error_msg += f"\n{err_tail}"
            return {
                'success': False,
                'input_raster': str(input_raster),
                'error': error_msg,
                'stage_times': stage_times
            }

    if not (pmtiles_path.exists() and pmtiles_path.stat().st_size > 0):
        return {
            'success': False,
            'input_raster': str(input_raster),
            'error': "pmtiles output missing",
            'stage_times': stage_times
        }

    upload_flags = list(rclone_flags)
    if quiet_external_tools:
        cleaned_flags = []
        skip_next = False
        for flag in upload_flags:
            if skip_next:
                skip_next = False
                continue
            if flag in {"-P", "--progress"}:
                continue
            if flag == "--stats":
                skip_next = True
                continue
            cleaned_flags.append(flag)
        upload_flags = cleaned_flags

    upload_cmd = ["rclone", "copyto", str(pmtiles_path), f"{remote}/{pmtiles_path.name}"] + upload_flags
    ok, rc, err_tail, elapsed_sec = run(upload_cmd)
    stage_times["rclone_upload"] = elapsed_sec
    if not ok:
        error_msg = f"rclone upload failed (exit {rc})"
        if err_tail:
            error_msg += f"\n{err_tail}"
        return {
            'success': False,
            'input_raster': str(input_raster),
            'error': error_msg,
            'stage_times': stage_times
        }

    deleted_input = False
    delete_error = None
    if delete_input and input_raster.suffix.lower() in {".tif", ".tiff"}:
        try:
            if input_raster.exists():
                input_raster.unlink()
                deleted_input = True
        except Exception as e:
            delete_error = str(e)

    return {
        'success': True,
        'input_raster': str(input_raster),
        'pmtiles': str(pmtiles_path),
        'stage_times': stage_times,
        'deleted_input': deleted_input,
        'delete_error': delete_error
    }


class ChartSlicer:
    """Process FAA charts and create COGs."""

    @staticmethod
    def _check_zstd_support() -> bool:
        """Check if GDAL supports ZSTD compression."""
        try:
            driver = gdal.GetDriverByName('GTiff')
            if driver:
                metadata = driver.GetMetadata()
                if metadata and 'ZSTD' in metadata.get('DMD_CREATIONOPTIONLIST', ''):
                    return True
        except:
            pass
        return False

    def __init__(self, source_dir: Path, output_dir: Path, csv_file: Path, shape_dir: Path,
                 preview_mode: bool = False, download_delay: float = 8.0, temp_dir: Path = None):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.csv_file = csv_file
        self.shape_dir = shape_dir
        self.temp_dir = temp_dir if temp_dir else Path("/Volumes/projects/.temp")
        self.dole_data = defaultdict(list)
        self.srs_cache = {}  # Cache for shapefile SRS lookups
        self.preview_mode = preview_mode  # If True, only report what would be downloaded
        self.download_delay = download_delay  # Seconds to wait between downloads (to avoid rate limiting)
        self.last_download_time = 0  # Track last download time for throttling
        self._date_cache: Dict[str, Optional[datetime.date]] = {}
        self.date_key_map: Dict[str, Tuple[str, str]] = {}
        self.upload_root = Path("/Users/ryanhemenway/Desktop/sync")
        self.upload_remote = "r2:charts/sectionals"
        self.upload_jobs = 1
        self.postprocess_threads: Union[int, str] = "AUTO"
        self.postprocess_cache_mb: Optional[int] = None
        self.skip_existing_postprocess = True
        self.quiet_external_tools = True
        self.delete_geotiff_after_mbtiles = True
        self.rclone_flags = [
            "--s3-upload-concurrency", "16",
            "--s3-chunk-size", "128M",
            "--buffer-size", "128M",
            "--s3-disable-checksum"
        ]
        if not self.quiet_external_tools:
            self.rclone_flags.extend(["-P", "--stats", "1s"])
        self.mbtiles_quality = 80
        self.max_postprocess_backlog = 10
        self.parallel_warp = 0
        # In threaded warp mode we already parallelize by job; avoid extra internal threads on Apple Silicon.
        self.warp_multithread = not IS_APPLE_SILICON
        self.abbreviations = {
            "hawaiian_is": "hawaiian_islands",
            "dallas_ft_worth": "dallas_ft_worth",
            "mariana_islands_inset": "mariana_islands",
        }
        # Use default PROJ transformer selection to avoid strict-op failures on Alaska/dateline charts.
        self.transformer_options = None

        # File index caches for O(1) lookups
        self.files_by_name: Dict[str, Path] = {}  # Direct TIFs and ZIPs
        self.tifs_by_dir: Dict[str, List[Path]] = {}  # Extracted folder -> TIF list
        self.shapefile_index: Dict[str, Path] = {}  # Normalized location -> shapefile path

        # Kept for backward compatibility with existing CLI options.
        self.compression = 'AUTO'

    def log(self, msg: str):
        """Print log message with timestamp."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")

    def normalize_name(self, name: str) -> str:
        """Normalize chart name consistently."""
        norm = name.strip().lower().replace(' ', '_').replace('-', '_').replace('.', '_')
        while '__' in norm:
            norm = norm.replace('__', '_')
        return self.abbreviations.get(norm, norm)

    def sanitize_filename(self, name: str) -> str:
        """Remove spaces and special characters from filename."""
        name = name.replace(' ', '_')
        name = name.replace('-', '_')
        name = name.replace('.', '_')
        name = name.replace('(', '')
        name = name.replace(')', '')
        name = name.replace(',', '')
        while '__' in name:
            name = name.replace('__', '_')
        return name.lower()

    def _date_from_str(self, date_str: str) -> Optional[datetime.date]:
        """Parse YYYY-MM-DD date strings with caching."""
        if date_str in self._date_cache:
            return self._date_cache[date_str]
        try:
            parsed = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            parsed = None
        self._date_cache[date_str] = parsed
        return parsed

    def _date_range_key(self, start_date: Optional[str], end_date: Optional[str]) -> Optional[str]:
        """Build a stable date key that includes both start and end dates."""
        if not start_date:
            return None
        if not end_date or end_date == start_date:
            return start_date
        return f"{start_date}_to_{end_date}"

    def _date_key_sort_key(self, date_key: str) -> datetime.date:
        """Sort by start date for range keys."""
        start_date = self.date_key_map.get(date_key, (date_key, None))[0]
        parsed = self._date_from_str(start_date)
        return parsed if parsed else datetime.date.min

    def _date_from_stem(self, stem: str) -> Optional[datetime.date]:
        """Best-effort parse of a date from postprocess output stems."""
        if not stem:
            return None
        if stem in self.date_key_map:
            start_date = self.date_key_map.get(stem, (None, None))[0]
            if start_date:
                return self._date_from_str(start_date)

        # Handles stems like YYYY-MM-DD and YYYY-MM-DD_to_YYYY-MM-DD.
        match = re.search(r"\d{4}-\d{2}-\d{2}", stem)
        if not match:
            return None
        return self._date_from_str(match.group(0))

    def _postprocess_sort_key(self, stem: str) -> Tuple[int, int, str]:
        """
        Sort postprocess jobs newest-to-oldest, starting from 2025 and going backward.
        Dates after 2025 are kept, but processed after the <=2025 backlog.
        """
        date_obj = self._date_from_stem(stem)
        if date_obj:
            if date_obj.year <= 2025:
                return (0, -date_obj.toordinal(), stem)
            return (1, -date_obj.toordinal(), stem)
        return (2, 0, stem)

    def _run_postprocess_upload_stage(
        self,
        input_dir: Optional[Path] = None,
        input_jobs: Optional[List[Tuple[Path, str]]] = None
    ) -> None:
        """Convert mosaic rasters to MBTiles/PMTiles and upload PMTiles."""
        output_dir = self.upload_root
        output_dir.mkdir(parents=True, exist_ok=True)

        discovered_jobs: List[Dict[str, Union[Path, str]]] = []
        skipped_up_to_date = 0

        def _add_job(input_path: Path, output_stem: Optional[str] = None) -> None:
            nonlocal skipped_up_to_date
            if not input_path.exists():
                return
            min_size = 128 if input_path.suffix.lower() == ".vrt" else 1024 * 1024
            if input_path.stat().st_size <= min_size:
                return
            stem = (str(output_stem).strip() if output_stem else input_path.stem).strip()
            if not stem:
                return
            pmtiles_path = output_dir / f"{stem}.pmtiles"
            if getattr(self, 'skip_existing_postprocess', True):
                input_mtime = input_path.stat().st_mtime
                if pmtiles_path.exists() and pmtiles_path.stat().st_size > 0 and pmtiles_path.stat().st_mtime >= input_mtime:
                    skipped_up_to_date += 1
                    return
            discovered_jobs.append({
                "input": input_path,
                "stem": stem
            })

        if input_jobs:
            for entry in input_jobs:
                if not entry:
                    continue
                input_path = Path(entry[0])
                output_stem = entry[1] if len(entry) > 1 else None
                _add_job(input_path, output_stem)
        else:
            scan_dir = Path(input_dir) if input_dir else self.output_dir
            for tif in sorted(scan_dir.glob("*.tif")):
                _add_job(tif, tif.stem)
            for tif in sorted(scan_dir.glob("*.tiff")):
                _add_job(tif, tif.stem)

            for mosaic_vrt in sorted(self.temp_dir.glob("*/mosaic_*.vrt")):
                stem = mosaic_vrt.stem
                if stem.startswith("mosaic_"):
                    stem = stem[len("mosaic_"):]
                _add_job(mosaic_vrt, stem)

        if not discovered_jobs:
            scan_desc = str(input_dir) if input_dir else str(self.output_dir)
            if skipped_up_to_date > 0:
                self.log(f"  ✓ Skipping {skipped_up_to_date} jobs with up-to-date local PMTiles")
            self.log(f"  ⊘ No postprocess inputs found (checked {scan_desc} and {self.temp_dir})")
            return

        jobs_by_stem: Dict[str, Dict[str, Union[Path, str]]] = {}
        for job in discovered_jobs:
            stem = str(job["stem"])
            existing = jobs_by_stem.get(stem)
            if not existing:
                jobs_by_stem[stem] = job
                continue
            current_input = Path(job["input"])
            existing_input = Path(existing["input"])
            if current_input.stat().st_mtime >= existing_input.stat().st_mtime:
                jobs_by_stem[stem] = job

        ordered_stems = sorted(jobs_by_stem.keys(), key=self._postprocess_sort_key)
        jobs: List[Dict[str, Union[Path, str]]] = [jobs_by_stem[k] for k in ordered_stems]
        if skipped_up_to_date > 0:
            self.log(f"  ✓ Skipping {skipped_up_to_date} jobs with up-to-date local PMTiles")

        worker_upper_bound = max(1, min(self.upload_jobs, len(jobs)))
        adaptive_workers = worker_upper_bound
        post_threads = 1
        post_cache_mb = 512
        quiet_tools = bool(getattr(self, 'quiet_external_tools', True))
        throughput_history = deque(maxlen=8)
        recent_failures = deque(maxlen=6)
        completions = 0
        last_tune_completion = -99
        next_job_index = 0

        def _recompute_postprocess_settings() -> None:
            nonlocal post_threads, post_cache_mb
            post_threads, post_cache_mb = _auto_postprocess_settings(
                worker_count=max(1, adaptive_workers),
                requested_threads=getattr(self, 'postprocess_threads', "AUTO"),
                requested_cache_mb=getattr(self, 'postprocess_cache_mb', None)
            )

        def _postprocess_memory_cap(cache_mb: int) -> Tuple[int, Dict[str, int], int, int]:
            mem_snapshot = _get_memory_snapshot_mb(
                default_available_mb=8192 if IS_APPLE_SILICON else 4096,
                default_total_mb=16384 if IS_APPLE_SILICON else 8192
            )
            reserve_mb = _recommended_free_ram_floor_mb(mem_snapshot["total_mb"])
            est_worker_mb = max(640, int(cache_mb or 0) + 192)
            budget_mb = max(0, mem_snapshot["available_mb"] - reserve_mb)
            cap = max(1, min(worker_upper_bound, (budget_mb // est_worker_mb) if budget_mb > 0 else 1))
            return cap, mem_snapshot, reserve_mb, est_worker_mb

        def _maybe_tune_postprocess(force: bool = False, hint: str = "") -> None:
            nonlocal adaptive_workers, post_threads, post_cache_mb, last_tune_completion
            if not force and (completions - last_tune_completion) < 4:
                return

            previous_workers = adaptive_workers
            reason = hint or "adaptive"

            memory_cap, mem_snapshot, reserve_mb, est_worker_mb = _postprocess_memory_cap(post_cache_mb)
            if adaptive_workers > memory_cap:
                adaptive_workers = memory_cap
                reason = f"memory pressure ({mem_snapshot['available_mb']}MB free)"

            queue_remaining = total_jobs - next_job_index
            if adaptive_workers > 1 and sum(recent_failures) >= 2:
                adaptive_workers = max(1, adaptive_workers - 1)
                reason = "recent worker failures"

            if adaptive_workers > 1 and len(throughput_history) >= 4 and queue_remaining > 0:
                recent_window = list(throughput_history)[-3:]
                recent_rate = sum(recent_window) / len(recent_window)
                baseline_rate = sum(throughput_history) / len(throughput_history)
                if recent_rate < (baseline_rate * 0.72):
                    adaptive_workers = max(1, adaptive_workers - 1)
                    reason = f"throughput drop ({recent_rate:.2f}MB/s)"

            if (
                adaptive_workers < worker_upper_bound
                and queue_remaining > adaptive_workers
                and len(throughput_history) >= 4
                and sum(recent_failures) == 0
            ):
                recent_window = list(throughput_history)[-3:]
                recent_rate = sum(recent_window) / len(recent_window)
                baseline_rate = sum(throughput_history) / len(throughput_history)
                if recent_rate >= (baseline_rate * 0.95):
                    trial_workers = adaptive_workers + 1
                    trial_threads, trial_cache = _auto_postprocess_settings(
                        worker_count=trial_workers,
                        requested_threads=getattr(self, 'postprocess_threads', "AUTO"),
                        requested_cache_mb=getattr(self, 'postprocess_cache_mb', None)
                    )
                    trial_cap, trial_mem_snapshot, _, _ = _postprocess_memory_cap(trial_cache)
                    if trial_workers <= trial_cap:
                        adaptive_workers = trial_workers
                        mem_snapshot = trial_mem_snapshot
                        post_threads = trial_threads
                        post_cache_mb = trial_cache
                        reason = f"throughput stable ({recent_rate:.2f}MB/s)"

            if adaptive_workers != previous_workers:
                post_threads, post_cache_mb = _auto_postprocess_settings(
                    worker_count=max(1, adaptive_workers),
                    requested_threads=getattr(self, 'postprocess_threads', "AUTO"),
                    requested_cache_mb=getattr(self, 'postprocess_cache_mb', None)
                )
                self.log(
                    f"  ↺ Autotune postprocess: workers {previous_workers}->{adaptive_workers}, "
                    f"threads/worker={post_threads}, cache/worker={post_cache_mb}MB "
                    f"({reason}; free={mem_snapshot['available_mb']}MB, floor={reserve_mb}MB, est/worker={est_worker_mb}MB)"
                )
                last_tune_completion = completions
            elif force and last_tune_completion < 0:
                last_tune_completion = completions

        _recompute_postprocess_settings()

        self.log(f"\n=== Post-Processing: Raster → MBTiles → Overviews → PMTiles → Upload ===")
        self.log(f"Output: {output_dir}")
        self.log(f"Remote: {self.upload_remote}")
        self.log(f"Workers upper bound: {worker_upper_bound}")
        self.log(f"Postprocess GDAL threads/worker: {post_threads}")
        self.log(f"Postprocess GDAL cache/worker: {post_cache_mb}MB")
        self.log(f"External tool output: {'quiet' if quiet_tools else 'verbose'}")

        successes = 0
        failures = 0
        live_status_enabled = bool(sys.stdout.isatty())
        heartbeat_sec = 3 if live_status_enabled else 60
        total_jobs = len(jobs)
        self.log(f"Jobs to process: {total_jobs}")

        live_line_len = 0
        live_line_active = False
        row_range_cache: Dict[Tuple[str, int], Optional[Tuple[int, int]]] = {}

        def _clear_live_status_line() -> None:
            nonlocal live_line_len, live_line_active
            if not live_line_active:
                return
            print()
            live_line_len = 0
            live_line_active = False

        def _emit_live_status_line(message: str) -> None:
            nonlocal live_line_len, live_line_active
            if not live_status_enabled:
                self.log(message)
                return
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            line = f"[{ts}] {message}"
            pad = max(0, live_line_len - len(line))
            print("\r" + line + (" " * pad), end="", flush=True)
            live_line_len = len(line)
            live_line_active = True

        def _mercator_y_to_lat(y_merc: float) -> float:
            return math.degrees(math.atan(math.sinh(float(y_merc) / 6378137.0)))

        def _lat_to_xyz_row(lat_deg: float, zoom: int) -> int:
            lat_clamped = max(-85.05112878, min(85.05112878, float(lat_deg)))
            lat_rad = math.radians(lat_clamped)
            n = float(1 << int(zoom))
            y = (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) * 0.5 * n
            row = int(y)
            return max(0, min((1 << int(zoom)) - 1, row))

        def _zoom_row_range_for_input(input_raster: Path, zoom_level: int) -> Optional[Tuple[int, int]]:
            cache_key = (str(input_raster), int(zoom_level))
            if cache_key in row_range_cache:
                return row_range_cache[cache_key]
            try:
                ds = gdal.Open(str(input_raster))
                if not ds:
                    row_range_cache[cache_key] = None
                    return None
                gt = ds.GetGeoTransform(can_return_null=True)
                if gt is None:
                    ds = None
                    row_range_cache[cache_key] = None
                    return None
                x_size = ds.RasterXSize
                y_size = ds.RasterYSize
                ds = None
                # VRT mosaics are expected in EPSG:3857 here; we only need vertical bounds for frontier percentage.
                y_top = float(gt[3])
                y_bottom = float(gt[3]) + (float(gt[4]) * float(x_size)) + (float(gt[5]) * float(y_size))
                y_max = max(y_top, y_bottom)
                y_min = min(y_top, y_bottom)
                lat_n = _mercator_y_to_lat(y_max)
                lat_s = _mercator_y_to_lat(y_min)
                row_n = _lat_to_xyz_row(lat_n, int(zoom_level))
                row_s = _lat_to_xyz_row(lat_s, int(zoom_level))
                if row_s < row_n:
                    row_n, row_s = row_s, row_n
                row_range_cache[cache_key] = (row_n, row_s)
                return row_range_cache[cache_key]
            except Exception:
                row_range_cache[cache_key] = None
                return None

        def _read_partial_frontier(partial_db_path: Path) -> Optional[Tuple[int, int, int, int, int]]:
            # Returns (min_zoom, max_zoom, min_row, max_row, count)
            if not partial_db_path.exists():
                return None
            conn = None
            try:
                conn = sqlite3.connect(
                    # nolock=1 avoids taking shared locks that can interfere with GDAL's writer.
                    f"file:{partial_db_path}?mode=ro&nolock=1",
                    uri=True,
                    timeout=0.2
                )
                conn.execute("PRAGMA query_only=ON")
                conn.execute("PRAGMA read_uncommitted=1")
                cur = conn.cursor()
                cur.execute(
                    "SELECT min(zoom_level), max(zoom_level), min(tile_row), max(tile_row), count(*) "
                    "FROM partial_tiles WHERE zoom_level >= 0 AND tile_row >= 0"
                )
                row = cur.fetchone()
                if not row:
                    return None
                min_zoom, max_zoom, min_row, max_row, count = row
                if max_zoom is None or max_row is None:
                    return None
                return (
                    int(min_zoom if min_zoom is not None else max_zoom),
                    int(max_zoom),
                    int(min_row if min_row is not None else max_row),
                    int(max_row),
                    int(count if count is not None else 0),
                )
            except Exception:
                return None
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass

        def _worker_status_segment(info: Dict, now_ts: float) -> str:
            idx = int(info.get('idx') or 0)
            input_path = Path(info.get('input_path'))
            stem = str(info.get('stem') or input_path.stem)
            elapsed = int(max(0.0, now_ts - float(info.get('start') or now_ts)))

            mbtiles_path = output_dir / f"{stem}.mbtiles"
            pmtiles_path = output_dir / f"{stem}.pmtiles"
            partial_db_path = output_dir / f"{stem}.partial_tiles.db"

            display_name = input_path.name
            if display_name.startswith("mosaic_"):
                display_name = display_name[len("mosaic_"):]
            if display_name.endswith(".vrt"):
                display_name = display_name[:-4]

            if partial_db_path.exists():
                frontier = _read_partial_frontier(partial_db_path)
                if frontier:
                    _min_z, max_z, min_row, max_row, _count = frontier
                    row_range = _zoom_row_range_for_input(input_path, int(max_z))
                    if row_range:
                        row_n, row_s = row_range
                        denom = max(1, (row_s - row_n + 1))
                        pct = ((max_row - row_n + 1) / denom) * 100.0
                        pct = max(0.0, min(100.0, pct))
                        return (
                            f"[{idx}/{total_jobs}] {display_name} "
                            f"translate {pct:.1f}% z{max_z} frontier y{min_row}->{max_row}/{row_s} ({elapsed}s)"
                        )
                    return (
                        f"[{idx}/{total_jobs}] {display_name} "
                        f"translate z{max_z} frontier y{min_row}->{max_row} ({elapsed}s)"
                    )
                return f"[{idx}/{total_jobs}] {display_name} translate (warming up) ({elapsed}s)"

            if pmtiles_path.exists():
                return f"[{idx}/{total_jobs}] {display_name} upload/finish ({elapsed}s)"
            if mbtiles_path.exists():
                return f"[{idx}/{total_jobs}] {display_name} gdaladdo/convert ({elapsed}s)"
            return f"[{idx}/{total_jobs}] {display_name} starting ({elapsed}s)"

        _maybe_tune_postprocess(force=True, hint="startup")
        self.log(f"Adaptive workers at start: {adaptive_workers}")

        with ThreadPoolExecutor(max_workers=worker_upper_bound) as executor:
            futures = {}
            next_job_index = 0

            def submit_next() -> bool:
                nonlocal next_job_index
                if next_job_index >= total_jobs:
                    return False

                job = jobs[next_job_index]
                input_path = Path(job["input"])
                output_stem = str(job["stem"])
                idx = next_job_index + 1
                next_job_index += 1
                input_size_mb = input_path.stat().st_size / (1024 * 1024) if input_path.exists() else 0
                self.log(
                    f"  → Starting [{idx}/{total_jobs}]: {input_path.name} ({input_size_mb:.1f} MB) "
                    f"[workers={adaptive_workers}, threads={post_threads}, cache={post_cache_mb}MB]"
                )
                future = executor.submit(
                    postprocess_tif_worker,
                    (
                        input_path,
                        output_dir,
                        self.upload_remote,
                        self.rclone_flags,
                        self.delete_geotiff_after_mbtiles,
                        self.mbtiles_quality,
                        post_threads,
                        post_cache_mb,
                        quiet_tools,
                        output_stem,
                        getattr(self, 'clip_projwin', None)
                    )
                )
                futures[future] = {
                    'input_path': input_path,
                    'idx': idx,
                    'start': time.time(),
                    'input_size_mb': input_size_mb,
                    'stem': output_stem
                }
                return True

            for _ in range(min(adaptive_workers, total_jobs)):
                submit_next()

            while futures:
                done, not_done = wait(futures, timeout=heartbeat_sec, return_when=FIRST_COMPLETED)
                if not done:
                    now = time.time()
                    queued = total_jobs - next_job_index
                    segments = []
                    for fut in sorted(not_done, key=lambda x: futures[x]['idx']):
                        segments.append(_worker_status_segment(futures[fut], now))
                    _emit_live_status_line(
                        f"  ⏳ Active: {' | '.join(segments)} | queued: {queued} | target workers: {adaptive_workers}"
                    )
                    continue

                for future in done:
                    _clear_live_status_line()
                    info = futures.pop(future)
                    input_path = info['input_path']
                    idx = info['idx']
                    total_elapsed = time.time() - info['start']

                    try:
                        result = future.result()
                    except Exception as e:
                        failures += 1
                        completions += 1
                        recent_failures.append(1)
                        self.log(f"  ✗ Post-process failed [{idx}/{total_jobs}] {input_path.name}: {e}")
                        _maybe_tune_postprocess(hint="worker exception")
                        while len(futures) < adaptive_workers and submit_next():
                            pass
                        continue

                    stage_times = result.get('stage_times') or {}
                    stage_summary = ", ".join(f"{k}={v:.1f}s" for k, v in stage_times.items())
                    if stage_summary:
                        stage_summary = f" [{stage_summary}]"

                    if result.get('success'):
                        successes += 1
                        msg = (
                            f"  ✓ Uploaded [{idx}/{total_jobs}]: "
                            f"{Path(result.get('pmtiles', '')).name} ({total_elapsed:.1f}s){stage_summary}"
                        )
                        if result.get('deleted_input'):
                            msg += " (deleted input TIFF)"
                        elif result.get('delete_error'):
                            msg += f" (delete failed: {result.get('delete_error')})"
                        self.log(msg)
                    else:
                        failures += 1
                        recent_failures.append(1)
                        self.log(
                            f"  ✗ Failed [{idx}/{total_jobs}]: {input_path.name} ({total_elapsed:.1f}s){stage_summary} - "
                            f"{result.get('error', 'unknown error')}"
                        )

                    if result.get('success'):
                        recent_failures.append(0)

                    output_size_mb = 0.0
                    pmtiles_path = result.get('pmtiles')
                    if pmtiles_path:
                        try:
                            output_size_mb = Path(pmtiles_path).stat().st_size / (1024 * 1024)
                        except Exception:
                            output_size_mb = 0.0
                    if output_size_mb <= 0:
                        output_size_mb = float(info.get('input_size_mb') or 0.0)
                    if output_size_mb > 0 and total_elapsed > 0:
                        throughput_history.append(output_size_mb / total_elapsed)

                    completions += 1
                    _maybe_tune_postprocess()

                    while len(futures) < adaptive_workers and submit_next():
                        pass

        _clear_live_status_line()
        self.log(f"Post-processing complete: {successes} succeeded, {failures} failed")

    def _postprocess_single_tiff(self, tiff_path: Path) -> bool:
        """Convert a single raster input to MBTiles/PMTiles and upload."""
        post_threads, post_cache_mb = _auto_postprocess_settings(
            worker_count=1,
            requested_threads=getattr(self, 'postprocess_threads', "AUTO"),
            requested_cache_mb=getattr(self, 'postprocess_cache_mb', None)
        )
        result = postprocess_tif_worker(
            (
                tiff_path,
                self.upload_root,
                self.upload_remote,
                self.rclone_flags,
                self.delete_geotiff_after_mbtiles,
                self.mbtiles_quality,
                post_threads,
                post_cache_mb,
                bool(getattr(self, 'quiet_external_tools', True)),
                tiff_path.stem,
                getattr(self, 'clip_projwin', None)
            )
        )
        if result.get('success'):
            msg = f"  ✓ Uploaded: {Path(result.get('pmtiles', '')).name}"
            if result.get('deleted_input'):
                msg += " (deleted input TIFF)"
            elif result.get('delete_error'):
                msg += f" (delete failed: {result.get('delete_error')})"
            self.log(msg)
            return True

        self.log(f"  ✗ Post-process failed for {tiff_path.name}: {result.get('error', 'unknown error')}")
        return False


    def load_dole_data(self):
        """Load CSV data."""
        self.log(f"Loading CSV: {self.csv_file.name}...")
        self.dole_data = defaultdict(list)
        self.date_key_map = {}

        with open(self.csv_file, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                start_date = row.get('date')
                end_date = row.get('end_date') or start_date
                if not start_date:
                    continue

                # Support newer CSV schema where filename is stored as tif_filename
                if not row.get('filename') and row.get('tif_filename'):
                    row['filename'] = row.get('tif_filename')

                link = row.get('download_link', '')
                timestamp = None
                if 'web.archive.org/web/' in link:
                    match = re.search(r'/web/(\d{14})/', link)
                    if match:
                        timestamp = match.group(1)
                row['wayback_ts'] = timestamp
                row['start_date'] = start_date
                row['end_date'] = end_date
                date_key = self._date_range_key(start_date, end_date)
                if not date_key:
                    continue
                row['date_key'] = date_key
                self.dole_data[date_key].append(row)
                if date_key not in self.date_key_map:
                    self.date_key_map[date_key] = (start_date, end_date)

        self.log(f"Loaded {len(self.dole_data)} date ranges.")

    def _build_source_index(self):
        """
        Build file index for O(1) lookups instead of repeated os.walk calls.

        Creates:
        - files_by_name: Dict[str, Path] for direct TIFs and ZIPs
        - tifs_by_dir: Dict[str, List[Path]] keyed by extracted folder (ZIP stem)
        """
        self.log("Building source file index...")

        # Walk source_dir once and cache all files
        for root, dirs, files in os.walk(self.source_dir):
            root_path = Path(root)

            for filename in files:
                file_path = root_path / filename

                # Cache direct TIF and ZIP files by name
                if filename.endswith('.tif') or filename.endswith('.zip'):
                    self.files_by_name[filename] = file_path

                # For TIFs inside extracted directories, also index by parent directory name
                if filename.endswith('.tif'):
                    parent_dir = root_path.name
                    # Only index if parent is not source_dir itself
                    if root_path != self.source_dir:
                        if parent_dir not in self.tifs_by_dir:
                            self.tifs_by_dir[parent_dir] = []
                        self.tifs_by_dir[parent_dir].append(file_path)

        self.log(f"  Indexed {len(self.files_by_name)} files and {len(self.tifs_by_dir)} directories")

    def _build_shapefile_index(self):
        """
        Build shapefile index for O(1) lookups instead of repeated globbing.

        Creates shapefile_index: Dict[str, Path] keyed by normalized location name.
        """
        self.log("Building shapefile index...")

        sectional_dir = self.shape_dir / "sectional"
        if not sectional_dir.exists():
            self.log(f"  WARNING: Sectional shapefile directory not found at {sectional_dir}")
            return

        # One controlled glob to find all shapefiles
        candidates = list(sectional_dir.glob("**/*.shp"))

        for shp in candidates:
            # Normalize the shapefile name once
            shp_norm = self.normalize_name(shp.stem)
            # Store with normalized key
            self.shapefile_index[shp_norm] = shp

            # Also check for partial matches (for locations that might be substrings)
            # We'll do exact matches first in find_shapefile, then partial matches

        self.log(f"  Indexed {len(self.shapefile_index)} shapefiles")

    def download_file(self, url: str, target_path: Path, timeout: int = 300, max_retries: int = 3) -> bool:
        """
        Download a file from URL to target path with throttling and retry logic.

        Implements:
        - Throttling to avoid rate limits (configurable download_delay)
        - Exponential backoff retry logic on failure
        - Progress reporting

        Returns: True if successful, False otherwise
        """
        if requests is None:
            self.log("ERROR: requests is required for downloading files. Install with: pip install requests")
            return False

        # Throttle downloads to avoid rate limiting
        elapsed = time.time() - self.last_download_time
        if elapsed < self.download_delay:
            wait_time = self.download_delay - elapsed
            self.log(f"    Waiting {wait_time:.1f}s to avoid rate limiting...")
            time.sleep(wait_time)

        retry_count = 0
        backoff_delay = 5  # Start with 5 second backoff

        while retry_count < max_retries:
            try:
                self.log(f"    Downloading: {url}")
                response = requests.get(url, timeout=timeout, stream=True)
                response.raise_for_status()

                # Get total file size
                total_size = int(response.headers.get('content-length', 0))

                # Download with progress
                downloaded = 0
                last_logged_percent = -10  # Track last logged percentage
                with open(target_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                current_percent = int((downloaded / total_size) * 100)
                                if current_percent >= last_logged_percent + 10:
                                    self.log(f"      Progress: {current_percent}%")
                                    last_logged_percent = current_percent

                file_size_mb = target_path.stat().st_size / (1024 * 1024)
                self.log(f"    ✓ Downloaded: {target_path.name} ({file_size_mb:.1f} MB)")

                # Update file index with newly downloaded file
                self.files_by_name[target_path.name] = target_path

                # Update last download time for throttling
                self.last_download_time = time.time()
                return True

            except Exception as e:
                retry_count += 1
                error_str = str(e)
                self.log(f"    ✗ Download failed (attempt {retry_count}/{max_retries}): {error_str[:100]}")

                # Detect connection reset errors - server is rejecting us, slow down aggressively
                if 'Connection reset' in error_str or 'Connection aborted' in error_str:
                    # Increase global download delay to back off more for future requests
                    old_delay = self.download_delay
                    self.download_delay = max(self.download_delay * 1.5, self.download_delay + 5)
                    self.log(f"    ⚠ Connection error detected - increasing delay from {old_delay:.1f}s to {self.download_delay:.1f}s")

                if retry_count < max_retries:
                    self.log(f"    Retrying in {backoff_delay} seconds...")
                    time.sleep(backoff_delay)
                    backoff_delay *= 2  # Exponential backoff
                    # Remove partial file before retry
                    if target_path.exists():
                        target_path.unlink()
                else:
                    self.log(f"    Giving up after {max_retries} attempts")
                    if target_path.exists():
                        target_path.unlink()
                    return False

        return False

    def extract_zip(self, zip_path: Path, extract_dir: Path) -> bool:
        """
        Extract ZIP file to a directory with the same name (minus .zip).

        Returns: True if successful, False otherwise
        """
        try:
            if extract_dir.exists():
                self.log(f"    Directory already exists: {extract_dir.name}")
                return True

            self.log(f"    Extracting: {zip_path.name}")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            # Count extracted files and update index
            tif_files = list(extract_dir.glob('**/*.tif'))
            tif_count = len(tif_files)

            # Update tifs_by_dir index with newly extracted files
            dir_name = extract_dir.name
            self.tifs_by_dir[dir_name] = tif_files

            # Also add individual TIF files to files_by_name
            for tif_file in tif_files:
                self.files_by_name[tif_file.name] = tif_file

            self.log(f"    ✓ Extracted {tif_count} files to: {extract_dir.name}")
            return True

        except Exception as e:
            self.log(f"    ✗ Extraction failed: {e}")
            if extract_dir.exists():
                import shutil
                shutil.rmtree(extract_dir)
            return False

    def _download_missing_files(self, allowed_dates: Optional[set] = None):
        """
        Preparation phase: Download all missing files upfront before processing begins.

        This collects all files referenced in self.dole_data that don't exist locally,
        then downloads them sequentially with proper throttling. This prevents burst
        requests to web.archive.org when processing dates with many missing files.

        Respects self.download_delay and self.preview_mode settings.
        """
        # Collect all files that need downloading
        missing_files = []  # List of (filename, download_link, location, date, is_zip)

        for date in self.dole_data:
            if allowed_dates is not None and date not in allowed_dates:
                continue
            for record in self.dole_data[date]:
                filename = record.get('filename', '')
                download_link = record.get('download_link', '')
                location = record.get('location', '')

                if not filename or not download_link:
                    continue

                # Check if file exists locally
                if filename.endswith('.zip'):
                    # ZIP files become directories after extraction
                    dir_name = filename[:-4]
                    dir_path = self.source_dir / dir_name
                    if not dir_path.exists():
                        missing_files.append((filename, download_link, location, date, True))
                else:
                    # Direct TIF files
                    file_path = self.source_dir / filename
                    if not file_path.exists():
                        missing_files.append((filename, download_link, location, date, False))

        if not missing_files:
            self.log("No missing files to download.")
            return

        # Sort by date (latest first to match processing order)
        missing_files.sort(key=lambda x: self._date_key_sort_key(x[3]), reverse=True)

        self.log(f"Found {len(missing_files)} missing files to download (sorted latest→oldest)")
        self.log(f"Download throttle: {self.download_delay}s between requests")

        # Estimate total time
        estimated_secs = len(missing_files) * self.download_delay
        estimated_mins = estimated_secs / 60
        self.log(f"Estimated time: ~{estimated_mins:.0f} minutes (assuming {self.download_delay}s per file)\n")

        # Download each file
        downloaded_count = 0
        failed_count = 0

        for idx, (filename, download_link, location, date, is_zip) in enumerate(missing_files, 1):
            self.log(f"  [{idx}/{len(missing_files)}] {date} - {location}")

            target_path = self.source_dir / filename

            # Download the file
            if self.download_file(download_link, target_path):
                downloaded_count += 1

                # If ZIP, extract it
                if is_zip:
                    dir_name = filename[:-4]
                    extract_dir = self.source_dir / dir_name
                    if self.extract_zip(target_path, extract_dir):
                        self.log(f"    ✓ Downloaded and extracted: {filename}")
                    else:
                        self.log(f"    ⚠ Downloaded but extraction failed: {filename}")
                        failed_count += 1
                else:
                    self.log(f"    ✓ Downloaded: {filename}")

                # Add extra pause after successful download to avoid rate limits
                # This is in addition to the throttling in download_file()
                if idx < len(missing_files):  # Don't pause after last file
                    self.log(f"    Pausing {self.download_delay}s before next request...")
                    time.sleep(self.download_delay)
            else:
                self.log(f"    ✗ Failed to download: {filename}")
                failed_count += 1
                # Still pause before retry to avoid hammering the server
                if idx < len(missing_files):
                    time.sleep(self.download_delay)

        # Summary
        self.log(f"\n=== Download Preparation Complete ===")
        self.log(f"Downloaded: {downloaded_count}/{len(missing_files)}")
        if failed_count > 0:
            self.log(f"Failed: {failed_count}")
        self.log(f"Processing will now use these files as they're available locally.\n")

    def resolve_filename(self, filename: str, download_link: str = '') -> List[Tuple[Path, Optional[str]]]:
        """
        Resolve filename from CSV to actual file path(s) using index cache.

        Supports:
        1. Direct .tif files: "file.tif" → finds file.tif
        2. Unzipped .zip directories: "file.zip" → finds all .tif files in file/ directory
        3. Auto-download if not found locally (when download_link provided)

        Returns: List of (file_path, None) tuples
        """
        if not filename:
            return []

        results = []

        # Case 1: .zip file (after unzipping, this becomes a directory)
        if filename.endswith('.zip'):
            # Remove .zip extension to get directory name
            dir_name = filename[:-4]  # Remove '.zip'

            # Check index FIRST (O(1) lookup)
            if dir_name in self.tifs_by_dir:
                # Found in index - return cached list
                for tif_file in self.tifs_by_dir[dir_name]:
                    results.append((tif_file, None))
                return results

            # Not in index - check if directory exists but wasn't indexed
            direct_path = self.source_dir / dir_name
            if direct_path.exists() and direct_path.is_dir():
                tif_files = list(direct_path.glob('**/*.tif'))
                # Update index for future lookups
                self.tifs_by_dir[dir_name] = tif_files
                for tif_file in tif_files:
                    if tif_file.is_file():
                        results.append((tif_file, None))
                if results:
                    return results

            # If not found, try to download (but not in preview mode)
            if not results and download_link and not self.preview_mode:
                self.log(f"    File not found locally: {filename}")
                zip_path = self.source_dir / filename
                extract_dir = self.source_dir / dir_name

                # Download the ZIP file
                if self.download_file(download_link, zip_path):
                    # Extract it (this will update the index)
                    if self.extract_zip(zip_path, extract_dir):
                        # Get the updated index entry
                        if dir_name in self.tifs_by_dir:
                            for tif_file in self.tifs_by_dir[dir_name]:
                                results.append((tif_file, None))
                        return results
                    else:
                        self.log(f"    Failed to extract: {filename}")
                else:
                    self.log(f"    Failed to download: {download_link}")

        else:
            # Case 2: Direct .tif file
            # Check index FIRST (O(1) lookup)
            if filename in self.files_by_name:
                results.append((self.files_by_name[filename], None))
                return results

            # Not in index - check if file exists but wasn't indexed
            direct_path = self.source_dir / filename
            if direct_path.exists() and direct_path.is_file():
                # Update index for future lookups
                self.files_by_name[filename] = direct_path
                results.append((direct_path, None))
                return results

            # If not found, try to download (but not in preview mode)
            if not results and download_link and not self.preview_mode:
                self.log(f"    File not found locally: {filename}")
                tif_path = self.source_dir / filename

                if self.download_file(download_link, tif_path):
                    results.append((tif_path, None))
                else:
                    self.log(f"    Failed to download: {download_link}")

        return results

    def find_files(self, location: str, edition: str, timestamp: Optional[str], filename: Optional[str] = None, download_link: str = '') -> List[Union[Path, Tuple[Path, str]]]:
        """
        Find files from CSV filename.
        Filename must be provided in master_dole.csv.

        If file not found locally and download_link provided, will attempt download and extract.

        Returns:
            - List of (file_path, None) tuples for files
        """
        if not filename:
            return []

        resolved = self.resolve_filename(filename, download_link)
        return resolved

    def find_shapefile(self, location: str) -> Optional[Path]:
        """Find shapefile for location using index cache (O(1) lookup)."""
        norm_loc = self.normalize_name(location)

        # Check for exact match first (O(1))
        if norm_loc in self.shapefile_index:
            return self.shapefile_index[norm_loc]

        # Check for partial matches (substring search)
        for shp_norm, shp_path in self.shapefile_index.items():
            if norm_loc in shp_norm:
                return shp_path

        return None

    def get_shapefile_srs(self, shapefile: Path) -> str:
        """Get the actual SRS of a shapefile in Proj4 format. Results are cached."""
        # Check cache first
        shapefile_key = str(shapefile)
        if shapefile_key in self.srs_cache:
            return self.srs_cache[shapefile_key]

        try:
            driver = ogr.GetDriverByName("ESRI Shapefile")
            dataset = driver.Open(str(shapefile), 0)
            if not dataset:
                result = 'EPSG:4326'  # Fallback to WGS84
            else:
                layer = dataset.GetLayer(0)
                srs = layer.GetSpatialRef()
                dataset = None

                if srs:
                    proj4 = srs.ExportToProj4()
                    result = proj4 if proj4 else 'EPSG:4326'
                else:
                    result = 'EPSG:4326'  # Fallback
        except Exception:
            result = 'EPSG:4326'  # Fallback on any error

        # Cache the result
        self.srs_cache[shapefile_key] = result
        return result

    def _pick_shapefile_for_source(self, location: str, src_file: Path, default_shp: Optional[Path]) -> Optional[Path]:
        """
        Prefer directional shapefile variants (east/west/north/south) when the source filename
        encodes a directional split chart, e.g. Western Aleutian Islands East/West.
        """
        if default_shp is None:
            return None

        norm_loc = self.normalize_name(location)
        stem_norm = self.normalize_name(src_file.stem)
        for suffix in ("_east", "_west", "_north", "_south"):
            if suffix in stem_norm:
                variant_key = f"{norm_loc}{suffix}"
                variant = self.shapefile_index.get(variant_key)
                if variant:
                    return variant
        return default_shp

    def _antimeridian_split_bounds_3857(self, input_tiff: Path) -> Optional[List[Tuple[float, float, float, float]]]:
        """
        Return split output bounds (EPSG:3857) for sources that cross the antimeridian.
        When a source spans +180/-180 in geographic space, a single warp can balloon to nearly
        world width. Splitting into east/west output windows keeps extents narrow.
        """
        try:
            ds = gdal.Open(str(input_tiff))
            if not ds:
                return None

            gt = ds.GetGeoTransform(can_return_null=True)
            projection = ds.GetProjection()
            if gt is None or not projection:
                ds = None
                return None

            x_size = ds.RasterXSize
            y_size = ds.RasterYSize
            ds = None

            src_srs = osr.SpatialReference()
            if src_srs.ImportFromWkt(projection) != 0:
                return None
            geo_srs = osr.SpatialReference()
            if geo_srs.ImportFromEPSG(4326) != 0:
                return None
            merc_srs = osr.SpatialReference()
            if merc_srs.ImportFromEPSG(3857) != 0:
                return None

            if hasattr(src_srs, "SetAxisMappingStrategy") and hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
                src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                geo_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                merc_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

            to_geo = osr.CoordinateTransformation(src_srs, geo_srs)
            to_merc = osr.CoordinateTransformation(geo_srs, merc_srs)

            corners_px = [
                (0.0, 0.0),
                (float(x_size), 0.0),
                (float(x_size), float(y_size)),
                (0.0, float(y_size)),
            ]
            lon_lat: List[Tuple[float, float]] = []
            for px, py in corners_px:
                x = gt[0] + (px * gt[1]) + (py * gt[2])
                y = gt[3] + (px * gt[4]) + (py * gt[5])
                lon, lat, _ = to_geo.TransformPoint(float(x), float(y))
                while lon > 180.0:
                    lon -= 360.0
                while lon < -180.0:
                    lon += 360.0
                lon_lat.append((lon, lat))

            lons = [p[0] for p in lon_lat]
            lats = [p[1] for p in lon_lat]
            if not lons or not lats:
                return None

            # Detect dateline crossing by large longitudinal jump between neighboring corners.
            max_jump = 0.0
            for idx in range(len(lons)):
                prev = lons[idx - 1]
                curr = lons[idx]
                jump = abs(curr - prev)
                if jump > max_jump:
                    max_jump = jump
            if max_jump < 180.0:
                return None

            pos_lons = [lon for lon in lons if lon >= 0.0]
            neg_lons = [lon for lon in lons if lon < 0.0]
            if not pos_lons or not neg_lons:
                return None

            lat_margin = 0.25
            lon_margin = 0.25
            lat_min = max(-85.0, min(lats) - lat_margin)
            lat_max = min(85.0, max(lats) + lat_margin)
            east_lon_min = max(-180.0, min(pos_lons) - lon_margin)
            west_lon_max = min(180.0, max(neg_lons) + lon_margin)

            lon_windows = [
                (east_lon_min, 180.0),
                (-180.0, west_lon_max),
            ]

            split_bounds: List[Tuple[float, float, float, float]] = []
            for lon_min, lon_max in lon_windows:
                if lon_max <= lon_min:
                    continue
                min_x, min_y, _ = to_merc.TransformPoint(float(lon_min), float(lat_min))
                max_x, max_y, _ = to_merc.TransformPoint(float(lon_max), float(lat_max))
                bounds = (
                    min(min_x, max_x),
                    min(min_y, max_y),
                    max(min_x, max_x),
                    max(min_y, max_y),
                )
                if (bounds[2] - bounds[0]) > 1000 and (bounds[3] - bounds[1]) > 1000:
                    split_bounds.append(bounds)

            return split_bounds if split_bounds else None
        except Exception:
            return None

    def _is_world_width_warp(self, output_raster: Path) -> bool:
        """Detect pathological outputs that balloon to near full-world width in EPSG:3857."""
        try:
            ds = gdal.Open(str(output_raster))
            if not ds:
                return False
            gt = ds.GetGeoTransform(can_return_null=True)
            if gt is None:
                ds = None
                return False
            width_m = abs(float(gt[1])) * float(ds.RasterXSize)
            ds = None
            return width_m >= 35000000.0
        except Exception:
            return False

    def _prepare_warp_job(self, src_file_info: Union[Path, Tuple[Path, Optional[str]]], location: str, edition: str,
                          shp_path: Path, temp_dir: Path, date: str) -> List[Dict]:
        """
        Prepare warp job parameters for parallel processing.

        With pre-unzipped files, src_file_info is always a direct Path to a .tif file.
        No ZIP extraction needed.
        """
        norm_loc = self.normalize_name(location)
        jobs = []

        # Unpack the file info (should always be (Path, None) from unzipped structure)
        if isinstance(src_file_info, tuple):
            src_file, internal_file = src_file_info
        else:
            src_file = src_file_info
            internal_file = None

        # With pre-unzipped files, src_file is always a direct TIF path
        tiffs_to_warp = [src_file]

        # Decide intermediate format for warp output (TIFF default, VRT optional)
        warp_out = getattr(self, 'warp_output', 'tif').lower()
        suffix = '.vrt' if warp_out == 'vrt' else '.tif'

        # Create job entries for each TIFF
        for tiff in tiffs_to_warp:
            sanitized_stem = self.sanitize_filename(tiff.stem)
            temp_tif = temp_dir / f"{norm_loc}_{sanitized_stem}{suffix}"
            selected_shp = self._pick_shapefile_for_source(location, tiff, shp_path)
            jobs.append({
                'input_tiff': tiff,
                'shapefile': selected_shp,
                'output_tiff': temp_tif,
                'location': norm_loc,
                'date': date
            })

        return jobs

    def _warp_worker(self, job: Dict) -> Dict:
        """Worker function for parallel warping operations."""
        success = self.warp_and_cut(
            job['input_tiff'],
            job['shapefile'],
            job['output_tiff']
        )

        return {
            'success': success,
            'output_tiff': job['output_tiff'] if success else None,
            'location': job['location'],
            'date': job['date'],
            'attempt_id': job.get('attempt_id')
        }

    def warp_and_cut(self, input_tiff: Path, shapefile: Optional[Path], output_tiff: Path) -> bool:
        """Warp and cut TIFF using shapefile as cutline.

        If self.warp_output is set to "vrt", write a VRT instead of a TIFF
        to avoid the intermediate write/read of warped GeoTIFFs.
        """
        if not shapefile:
            self.log(f"      ✗ No shapefile available for {input_tiff.name}")
            return False

        try:
            warp_out = getattr(self, 'warp_output', 'tif').lower()
            to_vrt = warp_out == 'vrt'
            self.log(f"    Warping {input_tiff.name} to {'VRT' if to_vrt else 'TIFF'}...")

            # Step 1: Build an RGBA VRT source only for paletted inputs.
            # For non-paletted rasters, warp directly from source TIFF to skip an extra translate pass.
            sanitized_stem = self.sanitize_filename(input_tiff.stem)
            temp_vrt_name = None
            warp_source = str(input_tiff)

            src_ds = gdal.Open(str(input_tiff))
            if not src_ds:
                self.log(f"      ✗ Could not open {input_tiff.name}")
                return False
            first_band = src_ds.GetRasterBand(1)
            has_color_table = bool(first_band and first_band.GetColorTable())
            src_ds = None

            if has_color_table:
                # Keep source VRT on disk only when downstream output is VRT.
                if to_vrt:
                    temp_vrt_path = output_tiff.parent / f"{sanitized_stem}_src.vrt"
                    temp_vrt_name = str(temp_vrt_path)
                else:
                    temp_vrt_name = f"/vsimem/{sanitized_stem}_src.vrt"

                translate_options = gdal.TranslateOptions(format="VRT", rgbExpand="rgba")
                ds_vrt = gdal.Translate(temp_vrt_name, str(input_tiff), options=translate_options)
                if not ds_vrt:
                    self.log(f"      ✗ Failed to prepare RGBA source VRT for {input_tiff.name}")
                    return False
                ds_vrt = None
                warp_source = temp_vrt_name

            # Get the actual SRS of the shapefile (don't hardcode EPSG:4326)
            shapefile_srs = self.get_shapefile_srs(shapefile)
            split_bounds = self._antimeridian_split_bounds_3857(input_tiff)
            if split_bounds:
                self.log(f"      ↺ Antimeridian source detected; splitting warp into {len(split_bounds)} windows")

            # Step 2: Warp with cutline. Write GTiff (default) or VRT when requested
            # Note: Not specifying xRes/yRes allows GDAL to maintain source resolution during warp
            # Don't compress intermediates to reduce CPU load and improve parallel performance
            def _safe_unlink(path: Path) -> None:
                try:
                    if path.exists():
                        path.unlink()
                except Exception:
                    pass

            def _run_warp(
                transformer_opts,
                output_path: Path,
                output_bounds: Optional[Tuple[float, float, float, float]] = None,
                force_vrt_output: bool = False
            ) -> bool:
                output_format = "VRT" if (to_vrt or force_vrt_output) else "GTiff"
                crop_to_cutline = output_bounds is None
                if output_format == "GTiff":
                    creation_opts = ['TILED=YES', 'BIGTIFF=YES']
                else:
                    creation_opts = None
                _safe_unlink(output_path)
                warp_options = gdal.WarpOptions(
                    format=output_format,
                    dstSRS='EPSG:3857',
                    cutlineDSName=str(shapefile),
                    cutlineSRS=shapefile_srs,
                    cropToCutline=crop_to_cutline,
                    dstAlpha=True,
                    resampleAlg=getattr(self, 'resample_alg', gdal.GRA_Bilinear),
                    creationOptions=creation_opts,
                    multithread=getattr(self, 'warp_multithread', True),
                    transformerOptions=transformer_opts or None,
                    warpOptions=['CUTLINE_ALL_TOUCHED=TRUE'],
                    outputBounds=output_bounds
                )
                ds_out = gdal.Warp(str(output_path), warp_source, options=warp_options)
                if not ds_out:
                    return False
                ds_out = None
                return output_path.exists() and output_path.stat().st_size > 1024

            def _translate_vrt_to_tiff(src_vrt: Path, dst_tiff: Path) -> bool:
                _safe_unlink(dst_tiff)
                translate_opts = gdal.TranslateOptions(
                    format="GTiff",
                    creationOptions=['TILED=YES', 'BIGTIFF=YES']
                )
                ds_out = gdal.Translate(str(dst_tiff), str(src_vrt), options=translate_opts)
                if not ds_out:
                    return False
                ds_out = None
                return dst_tiff.exists() and dst_tiff.stat().st_size > 1024

            def _raster_pixel_area(path: Path) -> int:
                try:
                    ds_area = gdal.Open(str(path))
                    if not ds_area:
                        return 0
                    area = int(ds_area.RasterXSize) * int(ds_area.RasterYSize)
                    ds_area = None
                    return area
                except Exception:
                    return 0

            def _run_split_warp(transformer_opts, bounds_list: List[Tuple[float, float, float, float]]) -> bool:
                if not bounds_list:
                    return False

                part_vrts: List[Path] = []
                part_items: List[Tuple[Path, Tuple[float, float, float, float]]] = []
                merged_vrt: Optional[Path] = None
                try:
                    for idx, bounds in enumerate(bounds_list):
                        part_vrt = output_tiff.parent / f"{output_tiff.stem}_am_part{idx}.vrt"
                        if _run_warp(transformer_opts, part_vrt, output_bounds=bounds, force_vrt_output=True):
                            part_vrts.append(part_vrt)
                            part_items.append((part_vrt, bounds))
                        else:
                            _safe_unlink(part_vrt)

                    if not part_vrts:
                        return False

                    if len(part_vrts) == 1:
                        if to_vrt:
                            _safe_unlink(output_tiff)
                            part_vrts[0].replace(output_tiff)
                            part_vrts.clear()
                            return output_tiff.exists() and output_tiff.stat().st_size > 0
                        return _translate_vrt_to_tiff(part_vrts[0], output_tiff)

                    # If parts are on opposite sides of the WebMercator antimeridian,
                    # merging them into one VRT yields a full-world canvas. Pick one side.
                    has_pos = any(bounds[0] >= 0.0 for _, bounds in part_items)
                    has_neg = any(bounds[2] <= 0.0 for _, bounds in part_items)
                    if has_pos and has_neg:
                        stem_norm = self.normalize_name(input_tiff.stem)
                        preferred = None
                        if "_east" in stem_norm:
                            preferred = "neg"
                        elif "_west" in stem_norm:
                            preferred = "pos"

                        selected_item: Optional[Tuple[Path, Tuple[float, float, float, float]]] = None
                        if preferred == "neg":
                            neg_items = [item for item in part_items if item[1][2] <= 0.0]
                            if neg_items:
                                selected_item = max(neg_items, key=lambda item: _raster_pixel_area(item[0]))
                        elif preferred == "pos":
                            pos_items = [item for item in part_items if item[1][0] >= 0.0]
                            if pos_items:
                                selected_item = max(pos_items, key=lambda item: _raster_pixel_area(item[0]))

                        if selected_item is None:
                            selected_item = max(part_items, key=lambda item: _raster_pixel_area(item[0]))

                        selected_vrt = selected_item[0]
                        side = "west-hemisphere" if selected_item[1][2] <= 0.0 else "east-hemisphere"
                        self.log(f"      ↺ Antimeridian split spans both world edges; using {side} window for {input_tiff.name}")

                        if to_vrt:
                            _safe_unlink(output_tiff)
                            selected_vrt.replace(output_tiff)
                            part_vrts.remove(selected_vrt)
                            return output_tiff.exists() and output_tiff.stat().st_size > 0
                        return _translate_vrt_to_tiff(selected_vrt, output_tiff)

                    merged_vrt = output_tiff if to_vrt else (output_tiff.parent / f"{output_tiff.stem}_am_merge.vrt")
                    if (not to_vrt) and merged_vrt.exists():
                        _safe_unlink(merged_vrt)

                    merge_ds = gdal.BuildVRT(str(merged_vrt), [str(p) for p in part_vrts])
                    if not merge_ds:
                        return False
                    merge_ds = None

                    if to_vrt:
                        return merged_vrt.exists() and merged_vrt.stat().st_size > 0
                    return _translate_vrt_to_tiff(merged_vrt, output_tiff)
                finally:
                    for part in part_vrts:
                        _safe_unlink(part)
                    if merged_vrt and (not to_vrt):
                        _safe_unlink(merged_vrt)

            def _run_once(transformer_opts) -> bool:
                if split_bounds:
                    return _run_split_warp(transformer_opts, split_bounds)
                return _run_warp(transformer_opts, output_tiff)

            transformer_opts = getattr(self, 'transformer_options', None)
            success = False
            try:
                success = _run_once(transformer_opts)
            except Exception as e:
                if transformer_opts:
                    self.log(f"      ⚠ Warp failed with strict transformer options; retrying with relaxed options...")
                    try:
                        success = _run_once(['ALLOW_BALLPARK=YES'])
                    except Exception:
                        raise e
                else:
                    raise e
            if (not success) and transformer_opts:
                self.log(f"      ⚠ Warp returned no dataset with strict transformer options; retrying with relaxed options...")
                success = _run_once(['ALLOW_BALLPARK=YES'])

            # Safety guard: if output still spans near full-world width, retry with split bounds.
            if success and self._is_world_width_warp(output_tiff):
                self.log("      ⚠ Detected near-world-width warp output; forcing antimeridian split retry...")
                forced_bounds = split_bounds or self._antimeridian_split_bounds_3857(input_tiff)
                if forced_bounds:
                    success = _run_split_warp(['ALLOW_BALLPARK=YES'], forced_bounds)
                if success and self._is_world_width_warp(output_tiff):
                    self.log("      ✗ Warp still produced pathological full-world width after retry")
                    success = False

            # Cleanup
            if temp_vrt_name and (not to_vrt) and temp_vrt_name.startswith("/vsimem/"):
                try:
                    gdal.Unlink(temp_vrt_name)
                except:
                    pass

            if success:
                file_size_mb = output_tiff.stat().st_size / (1024 * 1024) if output_tiff.exists() else 0
                size_label = f"{file_size_mb:.1f} MB" if not to_vrt else f"{output_tiff.stat().st_size} bytes"
                self.log(f"      ✓ Created {output_tiff.name} ({size_label})")
                return True

            _safe_unlink(output_tiff)
            self.log(f"      ✗ Warp failed for {input_tiff.name}")
            return False

        except Exception as e:
            try:
                if output_tiff.exists():
                    output_tiff.unlink()
            except Exception:
                pass
            self.log(f"      ✗ Error warping {input_tiff.name}: {e}")
            return False

    def build_vrt(self, input_files: List[Path], output_vrt: Path) -> bool:
        """Combine multiple TIFFs/VRTs into a single VRT."""
        try:
            # Validate input files before building VRT
            valid_files = []
            for f in input_files:
                if not f.exists():
                    self.log(f"      Warning: Input file not found: {f.name}")
                    continue
                file_size = f.stat().st_size
                if file_size < 1024:  # Skip files smaller than 1KB (likely corrupt/empty)
                    self.log(f"      Warning: Skipping {f.name} (too small: {file_size} bytes)")
                    continue
                valid_files.append(f)

            if not valid_files:
                self.log(f"      ✗ No valid input files for VRT")
                return False

            input_strs = [str(f) for f in valid_files]
            # Don't add alpha - input files already have alpha from warp_and_cut
            resample_alg = getattr(self, 'resample_alg', gdal.GRA_Bilinear)
            vrt_options = gdal.BuildVRTOptions(resampleAlg=resample_alg, resolution='highest')
            ds = gdal.BuildVRT(str(output_vrt), input_strs, options=vrt_options)

            if ds:
                ds = None
                if output_vrt.exists():
                    vrt_size_kb = output_vrt.stat().st_size / 1024
                    self.log(f"      ✓ Built VRT: {output_vrt.name} ({vrt_size_kb:.1f} KB)")
                return True
            else:
                self.log(f"      ✗ Failed to build VRT")
                return False

        except Exception as e:
            self.log(f"      ✗ Error building VRT: {e}")
            import traceback
            self.log(f"      {traceback.format_exc()}")
            return False

    def create_geotiff(self, input_vrt: Path, output_tiff: Path, compress: str = 'LZW') -> Tuple[bool, Optional[float]]:
        """Convert VRT to optimized GeoTIFF using Translate with multithreading.

        Uses gdal.Translate instead of gdal.Warp because:
        - VRT is already georeferenced, no reprojection needed
        - Translate is faster for simple copy+compress operations
        - NUM_THREADS creation option enables parallel compression
        """
        try:
            # Log compression status
            num_threads = getattr(self, 'num_threads', '4')
            if compress == 'NONE':
                self.log(f"    Creating GeoTIFF with no compression ({num_threads} threads)...")
            else:
                self.log(f"    Creating GeoTIFF with {compress} compression ({num_threads} threads)...")

            # Configure GDAL for single-process GeoTIFF creation (use all CPUs)
            _configure_gdal_for_phase('geotiff', workers=1)

            # Build creation options based on compression
            creation_opts = [
                'TILED=YES',
                'BIGTIFF=YES',
                'BLOCKXSIZE=1024',
                'BLOCKYSIZE=1024',
                f'NUM_THREADS={num_threads}'
            ]

            if compress != 'NONE':
                creation_opts.extend([
                    f'COMPRESS={compress}',
                    'PREDICTOR=2'
                ])

            # Optional clipping window (projWin) if user supplied --clip-projwin
            projwin = getattr(self, 'clip_projwin', None)
            if projwin:
                ulx, uly, lrx, lry = projwin
                self.log(f"    Applying projWin clip: UL({ulx}, {uly}) LR({lrx}, {lry})")

            # Simple progress reporter for long translates (logs every 5%)
            start_time = time.time()
            progress_state = {'last_pct': -5}

            def progress_cb(complete, message, cb_data):
                pct = int(complete * 100)
                if pct - progress_state['last_pct'] >= 5:
                    progress_state['last_pct'] = pct
                    elapsed = time.time() - start_time
                    # Per-job ETA
                    if pct > 0:
                        est_total = elapsed * 100 / pct
                        eta_sec = max(0, est_total - elapsed)
                        eta_min = eta_sec / 60
                        self.log(f"      GeoTIFF progress: {pct}% | ETA {eta_min:.1f} min")
                    else:
                        self.log(f"      GeoTIFF progress: {pct}%")

                    # Overall ETA (based on completed GeoTIFFs)
                    # remaining = total_planned - completed (current job counts as in-progress, not completed)
                    completed = getattr(self, 'geotiff_completed', 0)
                    total_planned = getattr(self, 'geotiff_total_planned', 0)
                    remaining_jobs = max(0, total_planned - completed)
                    durations = getattr(self, 'geotiff_times', [])
                    if durations and remaining_jobs > 0:
                        avg_sec = sum(durations) / len(durations)
                        # Current job remaining + future jobs
                        overall_sec = eta_sec + (remaining_jobs - 1) * avg_sec if pct > 0 else remaining_jobs * avg_sec
                        overall_hours = overall_sec / 3600
                        if overall_hours >= 1:
                            self.log(f"      Overall ETA: {overall_hours:.1f} hours ({remaining_jobs} GeoTIFFs remaining, avg {avg_sec/60:.1f} min each)")
                        else:
                            self.log(f"      Overall ETA: {overall_sec/60:.1f} min ({remaining_jobs} GeoTIFFs remaining)")
                return 1

            # Use gdal.Translate - faster than Warp when no reprojection needed
            translate_options = gdal.TranslateOptions(
                format='GTiff',
                creationOptions=creation_opts,
                projWin=projwin if projwin else None,
                callback=progress_cb
            )

            ds = gdal.Translate(str(output_tiff), str(input_vrt), options=translate_options)

            if ds:
                ds = None
                file_size_mb = output_tiff.stat().st_size / (1024 * 1024) if output_tiff.exists() else 0
                elapsed = time.time() - start_time
                rate = file_size_mb / elapsed if elapsed > 0 else 0
                self.log(f"      ✓ GeoTIFF created: {output_tiff.name} ({file_size_mb:.1f} MB in {elapsed:.1f}s, {rate:.1f} MB/s)")
                return True, elapsed
            else:
                self.log(f"      ✗ Failed to create GeoTIFF")
                return False, None

        except Exception as e:
            error_str = str(e).lower()
            # If ZSTD decompression fails, try with LZW compression instead
            if compress == 'ZSTD' and ('zstd' in error_str or 'frame descriptor' in error_str or 'data corruption' in error_str):
                self.log(f"    ZSTD decompression error detected, retrying with LZW compression...")
                return self.create_geotiff(input_vrt, output_tiff, compress='LZW')

            self.log(f"    Error creating GeoTIFF: {e}")
            import traceback
            self.log(f"    {traceback.format_exc()}")
            return False, None

    def _save_review_csv(self, report_data: List[Dict], output_path: Path) -> None:
        """Save review report to CSV file in matrix format (dates × locations)."""
        try:
            import csv as csv_module

            # Build data structures (same as console)
            dates = sorted(set(e['date'] for e in report_data), reverse=True)
            locations = sorted(set(e['location'] for e in report_data))

            # Build lookup: date -> location -> cell content
            matrix = defaultdict(dict)
            for entry in report_data:
                date = entry['date']
                loc = entry['location']

                # Determine cell content based on matches
                if entry['num_files'] > 0:
                    # Extract just filename
                    files = entry['source_files'].split(',')
                    if len(files) > 1:
                        cell = f"({len(files)} files)"
                    else:
                        filename = Path(files[0].strip()).name
                        # Truncate if too long
                        if len(filename) > 12:
                            filename = filename[:9] + "..."
                        cell = filename
                else:
                    cell = "---"

                matrix[date][loc] = cell

            # Write matrix format CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv_module.writer(f)

                # Write header row with location names
                header = ['Date'] + locations
                writer.writerow(header)

                # Write data rows
                for date in dates:
                    row = [date]
                    for loc in locations:
                        cell_value = matrix[date].get(loc, "---")
                        row.append(cell_value)
                    writer.writerow(row)

            self.log(f"Review report saved to: {output_path}")
        except Exception as e:
            self.log(f"Warning: Could not save review CSV: {e}")

    def generate_review_report(self) -> bool:
        """
        Generate review report showing file matches for all dates.
        Saves CSV report and displays formatted summary.
        Returns True if user confirms to proceed, False otherwise.
        """
        self.log("\n=== Generating Processing Review ===")

        # Get all locations from CSV (not shapefiles)
        all_locations = set()
        for date, records in self.dole_data.items():
            for rec in records:
                norm_loc = self.normalize_name(rec.get('location', ''))
                if norm_loc:
                    all_locations.add(norm_loc)

        all_locations = sorted(all_locations)

        # Build CSV records lookup: date -> location -> record
        csv_lookup = defaultdict(dict)
        for date, records in self.dole_data.items():
            for rec in records:
                norm_loc = self.normalize_name(rec.get('location', ''))
                csv_lookup[date][norm_loc] = rec

        # Build report data for ALL locations on ALL dates
        sorted_dates = sorted(self.dole_data.keys(), key=self._date_key_sort_key, reverse=True)
        report_data = []

        for date in sorted_dates:
            for location in all_locations:
                # Check if this location has a CSV record for this date
                if location in csv_lookup[date]:
                    rec = csv_lookup[date][location]
                    edition = rec.get('edition', '')
                    timestamp = rec.get('wayback_ts')

                    # Find shapefile
                    shp_path = self.find_shapefile(location)
                    shapefile_found = "YES" if shp_path else "NO"

                    # Find source files
                    filename = rec.get('filename', '')
                    download_link = rec.get('download_link', '')
                    found_files = self.find_files(location, edition, timestamp, filename, download_link)

                    if found_files:
                        # Handle both Path and (Path, internal_file) tuple formats
                        file_names = []
                        for f in found_files:
                            if isinstance(f, tuple):
                                path, internal = f
                                if internal:
                                    file_names.append(f"{path.name}/{internal}")
                                else:
                                    file_names.append(path.name)
                            else:
                                file_names.append(f.name)
                        source_files_str = ', '.join(file_names)
                    else:
                        # Check if filename exists in CSV but just not on disk yet
                        if filename:
                            # File is in CSV but not on disk - will be downloaded during processing
                            source_files_str = f"📥 WILL DOWNLOAD: {filename}"
                        else:
                            source_files_str = "NOT IN CSV"

                    report_entry = {
                        'date': date,
                        'location': location,
                        'edition': edition,
                        'shapefile_found': shapefile_found,
                        'source_files': source_files_str,
                        'num_files': len(found_files)
                    }
                else:
                    # No CSV record for this location on this date
                    report_entry = {
                        'date': date,
                        'location': location,
                        'edition': 'N/A',
                        'shapefile_found': 'N/A',
                        'source_files': 'NOT FOUND',
                        'num_files': 0
                    }

                report_data.append(report_entry)

        # Save CSV report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"processing_review_{timestamp}.csv"
        self._save_review_csv(report_data, csv_path)

        # Get user confirmation
        while True:
            response = input("\nProceed with processing? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'n'")

    def _rebuild_vrt_library(self, all_locations: List[str]) -> Dict[str, Dict[str, List[Path]]]:
        """
        Rebuild vrt_library from existing temp VRTs for resume support.

        Scans temp_dir for previously created location VRTs so that
        resuming an interrupted run can reuse existing intermediates.

        Returns:
            Dict mapping location -> date -> list of VRT/TIF paths
        """
        vrt_library: Dict[str, Dict[str, List[Path]]] = defaultdict(dict)

        if not self.temp_dir.exists():
            return vrt_library

        self.log("Rebuilding VRT library from existing temp files...")

        # Scan date subdirectories in temp_dir
        date_dirs = [d for d in self.temp_dir.iterdir() if d.is_dir()]

        found_count = 0
        skipped_pathological = 0
        for date_dir in date_dirs:
            date = date_dir.name

            # Skip non-date directories (simple validation: should contain hyphen)
            if '-' not in date:
                continue

            # Collect per-location warped outputs (TIF/VRT) and fall back to combined VRTs
            combined_vrt_by_loc = {}

            for vrt_file in date_dir.glob("*.vrt"):
                # Skip mosaic VRTs
                if vrt_file.name.startswith("mosaic_"):
                    continue
                # Skip RGBA intermediate VRTs
                if "_rgba.vrt" in vrt_file.name:
                    continue
                if self._is_world_width_warp(vrt_file):
                    skipped_pathological += 1
                    continue

                # Track per-location combined VRTs: {location}_{date}.vrt
                stem = vrt_file.stem
                if stem.endswith(f"_{date}"):
                    location = stem[:-len(f"_{date}")]
                    if location in all_locations or self.normalize_name(location) in all_locations:
                        norm_loc = self.normalize_name(location)
                        combined_vrt_by_loc[norm_loc] = vrt_file
                    continue

                # Treat other VRTs as individual warped outputs: {location}_{original_stem}.vrt
                for loc in all_locations:
                    if stem.startswith(f"{loc}_"):
                        if date not in vrt_library.get(loc, {}):
                            vrt_library[loc][date] = []
                        vrt_library[loc][date].append(vrt_file)
                        found_count += 1
                        break

            # Collect warped TIFs: {location}_{original_stem}.tif
            for tif_file in date_dir.glob("*.tif"):
                if self._is_world_width_warp(tif_file):
                    skipped_pathological += 1
                    continue
                stem = tif_file.stem
                for loc in all_locations:
                    if stem.startswith(f"{loc}_"):
                        if date not in vrt_library.get(loc, {}):
                            vrt_library[loc][date] = []
                        vrt_library[loc][date].append(tif_file)
                        found_count += 1
                        break

            # If no individual warped outputs were found, fall back to per-location combined VRTs
            for norm_loc, loc_vrt in combined_vrt_by_loc.items():
                if date in vrt_library.get(norm_loc, {}):
                    continue
                vrt_library[norm_loc][date] = [loc_vrt]
                found_count += 1

        if found_count > 0:
            # Count unique location-date pairs
            total_pairs = sum(len(dates) for dates in vrt_library.values())
            self.log(f"  Restored {total_pairs} location-date entries from {len(vrt_library)} locations")
        else:
            self.log("  No existing VRTs found (fresh run)")
        if skipped_pathological > 0:
            self.log(f"  Skipped {skipped_pathological} pathological intermediates with near-world extents")

        return vrt_library

    def process_all_dates(self, date_range: Optional[Tuple[datetime.date, datetime.date]] = None):
        """Process date ranges in CSV and publish PMTiles, optionally limited by start date."""
        self.log("\n=== Starting Chart Processing ===")

        # Get all locations from CSV (not shapefiles)
        all_locations = set()
        for date, records in self.dole_data.items():
            for rec in records:
                norm_loc = self.normalize_name(rec.get('location', ''))
                if norm_loc:
                    all_locations.add(norm_loc)

        all_locations = sorted(all_locations)

        # Count how many have shapefiles
        shapefile_count = sum(1 for loc in all_locations if self.find_shapefile(loc))
        self.log(f"Found {len(all_locations)} locations from CSV ({shapefile_count} with shapefiles).")

        # Build processing date list (optionally filtered)
        sorted_dates = sorted(self.dole_data.keys(), key=self._date_key_sort_key)
        if date_range:
            start_date, end_date = date_range
            if start_date and end_date and start_date > end_date:
                start_date, end_date = end_date, start_date
            filtered = []
            for date_key in sorted_dates:
                start_str = self.date_key_map.get(date_key, (date_key, None))[0]
                date_obj = self._date_from_str(start_str)
                if not date_obj:
                    continue
                if start_date <= date_obj <= end_date:
                    filtered.append(date_key)
            sorted_dates = filtered

            if not sorted_dates:
                self.log("No dates found in the requested range. Nothing to process.")
                return

            self.log(f"Limiting processing to {len(sorted_dates)} date ranges ({start_date} to {end_date})")

        # === PREPARATION PHASE: Download all missing files upfront ===
        self.log("\n=== Preparing: Downloading missing files ===")
        self._download_missing_files(allowed_dates=set(sorted_dates))

        # Track all warped VRTs per location-date
        # Rebuild from existing temp files to support resume
        vrt_library = self._rebuild_vrt_library(all_locations)

        # Process each date range (earliest to latest)
        total_dates = len(sorted_dates)

        postprocess_jobs: List[Tuple[Path, str]] = []

        for date_idx, date in enumerate(sorted_dates, 1):
            self.log(f"\n[{date_idx}/{total_dates}] Processing range: {date}")

            temp_dir = self.temp_dir / date
            temp_dir.mkdir(parents=True, exist_ok=True)

            mosaic_vrt = temp_dir / f"mosaic_{date}.vrt"
            if mosaic_vrt.exists() and mosaic_vrt.stat().st_size > 0:
                self.log(f"  ✓ Mosaic VRT already exists - skipping warp/mosaic build")
                postprocess_jobs.append((mosaic_vrt, date))
                self.log(f"  Queued MBTiles/PMTiles post-process")
                continue

            records = self.dole_data[date]

            # Group records by exact (location, edition) to handle duplicate rows with fallback
            record_groups = []
            record_group_map = {}
            for rec in records:
                location = (rec.get('location') or '').strip()
                edition = (rec.get('edition') or '').strip()
                key = (location, edition)
                if key not in record_group_map:
                    record_group_map[key] = {
                        'location': location,
                        'edition': edition,
                        'records': []
                    }
                    record_groups.append(record_group_map[key])
                record_group_map[key]['records'].append(rec)

            # Track warped outputs selected per location for this date
            temp_warped_by_location = defaultdict(list)

            # Prepare group state for fallback attempts
            group_states = []
            for group in record_groups:
                norm_loc = self.normalize_name(group['location'])
                group_states.append({
                    'location': group['location'],
                    'edition': group['edition'],
                    'norm_loc': norm_loc,
                    'records': group['records'],
                    'next_index': 0,
                    'done': False
                })

            # Execute warp jobs in parallel, with fallback to duplicate rows on failure
            if group_states:
                requested_warp_workers = getattr(self, 'parallel_warp', 0)
                if requested_warp_workers and requested_warp_workers > 0:
                    warp_worker_upper_bound = max(1, min(requested_warp_workers, cpu_count))
                else:
                    warp_worker_upper_bound = max(2, cpu_count - 2)
                    if IS_APPLE_SILICON:
                        # Keep some headroom for system + upload threads on M-series SoCs.
                        warp_worker_upper_bound = min(warp_worker_upper_bound, 6)

                warp_worker_target = warp_worker_upper_bound
                warp_round_rate_ema = None
                warp_mem_per_worker_mb = 1200 if getattr(self, 'warp_output', 'tif').lower() == 'vrt' else 900
                if getattr(self, 'warp_multithread', True):
                    warp_mem_per_worker_mb += 256

                round_idx = 0
                while True:
                    warp_jobs = []
                    attempt_map = {}
                    made_progress = False

                    for state in group_states:
                        if state['done'] or state['next_index'] >= len(state['records']):
                            continue

                        rec = state['records'][state['next_index']]
                        location = (rec.get('location') or '').strip()
                        edition = (rec.get('edition') or '').strip()
                        timestamp = rec.get('wayback_ts')
                        filename = rec.get('filename', '')  # Get filename from CSV

                        attempt_num = state['next_index'] + 1
                        attempt_total = len(state['records'])
                        attempt_label = f" (option {attempt_num}/{attempt_total})" if attempt_total > 1 else ""

                        # Find shapefile
                        shp_path = self.find_shapefile(location)
                        if not shp_path:
                            self.log(f"    {location}: No shapefile found")
                            state['done'] = True
                            made_progress = True
                            continue

                        # Find source files (use filename from CSV if available)
                        download_link = rec.get('download_link', '')
                        found_files = self.find_files(location, edition, timestamp, filename, download_link)
                        if not found_files:
                            self.log(
                                f"    {location} (edition {edition}){attempt_label}: No source files found [CSV: {filename if filename else 'no match'}]"
                            )
                            state['next_index'] += 1
                            made_progress = True
                            if state['next_index'] < len(state['records']):
                                self.log(f"    {location} (edition {edition}): trying next duplicate option")
                            continue

                        self.log(f"    {location} (edition {edition}){attempt_label}: {filename if filename else 'pattern match'}")

                        attempt_id = (state['location'], state['edition'], state['next_index'])
                        attempt_map[attempt_id] = state

                        # Prepare warp jobs for this record attempt
                        for src_file_info in found_files:
                            jobs = self._prepare_warp_job(src_file_info, location, edition, shp_path, temp_dir, date)
                            for job in jobs:
                                job['attempt_id'] = attempt_id
                            warp_jobs.extend(jobs)

                    if not warp_jobs:
                        if made_progress:
                            continue
                        break

                    round_idx += 1
                    warp_mem_snapshot = _get_memory_snapshot_mb(
                        default_available_mb=8192 if IS_APPLE_SILICON else 4096,
                        default_total_mb=16384 if IS_APPLE_SILICON else 8192
                    )
                    warp_reserve_mb = _recommended_free_ram_floor_mb(warp_mem_snapshot["total_mb"])
                    warp_mem_budget_mb = max(0, warp_mem_snapshot["available_mb"] - warp_reserve_mb)
                    warp_memory_cap = max(
                        1,
                        min(
                            warp_worker_upper_bound,
                            (warp_mem_budget_mb // warp_mem_per_worker_mb) if warp_mem_budget_mb > 0 else 1
                        )
                    )
                    current_warp_workers = max(
                        1,
                        min(warp_worker_target, warp_memory_cap, len(warp_jobs))
                    )
                    if current_warp_workers != warp_worker_target:
                        self.log(
                            f"  ↺ Autotune warp: workers {warp_worker_target}->{current_warp_workers} "
                            f"(free={warp_mem_snapshot['available_mb']}MB, floor={warp_reserve_mb}MB)"
                        )
                    warp_worker_target = current_warp_workers
                    _configure_gdal_for_phase('warp', workers=current_warp_workers)
                    self.log(
                        f"  Processing {len(warp_jobs)} warp jobs with {current_warp_workers} parallel workers "
                        f"(round {round_idx}, free RAM {warp_mem_snapshot['available_mb']}MB)..."
                    )
                    round_start = time.time()
                    results = []

                    with ThreadPoolExecutor(max_workers=current_warp_workers) as executor:
                        futures = {executor.submit(self._warp_worker, job): job for job in warp_jobs}

                        for future in as_completed(futures):
                            try:
                                result = future.result()
                                results.append(result)
                            except Exception as e:
                                self.log(f"    Error in parallel warp: {e}")

                    round_elapsed = max(0.001, time.time() - round_start)
                    round_successes = sum(1 for result in results if result.get('success'))
                    round_rate = round_successes / round_elapsed

                    if warp_round_rate_ema is None:
                        warp_round_rate_ema = round_rate
                    else:
                        if round_successes == 0 and warp_worker_target > 1:
                            next_workers = max(1, warp_worker_target - 1)
                            self.log(
                                f"  ↺ Autotune warp: workers {warp_worker_target}->{next_workers} "
                                f"(no successes in round {round_idx})"
                            )
                            warp_worker_target = next_workers
                        elif round_rate < (warp_round_rate_ema * 0.75) and warp_worker_target > 1:
                            next_workers = max(1, warp_worker_target - 1)
                            self.log(
                                f"  ↺ Autotune warp: workers {warp_worker_target}->{next_workers} "
                                f"(throughput drop to {round_rate:.3f} jobs/s)"
                            )
                            warp_worker_target = next_workers
                        elif (
                            round_rate >= (warp_round_rate_ema * 1.05)
                            and warp_worker_target < warp_worker_upper_bound
                            and len(warp_jobs) > warp_worker_target
                        ):
                            projected_workers = warp_worker_target + 1
                            projected_need_mb = warp_reserve_mb + (projected_workers * warp_mem_per_worker_mb)
                            if warp_mem_snapshot["available_mb"] >= projected_need_mb:
                                self.log(
                                    f"  ↺ Autotune warp: workers {warp_worker_target}->{projected_workers} "
                                    f"(throughput stable at {round_rate:.3f} jobs/s)"
                                )
                                warp_worker_target = projected_workers

                        warp_round_rate_ema = (warp_round_rate_ema * 0.7) + (round_rate * 0.3)

                    # Group results by attempt (validate output files)
                    outputs_by_attempt = defaultdict(list)
                    for result in results:
                        if result['success'] and result['output_tiff']:
                            output_file = result['output_tiff']
                            if output_file.exists() and output_file.stat().st_size > 1024:
                                if self._is_world_width_warp(output_file):
                                    self.log(f"      Warning: Skipping pathological full-world warp: {output_file.name}")
                                    continue
                                outputs_by_attempt[result.get('attempt_id')].append(output_file)
                            else:
                                self.log(f"      Warning: Skipping invalid warped file: {output_file.name}")

                    # Apply results and schedule fallback attempts as needed
                    for attempt_id, state in attempt_map.items():
                        attempt_outputs = outputs_by_attempt.get(attempt_id, [])
                        if attempt_outputs:
                            temp_warped_by_location[state['norm_loc']].extend(attempt_outputs)
                            state['done'] = True
                        else:
                            state['next_index'] += 1
                            made_progress = True
                            if state['next_index'] < len(state['records']):
                                self.log(
                                    f"    {state['location']} (edition {state['edition']}): warp failed; trying next duplicate option"
                                )

                # Register warped outputs directly for mosaic (no per-location VRT combining)
                location_count = 0
                for norm_loc, temp_warped in temp_warped_by_location.items():
                    if not temp_warped:
                        continue
                    location_count += 1
                    if len(temp_warped) > 1:
                        self.log(f"      Using {len(temp_warped)} warped files for {norm_loc}")
                    else:
                        self.log(f"      Single file, using directly")

                    existing = vrt_library.get(norm_loc, {}).get(date, [])
                    if isinstance(existing, list):
                        combined = list(existing)
                    elif existing:
                        combined = [existing]
                    else:
                        combined = []

                    combined.extend(temp_warped)

                    # De-duplicate while preserving order
                    seen = set()
                    deduped = []
                    for p in combined:
                        key = str(p)
                        if key in seen:
                            continue
                        seen.add(key)
                        deduped.append(p)

                    vrt_library[norm_loc][date] = deduped

                if temp_warped_by_location:
                    self.log(f"  Processed {location_count} locations with data")
                else:
                    self.log(f"  No valid locations found with source files")
            else:
                self.log(f"  No valid locations found with source files")

            # Build mosaic inputs from exact date range matches only
            mosaic_sources = []
            for location in all_locations:
                entries = vrt_library.get(location, {}).get(date)
                if not entries:
                    continue
                if isinstance(entries, list):
                    mosaic_sources.extend(entries)
                else:
                    mosaic_sources.append(entries)

            # Create mosaic VRT and queue direct MBTiles/PMTiles post-process
            if mosaic_sources:
                self.log(f"  Creating mosaic VRT from {len(mosaic_sources)} sources...")

                temp_dir.mkdir(parents=True, exist_ok=True)

                mosaic_vrt = temp_dir / f"mosaic_{date}.vrt"
                if self.build_vrt(mosaic_sources, mosaic_vrt):
                    self.log(f"  ✓ Mosaic VRT created")

                    postprocess_jobs.append((mosaic_vrt, date))
                    self.log(f"  Queued MBTiles/PMTiles post-process")
                    self.log(f"  ✓✓✓ {date.upper()} VRT COMPLETE")
                else:
                    self.log(f"  ✗ Mosaic creation failed")
            else:
                self.log(f"  Skipping {date} - no data available")

        if postprocess_jobs:
            # Post-process: convert mosaic rasters directly to MBTiles/PMTiles and upload
            self._run_postprocess_upload_stage(input_dir=self.temp_dir, input_jobs=postprocess_jobs)
        else:
            self.log("No mosaics queued for post-processing.")

        self.log("\n=== Processing Complete ===")


def main():
    parser = argparse.ArgumentParser(
        description="FAA Chart Newslicer - Process sectional charts to COGs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s -s /path/to/charts -o /path/to/output -c master_dole.csv -b shapefiles
        """
    )

    parser.add_argument(
        "-s", "--source",
        type=Path,
        default=Path("/Volumes/drive/newrawtiffs"),
        help="Source directory containing TIFF/ZIP files (default: /Volumes/drive/newrawtiffs)"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("/Volumes/drive/sync"),
        help="Output directory for COGs (default: /Volumes/drive/sync)"
    )

    parser.add_argument(
        "-c", "--csv",
        type=Path,
        default=Path("/Users/ryanhemenway/archive.aero/master_dole.csv"),
        help="Master Dole CSV file (default: /Users/ryanhemenway/archive.aero/master_dole.csv)"
    )

    parser.add_argument(
        "-b", "--shapefiles",
        type=Path,
        default=Path("/Users/ryanhemenway/archive.aero/shapefiles"),
        help="Directory containing shapefiles (default: /Users/ryanhemenway/archive.aero/shapefiles)"
    )

    parser.add_argument(
        "-t", "--temp-dir",
        type=Path,
        default=Path("/Volumes/projects/.temp"),
        help="Temporary directory for intermediate files (default: /Volumes/projects/.temp)"
    )

    parser.add_argument(
        "-r", "--resample",
        type=str,
        choices=['nearest', 'bilinear', 'cubic', 'cubicspline'],
        default='nearest',
        help="Resampling algorithm for warping (default: nearest). Use 'nearest' for faster processing with charts"
    )

    parser.add_argument(
        "--warp-output",
        type=str,
        choices=['tif', 'vrt'],
        default='vrt',
        help="Intermediate warp output format: 'tif' writes cutlines to disk, 'vrt' (default) keeps them virtual to reduce I/O"
    )

    parser.add_argument(
        "--clip-projwin",
        type=float,
        nargs=4,
        metavar=('ULX', 'ULY', 'LRX', 'LRY'),
        default=None,
        help="Optional projWin window (in target CRS, EPSG:3857) applied at VRT→MBTiles translate. "
             "Order: ULX ULY LRX LRY. Example for CONUS approx: --clip-projwin -13914936 6446276 -7347086 2753408"
    )

    parser.add_argument(
        "--compression",
        type=str,
        choices=['AUTO', 'ZSTD', 'LZW', 'DEFLATE', 'NONE'],
        default='AUTO',
        help="Deprecated (GeoTIFF phase removed). Retained for CLI compatibility."
    )

    parser.add_argument(
        "--num-threads",
        type=str,
        default='4',
        help="Deprecated (GeoTIFF phase removed). Retained for CLI compatibility."
    )

    parser.add_argument(
        "--parallel-geotiff",
        type=int,
        default=4,
        help="Deprecated (GeoTIFF phase removed). Retained for CLI compatibility."
    )

    parser.add_argument(
        "--parallel-warp",
        type=int,
        default=0,
        help="Number of parallel warp jobs (default: auto; Apple Silicon auto mode caps at 6 for better perf/watt)"
    )

    parser.add_argument(
        "--warp-multithread",
        action="store_true",
        help="Enable GDAL internal multithreading inside each warp job (off by default on Apple Silicon)"
    )

    parser.add_argument(
        "--mbtiles-quality",
        type=int,
        default=80,
        help="WEBP quality for MBTiles tiles (1-100, default: 80)"
    )

    parser.add_argument(
        "--upload-jobs",
        type=int,
        default=1,
        help="Parallel postprocess/upload jobs (default: 1)"
    )

    parser.add_argument(
        "--postprocess-threads",
        type=str,
        default="AUTO",
        help="GDAL threads per postprocess worker: AUTO or integer (default: AUTO)"
    )

    parser.add_argument(
        "--postprocess-cache-mb",
        type=int,
        default=0,
        help="GDAL cache per postprocess worker in MB (default: auto)"
    )

    parser.add_argument(
        "--verbose-external-tools",
        action="store_true",
        help="Show raw output from gdal/pmtiles/rclone subprocesses"
    )

    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Skip warp/mosaic creation and only post-process existing mosaic rasters to MBTiles/PMTiles + upload"
    )

    parser.add_argument(
        "--reprocess-existing",
        action="store_true",
        help="Do not skip jobs that already have up-to-date local PMTiles; useful when forcing re-upload"
    )

    parser.add_argument(
        "--download-delay",
        type=float,
        default=8.0,
        help="Seconds to wait between downloads to avoid rate limiting (default: 8.0, try 12.0-20.0 if hitting connection errors)"
    )

    args = parser.parse_args()

    # Validate paths
    if not args.source.exists():
        print(f"ERROR: Source directory not found: {args.source}")
        sys.exit(1)

    if not args.csv.exists():
        print(f"ERROR: CSV file not found: {args.csv}")
        sys.exit(1)

    if not args.shapefiles.exists():
        print(f"ERROR: Shapefiles directory not found: {args.shapefiles}")
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Create temp directory
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    # Process
    slicer = ChartSlicer(args.source, args.output, args.csv, args.shapefiles,
                         preview_mode=False, download_delay=args.download_delay,
                         temp_dir=args.temp_dir)
    slicer.warp_output = args.warp_output
    slicer.clip_projwin = tuple(args.clip_projwin) if args.clip_projwin else None
    slicer.parallel_warp = args.parallel_warp
    if args.warp_multithread:
        slicer.warp_multithread = True

    slicer.upload_jobs = max(1, args.upload_jobs)
    slicer.postprocess_threads = args.postprocess_threads
    slicer.postprocess_cache_mb = args.postprocess_cache_mb if args.postprocess_cache_mb > 0 else None
    slicer.skip_existing_postprocess = not args.reprocess_existing
    slicer.quiet_external_tools = not args.verbose_external_tools
    if args.verbose_external_tools:
        if "-P" not in slicer.rclone_flags:
            slicer.rclone_flags.insert(0, "-P")
        if "--stats" not in slicer.rclone_flags:
            slicer.rclone_flags.extend(["--stats", "1s"])
    slicer.mbtiles_quality = args.mbtiles_quality
    if args.compression != 'AUTO':
        slicer.log("Warning: --compression is ignored (GeoTIFF phase removed).")
    if args.num_threads != '4':
        slicer.log("Warning: --num-threads is ignored (GeoTIFF phase removed).")
    if args.parallel_geotiff != 4:
        slicer.log("Warning: --parallel-geotiff is ignored (GeoTIFF phase removed).")

    # Map resampling algorithm
    resample_map = {
        'nearest': gdal.GRA_NearestNeighbour,
        'bilinear': gdal.GRA_Bilinear,
        'cubic': gdal.GRA_Cubic,
        'cubicspline': gdal.GRA_CubicSpline
    }
    slicer.resample_alg = resample_map[args.resample]

    if args.postprocess_only:
        slicer.log("Post-process only: skipping warp/mosaic creation.")
        slicer._run_postprocess_upload_stage(input_dir=args.output)
        return

    slicer.load_dole_data()

    # Build indexes for O(1) lookups (after loading CSV data)
    slicer._build_source_index()
    slicer._build_shapefile_index()

    # During review phase, disable downloads (don't try to validate by downloading)
    # This prevents rate limiting during the review report generation
    slicer.preview_mode = True

    # Generate review report and get user confirmation (without downloading)
    if not slicer.generate_review_report():
        print("Processing cancelled by user.")
        sys.exit(0)

    def _prompt_date_range() -> Optional[Tuple[datetime.date, datetime.date]]:
        while True:
            choice = input("Process all dates or a range? (all/range): ").strip().lower()
            if choice in {"all", "a", ""}:
                return None
            if choice in {"range", "r"}:
                start_str = input("Start date (YYYY-MM-DD): ").strip()
                end_str = input("End date (YYYY-MM-DD): ").strip()
                start_dt = slicer._date_from_str(start_str)
                end_dt = slicer._date_from_str(end_str)
                if not start_dt or not end_dt:
                    print("Please enter valid dates in YYYY-MM-DD format.")
                    continue
                if start_dt > end_dt:
                    start_dt, end_dt = end_dt, start_dt
                    print(f"Swapped range to {start_dt} - {end_dt}")
                return (start_dt, end_dt)
            print("Please enter 'all' or 'range'.")

    # Now enable downloads for actual processing
    slicer.preview_mode = False
    date_range = _prompt_date_range()
    slicer.process_all_dates(date_range=date_range)


if __name__ == '__main__':
    main()
