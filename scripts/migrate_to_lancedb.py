import argparse
import logging
from pathlib import Path
import shutil

import h5py
import lancedb
import pyarrow as pa
import numpy as np

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MigrateToLanceDB")


def migrate(h5_path: str, batch_size: int = 128, min_index_frames: int = 256):
    h5_file = Path(h5_path)
    if not h5_file.exists() or not h5_file.is_file():
        logger.error(f"File not found: {h5_path}")
        return

    logger.info(f"Opening HDF5 file: {h5_path}")
    lance_dir = h5_file.parent / "vectors.lance"
    
    with h5py.File(h5_file, "r") as f:
        if "global_descriptors" not in f:
            logger.error("No 'global_descriptors' group found. Not a valid topometric database.")
            return
            
        grp = f["global_descriptors"]
        if "descriptors" not in grp:
            logger.error("No 'descriptors' dataset found in 'global_descriptors'. Already migrated?")
            return
            
        descriptors = grp["descriptors"]
        num_frames, dim = descriptors.shape
        logger.info(f"Found {num_frames} global descriptors of dimension {dim}.")
        
        if lance_dir.exists():
            logger.warning(f"LanceDB directory already exists at {lance_dir}. Removing it.")
            shutil.rmtree(lance_dir)
            
        logger.info("Initializing LanceDB...")
        db = lancedb.connect(str(lance_dir))
        
        schema = pa.schema([
            pa.field("frame_id", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), dim))
        ])
        
        table = db.create_table("global_vectors", schema=schema, mode="create")
        
        logger.info("Starting migration to LanceDB...")
        for i in range(0, num_frames, batch_size):
            end_idx = min(i + batch_size, num_frames)
            subset = descriptors[i:end_idx]
            
            batch = []
            for j in range(len(subset)):
                frame_id = i + j
                # Check for zero vectors (unfilled slots if pre-allocated but unfilled)
                if np.sum(np.abs(subset[j])) < 1e-6:
                    continue
                batch.append({
                    "frame_id": frame_id,
                    "vector": subset[j]
                })
                
            if batch:
                table.add(batch)
                
            logger.info(f"Processed {end_idx}/{num_frames} frames...")
            
        actual_count = table.count_rows()
        logger.info(f"Inserted {actual_count} valid vectors into LanceDB.")
        
        if actual_count >= min_index_frames:
            logger.info("Building LanceDB IVF-PQ index...")
            table.create_index(
                metric="cosine",
                num_partitions=min(256, actual_count // 8),
                num_sub_vectors=32
            )
            logger.info("Index built successfully.")
        else:
            logger.info("Skipped index building (not enough frames).")
            
        logger.info(f"Migration complete! Lance database stored safely in {lance_dir}.")
        logger.info("Note: The old descriptors were kept in the HDF5 file for safety.")
        logger.info("You can repack the HDF5 manually using 'h5repack' if disk space is critical, or leave it as is.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate HDF5 descriptors to LanceDB.")
    parser.add_argument("h5_path", help="Path to the database.h5 file")
    parser.add_argument("--batch", type=int, default=128, help="Batch size for writing to LanceDB")
    
    args = parser.parse_args()
    migrate(args.h5_path, batch_size=args.batch)
