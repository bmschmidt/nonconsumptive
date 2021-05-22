from hashlib import new
import pyarrow as pa
from pyarrow import compute as pc
from pyarrow import feather, ipc, csv, parquet
from tempfile import TemporaryDirectory
from pathlib import Path
import uuid
import argparse
import random
from typing import List, Set, Dict, Tuple, Optional, Iterator, Union



def split(batch, key):
    """
    Partition (not sort) a table into front half, back half.
    Time complexity is greatly reduced by using partition rather than sort.
    
    Probably makes a bunch of extra copies.
    """
    if len(batch) < 3:
        return [batch]
    batch = batch.combine_chunks()
    mid = random.randint(1, len(batch) - 1)
    mid = len(batch) // 2
    batch.take([mid])
    try:
        sort_order = pc.partition_nth_indices(batch[key], options=pc.PartitionNthOptions(mid))
    except:
        print(mid, batch[key])
        raise
    front = batch.take(sort_order[0:mid])
    end = batch.take(sort_order[mid:])
    return [front, end]

def partition(batch, key, times = 4):
    """
    times: number of splits. Power of 2.
    
    Split an array into parts.
    """
    array = [batch]
    for _ in range(2**times):
        first = array.pop(0)
        new_items = split(first, key)
        array += new_items
        del first
    return array

def is_ordered(new_files, batch_size):
    """
    Pass over the memorized files to see if the batch
    size could conceivable write a lower to number to
    disk before a higher number appears.
    """
    queue = []
    buff_size = 0
    for (min, max, size, _) in new_files:
        queue.append((min, max, size))
        buff_size += size
        if buff_size > batch_size:
            _, f_max, f_size = queue.pop(0)
            if f_max > min:
                return False
            buff_size -= f_size
        
    return True


# Better Key: pc.add(tb['bookid'], pc.multiply(tb['wordid'], pc.add(1, pc.min_max(tb['bookid'])['max'])))

def from_csv(input, keys, output, block_size):
    output = Path(output)
    input = Path(input)
    print("Parsing CSV")
    f = csv.open_csv(input, read_options = csv.ReadOptions(block_size = block_size),
            parse_options = csv.ParseOptions(delimiter = ","), convert_options= csv.ConvertOptions(
            column_types={'bookid': pa.uint32(), 'wordid': pa.uint32(), 'count': pa.uint32()}))
    quacksort(f, keys, output, block_size)

def from_feather(input, keys, output, block_size):
    output = Path(output)
    input = Path(input)
    print("Parsing feather")
    inp = ipc.open_file(input)

    def yielder():
        for i in range(inp.num_record_batches):
            yield inp.get_batch(i)
    
    quacksort(yielder(), keys, output, block_size)

def parse_args():
    parser = argparse.ArgumentParser(description='Sort a file.')
    parser.add_argument('--keys', type=str, nargs='+',
                        help='The names of the columns to sort on.')
    parser.add_argument('--block-size', type=int, nargs=1,
                        default = 2_500_000_000,
                        help="The maximum size of tables to hold in memory, in bytes. Performance "
                        "depends on making this as big as possible. Default 2_500_000_000 (2.5 gigabytes)")                        
    parser.add_argument('--format', type=str, nargs=1, help="The format of the output file. "
    "Currently only 'feather', but it would be easy to add more.")
    parser.add_argument('input', type=Path, help = "The file to sort.")
    parser.add_argument('output', type=Path, help = "The file to write into. Type will be gleaned from suffix--must be '.parquet' or '.feather'")
    return parser.parse_args()

def ducksauce(input, **args):
  """

  """
  if input.suffix == ".csv":
      from_csv(input, **args)
  if input.suffix == ".feather":
      from_feather(input, **args)

def args_ducksauce():
    assert output.suffix in {".feather", ".parquet"}
    from_file(**parse_args())

def sort(iterator, keys, tmp_dir, output, block_size = 2_500_000_000):
    old_files = []
    n_records = 0
    # First pass--simply write to disk.
    cache_size = 0
    total_bytes = 0
    cache = []
    key = keys[0]
    for i, batch in enumerate(iterator):
        n_records += len(batch)
        cache_size += batch.nbytes
        total_bytes += batch.nbytes
        cache.append(batch)        
        if cache_size > block_size:
            block = pa.Table.from_batches(cache)
            array = partition(block, key, 3)
            for subbatch in array:
                m = pc.min_max(subbatch[key]).as_py()
                path = Path(tmp_dir / (str(uuid.uuid1()) + ".feather"))
                feather.write_feather(subbatch, path)
                old_files.append((m['min'], m['max'], len(subbatch), path))
            cache = []
            cache_size = 0

    if len(cache) > 0:
        block = pa.Table.from_batches(cache)
        array = partition(block, key, 3)
        for subbatch in array:
            m = pc.min_max(sortkey(subbatch, key)).as_py()
            path = Path(tmp_dir / (str(uuid.uuid1()) + ".feather"))
            feather.write_feather(subbatch, path)
            old_files.append((m['min'], m['max'], len(subbatch), path))

    assert(n_records == sum([f[2] for f in old_files]))
    
    old_files.sort()

    n_splits = 3
    print(f"Proceeding to split {total_bytes / 1024 / 1024:.1f}MB into {2**n_splits} chunks per {block_size / 1024 / 1024:.2f}MB block, {total_bytes // block_size} total chunks.")
    
    iter_num = 0
    while not is_ordered(old_files, block_size):
        iter_num += 1
        # If it's not sorted, break each of the current chunks into 
        cache_size = 0
        cache = []
        new_files = []
        for i, (min, max, size, path) in enumerate(old_files):
            t = feather.read_table(path)
            path.unlink()
            cache_size += t.nbytes
            cache.append(t)
            if i == (len(old_files) - 1) or cache_size >= block_size or (cache_size >= block_size // 2 and iter_num % 2 == 1) :
                tab = pa.concat_tables(cache).combine_chunks()
                array = partition(tab, key, n_splits)
                for subbatch in array:
                    m = pc.min_max(sortkey(subbatch, key)).as_py()
                    path = Path(tmp_dir / (str(uuid.uuid1()) + ".feather"))
                    feather.write_feather(subbatch, path)
                    new_files.append((m['min'], m['max'], len(subbatch), path))       
                cache_size = 0
                cache = []
        old_files = new_files
        old_files.sort()
        assert(n_records == sum([f[2] for f in old_files]))    
        overlaps = 0
        for (mina, maxa, e, f), (minb, maxb, c, d) in zip(old_files[:-1], old_files[1:]):
            overlaps += (maxa - minb) / (maxa - mina)
        overlaps  = overlaps / (len(old_files) - 1)
        print(f"average overlap of adjacent batches is {overlaps:.3%}")

    cache_size = 0
    cache = []
    out_num = 0
    written = 0
    print("Writing final data.")
    final_outfile = None
    for i, (min, max, size, path) in enumerate(old_files):
        t = feather.read_table(path)
        path.unlink()
        cache_size += t.nbytes
        cache.append(t)
        
        try:
            next_min = old_files[i + 1][0]
        except IndexError:
            # Max 32-byte float. will need adjustment if ever want to do this on 64 bits.
            next_min = int(0x7FFFFFFFFFFFFFFF)
        if cache_size >= block_size or next_min == int(0x7FFFFFFFFFFFFFFF):
            if final_outfile is None:
                if output.suffix == ".feather":
                    final_outfile = ipc.new_file(output, schema = cache[0].schema)
                if output.suffix == ".parquet":
                    final_outfile = parquet.ParquetWriter(output, schema=cache[0].schema)
            tab = pa.concat_tables(cache)
            sort_order = pc.sort_indices(sort_keys = keys)
            tab = tab.take(sort_order)
            out_num += 1
            mask = pc.less(tab[key], pa.scalar(next_min, pa.int64()))
            done = tab.filter(mask)
            if output.suffix == ".feather":
                for record_batch in done.to_batches():
                    final_outfile.write_batch(record_batch)
            elif output.suffix == ".parquet":
                final_outfile.write_table(done)
            written += done.nbytes
            leftover = tab.filter(pc.invert(mask))

            # The cache are values that might be part of the next item.
            cache = [leftover]
            cache_size = leftover.nbytes
    # No need for a final flush
    final_outfile.close()

class MyTable():
    def __init__(self, table, dir, key):
        self.path = Path(dir) / (str(uuid.uuid1()) + ".feather")
        self.minmax = None
        self.set_min_max(table, key)
        if self.minmax['min'] is None:
            raise("foo")
        self.nbytes = table.nbytes
        pa.feather.write_feather(table, self.path)
        self.length = len(table)
        self._table = None

    @property 
    def table(self):
        if self._table:
            return self._table
        self._table = feather.read_table(self.path, memory_map = True)
        return self._table
    
    def set_min_max(self, table, key):
        self.minmax = pc.min_max(table[key]).as_py()

    def destroy(self):
        self.path.unlink()

def quacksort(iterator: Iterator[pa.RecordBatch], keys: List[str], output: Union[Path, str], block_size = 2_500_000_000):
    """
    Some kind of multi-pass sorting algorithm that aims to reduce useless ahead-
    of time sorting. 

    iterator: something that yields an iterator over arrow recordbatches.
    keys: an ordered list of sort keys.
    output: the destination file for a parquet file.
    block_size: the block size in bytes. I wouldn't be shocked if 
    actual memory consumption doubles this on occasion. 
    """

    output = Path(output)
    n_records = 0
    # First pass--simply write to disk.
    cache_size = 0
    total_bytes = 0
    cache = []
    tables = []
    """
    First pass--chunk into files of 1/8 the block size.
    """ 
    key = keys[0]
    n_written = 0
    with TemporaryDirectory() as tmp_dir:
        print("Reading input")
        for i, batch in enumerate(iterator):
            n_records += len(batch)
            cache_size += batch.nbytes
            total_bytes += batch.nbytes
            cache.append(batch)        
            if cache_size > block_size:
                block = pa.Table.from_batches(cache)
                array = partition(block, key, 3)
                for subbatch in array:
                    tables.append(MyTable(subbatch, tmp_dir, key))
                    n_written += 1
                print(n_written, end = "\r")
                cache = []
                cache_size = 0
        # Flush the cache
        if len(cache) > 0:
            block = pa.Table.from_batches(cache)
            array = partition(block, key, 2)
            for subbatch in array:
                tables.append(MyTable(subbatch, tmp_dir, key))

        assert(n_records == sum([f.length for f in tables]))
        
        n_splits = 3
        print("Preparing for shuffle sort.")
        while True:
            tables.sort(key = lambda x: x.minmax['min'])
            malordered = malordered_ranges(tables, block_size)
            if len(malordered) == 0:
                break
            worst = malordered[0]
            score = sum([m[0] for m in malordered])
            print(f"{score} bad, reordering {worst} ", end = "\r")
            head = tables[:worst[1][0]]
            to_fix = tables[worst[1][0]:worst[1][1]]
            tail = tables[worst[1][1]:]
            if len(to_fix) == 0:
                to_fix = [tables[worst[1][0]]]
            reorder_table = pa.concat_tables([f.table for f in to_fix])
            new_parts = partition(reorder_table, key, n_splits)
            new_mid = [MyTable(subbatch, tmp_dir, key) for subbatch in new_parts]
            # Cleanup.
            for f in to_fix:
                try:
                    f.destroy()
                except FileNotFoundError:
                    "WTF?"
                    continue
            tables = head + new_mid + tail
        cache_size = 0
        cache = []
        out_num = 0
        written = 0
        print("Writing final data.")
        final_outfile = None
        for i, tab in enumerate(tables):
            cache.append(tab.table)
            cache_size += tab.nbytes
            tab.destroy()
            try:
                next_min = tables[i + 1].minmax['min']
            except IndexError:
                # Max 32-byte float. will need adjustment if ever want to do this on 64 bits.
                next_min = int(0x7FFFFFFFFFFFFFFF)
            if cache_size >= block_size or next_min == int(0x7FFFFFFFFFFFFFFF):
                if final_outfile is None:
                    if output.suffix == ".feather":
                        final_outfile = ipc.new_file(output, schema = cache[0].schema)
                    if output.suffix == ".parquet":
                        final_outfile = parquet.ParquetWriter(output, schema=cache[0].schema)
                tab = pa.concat_tables(cache)
                sort_order = pc.sort_indices(tab[key])
                tab = tab.take(sort_order)
                out_num += 1
                mask = pc.less(tab[key], pa.scalar(next_min, pa.int64()))
                done = tab.filter(mask)
                if output.suffix == ".feather":
                    for record_batch in done.to_batches():
                        final_outfile.write_batch(record_batch)
                elif output.suffix == ".parquet":
                    final_outfile.write_table(done)
                written += done.nbytes
                leftover = tab.filter(pc.invert(mask))

                # The cache are values that might be part of the next item.
                cache = [leftover]
                cache_size = leftover.nbytes
        # No need for a final flush
        final_outfile.close()


def malordered_ranges(files, batch_size):
    """
    Pass over the memorized files to see if the batch
    size could conceivable write a lower to number to
    disk before a higher number appears.
    """
    queue = []
    buff_size = 0
    info = []
    for i, f in enumerate(files):
        right_min = f.minmax['min']
        right_max = f.minmax['max']
        right_size = f.nbytes
        queue.append((right_min, right_max, right_size, i))
        buff_size += f.nbytes
        if buff_size > batch_size:
            left_min, left_max, left_size, left_i = queue.pop(0)
            buff_size -= left_size
            if left_max > right_min:
                info.append(((left_max - left_min)/(right_max - left_min), (left_i, i + 1), "A"))
                info.append(((right_max - right_min)/(right_max - left_min), (left_i, i + 1), "B"))
    info.sort(reverse = True)
    return info



if __name__=="__main__":
    main()