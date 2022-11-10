import argparse
import functools
import multiprocessing
import os
import shutil
from glob import glob
import subprocess
import tqdm
import gdown

from .utils import download, parse_str_of_num_bytes
from lddl.utils import expand_outdir_and_mkdir, mkdir, get_all_files_paths_under, attach_bool_arg


def attach_args(parser=argparse.ArgumentParser()):
  parser.add_argument(
      '--outdir',
      type=str,
      default=None,
      required=True,
      help='path to the output dir',
  )
  attach_bool_arg(parser, 'download', default=True)
  attach_bool_arg(parser, 'unzip', default=True)
  attach_bool_arg(parser, 'shard', default=True)
  parser.add_argument(
      '--num-shards',
      type=int,
      default=10,
      help='number of shards',
  )
  parser.add_argument(
      '--shard-num-processes',
      type=int,
      default=os.cpu_count(),
      help='num of processes used to shard all books',
  )
  return parser


def _shard_book(shard):
  shard_path, books = shard
  with open(shard_path, 'w', newline='\n') as shard_file:
    one_line_books = []
    for book in books:
      text_paths = [
        f for f in get_all_files_paths_under(book)
        if os.path.splitext(f)[1] == '.txt'
      ]
      book_lines = []
      for text in text_paths:
        with open(text, 'r', encoding='utf-8-sig', newline='\n') as book_file:
          sub_book_lines = (bl.strip() for bl in book_file)
          sub_book_lines = [bl for bl in sub_book_lines if len(bl) > 0]
          book_lines.extend(sub_book_lines)
      # The first token is the name of the book.
      book_name = os.path.splitext(os.path.basename(book))[0]
      one_line_books.append(' '.join([book_name] + book_lines))
    shard_file.write('\n'.join(one_line_books))

def unzip_subset(subset, text_dir):
  try:
    subdir_name = subset.split('.xz')[0].split('/')[-1]
    tmpdir_name = os.path.join('/tmp', subdir_name)
    subdir_name = os.path.join(text_dir, subdir_name)
    mkdir(subdir_name)
    mkdir(tmpdir_name)
    out_path = os.path.join(tmpdir_name, 'tar.out')
    err_path = os.path.join(tmpdir_name, 'tar.err')
    subprocess.run(
        ['tar', '-xvf', subset, '-C', subdir_name],
        check=True,
        stdout=open(out_path, 'w'),
        stderr=open(err_path, 'w'),
    )
    shutil.rmtree(tmpdir_name)
  except subprocess.CalledProcessError as e:
    print(e, 'Please check {} and {}'.format(out_path, err_path))
    raise

def unzip_merge_txt(openweb_dir, text_dir, num_processes):
  subset_paths = [
      f for f in get_all_files_paths_under(openweb_dir)
      if os.path.splitext(f)[1] == '.xz'
  ]
  with multiprocessing.Pool(num_processes) as p:
    list(tqdm.tqdm(p.map(functools.partial(unzip_subset, text_dir=text_dir), subset_paths), total=len(subset_paths)))


def _shard_openwebs(text_dir, shards_dir, num_shards, num_processes):
  dir_paths = [
    d for d in glob(text_dir+'/*')
  ]
  shards = [(
      os.path.join(shards_dir, '{}.txt'.format(shard_idx)),
      dir_paths[shard_idx::num_shards],
  ) for shard_idx in range(num_shards)]
  with multiprocessing.Pool(num_processes) as p:
    list(tqdm.tqdm(p.imap(_shard_book, shards), total=len(shards)))


def main(args):
  args.outdir = expand_outdir_and_mkdir(args.outdir)
  target_path = os.path.join(args.outdir, 'openwebtext.tar.xz')
  if args.download:
    gdown.download('https://drive.google.com/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx',
      target_path, quiet=False)
  if args.unzip:
    print('Unzipping {} ...'.format(target_path))
    out_path = os.path.join(args.outdir, 'tar.out')
    err_path = os.path.join(args.outdir, 'tar.err')
    try:
      subprocess.run(
          ['tar', '-xvf', target_path, '-C', args.outdir],
          check=True,
          stdout=open(out_path, 'w'),
          stderr=open(err_path, 'w'),
      )
    except subprocess.CalledProcessError as e:
      print(e, 'Please check {} and {}'.format(out_path, err_path))
      raise
    openweb_dir = os.path.join(args.outdir, 'openwebtext')
    text_dir = os.path.join(args.outdir, 'txt')
    mkdir(text_dir)
    unzip_merge_txt(openweb_dir, text_dir, args.shard_num_processes)

  if args.shard:
    text_dir = os.path.join(args.outdir, 'txt')
    print('Sharding {} ...'.format(text_dir))
    dask_source_path = os.path.join(args.outdir, 'source')
    mkdir(dask_source_path)
    _shard_openwebs(
        text_dir,
        dask_source_path,
        args.num_shards,
        args.shard_num_processes,
    )
    print('Dask source prepared at {} !'.format(dask_source_path))


def console_script():
  main(attach_args().parse_args())

if __name__ == '__main__':
  console_script()
