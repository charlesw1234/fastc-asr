# -*- python -*-
from pathlib import Path

with_kaldi2 = True

env = Environment(CCFLAGS = ['-g', '-Wall'], CPPPATH = str(Path('srcs')), PROGSUFFIX = '.exe',
                  LIBS = ['sndfile', 'fftw3f', 'blas', 'crypto', 'm'])
srcs = [*Path('srcs').glob('*.c')]
if with_kaldi2: srcs.extend(Path('srcs', 'kaldi2').glob('*.c'))
for test_main in Path('tests').glob('*.c'):
    env.Program(str(test_main.parent.joinpath(test_main.stem)),
                [str(test_main), *map(lambda src_path: str(src_path), srcs)])
