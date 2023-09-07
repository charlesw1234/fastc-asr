# -*- python -*-
from pathlib import Path

env = Environment(CCFLAGS = ['-g', '-Wall'], CPPPATH = str(Path('srcs')), PROGSUFFIX = '.elf',
                  LIBS = ['sndfile', 'fftw3f', 'blas', 'crypto', 'm'])

srcs_c = [*Path('srcs.c').glob('*.c')]
srcs_c.extend(Path('srcs.c', 'kaldi2').glob('*.c'))
for test_main in Path('tests').glob('*.c'):
    env.Program(str(test_main.parent.joinpath(test_main.stem)),
                [str(test_main), *map(lambda src_path: str(src_path), srcs_c)],
                CPPPATH = str(Path('srcs.c')))

srcs_cxx = [*Path('srcs.cxx').glob('*.cxx'), *Path('srcs.cxx').glob('*.c')]
srcs_cxx.extend(Path('srcs.cxx', 'kaldi2').glob('*.cxx'))
for test_main in Path('tests').glob('*.cxx'):
    env.Program(str(test_main.parent.joinpath(test_main.stem)),
                [str(test_main), *map(lambda src_path: str(src_path), srcs_cxx)],
                CPPPATH = str(Path('srcs.cxx')))
