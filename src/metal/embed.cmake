# Turn kernels.metal into a C++ header holding the source as a raw string literal.
file(READ ${IN} SRC)
file(WRITE ${OUT} "// generated from kernels.metal, do not edit\n#pragma once\nstatic const char *kAtomKernelSource = R\"ATOMMTL(\n${SRC}\n)ATOMMTL\";\n")
