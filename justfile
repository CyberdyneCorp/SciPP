# SciPP developer tasks — run `just` (or `just --list`) to see everything.
# Requires: cmake >= 3.25, clang++ or g++, ninja (for the gcc/asan recipes),
# a pinned NumPP install under .deps/numpp (see `just bootstrap`), and
# python3 + scipy (only for `just oracle`, to regenerate the frozen golden data).

build_dir := "build"
cxx       := "clang++"
numpp_src := "../NumPP"
numpp_lib := ".deps/numpp/lib"

# Show the available recipes (default).
default:
    @just --list

# Build + install the pinned NumPP dependency into .deps/numpp so that
# find_package(NumPP) resolves it. Pass a source path to override ../NumPP.
bootstrap src=numpp_src:
    ./scripts/bootstrap_numpp.sh {{src}}

# Configure the CPU-only Release build. Pass extra cmake flags, e.g.
#   just configure -DSCIPP_WITH_CUDA=ON
configure *FLAGS:
    cmake -S . -B {{build_dir}} -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER={{cxx}} {{FLAGS}}

# Compile the library and the test suite.
build: configure
    cmake --build {{build_dir}} -j

# Compile only the library.
lib: configure
    cmake --build {{build_dir}} --target scipp -j

# Run the SciPy-oracle test suite (against the frozen golden data).
test: build
    LD_LIBRARY_PATH={{numpp_lib}} ./{{build_dir}}/tests/scipp_tests
alias unit := test

# Run the test suite through CTest.
ctest: build
    ctest --test-dir {{build_dir}} --output-on-failure

# Configure + build + test with debug symbols and assertions (separate dir).
debug:
    cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER={{cxx}}
    cmake --build build-debug -j
    LD_LIBRARY_PATH={{numpp_lib}} ./build-debug/tests/scipp_tests

# Build + test with GCC (separate build dir).
gcc:
    cmake -S . -B build-gcc -G Ninja -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release
    cmake --build build-gcc -j
    LD_LIBRARY_PATH={{numpp_lib}} ./build-gcc/tests/scipp_tests

# Build + test under AddressSanitizer / UBSan (separate build dir).
asan:
    cmake -S . -B build-asan -G Ninja -DCMAKE_CXX_COMPILER={{cxx}} -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
      -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
    cmake --build build-asan -j
    LD_LIBRARY_PATH={{numpp_lib}} ./build-asan/tests/scipp_tests

# Regenerate the frozen oracle golden data (requires python3 + scipy).
oracle:
    python3 tests/oracle/generate.py

# Validate all OpenSpec specs and changes.
spec:
    openspec validate --all --strict

# Full local CI: clang tests + gcc + spec.
ci: test gcc spec

# Remove all build directories.
clean:
    rm -rf {{build_dir}} build-debug build-gcc build-asan
