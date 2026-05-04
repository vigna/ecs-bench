# Makefile for the BuRR-VL + coding benchmark (bench_csf).
#
# Uses BuRR-VL headers from the BuRR-VL/ submodule and coding headers
# from LearnedStaticFunction/include/.  The BuRR-VL library needs tlx
# and ips2ra (pulled in as submodules of BuRR-VL/).

CXX = clang++

BURR_VL = BuRR-VL
LSF = LearnedStaticFunction

INCLUDES = -I$(BURR_VL) \
           -I$(BURR_VL)/tlx \
           -isystem $(BURR_VL)/ips2ra/include \
           -I$(BURR_VL)/DySECT \
           -I$(LSF)/include

CFLAGS = -std=c++20 -march=native -D_REENTRANT $(INCLUDES)
OPTFLAGS = -O3 -DNDEBUG -march=native -fPIE

LDFLAGS = $(BURR_VL)/sorter.o $(BURR_VL)/tlx/build/tlx/libtlx.a \
          -lpthread -ltbb -latomic

all: bench_csf

# Initialize submodules without pulling LearnedStaticFunction's nested
# tensorflow (4GB+).  Use this instead of git clone --recurse-submodules.
init:
	git submodule update --init --recursive BuRR BuRR-VL sux-rs
	git submodule update --init LearnedStaticFunction

# BuRR-VL dependencies: sorter object and tlx library
$(BURR_VL)/sorter.o: $(BURR_VL)/sorter.cpp $(BURR_VL)/sorter.hpp
	$(CXX) $(OPTFLAGS) $(CFLAGS) -Wno-unused-function -c -o $@ $<

$(BURR_VL)/tlx/build/tlx/libtlx.a:
	@mkdir -p $(BURR_VL)/tlx/build
	cd $(BURR_VL)/tlx/build && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-g0 -G Ninja .. && ninja -j4

bench_csf: $(BURR_VL)/sorter.o bench_csf.cpp $(BURR_VL)/tlx/build/tlx/libtlx.a
	$(CXX) $(OPTFLAGS) $(CFLAGS) -o $@ bench_csf.cpp $(LDFLAGS)

clean:
	rm -f bench_csf $(BURR_VL)/sorter.o
	rm -rf $(BURR_VL)/tlx/build
	rm -f BuRR/bench[0-9]* BuRR/ribbon[0-9]* BuRR/debug[0-9]* \
	      BuRR/parbench[0-9]* BuRR/parrun[0-9]* BuRR/pardbg[0-9]* \
	      BuRR/sorter.o BuRR/tests
	rm -rf BuRR/tlx/build
	cd sux-rs && cargo clean 2>/dev/null || true

.PHONY: all init clean
