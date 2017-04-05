/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Dan Ibanez (SNL)
------------------------------------------------------------------------- */

#include "fix_ave_histo_kokkos.h"
#include "kokkos_type.h"
#include "atom.h"
#include "atom_kokkos.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{X,V,F,COMPUTE,FIX,VARIABLE};
enum{ONE,RUNNING};
enum{SCALAR,VECTOR,WINDOW};
enum{GLOBAL,PERATOM,LOCAL};
enum{IGNORE,END,EXTRA};

#define INVOKED_SCALAR 1
#define INVOKED_VECTOR 2
#define INVOKED_ARRAY 4
#define INVOKED_PERATOM 8
#define INVOKED_LOCAL 16

#define BIG 1.0e20
/* ---------------------------------------------------------------------- */

FixAveHistoKokkos::FixAveHistoKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixAveHisto(lmp, narg, arg)
{
}

/* ---------------------------------------------------------------------- */

FixAveHistoKokkos::~FixAveHistoKokkos()
{
}

/* ---------------------------------------------------------------------- */

/* the main conflict here is between the fact that
   we can't have variable-sized reduction value types
   and the fact that doing one bin at a time would slow
   this down considerably compared to the non-Kokkos version
   in serial.
   as a compromise, we'll process a compile-time-defined
   number of bins at a time, and if there are more then we'll
   have to repeat this process but if there are fewer
   then our runtime should be similar */

/* shamelessly setting this to the number of bins used for the
   Gordon Bell attempt */

constexpr int block_size = 100;

struct BinOut {
  double minval;
  double maxval;
  double below;
  double above;
  double bins[block_size];
  double total;
};

using DView = Kokkos::View<double*, LMPDeviceType>;
using HView = typename DView::HostMirror;

struct BinFunctor {
  using value_type = BinOut;
  int stride;
  double lo;
  double hi;
  double bininv;
  DView view;
  BinFunctor(int stride_in, double lo_in, double hi_in, DView view_in) {
    stride = stride_in,
    lo = lo_in;
    hi = hi_in;
    bininv = double(block_size) / (hi - lo);
    view = view_in;
  }
  KOKKOS_INLINE_FUNCTION void init(value_type& update) const {
    update.minval = BIG;
    update.maxval = -BIG;
    update.below = 0.0;
    update.above = 0.0;
    for (int j = 0; j < block_size; ++j) {
      update.bins[j] = 0.0;
    }
    update.total = 0.0;
  }
  KOKKOS_INLINE_FUNCTION void join(volatile value_type& update,
      const volatile value_type& input) const {
    update.minval = MIN(update.minval, input.minval);
    update.maxval = MAX(update.maxval, input.maxval);
    update.below += input.below;
    update.above += input.above;
    for (int j = 0; j < block_size; ++j) {
      update.bins[j] += input.bins[j];
    }
    update.total += input.total;
  }
  KOKKOS_INLINE_FUNCTION void bin_one(int i, BinOut& out) const {
    double value = view(i * stride); 
    out.minval = MIN(out.minval, value);
    out.maxval = MAX(out.maxval, value);
    if (value < lo) {
      out.below += 1.0;
    } else if (value > hi) {
      out.above += 1.0;
    } else {
      double rbin = (value - lo) * bininv;
      int ibin = static_cast<int>(rbin);
      ibin = MIN(ibin, block_size - 1);
      out.bins[ibin] += 1.0;
    }
    out.total += 1.0;
  }
};

struct BinVectorFunctor : public BinFunctor {
  using value_type = BinFunctor::value_type;
  BinVectorFunctor(int a, double b, double c, DView d):BinFunctor(a, b, c, d) {}
  KOKKOS_INLINE_FUNCTION void operator()(int i, BinOut& out) const {
    BinFunctor::bin_one(i, out);
  }
};

using Mask = ::DAT::t_int_1d;

struct BinAtomsFunctor : public BinFunctor {
  using value_type = BinFunctor::value_type;
  Mask mask;
  int groupbit;
  BinAtomsFunctor(int a, double b, double c, DView d,
      Mask mask_in, int groupbit_in):
      BinFunctor(a, b, c, d) {
    mask = mask_in;
    groupbit = groupbit_in;
  }
  KOKKOS_INLINE_FUNCTION void operator()(int i, BinOut& out) const {
    if (mask[i] & groupbit) BinFunctor::bin_one(i, out);
  }
};

static BinOut bin_block(int n, int stride, double lo, double hi,
    DView view, bool has_mask, Mask mask, int groupbit) {
  BinOut out;
  if (has_mask) {
    Kokkos::parallel_reduce(n, BinAtomsFunctor(stride, lo, hi, view, mask, groupbit), out);
  } else  {
    Kokkos::parallel_reduce(n, BinVectorFunctor(stride, lo, hi, view), out);
  }
  return out;
}

static void bin_any(int n, int stride, double lo, double hi,
    DView view, bool has_mask, Mask mask, int groupbit,
    int beyond, int nbins, double* bins, double* stats) {
  auto actual_nbins = nbins;
  if (beyond == EXTRA) actual_nbins -= 2;
  auto binsize = (hi - lo) / actual_nbins;
  double block_lo = lo;
  for (int ndone = 0; ndone < actual_nbins; ndone += block_size) {
    auto block_hi = block_lo + (block_size * binsize);
    if (ndone + block_size == actual_nbins) {
      block_hi = hi;
    }
    auto block_out = bin_block(n, stride, block_lo, block_hi,
        view, has_mask, mask, groupbit);
    stats[0] = block_out.total;
    stats[2] = block_out.minval;
    stats[3] = block_out.maxval;
    for (int block_bin = 0; block_bin < block_size; ++block_bin) {
      auto bin = block_bin + ndone;
      if (bin < actual_nbins) {
        if (beyond == EXTRA) ++bin;
        bins[bin] = block_out.bins[block_bin];
      } else {
        block_out.above += block_out.bins[block_bin];
      }
    }
    if (ndone == 0) {
      if (beyond == IGNORE) {
        stats[1] += block_out.below;
      } else {
        bins[0] += block_out.below;
      }
    }
    if (ndone + block_size >= actual_nbins) {
      if (beyond == IGNORE) {
        stats[1] += block_out.above;
      } else {
        bins[nbins-1] += block_out.above;
      }
    }
    block_lo = block_hi;
  }
}

static void bin_any(int n, int stride, double lo, double hi,
    double* values, bool has_mask, Mask mask, int groupbit,
    int beyond, int nbins, double* bins, double* stats) {
  DView d_values;
  auto view_size = (n - 1) * stride + 1;
  HView h_values(values, view_size);
#ifdef KOKKOS_HAVE_CUDA
  d_values = DView(view_size);
  Kokkos::deep_copy(d_values, h_values);
#else
  d_values = h_values;
#endif
  bin_any(n, stride, lo, hi, d_values, has_mask, mask, groupbit,
      beyond, nbins, bins, stats);
}

/* ----------------------------------------------------------------------
   bin a vector of values with stride
------------------------------------------------------------------------- */

void FixAveHistoKokkos::bin_vector(int n, double *values, int stride)
{
  bin_any(n, stride, lo, hi, values, false, Mask(), -1,
      beyond, nbins, bin, stats);
}

/* ----------------------------------------------------------------------
   bin a per-atom vector of values with stride
   only bin if atom is in group
------------------------------------------------------------------------- */

void FixAveHistoKokkos::bin_atoms(double *values, int stride)
{
  int nlocal = atom->nlocal;
  auto atomKK = dynamic_cast<AtomKokkos*>(atom);
  atomKK->k_mask.sync<LMPDeviceType>();
  auto mask = atomKK->k_mask.view<LMPDeviceType>();

  bin_any(nlocal, stride, lo, hi, values, true, mask, groupbit,
      beyond, nbins, bin, stats);
}
