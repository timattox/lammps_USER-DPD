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

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "fix_ave_histo.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "group.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"

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

FixAveHisto::FixAveHistoKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixAveHisto(lmp, narg, arg)
{
}

/* ---------------------------------------------------------------------- */

FixAveHistoKokkos::~FixAveHistoKokkos()
{
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   bin a single value
------------------------------------------------------------------------- */

void FixAveHisto::bin_one(double value)
{
  stats[2] = MIN(stats[2],value);
  stats[3] = MAX(stats[3],value);

  if (value < lo) {
    if (beyond == IGNORE) {
      stats[1] += 1.0;
      return;
    } else bin[0] += 1.0;
  } else if (value > hi) {
    if (beyond == IGNORE) {
      stats[1] += 1.0;
      return;
    } else bin[nbins-1] += 1.0;
  } else {
    int ibin = static_cast<int> ((value-lo)*bininv);
    ibin = MIN(ibin,nbins-1);
    if (beyond == EXTRA) ibin++;
    bin[ibin] += 1.0;
  }

  stats[0] += 1.0;
}

/* the main conflict here is between the fact that
   we can't have variable-sized reduction value types
   and the fact that doing one bin at a time would slow
   this down considerably compared to the non-Kokkos version
   in serial.
   as a compromise, we'll process a compile-time-defined
   number of bins at a time, and if there are more then we'll
   have to repeat this process but if there are fewer
   then our runtime should be similar */

constexpr int block_size = 16;

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
  using value_type = BinOut;
  KOKKOS_INLINE_FUNCTION init(value_type& update) const {
    update.minval = BIG;
    update.maxval = -BIG;
    update.below = 0.0;
    update.above = 0.0;
    for (int j = 0; j < block_size; ++j) {
      update.bins[j] = 0.0;
    }
    update.total = 0.0;
  }
  KOKKOS_INLINE_FUNCTION join(volatile value_type& update,
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
    out.maxval = MAX(out.minval, value);
    if (value < lo) {
      out.below += 1.0;
    } else if (value > hi) {
      out.above += 1.0;
    } else {
      double rbin = (value - lo) / bininv;
      int ibin = static_cast<int>(rbin);
      ibin = MIN(ibin, block_size - 1);
      out.bins[ibin] += 1.0;
    }
    out.total += 1.0;
  }
};

struct BinVectorFunctor : public BinFunctor {
  BinVectorFunctor(int a, double b, double c, DView d):BinFunctor(a, b, c, d) {}
  KOKKOS_INLINE_FUNCTION void operator()(int i, BinFunctor& out) const {
    BinFunctor::bin_one(i, out);
  }
};

using Mask = LAMMPS_NS::DAT::tdual_int_1d;

struct BinAtomsFunctor : public BinFunctor {
  Mask mask;
  int groupbit;
  BinAtomsFunctor(int a, double b, double c, DView d,
      Mask mask_in, int groupbit_in):
      BinFunctor(a, b, c, d) {
    mask = mask_in;
    groupbit = groupbit_in;
  }
  KOKKOS_INLINE_FUNCTION void operator()(int i, BinFunctor& out) const {
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

static void bin_vector_kokkos(int n, double* values, int stride) {
  DView d_values;
  auto view_size = (n - 1) * stride + 1;
  HView h_values(values, view_size);
#ifdef KOKKOS_HAVE_CUDA
  d_values = DView(view_size);
  Kokkos::deep_copy(d_values, h_values);
#else
  d_values = h_values;
#endif
}

/* ----------------------------------------------------------------------
   bin a vector of values with stride
------------------------------------------------------------------------- */

void FixAveHisto::bin_vector(int n, double *values, int stride)
{
  int m = 0;
  for (int i = 0; i < n; i++) {
    bin_one(values[m]);
    m += stride;
  }
}

/* ----------------------------------------------------------------------
   bin a per-atom vector of values with stride
   only bin if atom is in group
------------------------------------------------------------------------- */

void FixAveHisto::bin_atoms(double *values, int stride)
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  int m = 0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) bin_one(values[m]);
    m += stride;
  }
}

/* ----------------------------------------------------------------------
   parse optional args
------------------------------------------------------------------------- */

void FixAveHisto::options(int iarg, int narg, char **arg)
{
  // option defaults

  fp = NULL;
  ave = ONE;
  startstep = 0;
  mode = SCALAR;
  beyond = IGNORE;
  overwrite = 0;
  title1 = NULL;
  title2 = NULL;
  title3 = NULL;

  // optional args

  while (iarg < narg) {
    if (strcmp(arg[iarg],"file") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/histo command");
      if (me == 0) {
        fp = fopen(arg[iarg+1],"w");
        if (fp == NULL) {
          char str[128];
          sprintf(str,"Cannot open fix ave/histo file %s",arg[iarg+1]);
          error->one(FLERR,str);
        }
      }
      iarg += 2;
    } else if (strcmp(arg[iarg],"ave") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/histo command");
      if (strcmp(arg[iarg+1],"one") == 0) ave = ONE;
      else if (strcmp(arg[iarg+1],"running") == 0) ave = RUNNING;
      else if (strcmp(arg[iarg+1],"window") == 0) ave = WINDOW;
      else error->all(FLERR,"Illegal fix ave/histo command");
      if (ave == WINDOW) {
        if (iarg+3 > narg) error->all(FLERR,"Illegal fix ave/histo command");
        nwindow = force->inumeric(FLERR,arg[iarg+2]);
        if (nwindow <= 0) error->all(FLERR,"Illegal fix ave/histo command");
      }
      iarg += 2;
      if (ave == WINDOW) iarg++;
    } else if (strcmp(arg[iarg],"start") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/histo command");
      startstep = force->inumeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"mode") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/histo command");
      if (strcmp(arg[iarg+1],"scalar") == 0) mode = SCALAR;
      else if (strcmp(arg[iarg+1],"vector") == 0) mode = VECTOR;
      else error->all(FLERR,"Illegal fix ave/histo command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"beyond") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/histo command");
      if (strcmp(arg[iarg+1],"ignore") == 0) beyond = IGNORE;
      else if (strcmp(arg[iarg+1],"end") == 0) beyond = END;
      else if (strcmp(arg[iarg+1],"extra") == 0) beyond = EXTRA;
      else error->all(FLERR,"Illegal fix ave/histo command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"overwrite") == 0) {
      overwrite = 1;
      iarg += 1;
    } else if (strcmp(arg[iarg],"title1") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/histo command");
      delete [] title1;
      int n = strlen(arg[iarg+1]) + 1;
      title1 = new char[n];
      strcpy(title1,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"title2") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/histo command");
      delete [] title2;
      int n = strlen(arg[iarg+1]) + 1;
      title2 = new char[n];
      strcpy(title2,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"title3") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix ave/histo command");
      delete [] title3;
      int n = strlen(arg[iarg+1]) + 1;
      title3 = new char[n];
      strcpy(title3,arg[iarg+1]);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix ave/histo command");
  }
}

/* ----------------------------------------------------------------------
   calculate nvalid = next step on which end_of_step does something
   can be this timestep if multiple of nfreq and nrepeat = 1
   else backup from next multiple of nfreq
   startstep is lower bound on nfreq multiple
------------------------------------------------------------------------- */

bigint FixAveHisto::nextvalid()
{
  bigint nvalid = (update->ntimestep/nfreq)*nfreq + nfreq;
  while (nvalid < startstep) nvalid += nfreq;
  if (nvalid-nfreq == update->ntimestep && nrepeat == 1)
    nvalid = update->ntimestep;
  else
    nvalid -= (nrepeat-1)*nevery;
  if (nvalid < update->ntimestep) nvalid += nfreq;
  return nvalid;
}

