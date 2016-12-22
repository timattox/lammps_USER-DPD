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
   Contributing author: Stan Moore (SNL)
------------------------------------------------------------------------- */

#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "pair_table_rx_kokkos.h"
#include "kokkos.h"
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include "atom_masks.h"

using namespace LAMMPS_NS;

enum{NONE,RLINEAR,RSQ,BMP};
enum{FULL,HALFTHREAD,HALF};

#define MAXLINE 1024

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairTableRXKokkos<DeviceType>::PairTableRXKokkos(LAMMPS *lmp) : PairTableRX(lmp)
{
  update_table = 0;
  atomKK = (AtomKokkos *) atom;
  ntables = 0;
  tables = NULL;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
  h_table = new TableHost();
  d_table = new TableDevice();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairTableRXKokkos<DeviceType>::~PairTableRXKokkos()
{
/*  for (int m = 0; m < ntables; m++) free_table(&tables[m]);
  memory->sfree(tables);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(tabindex);
  }*/
  delete h_table;
  delete d_table;

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairTableRXKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  if(update_table)
    create_kokkos_tables();
  if(tabstyle == LOOKUP)
    compute_style<LOOKUP>(eflag_in,vflag_in);
  if(tabstyle == LINEAR)
    compute_style<LINEAR>(eflag_in,vflag_in);
  if(tabstyle == SPLINE)
    compute_style<SPLINE>(eflag_in,vflag_in);
  if(tabstyle == BITMAP)
    compute_style<BITMAP>(eflag_in,vflag_in);
}

template<class DeviceType>
template<int TABSTYLE>
void PairTableRXKokkos<DeviceType>::compute_style(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;


  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  atomKK->sync(execution_space,datamask_read);
  //k_cutsq.template sync<DeviceType>();
  //k_params.template sync<DeviceType>();
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);

  x = c_x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  special_lj[0] = force->special_lj[0];
  special_lj[1] = force->special_lj[1];
  special_lj[2] = force->special_lj[2];
  special_lj[3] = force->special_lj[3];
  newton_pair = force->newton_pair;
  d_cutsq = d_table->cutsq;
  // loop over neighbors of my atoms

  EV_FLOAT ev;
  if(atom->ntypes > MAX_TYPES_STACKPARAMS) {
    if (neighflag == FULL) {
      PairComputeFunctor<PairTableRXKokkos<DeviceType>,FULL,false,S_TableRXCompute<DeviceType,TABSTYLE> >
        ff(this,(NeighListKokkos<DeviceType>*) list);
      if (eflag || vflag) Kokkos::parallel_reduce(list->inum,ff,ev);
      else Kokkos::parallel_for(list->inum,ff);
    } else if (neighflag == HALFTHREAD) {
      PairComputeFunctor<PairTableRXKokkos<DeviceType>,HALFTHREAD,false,S_TableRXCompute<DeviceType,TABSTYLE> >
        ff(this,(NeighListKokkos<DeviceType>*) list);
      if (eflag || vflag) Kokkos::parallel_reduce(list->inum,ff,ev);
      else Kokkos::parallel_for(list->inum,ff);
    } else if (neighflag == HALF) {
      PairComputeFunctor<PairTableRXKokkos<DeviceType>,HALF,false,S_TableRXCompute<DeviceType,TABSTYLE> >
        f(this,(NeighListKokkos<DeviceType>*) list);
      if (eflag || vflag) Kokkos::parallel_reduce(list->inum,f,ev);
      else Kokkos::parallel_for(list->inum,f);
    } else if (neighflag == N2) {
      PairComputeFunctor<PairTableRXKokkos<DeviceType>,N2,false,S_TableRXCompute<DeviceType,TABSTYLE> >
        f(this,(NeighListKokkos<DeviceType>*) list);
      if (eflag || vflag) Kokkos::parallel_reduce(nlocal,f,ev);
      else Kokkos::parallel_for(nlocal,f);
    }
  } else {
    if (neighflag == FULL) {
      PairComputeFunctor<PairTableRXKokkos<DeviceType>,FULL,true,S_TableRXCompute<DeviceType,TABSTYLE> >
        f(this,(NeighListKokkos<DeviceType>*) list);
      if (eflag || vflag) Kokkos::parallel_reduce(list->inum,f,ev);
      else Kokkos::parallel_for(list->inum,f);
    } else if (neighflag == HALFTHREAD) {
      PairComputeFunctor<PairTableRXKokkos<DeviceType>,HALFTHREAD,true,S_TableRXCompute<DeviceType,TABSTYLE> >
        f(this,(NeighListKokkos<DeviceType>*) list);
      if (eflag || vflag) Kokkos::parallel_reduce(list->inum,f,ev);
      else Kokkos::parallel_for(list->inum,f);
    } else if (neighflag == HALF) {
      PairComputeFunctor<PairTableRXKokkos<DeviceType>,HALF,true,S_TableRXCompute<DeviceType,TABSTYLE> >
        f(this,(NeighListKokkos<DeviceType>*) list);
      if (eflag || vflag) Kokkos::parallel_reduce(list->inum,f,ev);
      else Kokkos::parallel_for(list->inum,f);
    } else if (neighflag == N2) {
      PairComputeFunctor<PairTableRXKokkos<DeviceType>,N2,true,S_TableRXCompute<DeviceType,TABSTYLE> >
        f(this,(NeighListKokkos<DeviceType>*) list);
      if (eflag || vflag) Kokkos::parallel_reduce(nlocal,f,ev);
      else Kokkos::parallel_for(nlocal,f);
    }
  }

  if (eflag) eng_vdwl += ev.evdwl;
  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);
}

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairTableRXKokkos<DeviceType>::
compute_fpair(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const {
  (void) i;
  (void) j;
  union_int_float_t rsq_lookup;
  double fpair;
  const int tidx = d_table_const.tabindex(itype,jtype);
  //const Table* const tb = &tables[tabindex[itype][jtype]];

  //if (rsq < d_table_const.innersq(tidx))
  //  error->one(FLERR,"Pair distance < table inner cutoff");


  if (Specialisation::TabStyle == LOOKUP) {
    const int itable = static_cast<int> ((rsq - d_table_const.innersq(tidx)) * d_table_const.invdelta(tidx));
    //if (itable >= tlm1)
    //  error->one(FLERR,"Pair distance > table outer cutoff");
    fpair = d_table_const.f(tidx,itable);
  } else if (Specialisation::TabStyle == LINEAR) {
    const int itable = static_cast<int> ((rsq - d_table_const.innersq(tidx)) * d_table_const.invdelta(tidx));
    //if (itable >= tlm1)
    //  error->one(FLERR,"Pair distance > table outer cutoff");
    const double fraction = (rsq - d_table_const.rsq(tidx,itable)) * d_table_const.invdelta(tidx);
    fpair = d_table_const.f(tidx,itable) + fraction*d_table_const.df(tidx,itable);
  } else if (Specialisation::TabStyle == SPLINE) {
    const int itable = static_cast<int> ((rsq - d_table_const.innersq(tidx)) * d_table_const.invdelta(tidx));
    //if (itable >= tlm1)
    //  error->one(FLERR,"Pair distance > table outer cutoff");
    const double b = (rsq - d_table_const.rsq(tidx,itable)) * d_table_const.invdelta(tidx);
    const double a = 1.0 - b;
    fpair = a * d_table_const.f(tidx,itable) + b * d_table_const.f(tidx,itable+1) +
      ((a*a*a-a)*d_table_const.f2(tidx,itable) + (b*b*b-b)*d_table_const.f2(tidx,itable+1)) *
      d_table_const.deltasq6(tidx);
  } else {
    rsq_lookup.f = rsq;
    int itable = rsq_lookup.i & d_table_const.nmask(tidx);
    itable >>= d_table_const.nshiftbits(tidx);
    const double fraction = (rsq_lookup.f - d_table_const.rsq(tidx,itable)) * d_table_const.drsq(tidx,itable);
    fpair = d_table_const.f(tidx,itable) + fraction*d_table_const.df(tidx,itable);
  }
  return fpair;
}

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairTableRXKokkos<DeviceType>::
compute_evdwl(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const {
  (void) i;
  (void) j;
  double evdwl;
  union_int_float_t rsq_lookup;
  const int tidx = d_table_const.tabindex(itype,jtype);
  //const Table* const tb = &tables[tabindex[itype][jtype]];

  //if (rsq < d_table_const.innersq(tidx))
  //  error->one(FLERR,"Pair distance < table inner cutoff");

  if (Specialisation::TabStyle == LOOKUP) {
    const int itable = static_cast<int> ((rsq - d_table_const.innersq(tidx)) * d_table_const.invdelta(tidx));
    //if (itable >= tlm1)
    //  error->one(FLERR,"Pair distance > table outer cutoff");
    evdwl = d_table_const.e(tidx,itable);
  } else if (Specialisation::TabStyle == LINEAR) {
    const int itable = static_cast<int> ((rsq - d_table_const.innersq(tidx)) * d_table_const.invdelta(tidx));
    //if (itable >= tlm1)
    //  error->one(FLERR,"Pair distance > table outer cutoff");
    const double fraction = (rsq - d_table_const.rsq(tidx,itable)) * d_table_const.invdelta(tidx);
    evdwl = d_table_const.e(tidx,itable) + fraction*d_table_const.de(tidx,itable);
  } else if (Specialisation::TabStyle == SPLINE) {
    const int itable = static_cast<int> ((rsq - d_table_const.innersq(tidx)) * d_table_const.invdelta(tidx));
    //if (itable >= tlm1)
    //  error->one(FLERR,"Pair distance > table outer cutoff");
    const double b = (rsq - d_table_const.rsq(tidx,itable)) * d_table_const.invdelta(tidx);
    const double a = 1.0 - b;
    evdwl = a * d_table_const.e(tidx,itable) + b * d_table_const.e(tidx,itable+1) +
        ((a*a*a-a)*d_table_const.e2(tidx,itable) + (b*b*b-b)*d_table_const.e2(tidx,itable+1)) *
        d_table_const.deltasq6(tidx);
  } else {
    rsq_lookup.f = rsq;
    int itable = rsq_lookup.i & d_table_const.nmask(tidx);
    itable >>= d_table_const.nshiftbits(tidx);
    const double fraction = (rsq_lookup.f - d_table_const.rsq(tidx,itable)) * d_table_const.drsq(tidx,itable);
    evdwl = d_table_const.e(tidx,itable) + fraction*d_table_const.de(tidx,itable);
  }
  return evdwl;
}

template<class DeviceType>
void PairTableRXKokkos<DeviceType>::create_kokkos_tables()
{
  const int tlm1 = tablength-1;

  memory->create_kokkos(d_table->nshiftbits,h_table->nshiftbits,ntables,"Table::nshiftbits");
  memory->create_kokkos(d_table->nmask,h_table->nmask,ntables,"Table::nmask");
  memory->create_kokkos(d_table->innersq,h_table->innersq,ntables,"Table::innersq");
  memory->create_kokkos(d_table->invdelta,h_table->invdelta,ntables,"Table::invdelta");
  memory->create_kokkos(d_table->deltasq6,h_table->deltasq6,ntables,"Table::deltasq6");

  if(tabstyle == LOOKUP) {
    memory->create_kokkos(d_table->e,h_table->e,ntables,tlm1,"Table::e");
    memory->create_kokkos(d_table->f,h_table->f,ntables,tlm1,"Table::f");
  }

  if(tabstyle == LINEAR) {
    memory->create_kokkos(d_table->rsq,h_table->rsq,ntables,tablength,"Table::rsq");
    memory->create_kokkos(d_table->e,h_table->e,ntables,tablength,"Table::e");
    memory->create_kokkos(d_table->f,h_table->f,ntables,tablength,"Table::f");
    memory->create_kokkos(d_table->de,h_table->de,ntables,tlm1,"Table::de");
    memory->create_kokkos(d_table->df,h_table->df,ntables,tlm1,"Table::df");
  }

  if(tabstyle == SPLINE) {
    memory->create_kokkos(d_table->rsq,h_table->rsq,ntables,tablength,"Table::rsq");
    memory->create_kokkos(d_table->e,h_table->e,ntables,tablength,"Table::e");
    memory->create_kokkos(d_table->f,h_table->f,ntables,tablength,"Table::f");
    memory->create_kokkos(d_table->e2,h_table->e2,ntables,tablength,"Table::e2");
    memory->create_kokkos(d_table->f2,h_table->f2,ntables,tablength,"Table::f2");
  }

  if(tabstyle == BITMAP) {
    int ntable = 1 << tablength;
    memory->create_kokkos(d_table->rsq,h_table->rsq,ntables,ntable,"Table::rsq");
    memory->create_kokkos(d_table->e,h_table->e,ntables,ntable,"Table::e");
    memory->create_kokkos(d_table->f,h_table->f,ntables,ntable,"Table::f");
    memory->create_kokkos(d_table->de,h_table->de,ntables,ntable,"Table::de");
    memory->create_kokkos(d_table->df,h_table->df,ntables,ntable,"Table::df");
    memory->create_kokkos(d_table->drsq,h_table->drsq,ntables,ntable,"Table::drsq");
  }

  for(int i=0; i < ntables; i++) {
    Table* tb = &tables[i];

    h_table->nshiftbits[i] = tb->nshiftbits;
    h_table->nmask[i] = tb->nmask;
    h_table->innersq[i] = tb->innersq;
    h_table->invdelta[i] = tb->invdelta;
    h_table->deltasq6[i] = tb->deltasq6;

    for(int j = 0; j<h_table->rsq.dimension_1(); j++)
      h_table->rsq(i,j) = tb->rsq[j];
    for(int j = 0; j<h_table->drsq.dimension_1(); j++)
      h_table->drsq(i,j) = tb->drsq[j];
    for(int j = 0; j<h_table->e.dimension_1(); j++)
      h_table->e(i,j) = tb->e[j];
    for(int j = 0; j<h_table->de.dimension_1(); j++)
      h_table->de(i,j) = tb->de[j];
    for(int j = 0; j<h_table->f.dimension_1(); j++)
      h_table->f(i,j) = tb->f[j];
    for(int j = 0; j<h_table->df.dimension_1(); j++)
      h_table->df(i,j) = tb->df[j];
    for(int j = 0; j<h_table->e2.dimension_1(); j++)
      h_table->e2(i,j) = tb->e2[j];
    for(int j = 0; j<h_table->f2.dimension_1(); j++)
      h_table->f2(i,j) = tb->f2[j];
  }


  Kokkos::deep_copy(d_table->nshiftbits,h_table->nshiftbits);
  Kokkos::deep_copy(d_table->nmask,h_table->nmask);
  Kokkos::deep_copy(d_table->innersq,h_table->innersq);
  Kokkos::deep_copy(d_table->invdelta,h_table->invdelta);
  Kokkos::deep_copy(d_table->deltasq6,h_table->deltasq6);
  Kokkos::deep_copy(d_table->rsq,h_table->rsq);
  Kokkos::deep_copy(d_table->drsq,h_table->drsq);
  Kokkos::deep_copy(d_table->e,h_table->e);
  Kokkos::deep_copy(d_table->de,h_table->de);
  Kokkos::deep_copy(d_table->f,h_table->f);
  Kokkos::deep_copy(d_table->df,h_table->df);
  Kokkos::deep_copy(d_table->e2,h_table->e2);
  Kokkos::deep_copy(d_table->f2,h_table->f2);
  Kokkos::deep_copy(d_table->tabindex,h_table->tabindex);

  d_table_const.nshiftbits = d_table->nshiftbits;
  d_table_const.nmask = d_table->nmask;
  d_table_const.innersq = d_table->innersq;
  d_table_const.invdelta = d_table->invdelta;
  d_table_const.deltasq6 = d_table->deltasq6;
  d_table_const.rsq = d_table->rsq;
  d_table_const.drsq = d_table->drsq;
  d_table_const.e = d_table->e;
  d_table_const.de = d_table->de;
  d_table_const.f = d_table->f;
  d_table_const.df = d_table->df;
  d_table_const.e2 = d_table->e2;
  d_table_const.f2 = d_table->f2;


  Kokkos::deep_copy(d_table->cutsq,h_table->cutsq);
  update_table = 0;
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void PairTableRXKokkos<DeviceType>::allocate()
{
  allocated = 1;
  const int nt = atom->ntypes + 1;

  memory->create(setflag,nt,nt,"pair:setflag");
  memory->create_kokkos(d_table->cutsq,h_table->cutsq,cutsq,nt,nt,"pair:cutsq");
  memory->create_kokkos(d_table->tabindex,h_table->tabindex,tabindex,nt,nt,"pair:tabindex");

  d_table_const.cutsq = d_table->cutsq;
  d_table_const.tabindex = d_table->tabindex;
  memset(&setflag[0][0],0,nt*nt*sizeof(int));
  memset(&cutsq[0][0],0,nt*nt*sizeof(double));
  memset(&tabindex[0][0],0,nt*nt*sizeof(int));
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

template<class DeviceType>
void PairTableRXKokkos<DeviceType>::settings(int narg, char **arg)
{
  if (narg < 2) error->all(FLERR,"Illegal pair_style command");

  // new settings

  if (strcmp(arg[0],"lookup") == 0) tabstyle = LOOKUP;
  else if (strcmp(arg[0],"linear") == 0) tabstyle = LINEAR;
  else if (strcmp(arg[0],"spline") == 0) tabstyle = SPLINE;
  else if (strcmp(arg[0],"bitmap") == 0) tabstyle = BITMAP;
  else error->all(FLERR,"Unknown table style in pair_style command");

  tablength = force->inumeric(FLERR,arg[1]);
  if (tablength < 2) error->all(FLERR,"Illegal number of pair table entries");

  // optional keywords
  // assert the tabulation is compatible with a specific long-range solver

  int iarg = 2;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"ewald") == 0) ewaldflag = 1;
    else if (strcmp(arg[iarg],"pppm") == 0) pppmflag = 1;
    else if (strcmp(arg[iarg],"msm") == 0) msmflag = 1;
    else if (strcmp(arg[iarg],"dispersion") == 0) dispersionflag = 1;
    else if (strcmp(arg[iarg],"tip4p") == 0) tip4pflag = 1;
    else error->all(FLERR,"Illegal pair_style command");
    iarg++;
  }

  // delete old tables, since cannot just change settings

  for (int m = 0; m < ntables; m++) free_table(&tables[m]);
  memory->sfree(tables);

  if (allocated) {
    memory->destroy(setflag);

    d_table_const.tabindex = d_table->tabindex = typename ArrayTypes<DeviceType>::t_int_2d();
    h_table->tabindex = typename ArrayTypes<LMPHostType>::t_int_2d();

    d_table_const.cutsq = d_table->cutsq = typename ArrayTypes<DeviceType>::t_ffloat_2d();
    h_table->cutsq = typename ArrayTypes<LMPHostType>::t_ffloat_2d();
  }
  allocated = 0;

  ntables = 0;
  tables = NULL;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template<class DeviceType>
double PairTableRXKokkos<DeviceType>::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  tabindex[j][i] = tabindex[i][j];

  if(i<MAX_TYPES_STACKPARAMS+1 && j<MAX_TYPES_STACKPARAMS+1) {
    m_cutsq[j][i] = m_cutsq[i][j] = tables[tabindex[i][j]].cut*tables[tabindex[i][j]].cut;
  }

  return tables[tabindex[i][j]].cut;
}

template<class DeviceType>
void PairTableRXKokkos<DeviceType>::init_style()
{
  neighbor->request(this,instance_me);
  neighflag = lmp->kokkos->neighflag;
  int irequest = neighbor->nrequest - 1;

  neighbor->requests[irequest]->
    kokkos_host = Kokkos::Impl::is_same<DeviceType,LMPHostType>::value &&
    !Kokkos::Impl::is_same<DeviceType,LMPDeviceType>::value;
  neighbor->requests[irequest]->
    kokkos_device = Kokkos::Impl::is_same<DeviceType,LMPDeviceType>::value;

  if (neighflag == FULL) {
    neighbor->requests[irequest]->full = 1;
    neighbor->requests[irequest]->half = 0;
  } else if (neighflag == HALF || neighflag == HALFTHREAD) {
    neighbor->requests[irequest]->full = 0;
    neighbor->requests[irequest]->half = 1;
  } else if (neighflag == N2) {
    neighbor->requests[irequest]->full = 0;
    neighbor->requests[irequest]->half = 0;
  } else {
    error->all(FLERR,"Cannot use chosen neighbor list style with lj/cut/kk");
  }
}

/*
template <class DeviceType> template<int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION
void PairTableRXKokkos<DeviceType>::
ev_tally(EV_FLOAT &ev, const int &i, const int &j, const F_FLOAT &fpair,
         const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz) const
{
  const int EFLAG = eflag;
  const int NEWTON_PAIR = newton_pair;
  const int VFLAG = vflag_either;

  if (EFLAG) {
    if (eflag_atom) {
      E_FLOAT epairhalf = 0.5 * (ev.evdwl + ev.ecoul);
      if (NEWTON_PAIR || i < nlocal) eatom[i] += epairhalf;
      if (NEWTON_PAIR || j < nlocal) eatom[j] += epairhalf;
    }
  }

  if (VFLAG) {
    const E_FLOAT v0 = delx*delx*fpair;
    const E_FLOAT v1 = dely*dely*fpair;
    const E_FLOAT v2 = delz*delz*fpair;
    const E_FLOAT v3 = delx*dely*fpair;
    const E_FLOAT v4 = delx*delz*fpair;
    const E_FLOAT v5 = dely*delz*fpair;

    if (vflag_global) {
      if (NEIGHFLAG) {
        if (NEWTON_PAIR) {
          ev.v[0] += v0;
          ev.v[1] += v1;
          ev.v[2] += v2;
          ev.v[3] += v3;
          ev.v[4] += v4;
          ev.v[5] += v5;
        } else {
          if (i < nlocal) {
            ev.v[0] += 0.5*v0;
            ev.v[1] += 0.5*v1;
            ev.v[2] += 0.5*v2;
            ev.v[3] += 0.5*v3;
            ev.v[4] += 0.5*v4;
            ev.v[5] += 0.5*v5;
          }
          if (j < nlocal) {
            ev.v[0] += 0.5*v0;
            ev.v[1] += 0.5*v1;
            ev.v[2] += 0.5*v2;
            ev.v[3] += 0.5*v3;
            ev.v[4] += 0.5*v4;
            ev.v[5] += 0.5*v5;
          }
        }
      } else {
        ev.v[0] += 0.5*v0;
        ev.v[1] += 0.5*v1;
        ev.v[2] += 0.5*v2;
        ev.v[3] += 0.5*v3;
        ev.v[4] += 0.5*v4;
        ev.v[5] += 0.5*v5;
      }
    }

    if (vflag_atom) {
      if (NEWTON_PAIR || i < nlocal) {
        d_vatom(i,0) += 0.5*v0;
        d_vatom(i,1) += 0.5*v1;
        d_vatom(i,2) += 0.5*v2;
        d_vatom(i,3) += 0.5*v3;
        d_vatom(i,4) += 0.5*v4;
        d_vatom(i,5) += 0.5*v5;
      }
      if (NEWTON_PAIR || (NEIGHFLAG && j < nlocal)) {
        d_vatom(j,0) += 0.5*v0;
        d_vatom(j,1) += 0.5*v1;
        d_vatom(j,2) += 0.5*v2;
        d_vatom(j,3) += 0.5*v3;
        d_vatom(j,4) += 0.5*v4;
        d_vatom(j,5) += 0.5*v5;
      }
    }
  }
}
*/
template<class DeviceType>
void PairTableRXKokkos<DeviceType>::cleanup_copy() {
  // WHY needed: this prevents parent copy from deallocating any arrays
  allocated = 0;
  cutsq = NULL;
  eatom = NULL;
  vatom = NULL;
  h_table=NULL; d_table=NULL;
}

namespace LAMMPS_NS {
template class PairTableRXKokkos<LMPDeviceType>;
#ifdef KOKKOS_HAVE_CUDA
template class PairTableRXKokkos<LMPHostType>;
#endif

}
