"""mirgecom driver for the Y0 demonstration.

Note: this example requires a *scaled* version of the Y0
grid. A working grid example is located here:
github.com:/illinois-ceesd/data@y0scaled
"""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import yaml
import logging
import sys
import numpy as np
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
from functools import partial

from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.simutil import (
    generate_and_distribute_mesh,
    write_visfile,
)
from mirgecom.restart import write_restart_file

from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools
from mirgecom.fluid import make_conserved

from logpyle import IntervalTimer, set_dt
from pytools.obj_array import make_obj_array

import time

import glob

from meshmode.array_context import (
    PyOpenCLArrayContext,
    PytatoPyOpenCLArrayContext
)

#######################################################################################

class InterpolateData:

    def __init__(self, dim=2):
        self._dim = dim

    def __call__(self, actx, old_soln, old_tseed, old_grid, new_grid,
                 is_mixture=False):

        if old_grid.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")
        if new_grid.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        old_x = old_grid[0]
#        actx = old_x.array_context

        old_x = actx.to_numpy(old_grid[0][0]).reshape(-1,1)
        old_y = actx.to_numpy(old_grid[1][0]).reshape(-1,1)

        old_mass = actx.to_numpy(old_soln.mass[0]).reshape(-1,1)
        old_momX = actx.to_numpy(old_soln.momentum[0][0]).reshape(-1,1)
        old_momY = actx.to_numpy(old_soln.momentum[1][0]).reshape(-1,1)
        old_ener = actx.to_numpy(old_soln.energy[0]).reshape(-1,1)
        old_tseed = actx.to_numpy(old_tseed[0]).reshape(-1,1)

        new_x = actx.to_numpy(new_grid[0][0]).reshape(-1,1)
        new_y = actx.to_numpy(new_grid[1][0]).reshape(-1,1)
       
        dummy = actx.to_numpy(new_grid[0][0])
        ngrid = dummy.shape[ 0]
        ncoll = dummy.shape[-1]
    
        from scipy.interpolate import griddata
        from meshmode.dof_array import DOFArray

        tic = time.time()
        print('Interpolating temperature')
        new_temp = DOFArray(actx, data=(actx.from_numpy( 
                   griddata((old_x[:,0], old_y[:,0]), old_tseed[:,0], 
                            (new_x[:,0], new_y[:,0]),
                            method='linear').reshape(ngrid,ncoll) ),
                   ))
        toc = time.time()
        print(toc - tic, 's')

        tic = time.time()
        print('Interpolating mass')
        new_mass = DOFArray(actx, data=(actx.from_numpy( 
                   griddata((old_x[:,0], old_y[:,0]), old_mass[:,0], 
                            (new_x[:,0], new_y[:,0]),
                            method='linear').reshape(ngrid,ncoll) ),
                   ))
        toc = time.time()
        print(toc - tic, 's')

        tic = time.time()
        print('Interpolating momentum X')
        new_momX = DOFArray(actx, data=(actx.from_numpy( 
                   griddata((old_x[:,0], old_y[:,0]), old_momX[:,0], 
                            (new_x[:,0], new_y[:,0]),
                            method='linear').reshape(ngrid,ncoll) ),
                   ))
        toc = time.time()
        print(toc - tic, 's')

        tic = time.time()
        print('Interpolating momentum Y')
        new_momY = DOFArray(actx, data=(actx.from_numpy( 
                   griddata((old_x[:,0], old_y[:,0]), old_momY[:,0],
                            (new_x[:,0], new_y[:,0]),
                            method='linear').reshape(ngrid,ncoll) ),
                   ))
        toc = time.time()
        print(toc - tic, 's')

        if self._dim == 2:
            new_momentum = make_obj_array([new_momX, new_momY])
        else:
            tic = time.time()
            print('Interpolating momentum Z')
            new_momY = DOFArray(actx, data=(actx.from_numpy( 
                       griddata((old_x[:,0], old_y[:,0]), old_momY[:,0],
                                (new_x[:,0], new_y[:,0]),
                                method='linear').reshape(ngrid,ncoll) ),
                       ))
            toc = time.time()
            print(toc - tic, 's')

            new_momentum = make_obj_array([new_momX, new_momY, new_momZ])

        tic = time.time()
        print('Interpolating energy')
        new_ener = DOFArray(actx, data=(actx.from_numpy( 
                   griddata((old_x[:,0], old_y[:,0]), old_ener[:,0], 
                            (new_x[:,0], new_y[:,0]),
                            method='linear').reshape(ngrid,ncoll) ),
                   ))
        toc = time.time()
        print(toc - tic, 's')


        if is_mixture:
            nspecies = len(old_soln.species_mass)

            old_spec = make_obj_array([
                           actx.to_numpy(old_soln.species_mass[i][0]).reshape(-1,1)
                           for i in range(nspecies)])
        
            tic = time.time()
            print('Interpolating all species')
            new_spec = make_obj_array([
                   DOFArray(actx, data=(actx.from_numpy( 
                            griddata((old_x[:,0], old_y[:,0]), old_spec[i][:,0], 
                                     (new_x[:,0], new_y[:,0]),
                                     method='linear').reshape(ngrid,ncoll) ),
                   ))
            for i in range(nspecies)])
            toc = time.time()
            print(toc - tic, 's')
        else:
            new_spec = np.empty((0,), dtype=object)
        

        return (
            make_conserved(dim=self._dim, mass=new_mass, energy=new_ener,
                           momentum=new_momentum, species_mass=new_spec),
            new_temp
        )

#class InterpolateData:

#    def __init__(self, dim=2):
#        self._dim = dim

#    def __call__(self, actx, old_soln, old_tseed, old_grid, new_grid,
#                 is_mixture=False):

#        if old_grid.shape != (self._dim,):
#            raise ValueError(f"Position vector has unexpected dimensionality,"
#                             f" expected {self._dim}.")
#        if new_grid.shape != (self._dim,):
#            raise ValueError(f"Position vector has unexpected dimensionality,"
#                             f" expected {self._dim}.")

#        old_x = old_grid[:,0]
#        old_y = old_grid[:,1]

#        old_mass = old_soln.mass
#        old_momX = old_soln.momentum[0]
#        old_momY = old_soln.momentum[1]
#        old_ener = old_soln.energy
#        old_tseed = old_tseed

#        new_x = (new_grid[0][0]).reshape(1,-1)
#        new_y = (new_grid[1][0]).reshape(1,-1)
#       
#        ngrid = (new_grid[0][0]).shape[ 0]
#        ncoll = (new_grid[0][0]).shape[-1]
#    
#        from scipy.interpolate import griddata
#        from meshmode.dof_array import DOFArray  

#        print(ngrid)
#        print(ncoll)

#        print((old_x, old_y))
#        print((new_x[0], new_y[0]))

#        tic = time.time()
#        print('Interpolating temperature')
##        new_temp = DOFArray(actx, data=(actx.from_numpy( 
##                   griddata((old_x[:], old_y[:]), old_tseed[:], 
##                            (new_x[:], new_y[:]),
##                            method='linear').reshape(ngrid,ncoll) ),
##                   ))
#        new_temp = griddata((old_x, old_y), old_tseed, (new_x[0], new_y[0]),
#                            method='linear')
#        toc = time.time()
#        print(toc - tic, 's')


#        sys.exit()

#        print(new_temp)
#        sys.exit()

#        tic = time.time()
#        print('Interpolating mass')
#        new_mass = DOFArray(actx, data=(actx.from_numpy( 
#                   griddata((old_x[:,0], old_y[:,0]), old_mass[:,0], 
#                            (new_x[:,0], new_y[:,0]),
#                            method='linear').reshape(ngrid,ncoll) ),
#                   ))
#        toc = time.time()
#        print(toc - tic, 's')

#        tic = time.time()
#        print('Interpolating momentum X')
#        new_momX = DOFArray(actx, data=(actx.from_numpy( 
#                   griddata((old_x[:,0], old_y[:,0]), old_momX[:,0], 
#                            (new_x[:,0], new_y[:,0]),
#                            method='linear').reshape(ngrid,ncoll) ),
#                   ))
#        toc = time.time()
#        print(toc - tic, 's')

#        tic = time.time()
#        print('Interpolating momentum Y')
#        new_momY = DOFArray(actx, data=(actx.from_numpy( 
#                   griddata((old_x[:,0], old_y[:,0]), old_momY[:,0],
#                            (new_x[:,0], new_y[:,0]),
#                            method='linear').reshape(ngrid,ncoll) ),
#                   ))
#        toc = time.time()
#        print(toc - tic, 's')

#        if self._dim == 2:
#            new_momentum = make_obj_array([new_momX, new_momY])
#        else:
#            tic = time.time()
#            print('Interpolating momentum Z')
#            new_momY = DOFArray(actx, data=(actx.from_numpy( 
#                       griddata((old_x[:,0], old_y[:,0]), old_momY[:,0],
#                                (new_x[:,0], new_y[:,0]),
#                                method='linear').reshape(ngrid,ncoll) ),
#                       ))
#            toc = time.time()
#            print(toc - tic, 's')

#            new_momentum = make_obj_array([new_momX, new_momY, new_momZ])

#        tic = time.time()
#        print('Interpolating energy')
#        new_ener = DOFArray(actx, data=(actx.from_numpy( 
#                   griddata((old_x[:,0], old_y[:,0]), old_ener[:,0], 
#                            (new_x[:,0], new_y[:,0]),
#                            method='linear').reshape(ngrid,ncoll) ),
#                   ))
#        toc = time.time()
#        print(toc - tic, 's')


#        if is_mixture:
#            nspecies = len(old_soln.species_mass)

#            old_spec = make_obj_array([
#                           actx.to_numpy(old_soln.species_mass[i][0]).reshape(-1,1)
#                           for i in range(nspecies)])
#        
#            tic = time.time()
#            print('Interpolating all species')
#            new_spec = make_obj_array([
#                   DOFArray(actx, data=(actx.from_numpy( 
#                            griddata((old_x[:,0], old_y[:,0]), old_spec[i][:,0], 
#                                     (new_x[:,0], new_y[:,0]),
#                                     method='linear').reshape(ngrid,ncoll) ),
#                   ))
#            for i in range(nspecies)])
#            toc = time.time()
#            print(toc - tic, 's')
#        else:
#            new_spec = np.empty((0,), dtype=object)
#        

#        return (
#            make_conserved(dim=self._dim, mass=new_mass, energy=new_ener,
#                           momentum=new_momentum, species_mass=new_spec),
#            new_temp
#        )

def get_mesh(dim, mesh_filename, read_mesh=True):
    """Get the mesh."""
    from meshmode.mesh.io import read_gmsh
    mesh = partial(read_gmsh, filename=mesh_filename, force_ambient_dim=dim)

    return mesh

class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


h1 = logging.StreamHandler(sys.stdout)
f1 = SingleLevelFilter(logging.INFO, False)
h1.addFilter(f1)
root_logger = logging.getLogger()
root_logger.addHandler(h1)
h2 = logging.StreamHandler(sys.stderr)
f2 = SingleLevelFilter(logging.INFO, True)
h2.addFilter(f2)
root_logger.addHandler(h2)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, casename=None):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    cl_ctx = ctx_factory()

    queue = cl.CommandQueue(cl_ctx)

    actx = actx_class(comm, queue,
           allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
           force_device_scalars=True)

    # ~~~~~~~~~~~~~~~~~~

    new_gridfile = "mesh_v22m.msh"

    step = 0 #step of the old solution file

    is_mixture = True
    nspecies = 7
    species_names = ['C2H4', 'O2', 'CO2', 'CO', 'H2O', 'H2', 'N2']

    new_order = 2
    dim = 2
    
    step = str('%06i' % step)
    old_casename = "burner"
    new_casename = "interp"

    snapshot_pattern = "{casename}-{step:06d}-{rank:04d}.pkl"
    new_vizname = new_casename

    filelist = glob.glob(old_casename + "-" + step + '-????.vtu')
    old_nranks = len(filelist)
    if rank == 0:
        print(old_casename)
        print(filelist)
        print(old_nranks)

    if old_nranks == 0:
        print('No matching files...')
        sys.exit()

##############################################################################

    """
    Old restart file
    """

    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    kk = 0
    for old_files in filelist:
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(old_files)
        reader.Update()
        output = reader.GetOutput()
        point_data = output.GetPointData()

        grid_points = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
        
        if kk == 0:

            if rank == 0: print(point_data)
            old_coordX = grid_points[:,0]
            old_coordY = grid_points[:,1]
            if dim == 3:
                old_coordZ = grid_points[:,2]
            old_mass = vtk_to_numpy(point_data.GetArray( 0))
            old_rhoU = vtk_to_numpy(point_data.GetArray( 1))[:,0]
            old_rhoV = vtk_to_numpy(point_data.GetArray( 1))[:,1]
            if dim == 3:
                old_rhoW = vtk_to_numpy(point_data.GetArray( 1))[:,2]
            old_rhoE = vtk_to_numpy(point_data.GetArray( 2))
            old_temp = vtk_to_numpy(point_data.GetArray( 4))
            if is_mixture:
                old_Y_0 = vtk_to_numpy(point_data.GetArray( 8))
                old_Y_1 = vtk_to_numpy(point_data.GetArray( 9))
                old_Y_2 = vtk_to_numpy(point_data.GetArray(10))
                old_Y_3 = vtk_to_numpy(point_data.GetArray(11))
                old_Y_4 = vtk_to_numpy(point_data.GetArray(12))
                old_Y_5 = vtk_to_numpy(point_data.GetArray(13))
                old_Y_6 = vtk_to_numpy(point_data.GetArray(14))

        else:            
            old_coordX = np.append(old_coordX, grid_points[:,0])
            old_coordY = np.append(old_coordY, grid_points[:,1])
            if dim == 3:
                old_coordZ = np.append(old_coordZ, grid_points[:,2])
            old_mass = np.append(old_mass, vtk_to_numpy(point_data.GetArray( 0)))
            old_rhoU = np.append(old_rhoU, vtk_to_numpy(point_data.GetArray( 1))[:,0])
            old_rhoV = np.append(old_rhoV, vtk_to_numpy(point_data.GetArray( 1))[:,1])
            if dim == 3:
                old_rhoW = np.append(cv_rhoW, vtk_to_numpy(point_data.GetArray( 1))[:,2])
            old_rhoE = np.append(old_rhoE, vtk_to_numpy(point_data.GetArray( 2)))
            old_temp = np.append(old_temp, vtk_to_numpy(point_data.GetArray( 4)))
            if is_mixture:
                old_Y_0 = np.append(old_Y_0, vtk_to_numpy(point_data.GetArray( 8)))
                old_Y_1 = np.append(old_Y_1, vtk_to_numpy(point_data.GetArray( 9)))
                old_Y_2 = np.append(old_Y_2, vtk_to_numpy(point_data.GetArray(10)))
                old_Y_3 = np.append(old_Y_3, vtk_to_numpy(point_data.GetArray(11)))
                old_Y_4 = np.append(old_Y_4, vtk_to_numpy(point_data.GetArray(12)))
                old_Y_5 = np.append(old_Y_5, vtk_to_numpy(point_data.GetArray(13)))
                old_Y_6 = np.append(old_Y_6, vtk_to_numpy(point_data.GetArray(14)))
    
        kk += 1

    ncoll = 6
    ngrid = int(old_coordX.shape[0]/ncoll)
 
    from meshmode.dof_array import DOFArray    
    old_coordX = DOFArray(actx, data=(actx.from_numpy( 
                           old_coordX.reshape(ngrid,ncoll) ),
                 ))
    old_coordY = DOFArray(actx, data=(actx.from_numpy( 
                           old_coordY.reshape(ngrid,ncoll) ),
                 ))
    old_mass = DOFArray(actx, data=(actx.from_numpy( 
                           old_mass.reshape(ngrid,ncoll) ),
                 ))
    old_rhoU = DOFArray(actx, data=(actx.from_numpy( 
                           old_rhoU.reshape(ngrid,ncoll) ),
                 ))
    old_rhoV = DOFArray(actx, data=(actx.from_numpy( 
                           old_rhoV.reshape(ngrid,ncoll) ),
                 ))
    old_rhoE = DOFArray(actx, data=(actx.from_numpy( 
                           old_rhoE.reshape(ngrid,ncoll) ),
                 ))
    old_temp = DOFArray(actx, data=(actx.from_numpy( 
                           old_temp.reshape(ngrid,ncoll) ),
                 ))   

    if is_mixture:
        old_spc = make_obj_array([
                  DOFArray(actx, data=(actx.from_numpy(old_Y_0.reshape(ngrid,ncoll) ), )),
                  DOFArray(actx, data=(actx.from_numpy(old_Y_1.reshape(ngrid,ncoll) ), )),
                  DOFArray(actx, data=(actx.from_numpy(old_Y_2.reshape(ngrid,ncoll) ), )),
                  DOFArray(actx, data=(actx.from_numpy(old_Y_3.reshape(ngrid,ncoll) ), )),
                  DOFArray(actx, data=(actx.from_numpy(old_Y_4.reshape(ngrid,ncoll) ), )),
                  DOFArray(actx, data=(actx.from_numpy(old_Y_5.reshape(ngrid,ncoll) ), )),
                  DOFArray(actx, data=(actx.from_numpy(old_Y_6.reshape(ngrid,ncoll) ), ))
                  ])
        del old_Y_0, old_Y_1, old_Y_2, old_Y_3, old_Y_4, old_Y_5, old_Y_6
    else:
        old_spc = np.empty((0,), dtype=object)
    
    if dim == 2:
        old_nodes = make_obj_array([old_coordX, old_coordY])
        old_mom = make_obj_array([old_rhoU, old_rhoV])
        del grid_points, old_coordX, old_coordY
        del old_rhoU, old_rhoV
    else:
        old_nodes = make_obj_array([old_coordX, old_coordY, old_coordZ])
        old_mom = make_obj_array([old_rhoU, old_rhoV, old_rhoW])
        del grid_points, old_coordX, old_coordY, old_coordZ
        del old_rhoU, old_rhoV, old_rhoW

    old_cv = make_conserved(dim=dim, mass=old_mass, energy=old_rhoE,
                            momentum=old_mom, species_mass=old_mass*old_spc)

    del old_mass, old_rhoE

##############################################################################

    """
    New restart file
    """

    local_new_mesh, global_new_nelements = generate_and_distribute_mesh(
        comm, get_mesh(dim, mesh_filename=new_gridfile))
    local_new_nelements = local_new_mesh.nelements

    from grudge.dof_desc import DISCR_TAG_QUAD
    from mirgecom.discretization import create_discretization_collection
    new_discr = create_discretization_collection(
                actx, local_new_mesh, new_order, mpi_communicator=comm)
    new_nodes = actx.thaw(new_discr.nodes())

    interpolation = InterpolateData(dim=dim)
    new_cv, new_tseed = interpolation(actx, old_cv, old_temp, old_nodes,
                                      new_nodes, is_mixture=is_mixture)

    new_t = 0.0
    new_step = 0

    new_visualizer = make_visualizer(new_discr)
    def write_new_viz(step, t, cv, tseed, is_mixture=False):

        viz_fields = [("CV_rho", cv.mass),
                      ("CV_rhoU", cv.momentum),
                      ("CV_rhoE", cv.energy),
                      ("tseed", tseed)]

        if is_mixture:
            viz_fields.extend(
                ("Y_" + species_names[i], cv.species_mass_fractions[i])
                for i in range(nspecies))

        print('Writing solution file...')
        write_visfile(new_discr, viz_fields, new_visualizer,
                      vizname=new_vizname, step=step, t=t, overwrite=True)
                      
        return    
 
    write_new_viz(new_step, new_t, new_cv, new_tseed, is_mixture=is_mixture)

##############################################################################

    def my_write_restart(step, t, cv, temperature_seed=None):
        rst_fname = snapshot_pattern.format(casename=new_casename, 
                                            step=new_step, rank=rank)
        if rst_fname != restart_file:
            rst_data = {
                "local_mesh": local_new_mesh,
                "state": cv,
                "temperature_seed": new_tseed,
                "t": new_t,
                "step": new_step,
                "order": new_order,
                "global_nelements": global_new_nelements,
                "num_parts": nparts
            }
            
            write_restart_file(actx, rst_data, rst_fname, comm)    

    my_write_restart(step=new_step, t=new_t, cv=new_cv,
                     temperature_seed=new_tseed)

#    sys.exit()

##################################################################

if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description="MIRGE-Com 1D Flame Driver")
    parser.add_argument("-r", "--restart_file",  type=ascii,
                        dest="restart_file", nargs="?", action="store",
                        help="simulation restart file")
    parser.add_argument("-i", "--input_file",  type=ascii,
                        dest="input_file", nargs="?", action="store",
                        help="simulation config file")
    parser.add_argument("-c", "--casename",  type=ascii,
                        dest="casename", nargs="?", action="store",
                        help="simulation case name")
    parser.add_argument("--profile", action="store_true", default=False,
        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=True,
        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
        help="enable lazy evaluation [OFF]")

    args = parser.parse_args()

    restart_file = None
    if args.restart_file:
        restart_file = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_file}")

    input_file = None
    if(args.input_file):
        input_file = (args.input_file).replace("'", "")
        print(f"Reading user input from {args.input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=args.lazy, distributed=True)

    main(actx_class)
