#srun --ntasks=1 --time=01:00:00 --mem=256000 --account=project_2010748 --partition=test --cpus-per-task=1 --x11=first --pty $SHELL
import numpy as np
import xarray as xr
import dask.distributed
#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
import pandas as pd
import xesmf as xe
#from joblib import Parallel, delayed
import glob
import dask
import os
from pathlib import Path
import socket
from sys import argv
#
from pyproj import Proj, Transformer
from scipy.ndimage import gaussian_filter
#from memory_profiler import profile

import py_wake as pw
from py_wake.site import XRSite
from py_wake.site.shear import PowerShear
#from numpy import newaxis as na
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit,IEA37SimpleBastankhahGaussianDeficit
from py_wake.wind_turbines import WindTurbine, WindTurbines
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.superposition_models import LinearSum
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.site.distance import StraightDistance
#from py_wake.utils.numpy_utils import Numpy32

def turbine_positions(region,wt_x_all,wt_y_all,df,dy,dx):
    wt_x=[]
    wt_y=[]
    #print(region)
    if region in ['GoB']:
        x_slice = slice(0.3E6,0.6E6)
        y_slice = slice(0.3E6,1.05E6)
    elif region in ['Baltic']:
        x_slice = slice(-0.26E6,0.7E6)
        y_slice = slice(-0.34E6,0.3E6)
    elif region in ['NorthSeaSouth_1']:
        x_slice1 = slice(-0.95E6,-0.62E6)
        x_slice2 = slice(-0.62E6,-0.26E6)
        y_slice1 = slice(-0.65E6,-0.1E6)
        y_slice2 = slice(-0.65E6,-0.32E6)
        # here we need outer bounding box anyway
        x_slice  = slice(-0.95E6,-0.26E6)
        y_slice  = slice(-0.65E6,-0.1E6)
    elif region in ['NorthSeaSouth_2']:
        x_slice = slice(-0.64E6,-0.28E6)
        y_slice = slice(-0.32E6,0.1E6)
    elif region in ['NorthSeaNorth']:
        x_slice=slice(-1.1E6,-0.62E6)
        y_slice=slice(-0.1E6,0.5E6)
    elif region in ['NorwegianCoast']:
        x_slice=slice(-0.62E6,-0.26E6)
        y_slice=slice(0.15,0.55E6)
    #
    mask = np.logical_and(np.logical_and(x2d>x_slice.start,x2d<x_slice.stop),np.logical_and(y2d>y_slice.start,y2d<y_slice.stop))
    for ii,i in enumerate(range(len(df))):
        if region in ['NorthSeaSouth_1']:
            if (((wt_x_all[i]>=x_slice1.start+dx and wt_x_all[i]<=x_slice1.stop-dx) and (wt_y_all[i]>=y_slice1.start+dy and wt_y_all[i]<=y_slice1.stop-dy)) or
                ((wt_x_all[i]>x_slice2.start+dx and wt_x_all[i]<x_slice2.stop-dx) and (wt_y_all[i]>y_slice2.start+dy and wt_y_all[i]<y_slice2.stop-dy))):
                wt_x.append(wt_x_all[i])
                wt_y.append(wt_y_all[i])
        else:
            if ((wt_x_all[i]>x_slice.start+dx and wt_x_all[i]<x_slice.stop-dx) and (wt_y_all[i]>y_slice.start+dy and wt_y_all[i]<y_slice.stop-dy)):
                wt_x.append(wt_x_all[i])
                wt_y.append(wt_y_all[i])
    #
    types=[0]*len(wt_x)
    return wt_x,wt_y,x_slice,y_slice,types,mask

def wake(data,region_turbines,windTurbines,tt,fname,x_reg,y_reg,
         output_dir='/users/nummelin/Projects/DTO4OWE/',
         latituderef=59.35,longituderef=20,r=6378e3):
    #
    # create output data
    ws_final = data.ws.values.copy()
    ds_final = xr.merge([xr.DataArray(data.ws.values,dims=('y','x')).to_dataset(name='ws'),
                         xr.DataArray(data.wdir.values,dims=('y','x')).to_dataset(name='wd'),
                         xr.DataArray(data.ws.values.copy(),dims=('y','x')).to_dataset(name='ws_orig'),
    ]).assign_coords({'latitude':data.latitude,'longitude':data.longitude,'x':data.x,'y':data.y})
    #
    # crop the input data
    for region in region_turbines.keys():
        wt_x    = region_turbines[region]["wt_x"]
        wt_y    = region_turbines[region]["wt_y"]
        x_slice = region_turbines[region]["x_slice"]
        y_slice = region_turbines[region]["y_slice"]
        mask    = region_turbines[region]["mask"]
        data_in = data.sel(x=x_slice,y=y_slice)
        x_reg2 = xr.DataArray(x_reg,coords={'x':x_reg}).sel(x=x_slice)
        y_reg2 = xr.DataArray(y_reg,coords={'y':y_reg}).sel(y=y_slice)
        # create output grid
        grid= pw.HorizontalGrid(y=y_reg2.astype(np.float32),x=x_reg2.astype(np.float32),h=h)
        #
        ###################################################################
        #FINALLY WORKS - THE DIMENSIONS NEED TO BE X-Y NOT Y-X !!!
        # create input dataset
        ds=xr.Dataset(data_vars={'WS':data_in.ws.T.astype(np.float32),
                             'WD':data_in.wdir.T.astype(np.float32),
                             'P':1, 
                             'TI':np.float32(0.05)},
        ).assign_coords({'x':data_in.ws.x.astype(np.float32),'y':data_in.ws.y.astype(np.float32)}).load()
        #print('define wind farm model')
        site     = XRSite(ds,distance=StraightDistance(wind_direction='WD_i'),bounds='ignore')
        wf_model = PropagateDownwind(site, windTurbines,
                                 wake_deficitModel=BastankhahGaussianDeficit(use_effective_ws=True))
        wf     = wf_model(wt_x.astype(np.float32), wt_y.astype(np.float32),ws=10)
        #print('return data on a grid')
        wf_out = wf.flow_map(grid)
        #print('actual calculation is done')
        #
        #final regrid
        #
        jinds,iinds = np.where(mask)
        ws_region   = wf_out.WS_eff.drop_vars(['wd','ws']).squeeze()
        # wake in the region
        wake_region = ws_region.values.flatten() - ds_final.ws.values[jinds,iinds]
        # add wake to the final wind speed. Since we loop over regions, the wake will be summed if there are overlapping regions. Do not allow for negative values.
        ws_final[jinds,iinds] = np.max(np.concatenate([(ws_final[jinds,iinds] + wake_region)[:,np.newaxis],np.zeros_like(wake_region)[:,np.newaxis]],axis=-1),-1)
    #
    ds_final['ws']        = xr.DataArray(ws_final,dims=('y','x'))
    #
    #wf_out.WS_eff.drop_vars(['wd','ws']).squeeze().to_dataset(name='ws').to_netcdf(output_dir+region+'_test.nc')
    #ds_final.to_netcdf(output_dir+'BalticSeaNorthSea_'+f'ws_{str(tt).zfill(3)}_'+fname.split('/')[-1])
    #print('regridding and netcdf done')
    #data.close()
    #del wf, wf_model
    return ds_final

def wake_old(data,wt_x,wt_y,windTurbines,tt,fname,
         x_slice,y_slice,x_reg,y_reg,
         output_dir='/users/nummelin/Projects/DTO4OWE/',
         latituderef=59.35,longituderef=20,r=6378e3):
    #
    # create output data
    ds_final = xr.merge([xr.DataArray(data.ws.values,dims=('y','x')).to_dataset(name='ws'),
                         xr.DataArray(data.wdir.values,dims=('y','x')).to_dataset(name='wd'),
                         xr.DataArray(data.mask.values,dims=('y','x')).to_dataset(name='mask'),
                         xr.DataArray(data.ws.values,dims=('y','x')).to_dataset(name='ws_orig'),
    ]).assign_coords({'latitude':data.latitude,'longitude':data.longitude,'x':data.x,'y':data.y})
    #
    # crop the input data
    data_in = data.sel(x=x_slice,y=y_slice)
    x_reg2 = xr.DataArray(x_reg,coords={'x':x_reg}).sel(x=x_slice)
    y_reg2 = xr.DataArray(y_reg,coords={'y':y_reg}).sel(y=y_slice)
    # create output grid
    grid= pw.HorizontalGrid(y=y_reg2.astype(np.float32),x=x_reg2.astype(np.float32),h=h)
    #
    #print('create input dataset')
    #
    ###################################################################
    #FINALLY WORKS - THE DIMENSIONS NEED TO BE X-Y NOT Y-X !!!
    # create input dataset
    ds=xr.Dataset(data_vars={'WS':data_in.ws.T.astype(np.float32),
                             'WD':data_in.wdir.T.astype(np.float32),
                             'P':1, 
                             'TI':np.float32(0.05)},
    ).assign_coords({'x':data_in.ws.x.astype(np.float32),'y':data_in.ws.y.astype(np.float32)}).load()
    #print(ds.y.dtype)
    #If shear is needed, add shear=PowerShear(h_ref=150, alpha=.2)
    #print('define wind farm model')
    site     = XRSite(ds,distance=StraightDistance(wind_direction='WD_i'),bounds='ignore')
    wf_model = PropagateDownwind(site, windTurbines,
                                 wake_deficitModel=BastankhahGaussianDeficit(use_effective_ws=True))
    #print('wind farm model starting')
    #
    #print(np.min(wt_x2),np.max(wt_x2),np.min(wt_y2),np.max(wt_y2))
    wf     = wf_model(wt_x.astype(np.float32), wt_y.astype(np.float32),ws=10)
    #print('return data on a grid')
    wf_out = wf.flow_map(grid) #.XYGrid(resolution=50000)) #.drop(['wd','ws'])
    #wf=wf_model(wt_x, wt_y,ws=10,wd=10,type=types).flow_map(grid).drop(['wd','ws'])
    #print('actual calculation is done')
    #
    #final regrid
    #
    jinds,iinds = np.where(ds_final.mask)
    ws_final    = ds_final.ws.values.copy()
    ws_region   = wf_out.WS_eff.drop_vars(['wd','ws']).squeeze()
    ws_final[jinds,iinds] = ws_region.values.flatten()
    ds_final['ws']        = xr.DataArray(ws_final,dims=('y','x'))
    #
    #wf_out.WS_eff.drop_vars(['wd','ws']).squeeze().to_dataset(name='ws').to_netcdf(output_dir+region+'_test.nc')
    ds_final.to_netcdf(output_dir+region+'_'+f'ws_{str(tt).zfill(3)}_'+fname.split('/')[-1])
    #print('regridding and netcdf done')
    #data.close()
    #del wf, wf_model
    return ds_final
    
if __name__ == '__main__':
    ######################3
    # OPEN DASK CLUSTER
    local_dir = '/users/nummelin/Projects/DTO4OWE/'
    if not Path(local_dir).is_dir():
        os.system('mkdir -p '+local_dir)
        print('created folder '+local_dir)
    #
    n_workers = 1
    n_threads = 1
    processes = True
    cluster   = dask.distributed.LocalCluster(n_workers=n_workers,
                                              threads_per_worker=n_threads,
                                              processes=processes,
                                              local_directory=local_dir,
                                              lifetime='12 hour',
                                              lifetime_stagger='10 minutes',
                                              lifetime_restart=True,
                                              dashboard_address=None,
                                              worker_dashboard_address=None)
    client  = dask.distributed.Client(cluster)
    ###########################
    # BASIC SETTINGS
    year    = int(argv[1])
    month   = int(argv[2])
    day0    = int(argv[3])
    print(year, month)
    if year<2015:
        exp='hist'
    else:
        exp='ssp3-7.0'
    ##################################################################
    # THERE ARE TOO MANY TURBINES TO PROCESS EVERYTHING ALL AT ONCE
    # THE WORKAROUND IS TO SPLIT TO DIFFERENT REGIONS.
    regions=['NorthSeaNorth','NorwegianCoast','NorthSeaSouth_1','NorthSeaSouth_2','Baltic','GoB'] 
    ##############################
    # OPEN DATA AND COMPUTE WIND SPEED AND DIRECTION
    #
    datapath = '/scratch/project_2010748/DTO4OWE/CLIMATE_DT/IFS-NEMO/'+exp+'/'
    fname100 = 'hl_IFS-NEMO_'+exp+'_'+str(year)+'_'+str(month).zfill(2)+'.nc' #'CERRA_winds_test.nc'
    fname10  = 'sfc_IFS-NEMO_'+exp+'_'+str(year)+'_'+str(month).zfill(2)+'.nc'
    IFS100orig = xr.open_dataset(datapath+fname100,chunks={"time": 24})
    IFS10orig  = xr.open_dataset(datapath+fname10, chunks={"time": 24})
    #
    
    IFS100ws   = (np.sqrt(IFS100orig.u**2+IFS100orig.v**2)).to_dataset(name='ws')
    IFS10ws    = (np.sqrt(IFS10orig['10u']**2+IFS10orig['10v']**2)).to_dataset(name='ws')
    #
    IFS100wdir =  np.degrees(xr.apply_ufunc(np.atan2,IFS100orig.v,IFS100orig.u,
                                            input_core_dims=[[],[]],
                                            output_core_dims=[[]],
                                            vectorize=True,
                                            dask='parallelized',
                                            output_dtypes=[np.float32],
                                            )).to_dataset(name='wdir')
    #
    IFS10wdir =  np.degrees(xr.apply_ufunc(np.atan2,IFS10orig['10v'],IFS10orig['10u'],
                                            input_core_dims=[[],[]],
                                            output_core_dims=[[]],
                                            vectorize=True,
                                            dask='parallelized',
                                            output_dtypes=[np.float32],
                                            )).to_dataset(name='wdir')
    #
    if not Path('/scratch/project_2010748/DTO4OWE/CLIMATE_DT/IFS-NEMO/'+exp+'/IFS100_'+str(year)+'_'+str(month).zfill(2)+'_tmp.nc').is_file():
        xr.merge([IFS100ws,IFS100wdir]).to_netcdf('/scratch/project_2010748/DTO4OWE/CLIMATE_DT/IFS-NEMO/'+exp+'/IFS100_'+str(year)+'_'+str(month).zfill(2)+'_tmp.nc')
    if not Path('/scratch/project_2010748/DTO4OWE/CLIMATE_DT/IFS-NEMO/'+exp+'/IFS10_'+str(year)+'_'+str(month).zfill(2)+'_tmp.nc').is_file():
        xr.merge([IFS10ws,IFS10wdir]).to_netcdf('/scratch/project_2010748/DTO4OWE/CLIMATE_DT/IFS-NEMO/'+exp+'/IFS10_'+str(year)+'_'+str(month).zfill(2)+'_tmp.nc')
    #
    IFS100   = xr.open_dataset('/scratch/project_2010748/DTO4OWE/CLIMATE_DT/IFS-NEMO/'+exp+'/IFS100_'+str(year)+'_'+str(month).zfill(2)+'_tmp.nc',chunks={"time": 24}).load()
    IFS10    = xr.open_dataset('/scratch/project_2010748/DTO4OWE/CLIMATE_DT/IFS-NEMO/'+exp+'/IFS10_'+str(year)+'_'+str(month).zfill(2)+'_tmp.nc',chunks={"time": 24})
    #
    ###########################################
    # CREATE GRIDS AND MAPPINGS BETWEEN THEM
    dlon=dlat=0.05 #hard coded here
    lon_reg   = IFS100.longitude.values
    lat_reg   = IFS100.latitude.values
    lon_reg_b = np.arange(lon_reg[0]-dlon/2,lon_reg[-1]+dlon,dlon)
    lat_reg_b =	np.arange(lat_reg[-1]-dlat/2,lat_reg[0]+dlat,dlat)[::-1]
    lon2d,lat2d       = np.meshgrid(lon_reg,lat_reg)
    ds_regular_latlon = xr.merge([xr.DataArray(lat_reg,dims=('y')).to_dataset(name='lat'),
                                  xr.DataArray(lon_reg,dims=('x')).to_dataset(name='lon'),
                                  xr.DataArray(lat_reg_b,dims=('y_b')).to_dataset(name='lat_b'),
                                  xr.DataArray(lon_reg_b,dims=('x_b')).to_dataset(name='lon_b'),
                                  ])
    # define a regular xy grid reference to the mid-point of the regular lat/lon grid
    # this needs to be checked again
    latituderef  = np.mean(lat_reg)
    longituderef = np.mean(lon_reg)
    #
    proj_string = f"+proj=aeqd +lat_0={latituderef} +lon_0={longituderef} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    proj_local  = Proj(proj_string)
    proj_latlon = Proj(proj="latlong", datum="WGS84")
    transformer_locTolatlon = Transformer.from_proj(proj_local, proj_latlon)
    transformer_latlonToloc = Transformer.from_proj(proj_latlon, proj_local)
    xmin,ymin = transformer_latlonToloc.transform(np.min(lon_reg),np.min(lat_reg))
    xmax,ymax = transformer_latlonToloc.transform(np.max(lon_reg),np.max(lat_reg))
    dx=dy=2E3
    x_reg   = np.arange(xmin,xmax+dx,dx)
    y_reg   = np.arange(ymin,ymax+dy,dy) #
    x_reg_b = np.arange(xmin-dx/2,xmax+1.5*dx,dx)
    y_reg_b = np.arange(ymin-dy/2,ymax+1.5*dy,dy)
    x2d,y2d = np.meshgrid(x_reg,y_reg)
    x2d_b,y2d_b = np.meshgrid(x_reg_b,y_reg_b)
    lon2dxy,lat2dxy = transformer_locTolatlon.transform(x2d,y2d)
    lon2dxy_b,lat2dxy_b = transformer_locTolatlon.transform(x2d_b,y2d_b)
    # create datasets for both
    ds_regular_xy = xr.merge([xr.DataArray(lat2dxy,dims=('y','x')).to_dataset(name='latitude'),
                              xr.DataArray(lon2dxy,dims=('y','x')).to_dataset(name='longitude'),
                              xr.DataArray(lat2dxy_b,dims=('y_b','x_b')).to_dataset(name='lat_b'),
                              xr.DataArray(lon2dxy_b,dims=('y_b','x_b')).to_dataset(name='lon_b'),
                              ]
    )
    # regrid to the regular xy grid
    regridder_IFS = xe.Regridder(IFS100, ds_regular_xy, "bilinear",filename='IFS_to_xy_dxdy_'+str(dx/1E3)+'km.nc',reuse_weights=True)
    #regridder_cerra2latlon = xe.Regridder(cerra, ds_regular_latlon, "bilinear",filename=
    #regridder_xy2latlon = xe.Regridder(ds_regular_xy,ds_regular_latlon, "bilinear",filename='XY_to_regular.nc',reuse_weights=True)
    regridder_xy2latlon = xe.Regridder(ds_regular_xy,ds_regular_latlon, "conservative",filename='XY_to_regular_conservative_dxdy_'+str(dx/1E3)+'km.nc',reuse_weights=True)
    #
    if True:#data.time.size>tt:
        output_dir = '/scratch/project_2010748/DTO4OWE/CLIMATE_DT/IFS-NEMO/'+exp+'_'+str(year)+'_'+str(month).zfill(2)+'/' # where do you want to put your output
        final_output_dir = '/scratch/project_2010748/DTO4OWE/CLIMATE_DT/IFS-NEMO/pywake/'+exp+'/'
        if not os.path.isdir(output_dir):
            os.system('mkdir -p '+output_dir)
            print('created folder '+output_dir)
        if not os.path.isdir(final_output_dir):
            os.system('mkdir -p '+final_output_dir)
            print('created folder '+final_output_dir)
        ###################################
        # READ TURBINE PROPERTIES
        #
        df = pd.read_csv('/users/nummelin/Projects/mahti_projects/DTO4OWE/DTO4WOE_polygons/NorthSea_BalticSea_future_turbine_positions_consistent.csv',delimiter=',')
        #
        #
        latitude=df['latitude'].values
        longitude=df['longitude'].values
        wt_x_all, wt_y_all = transformer_latlonToloc.transform(longitude,latitude) #
        #NREL 15MW wind turbine
        IEAturbine_carac = pd.read_csv('/users/nummelin/Projects/mahti_projects/DTO4OWE/IEA_Reference_15MW_240.csv',delimiter=',')
        power_IEA        = IEAturbine_carac['Power [kW]'].values
        ct               = IEAturbine_carac['Ct [-]'].values
        u                = IEAturbine_carac['Wind Speed [m/s]'].values
        h                = 150
        D                = 240
        P                = 15e6
        #
        IEA15MW = WindTurbine(name='IEA15MW', diameter=D,hub_height=h,
                              powerCtFunction=PowerCtTabular(u,power_IEA,'kW',ct))
        #
        windTurbines = WindTurbines.from_WindTurbine_lst([IEA15MW])
        #
        print('loop over turbines')
        region_turbines = {}
        for region in regions:
            wt_x, wt_y, x_slice, y_slice, types, mask = turbine_positions(region,wt_x_all,wt_y_all,df,dy,dx)
            region_turbines[region] = {
                "wt_x": np.asarray(wt_x, dtype=np.float32),
                "wt_y": np.asarray(wt_y, dtype=np.float32),
                "x_slice": x_slice,
                "y_slice": y_slice,
                "mask": mask,
                }
        #
        #
        for tt, t in enumerate(range(day0*24,IFS100.time.size,24)):
            print(tt,t,t//24,day0)
            if not Path(output_dir+'BalticSeaNorthSeaDaily_'+f'ws_{str(t//24).zfill(3)}_'+fname100.split('/')[-1]).is_file():
                IFS100_reg = regridder_IFS(IFS100.isel(time=slice(t,t+24)))
                IFS100_reg = IFS100_reg.assign_coords({'longitude':ds_regular_xy.longitude,'latitude':ds_regular_xy.latitude,
                                                       'x':xr.DataArray(x_reg,dims=('x')),'y':xr.DataArray(y_reg,dims=('y'))})
                data = xr.merge([IFS100_reg,xr.DataArray(xr.zeros_like(IFS100_reg.ws)).to_dataset(name='mask')]).load()
                for t0 in range(24):
                    if t0==0:
                        ds_final = wake(data.isel(time=t0),region_turbines,windTurbines,t0,fname100,x_reg,y_reg,output_dir=output_dir)
                    else:
                        ds_dum = wake(data.isel(time=t0),region_turbines,windTurbines,t0,fname100,x_reg,y_reg,output_dir=output_dir)
                        ds_final=xr.concat([ds_final,ds_dum],dim='time')
                ds_final.to_netcdf(output_dir+'BalticSeaNorthSeaDaily_'+f'ws_{str(t//24).zfill(3)}_'+fname100.split('/')[-1])
        #
        #
        ds_final_reg_all  = xr.open_mfdataset(sorted(glob.glob(output_dir+'BalticSeaNorthSeaDaily_ws_*_hl_IFS-NEMO_hist_'+str(year)+'_'+str(month).zfill(2)+'.nc')),concat_dim='time',combine='nested')
        #
        ds_final_reg = regridder_xy2latlon(ds_final_reg_all).assign_coords({'lon':ds_regular_latlon.lon,'lat':ds_regular_latlon.lat})
        ds_final_reg.to_netcdf(output_dir+'BalticSeaNorthSea_regular_ws100_IFS-NEMO_'+exp+'_'+str(year)+'_'+str(month).zfill(2)+'.nc')
        #
        #########################################################################################
        # EXTRAPOLATE TO THE SURFACE
        #
        all_winds = xr.open_mfdataset(glob.glob(output_dir+'BalticSeaNorthSea_regular_ws100_IFS-NEMO_'+exp+'_'+str(year)+'_'+str(month).zfill(2)+'.nc'))
        all_wakes = all_winds.ws-all_winds.ws_orig
        all_wakes=all_wakes.rename({'x':'longitude','y':'latitude'}).assign_coords({'latitude':IFS100.latitude,'longitude':IFS100.longitude})
        #
        # interpolate to the surface
        #
        wakemask    = 3 * np.abs(all_wakes/IFS100.ws)
        mean_ratio  = (IFS10.ws / IFS100.ws) #consider applying gaussian filter or something here.
        IFS10_wake  = (1 - wakemask) * IFS10.ws + wakemask * IFS10.ws * np.clip((3.266 * np.exp(-5.473 * (mean_ratio + 1.503)) * np.exp(5.341 * ((IFS100.ws-all_wakes)/IFS100.ws) + 0.7554) + 0.8849), 0, 3)
        v = -IFS10_wake * np.cos((np.arctan2(IFS10orig['10u'], IFS10orig['10v']) * 180/np.pi + 180)*np.pi/180)
        u = -IFS10_wake * np.sin((np.arctan2(IFS10orig['10u'], IFS10orig['10v']) * 180/np.pi + 180)*np.pi/180)
        #
        # final wake
        xr.merge([u.rename('u10').to_dataset(),v.rename('v10').to_dataset()]).to_netcdf(final_output_dir+'Baltic_NorthSea_regular_ws_IFS-NEMO_'+exp+'_'+str(year)+'_'+str(month).zfill(2)+'.nc')
