#!/usr/bin/env python 

'''
from 
https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/distance_to_object.ipynb
'''
try:
    import cv2                                # state of the art computer vision algorithms library
except:
    print("no cv2 available")

import numpy as np                        # fundamental package for scientific computing
import matplotlib
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import json
from pathlib import Path

try:
    import pyglet
    import pyglet.gl as gl
except:
    print("no pyglet or pyglet.gl available")


class lidar_control():
    '''
    librealsense2 wrappers for camera 
    control functions

    Author: P. Lewis
    Email:  p.lewis@ucl.ac.uk

    '''
 
    def __init__(self,decimate_scale=0,postprocessing=None,\
                      color=True,colour=True,\
                      nir=True,depth=True,\
                      decimate=0,\
                      colourised=True,colorized=False,\
                      verbose=False):
        '''
        setup
        '''
        self.decimate_scale = 0 or decimate_scale
        self.postprocessing = False or postprocessing
        self.depth = depth
        self.color = (colour or color)
        self.nir = nir
        self.settings = {}
        self.dev = None
        self.verbose = verbose
        self.colorized = (colourised or colorized)
        self.decimate = decimate

        # active device lists
        self.active_devices = []

    def init(self,stop=True,self_test=True):
        '''
        find attached devices as start pipelines

        context derived from
        https://github.com/IntelRealSense/librealsense/tree/master/examples/multicam

        '''
        self.context = rs.context()
        self.config = rs.config()

        #
        # possible multiple devices
        # we store this as text in self.dev_info
        #
        self.active_devices = self.start_all_devices()
       
        if stop:
            self.stop_all_devices(self.active_devices)
            return 1
        
    def start_all_devices(self,reload=True,verbose=False,active_devices=None):
        '''
        start all attached devices

        reload:
            - search again for all attached devices

        '''
        self.dev = (not reload and self.dev) or self.context.query_devices()
        verbose = verbose or self.verbose
        active_devices = active_devices or self.active_devices

        #
        # try to stop any hanging devices
        #
        self.stop_all_devices(active_devices,verbose)

        #
        # loop over devices
        #
        for i,d in enumerate(self.dev):
            if(verbose):
                print(f'device {i} of {len(self.dev)}')
            device = {'device':d}
            device['pipe'] = rs.pipeline(self.context)
            device['device_info'] = self.get_device_info(d)
            self.config.enable_device(device['device_info']['serial_number'])
            device['pipeline_profile'] = self.start(pipeline=device['pipe'],config=self.config)
            #
            # get the sensors on this device
            #
            device['sensors'] = device['pipeline_profile'].get_device().query_sensors()
            if (verbose):
                print('started {name} S/N: {serial_number}'.format(**device['device_info']))
            device['camera'] = self.enable_stream(device)
            active_devices.append(device)

        self.active_devices = active_devices
        return active_devices
       
 
    def stop_all_devices(self,active_devices=None,verbose=False):
        active_devices = active_devices or self.active_devices
        verbose = self.verbose or verbose
       
        for device in active_devices:
             try:
                self.stop(pipeline=device['pipe'])
                if verbose:
                    print('stopped {name} S/N: {serial_number}'.format(**device['device_info']))    
             except:
                pass

        self.active_devices = []
        return active_devices

    def get_device_info(self,dev):
        camera_info = {}
        for k in rs.camera_info.__members__.keys():
            try:
                camera_info[k] = dev.get_info(rs.camera_info.__members__[k])
            except:
                pass
        return(camera_info)
        
    def start(self,pipeline=None,config=None):
        pipeline = pipeline or self.pipeline
        config = config or self.config
        try:
            self.pipeline_profile = pipeline.start(config)
            return self.pipeline_profile
        except:
            return None

    def stop(self,pipeline=None):
        pipeline = pipeline or self.pipeline
        try:
            pipeline.stop()
        except:
            print("failed attempt to stop")
        self.pipeline_profile = None
        return self.pipeline_profile

    def convert_fmt(self,fmt):
        '''
        from pyglet_pointcloud_viewer.py
        Processing blocks
        
        rs.format to pyglet format string
        ''' 
        return {
          rs.format.rgb8: 'RGB',
          rs.format.bgr8: 'BGR',
          rs.format.rgba8: 'RGBA',
          rs.format.bgra8: 'BGRA',
          rs.format.y8: 'L',
        }[fmt]


    def load_settings(self,filename,append=False):
        '''
        load settings from json file

        These can be written from the realsense viewer

        Use append=True to append to any existing settings
        '''
        try:
            settings = json.load(open(filename,'r'))
            if append:
                self.settings.update(settings)
            else:
                self.settings = settings
        except:
            print(f'failed to load settings file {filename}')


    def save_pointcloud(self,device,pc_file=None,\
                            verbose=False,format='npz'):
        '''
        get pointcloud data from depth and image frames

        option to save to ply file if pc_file is set

        '''
        verbose = verbose or self.verbose

        image = device['datasets'][device['camera']['bands'] == 'color']
        depth = device['datasets'][device['camera']['bands'] == 'depth']
    
        pc_file = pc_file or 'lidar'
        fn = "{:010d}".format(depth.frame_number)
        op_file = Path(f'{pc_file}_{fn}.{format}').as_posix()
            
        pc = rs.pointcloud()
        pc.map_to(image)
        points = pc.calculate(depth)

        h,w = device['camera']['h'], device['camera']['w']

        verts = np.asarray(points.get_vertices(2)).reshape(h*w, 3)
        texcoords = np.asarray(points.get_texture_coordinates(2)).reshape(h*w,2)
        mask = np.isclose((verts*verts).sum(axis=1),0)
        verts = verts[~mask]
        texcoords = texcoords[~mask]
        if(verbose):
            print(f'{op_file}: {verts.shape[0]} points')
        np.savez(op_file,verts=verts,texcoords=texcoords)
        return op_file

    def is_nir_sensor(self,s):
        '''
        This is missing from librealsense, so hack it
        supposing that if we arent color or depth, we are NIR

        Could refine this -- eg has same size as depth
        '''
        return (not s.is_depth_sensor()) and (not s.is_color_sensor())

    def first_nir_sensor(self,dev):
        '''
        placeholder like is_nir_sensor
        '''
        return dev.first_depth_sensor()

    def enable_stream(self,device,\
                           depth=True,nir=True,colour=True,color=False,\
                           size=(1024,768),image_size=None,\
                           verbose=False,\
                           fps=30,sample_rate=200):
        '''
        enable device stream for a single device
        for all sensors in device['sensors']
        '''
        depth = depth or self.depth
        color = (colour or color) or self.color
        nir = nir or self.nir
        verbose = verbose or self.verbose

        conf = self.config
        profile = device['pipeline_profile']
        dev = device['device']

        camera = {'depth_scale':0}

	# lets see what sensors we have
        # there is no self.is_nir_sensor()
        # so we have a placeholder
        info = {
            'none'  : (None,0,0,0,0,0),
            'nir'   : (rs.stream.infrared, 0, *size, rs.format.y8,fps),
            'color' : (rs.stream.color, 0, *(image_size or size), rs.format.rgb8,fps),
            'depth' : (rs.stream.depth, 0, *size, rs.format.z16,fps)
        }

        camera['sets'] = []
        camera['sensors'] = []
        camera['bands'] = []
        for i,s in enumerate(device['sensors']):

            if verbose:
                print(f'sensor {i} of {len(device["sensors"])}')

            itype =    ((depth and s.is_depth_sensor() and 'depth') or \
                        (nir and self.is_nir_sensor(s) and 'nir') or \
                        (color and s.is_color_sensor() and 'color') or \
                         'none')

            settings = list(info[itype])
            settings[1] = i

            if verbose:
                print(f'stream type: {itype}')
                print(f'stream settings: {settings}')
            conf.enable_stream(*settings)

            sensor  =  (depth and s.is_depth_sensor() and dev.first_depth_sensor()) or \
                        (nir and self.is_nir_sensor(s) and self.first_nir_sensor(dev)) or \
                        (color and s.is_color_sensor() and dev.first_color_sensor())

            print(f'sensor: {sensor}')

            try:
                camera['depth_scale'] = camera['depth_scale'] or (depth and s.is_depth_sensor() and sensor.get_depth_scale())
            except:
                pass
            camera['bands'].append(itype)
            camera['sets'].append(settings)
            camera['sensors'].append(sensor)
            if verbose:
                print(camera)

        imu_streams = (rs.stream.accel,rs.stream.gyro)
        imu_formats = (rs.format.motion_xyz32f, rs.format.motion_xyz32f)
        imu_rates   = (sample_rate,sample_rate)

        for s,f,r in zip(imu_streams,imu_formats,imu_rates):
            conf.enable_stream(s, f, r)

        camera['depth_profile'] = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        camera['depth_intrinsics'] = camera['depth_profile'].get_intrinsics()
        camera['w'], camera['h'] = camera['depth_intrinsics'].width, camera['depth_intrinsics'].height
        return camera

    def nono(self):
  
        if self.decimate_scale > 0:
            decimate = rs.decimation_filter()
            decimate.set_option(rs.option.filter_magnitude, int(2 ** decimate_level))
        
        colorizer = rs.colorizer()
    
        if self.postprocessing:
            camera['filters'] = [rs.disparity_transform(),
               rs.spatial_filter(),
               rs.temporal_filter(),
               rs.disparity_transform(False)]

    def click(self,verbose=False,save=True):
        '''
        click
        '''
        ofiles = []
        for device in self.active_devices:
            # grab
            self.get_frames(device,verbose=verbose)
            if save:
                sn = device['device_info']['serial_number']
                dn = device['device_info']['name'].replace(' ','_')
                pc_dir = Path(dn).joinpath(sn)
                pc_dir.mkdir(parents=True, exist_ok=True)
                pc_file = pc_dir.joinpath('lidar')
                
                ofile = self.save_pointcloud(device,pc_file=pc_file,\
                            verbose=verbose,format='npz')
                ofiles.append(ofile)
        return(ofiles)

    def plot_frames(self):
        for device in self.active_devices:
            self.plot(device['datasets'],file=None)

    def get_frames(self,device,decimate=0,verbose=False,colourised=True,colorized=True):

        '''
        grab the frames from device
        '''
        decimate = decimate or self.decimate
        verbose = verbose or self.verbose
        colorized = (colourised or colorized) or self.colorized

        camera = device['camera']

        try:
            p = device['pipe']
            frames = p.wait_for_frames()
            # loop over the sensors
            datasets = [frames.first(s[0]) for s in camera['sets']]


            # do a colorized depth frame
            
            depth = datasets[camera['bands'] == 'depth']
            if self.decimate_scale > 0:
                depth = decimate.process(depth)

            if self.postprocessing and 'filters' in camera.keys():
                for f in camera['filters']:
                    if verbose:
                        print(f'applying post-processing filter {f}')
                    depth = f.process(depth)

            # load back in
            datasets[camera['bands'] == 'depth'] = depth

            if colorized:
                colorizer = rs.colorizer()
                cdepth = colorizer.colorize(depth)
                datasets.append(cdepth)

            if self.decimate_scale > 0:
                # Grab new intrinsics (may be changed by decimation)
                camera['depth_profile'] = rs.video_stream_profile(depth.profile)
                camera['depth_intrinsics'] = camera['depth_profile'].get_intrinsics()
                camera['w'], camera['h'] = camera['depth_intrinsics'].width, camera['depth_intrinsics'].height

            # copy back in
            device['datasets'] = datasets
            device['camera'] = camera

        except:
            pass

        return device

    def to_np(self,frames):
        '''
        convert camera rs data to numpy arrays

        assume 
        '''
        results = [np.asanyarray(f.get_data()) for f in frames]
        return results
    
    # copy our data to pre-allocated buffers, this is faster than assigning...
    # pyglet will take care of uploading to GPU
    def copy(self,dst, src):
        """copy numpy array to pyglet array"""
        # timeit was mostly inconclusive, favoring slice assignment for safety
        np.array(dst, copy=False)[:] = src.ravel()
        # ctypes.memmove(dst, src.ctypes.data, src.nbytes)

    def plots(self,active_devices=None,verbose=False,\
                         centiles=None,\
                         titles=None,\
                         cmaps=None,\
                         figsize=None,\
                         file=None,\
                         dummy=False,\
                         format='png',\
                         transpose=False):
        '''
        plot for all devices
        '''
        active_devices = active_devices or self.active_devices 

        for device in active_devices:
            frames = device['datasets']
            self.plot(frames,verbose=verbose,\
                         centiles=centiles,\
                         titles=titles,\
                         cmaps=cmaps,\
                         figsize=figsize,\
                         file=file,\
                         dummy=dummy,\
                         format=format,\
                         transpose=transpose)

    def plot(self,frames,verbose=False,\
                         centiles=None,\
                         titles=None,\
                         cmaps=None,\
                         figsize=None,\
                         file=None,\
                         dummy=False,\
                         format='png',\
                         transpose=False):

        result = self.to_np(frames)
        verbose = verbose or self.verbose

        try:
            plt.clf()
        except:
            if verbose:
                print('unable to do interactive plot')
            dummy = True

        if dummy:
            matplotlib.use('Agg')
        figsize = figsize or (20,10)
        nr = (2 * ((len (result) +1)// 2))
        print(nr)
        nx = nr-nr//2
        shape = (nx,nr//nx)
        if transpose:
            shape = tuple(np.array(shape).T)

        fig, axs = plt.subplots(shape[1],shape[0],figsize=figsize)
        axs = np.array(axs).flatten()

        cmaps = cmaps or [None,None,plt.get_cmap('gray'),None]
        titles = titles or ['range','colour','NIR','colourised depth']
        print (shape)
        centiles = centiles or [(10,70),(25,75),(25,95),(5,95)]
      
        for i in range(nr):
            try:
                r = np.array(result[i].copy())
                '''r[r==0] = np.nan'''
                r_reste = r[r>0]
                rmin = np.percentile(r_reste,centiles[i][0])
                rmax = np.percentile(r,centiles[i][1])

                im = axs[i].imshow(r,vmin=rmin,vmax=rmax,cmap=cmaps[i])
                axs[i].title.set_text(titles[i])
                if i != 1:
                   divider = make_axes_locatable(axs[i])
                   cax = divider.append_axes('right', size='5%', pad=0.05)
                   fig.colorbar(im, cax=cax, orientation='vertical')
            except:
                pass
    
        if dummy:
            file = file or 'lidar'
            fn = "{:010d}".format(frames[0].frame_number)
            op_file = f'{file}_{fn}.{format}'
            if verbose:
                print(f'saving plot to {op_file}')
            fig.savefig(op_file)

def main():
    self = lidar_control()
    self.load_settings('short_range_settings.json')
    self.init(stop=False)

    self.click(save=True,verbose=True)

    # d stores information 
    d = self.stop_all_devices()
    self.plots(d,verbose=True,dummy=True)
    

if __name__== "__main__" :
    main()