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
                      nir=True,depth=True,confidence=True,\
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
        self.confidence = confidence
        self.settings = {}
        self.dev = None
        self.verbose = verbose
        self.colorized = (colourised or colorized)
        self.decimate = decimate
        self.store_meta_keys = None

        # active device lists
        self.active_devices = []

    def init(self,stop=True,self_test=True,verbose=False):
        '''
        find attached devices as start pipelines

        context derived from
        https://github.com/IntelRealSense/librealsense/tree/master/examples/multicam

        '''
        verbose = verbose or self.verbose
        self.context = rs.context()
        self.config = rs.config()

        #
        # possible multiple devices
        # we store this as text in self.dev_info
        #
        self.active_devices = self.start_all_devices()
        if verbose:
            print("Environment Ready")
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
  
            if (verbose):
                print('enabling {name}'.format(**device['device_info']))

            device['camera'] = self.enable_streams(device)
            self.config.enable_device(device['device_info']['serial_number'])
            device['pipeline_profile'] = self.start(pipeline=device['pipe'],config=self.config)

            if (verbose):
                print('starting sensors on {name} S/N: {serial_number}'.format(**device['device_info']))
            active_devices.append(device)

        self.active_devices = active_devices
        return active_devices
       
 
    def stop_all_devices(self,active_devices=None,verbose=False,fail=True):
        active_devices = active_devices or self.active_devices
        verbose = self.verbose or verbose
       
        for device in active_devices:
             try:
                self.stop(pipeline=device['pipe'],fail=fail)
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
        except Exception as e: 
            print(e)
            return None

    def stop(self,pipeline=None,fail=True):
        pipeline = pipeline or self.pipeline
        try:
            pipeline.stop()
        except Exception as e: 
            if fail:
                print(e)
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


    def load_settings(self,filename,append=False,apply=True,verbose=False):
        '''
        load settings from json file

        These can be written from the realsense viewer

        Use append=True to append to any existing settings

        use to set rs.option

        '''
        verbose = verbose or self.verbose
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

        if 'datasets' not in device:
            if verbose:
                print('unable to save pointcloud: no datasets')
            return None

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

    def enable_streams(self,device,\
                           depth=True,nir=True,confidence=True,\
                           colour=True,color=False,\
                           size=(1024,768),image_size=(1920,1080),\
                           verbose=False,\
                           fps=30,sample_rate=200):
        '''
        enable device stream for a single device
        '''
        dev = device['device']
        info = []

        # so we can modify
        # as you cant change tuple
        size = list(size)
        image_size = list(image_size)

       
        #
        # override if setting loaded from file
        #
        if "stream-width" in self.settings.keys():
            v = int(self.settings["stream-width"])
            size[0] = v
            if verbose:
                print(f'set "stream-width" to {v}')

        if "stream-height" in self.settings.keys():
            v = int(self.settings["stream-height"])
            size[1] = v
            if verbose:
                print(f'set "stream-height" to {v}')

        if "stream-fps" in self.settings.keys():
            v = int(self.settings["stream-fps"])
            fps = v
            if verbose:
                print(f'set "stream-fps" to {v}')
       
        size = tuple(size)
        image_size = tuple(image_size)
        #if confidence or self.confidence:
        #    fmt = rs.format.raw8
        #    info.append(['confidence', (rs.stream.confidence, *size, fmt ,fps),\
        #                           dev.first_depth_sensor()])
        if depth or self.depth:
            if "stream-depth-format" in self.settings.keys():
                v = self.settings["stream-depth-format"].lower()
                fmt = rs.format.__members__[v]
                if verbose:
                    print(f'set "stream-depth-format" to {v}')
            else:
                fmt = rs.format.z16
            info.append(['depth',       (rs.stream.depth, *size, fmt, fps),\
                                   dev.first_depth_sensor()])

        if (colour or color) or self.color:
            if "stream-color-format" in self.settings.keys():
                v = self.settings["stream-color-format"].lower()
                fmt = rs.format.__members__[v]
                if verbose:
                    print(f'set "stream-color-format" to {v}')
            else:
                fmt = rs.format.rgb8
            info.append(['color',       (rs.stream.color, *(image_size or size), fmt,fps),\
                                  dev.first_color_sensor()])
        if nir or self.nir:
            if "stream-ir-format" in self.settings.keys():
                v = self.settings["stream-ir-format"].lower()
                fmt = rs.format.__members__[v]
                if verbose:
                    print(f'set "stream-ir-format" to {v}')
            else:
                fmt = rs.format.y8
            info.append(['nir',         (rs.stream.infrared, *size, fmt,fps),\
                                   dev.first_depth_sensor()])

        verbose = verbose or self.verbose

        conf = self.config

        camera = {'depth_scale':0}

        camera['sets'] = []
        camera['sensors'] = []
        camera['bands'] = []

        for k,settings,sensor in info:

            if verbose:
                print(f'stream type: {k}')
                print(f'stream settings: {settings}')
                print(f'sensor: {sensor}')

            # apply settings
            for k2,v2 in self.settings.items():
                k_ = k2.replace(' ','_').lower()
                try:
                    if k_ in rs.option.__members__.keys():
                        sensor.set_option(rs.option.__members__[k_],v2)
                        if verbose:
                            print(k_,v2)
                except:
                    pass

            # this should set the parameters for the stream
            conf.enable_stream(*settings)
            try:
                camera['depth_scale'] = camera['depth_scale'] or sensor.get_depth_scale()
                if verbose:
                    print(f'depth scale: {camera["depth_scale"]}')
            except:
                pass

            camera['bands'].append(k)
            camera['sets'].append(settings)
            camera['sensors'].append(sensor)
            if verbose:
                print(camera)

        imu_streams = (rs.stream.accel,rs.stream.gyro)
        imu_formats = (rs.format.motion_xyz32f, rs.format.motion_xyz32f)
        imu_rates   = (sample_rate,sample_rate)

        for s,f,r in zip(imu_streams,imu_formats,imu_rates):
            conf.enable_stream(s, f, r)

        return camera

    def click(self,verbose=False,save=True,align=True,nframes=1):
        '''
        click
        '''
        ofiles = []
        
        for device in self.active_devices:
            # grab
            self.get_frames(device,nframes=nframes,verbose=verbose,align=align)
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

    def get_camera_meta(self,frames):
        '''
        get metadata
        new for each frame

        first pass, check all in
        rs.frame_metadata_value but cache for further calls

        '''
        
        meta = {}
        first_time = (self.store_meta_keys is None)
        keys = self.store_meta_keys or rs.frame_metadata_value.__members__.keys()
        self.store_meta_keys = self.store_meta_keys or []

        # then guess what might work
        for k in keys:
            try: 
                v = rs.frame_metadata_value.__members__[k]
                meta[k] = frames.get_frame_metadata(v)
                # this is good one so cache for next time
                if first_time:
                    self.store_meta_keys.append(k)
            except:
                pass
        return meta

    def get_frames(self,device,\
                        nframes=1,\
                        align=False,\
                        decimate=0,\
                        verbose=False,\
                        colourised=True,colorized=True):

        '''
        NB -- dont use nframes > 1 as it 
        bas blocking issues
      
        grab the frames from device
        and set in device['datasets']

        also set for camera = device['camera']:

            camera['bands']  
            - includes: 'depth', 'color', 'nir', 'cdepth'

            camera['intrinsics']
            camera['w']
            camera['h']

        Use camera['bands'] to select stream, e.g.:

            nd         = camera['bands'].index('depth')
            depth      = device['datasets'][nd]
            intrinsics = camera['intrinsics'][nd]

        '''
        decimate = decimate or self.decimate
        verbose = verbose or self.verbose
        colorized = (colourised or colorized) or self.colorized

        # Create an align object
        align_to = rs.stream.depth
        align = align and rs.align(align_to)

        try:
            del device['frames']
        except:
            pass

        # for storage of frames
        if nframes > 1:
            device['frames'] = {'camera'   : [], \
                                'datasets' : [] }

        # loop over frames
        for frame in range(nframes):
          camera = device['camera'].copy()
          if verbose:
              print(f'frame {frame}/{nframes} -------->')
          try:
            p = device['pipe']

            if verbose and not frame:
                print(f'entering capture loop using pipe {p}')
            if nframes > 1:
                frames = p.try_wait_for_frames()
            else:
                frames = p.wait_for_frames()

            # get metadata
            # new for each frame
            camera['meta'] = self.get_camera_meta(frames)

            if nframes == 1 and align:
                # Align the depth frame to color frame
                frames = align.process(frames)

            # loop over the sensors
            if verbose and not frame:
                print(f'gathering datasets for {camera["sets"]}')

            # need to try ... except in case asking for unobtainable
            datasets = []
            for i,s in enumerate(camera['sets']):
                try:
                    grab = frames.first(s[0])
                    datasets.append(grab)
                except:
                    if verbose and not frame:
                        print(f'error requesting {s} -- deleting')
                        del camera['sets'][i]
                        del camera['bands'][i]
                        del camera['sensors'][i]
            
            if verbose and not frame:
                print(f'{datasets}')
            
            nd = camera['bands'].index('depth')
            if verbose and not frame:
                print(f'depth frame index is {nd}')      

            depth = datasets[nd]
            if self.decimate_scale > 0:
                if verbose and not frame:
                    print(f'decimation by 2^{self.decimate_scale}')   
                depth = decimate.process(depth)

            if self.postprocessing and 'filters' in camera.keys():
                for f in camera['filters']:
                    if verbose and not frame:
                        print(f'applying post-processing filter {f}')
                    depth = f.process(depth)

            # load back in
            datasets[nd] = depth

            if colorized:
                if verbose and not frame:
                    print('adding colorized depth frame cdepth')
                colorizer = rs.colorizer()
                cdepth = colorizer.colorize(depth)
                datasets.append(cdepth)
                if 'cdepth' not in camera['bands']:
                    camera['bands'].append('cdepth')          

            # copy back in
            device['datasets'] = datasets
            device['camera'] = camera

            nd = camera['bands'].index('depth')
            
            # store profile
            camera['profile']    = [d.profile.as_video_stream_profile() for d in device['datasets']]
            # get intrinsics
            camera['intrinsics'] = [d.profile.as_video_stream_profile().intrinsics for d in device['datasets']]
            camera['w'] = camera['intrinsics'][nd].width
            camera['h'] = camera['intrinsics'][nd].height
            camera['timestamp'] = "{:016.6f}".format(depth.timestamp)
            camera['frame_number']     = "{:010d}".format(depth.frame_number)

            if nframes > 1:
                device['frames']['datasets'].append(datasets)
                device['frames']['camera'].append(camera)
            
          except:
            print('Warning: Failed to get frames')
            return device

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
            if 'datasets' in device:
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
            elif verbose:
                print('no dataset to plot')   

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
        figsize = figsize or ((10*1024)//768,10)
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
    self = lidar_control(verbose=True)
    self.load_settings('short_range_settings.json')
    self.init(stop=False)

    for frame in range(30):
        self.click(save=True,verbose=True)

    # d stores last frame information even when we stop devices
    d = self.stop_all_devices()
    self.plots(d,verbose=True,dummy=True)
    

if __name__== "__main__" :
    main()