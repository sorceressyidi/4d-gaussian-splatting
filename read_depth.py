import numpy as np
import os
import imageio.v2 as imageio
from pathlib import Path
from colmapUtils.read_write_model import *
from colmapUtils.read_write_dense import *
import json

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    '''
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    '''
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'tmp','dense','images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
        

def get_poses(images):
    poses = []
    for i in images:
        R = images[i].qvec2rotmat()
        t = images[i].tvec.reshape([3,1])
        bottom = np.array([0,0,0,1.]).reshape([1,4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        poses.append(c2w)
    return np.array(poses)


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses_arr = poses_arr[1:]
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # 3 x 5 x N
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'tmp','dense','images', f) for f in sorted(os.listdir(os.path.join(basedir, 'tmp','dense','images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

    
            

def load_colmap_depth(basedir, factor=8, bd_factor=.75):
    data_file = Path(basedir) / 'colmap_depth.npy'
    
    images = read_images_binary(Path(basedir) /'tmp'/'dense'/ 'sparse' /'images.bin')
    points = read_points3d_binary(Path(basedir)/'tmp'/'dense' / 'sparse' / 'points3D.bin')

    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)
    
    poses = get_poses(images)
    _, bds_raw, _ = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
    # print(bds_raw.shape)
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds_raw.min() * bd_factor)
    
    near = np.ndarray.min(bds_raw) * .9 * sc
    far = np.ndarray.max(bds_raw) * 1. * sc
    print('near/far:', near, far)

    data_list = []
    for id_im in range(1, len(images)+1):
        depth_list = []
        coord_list = []
        weight_list = []
        for i in range(len(images[id_im].xys)):
            point2D = images[id_im].xys[i]
            id_3D = images[id_im].point3D_ids[i]
            if id_3D == -1:
                continue
            point3D = points[id_3D].xyz
            depth = (poses[id_im-1,:3,2].T @ (point3D - poses[id_im-1,:3,3])) * sc
            if depth < bds_raw[id_im-1,0] * sc or depth > bds_raw[id_im-1,1] * sc:
                continue
            err = points[id_3D].error
            weight = 2 * np.exp(-(err/Err_mean)**2)
            depth_list.append(depth)
            coord_list.append(point2D/factor)
            weight_list.append(weight)
        if len(depth_list) > 0:
            print(id_im, len(depth_list), np.min(depth_list), np.max(depth_list), np.mean(depth_list))
            data_list.append({"depth":np.array(depth_list), "coord":np.array(coord_list), "error":np.array(weight_list)})
        else:
            print(id_im, len(depth_list))
    # json.dump(data_list, open(data_file, "w"))
    np.save(data_file, data_list)
    return data_list

if __name__ == '__main__':
    basedir = 'data/N3V/cook_spinach'
    load_colmap_depth(basedir)
