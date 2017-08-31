import argparse
import numpy as np
from datetime import datetime
from colorsys import rgb_to_hsv, rgb_to_yiq, rgb_to_hls
t1 = datetime.now()

def red(pixel):
    return pixel[0]
def grn(pixel):
    return pixel[1]
def blu(pixel):
    return pixel[2]

def hue(pixel):
    return rgb_to_hsv(pixel[0], pixel[1], pixel[2])[0]
def saturation(pixel):
    return rgb_to_hsv(pixel[0], pixel[1], pixel[2])[1]
def brightness(pixel):
    return rgb_to_hsv(pixel[0], pixel[1], pixel[2])[2]

def luminance(pixel):
    return rgb_to_hls(pixel[0], pixel[1], pixel[2])[1]
def hue_(pixel):
    return rgb_to_hls(pixel[0], pixel[1], pixel[2])[0]
def saturation_(pixel):
    return rgb_to_hls(pixel[0], pixel[1], pixel[2])[2]

def luma(pixel): #approx luma_alt2
    return rgb_to_yiq(pixel[0], pixel[1], pixel[2])[0]
def chroma_blu(pixel):
    return rgb_to_yiq(pixel[0], pixel[1], pixel[2])[1]
def chroma_red(pixel):
    return rgb_to_yiq(pixel[0], pixel[1], pixel[2])[2]

def luma_alt1(pixel): #relative
    return pixel[0]*0.2126 + pixel[1]*0.7152 + pixel[2]*0.0722
def luma_alt2(pixel): #perceived
    return pixel[0]*0.299 + pixel[1]*0.587 + pixel[2]*0.114
def intensity(pixel): #dummy def, this is slow; refers to np.sum instead!
    return pixel[0] + pixel[1] + pixel[2]

def noizsort(array):             # noisy fragments
    order = np.lexsort(array.T)
    return array[order]

def tonesort(array):             # tone sort, sorts along channel axis
    return np.sort(array,axis=0)

def messort(array):              # mess sort, sorts along pixel axis
    return np.sort(array,axis=1)

def sort_pixels(array,scale,rank): # sort list of pixels
    if scale==intensity:
        if rank!=intensity: #if NOT ranked by intensity, use this fast calculation
            sumline = np.sum(array,axis=1)
            order = np.argsort(sumline)
            return array[order]
        else: #if ranked by intensity, do linear sort with 'sorted'
            return np.array(sorted(array, key=np.sum))
    elif scale==min:
        return array[np.argsort(np.min(array,axis=1))]
    elif scale==np.median:
        return array[np.argsort(np.median(array,axis=1))]
    elif scale==max:
        return array[np.argsort(np.max(array,axis=1))]
    else:
        return np.array(sorted(array, key=scale)) # stable linear sort

def rank_map(imaj,scale): # rank image pixels
    if scale==intensity:
        imgset = np.sum(imaj,axis=2) # intensity map, fast calculation
    elif scale==min:
        imgset = np.min(imaj,axis=2)
    elif scale==np.median:
        imgset = np.median(imaj,axis=2)
    elif scale==max:
        imgset = np.max(imaj,axis=2)
    else:
        imgset=[]
        for i in range(imaj.shape[0]):
            imgset.append(np.apply_along_axis(scale,1,imaj[i,:])) # treat pixels by row
        imgset=np.array(imgset)
    return imgset.ravel().argsort().argsort().reshape(imgset.shape) # rank map

def limits(s):
    try:
        lower, upper = map(float, s.split(','))
        return lower,upper
    except:
        raise argparse.ArgumentTypeError("Limits must be floats from 0 to 1: lower, upper")

def main():
    parz = argparse.ArgumentParser(description = "2D image sort via rank map and linear sort")
    parz.add_argument("image", help = "input image")
    parz.add_argument("-r", "--rank", help = "rank criterion", default = "intensity")
    parz.add_argument("-s", "--sort", help = "sort criterion", default = "intensity")
    parz.add_argument("-pil", "--PIL", help = "use PIL (default CV2)", action="store_true")
    parz.add_argument("-rev", "--reverse", help = "reverse linear sort", action="store_true")
    parz.add_argument("-ton", "--tone", help = "optional 'tone' sort on 'mess','noise' sort", action="store_true")
    parz.add_argument("-noiz", "--noiz", help = "optional 'noiz' sort on 'mess' sort", action="store_true")
    parz.add_argument("-ran", "--random", help = "random replacement fraction", type=float, action="store")
    parz.add_argument("-rep", "--replace", help = "replacement threshold limits", type=limits, nargs='*', action="store")
    parz.add_argument("-ptl", "--partial", help = "partial threshold limits", type=limits, nargs='*', action="store")
    parz.add_argument("-alt", "--alternate", help = "characteristic length for alternating 2D intervals", type=float, action="store")
    args = parz.parse_args()

    dict = {"min": min,
            "med": np.median,
            "max": max,
            "red": red,
            "blu": blu,
            "grn": grn,
            "hue": hue,
            "sat": saturation,
            "val": brightness,
            "hue_": hue_,
            "sat_": saturation_,
            "lum": luminance,
            "luma": luma,
            "cblu": chroma_blu,
            "cred": chroma_red,
            "luma1": luma_alt1,
            "luma2": luma_alt2,
            "intensity": intensity}
    algs = {"noiz": noizsort,
            "tone": tonesort,
            "mess": messort}
    try:
        comb = dict.copy()
        comb.update(algs)
        rank = comb[args.rank]
    except KeyError:
        print("WARNING: Unknown rank choice given, defaulting to 'intensity'.")
        rank = intensity

    try:
        comb = dict.copy()
        comb.update(algs)
        sort = comb[args.sort]
    except KeyError:
        print("WARNING: Unknown sort choice given, defaulting to 'intensity'.")
        sort = intensity

    ''' OPEN IMAGE & DERIVE RANK MAP '''
    if args.PIL:
        from PIL import Image
        img = Image.open(args.image)
        null = np.asarray(img)      # pixels read as RGB
        y1,x1,channels = null.shape
        imgrank=rank_map(null,rank) # choose rank system
        colz=np.reshape(null,(x1*y1,channels))
    else:
        import cv2
        img=cv2.imread(args.image)  # pixels read as BGR
        img=img[...,[2,1,0]] # change pixels to read as RGB for proper processing
        y1,x1,channels = img.shape
        imgrank=rank_map(img,rank)  # choose rank system
        colz=np.reshape(img,(x1*y1,channels))

    ''' PIXEL TREATMENT '''
    if args.sort in algs: scolz = sort(colz)  # choose sort system
    else: scolz = sort_pixels(colz,sort,rank)

    if args.tone or args.noiz:  #'tone' or 'noiz' sort on 'mess' sort
        if sort not in [messort]:
            if args.tone:
                print("NOTE: '-ton','--tone' is optional with *mess* sort!\n"+
                  "      Use '-s tone' for *tone* sort.")
            elif args.noiz:
                print("NOTE: '-noiz','--noiz' is optional with *mess* sort!\n"+
                      "      Use '-s noiz' for *noiz* sort.")
            pass
        else:
            if args.tone:
                scolz = tonesort(scolz)
            elif args.noiz:
                scolz = noizsort(scolz)

    if args.random:             # partial: random replace linear sort
        frc = args.random
        randint = np.random.choice(x1*y1,size=int(round(x1*y1*frc)))
        ncolz = colz[np.argsort(np.sum(colz,axis=1))] #BASIC: sum, min, median, max
        scolz[randint] = ncolz[randint]

    if args.replace:            # partial: thresh. replace linear sort
        llim,ulim=args.replace[0]
        li,ui=int(round(x1*y1*llim)),int(round(x1*y1*ulim))
        ncolz = colz[np.argsort(np.max(colz,axis=1))] #BASIC: sum, min, median, max
        scolz[li:ui] = ncolz[li:ui]

    if args.reverse: scolz = scolz[::-1] # reverse order

    if args.PIL:
        pass
    else:
        scolz = scolz[...,[2,1,0]] # change to read BGR for saving in CV2

    if args.partial or args.alternate:
        img0=img.copy()[...,[2,1,0]]

    ''' PIXEL ASSIGNMENT '''
#### CV2 methods
#    for i in range(x1): # assign each pixel, CV2
#        for j in range(y1):
#            img[j,i] = scolz[imgrank[j,i]]

#    for i in range(x1): # assign each column of image, CV2
#        img[:,i] = scolz[imgrank[:,i]]

    img = scolz[imgrank] # transform

    if args.partial:            # partial: thresh. unsort rank map
        llim,ulim=args.partial[0]
        li,ui=int(round(x1*y1*llim)),int(round(x1*y1*ulim))
        mask = (li<=imgrank) & (imgrank<=ui)
        img[mask]=img0[mask]

    if args.alternate:          # partial: alternate sort/unsort
        altz=round(args.alternate)
        if altz<1e4:
            print("WARNING: Alternating can be very slow for 2D intervals <10^4 (or 1e4).\n"+
                  "Defaulting to characteristic length 1e4.\n"+
                  "Please choose a larger value, or disable this warning if you wish.")
            altz=1e4
        delt=int(round(altz/10.)) #+/-10% variation
        total,count=0,0
        while total<x1*y1:
            part = altz + np.random.randint(-delt,delt)
            if count%2!=0: #if odd
                mask = (total<=imgrank) & (imgrank<=total+part)
                img[mask]=img0[mask]
            total+=part
            count+=1

#### PIL method
#### NOTE: to test this, comment out cv2 methods & "img = Image.fromarray(img)"!
#    img = Image.new("RGB", (x1,y1))
#    for i in range(x1): # assign each pixel, PIL
#        for j in range(y1):
#            img.putpixel((i,j),tuple(scolz[imgrank[j,i]]))

    ''' SAVE IMAGE '''
    outfile = args.image[:-4] + "-2ds.png"
    if args.PIL:
        img = Image.fromarray(img) #convert array for PIL
        img.save(outfile,optimize=1)
    else:
        cv2.imwrite(outfile,img)
    t2 = datetime.now(); print t2-t1


if __name__ == "__main__":
    main()
