import os
import math
# def gen_2d_gt(name):
#     path='./data/gt/jrdb/jrdb_2d_box/'+name+'/gt/gt.txt'
#     new_path='./data/gt/jrdb/jrdb_2d_box_real/'+name+'/gt/gt.txt'
#     file=open(path,'r')
#     with open(new_path, 'a') as new_gt:
#         lines=file.readlines()
#         for l in lines:
#             l=l.split(',')[:6]+[l.split(',')[-1].split('\n')[0]]+[-1,-1,-1]
#             l=[str(i) for i in l]
#             l=','.join(l)+'\n'
#             new_gt.write(l)
# def gen_2d_tracker(name):
#     path='/pvol/jrdb_dev/jrdb_website_dev/backends_dev/TrackEval/data/trackers/jrdb/jrdb_2d_box/mytracker/data/'+name+'_image_stitched.txt'
#     new_path='./data/trackers/jrdb/jrdb_2d_box/mytracker/data_real/'+name+'.txt'
#     file=open(path,'r')
#     with open(new_path, 'a') as new_gt:
#         lines=file.readlines()
#         for l in lines:
#             #print(l)
#             l=l.split(',')[:6]+[l.split(',')[-1].split('\n')[0]]+[-1,-1,-1]
#             l=[str(i) for i in l]
#             l=','.join(l)+'\n'
#             new_gt.write(l)
if __name__ == '__main__':
    dirs=[x[0].split('/')[-1] for x in os.walk('./data/gt/jrdb/jrdb_2d_box/') if x[0][-2:]!='gt' and x[0]!='']
    for name in dirs:
        if name!='':
            path='./data/gt/jrdb/jrdb_2d_box/'+name+'/'
            file=open(path+'gt/gt.txt','r')
            lines=file.readlines()
            seqLength=0
            for l in lines:
                seqLength=max(seqLength,int(l.split(',')[0]))
                pass
            print(seqLength)
            ini_path='./data/gt/jrdb/jrdb_2d_box/'+name+'/seqinfo.ini'
            with open(ini_path, 'a') as ini:
                ini.write('[Sequence]\n')
                ini.write('name='+name+'\n')
                ini.write('imDir=img1\n')
                ini.write('frameRate=15\n')
                ini.write('seqLength='+str(seqLength)+'\n')
                ini.write('imWidth='+str(3760)+'\n')
                ini.write('imHeight='+str(480)+'\n')
                ini.write('imExt=.jpg\n')
    