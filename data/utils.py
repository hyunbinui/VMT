from pathlib import Path
import webvtt
import os, json
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def txt_to_list(txtpath):
    '''convert id_list.txt(list of video ids) to list'''
    id_list = open(txtpath, "r")
    id = id_list.read()

    id_to_list = id.replace('\n', ',').split(",")

    for id in id_to_list:
        if len(id) == 0:
            id_to_list.remove(id)

    id_list.close()

    # for validation. remove ids that do not contain both ko and en sub
    path = "./original_subs/"

    rmv = []
    for i in id_to_list:
        if (os.path.isfile(path+i+'.ko.vtt') == False or os.path.isfile(path+i+'.en.vtt') == False):
            rmv.append(i)

    id_to_list = [e for e in id_to_list if e not in rmv]

    return id_to_list


def sub_processing(idlist):
    '''get necessary info from vtt file + append them in list'''
    full = []
    text = []

    path = './original_subs/'

    for i in idlist:
        mid = []
        #read ko sub file
        for caption in webvtt.read(path+i+'.ko.vtt'):
            # basic text preprocessing
            caption.text=caption.text.strip().replace('\n', ' ')
            caption.text=caption.text.replace('-', '')
            caption.text=caption.text.replace('- ', '')

            time_text_list=[]
            time_text_list.append(i)
            time_text_list.append(caption.start[:-4])
            time_text_list.append(caption.end[:-4])
            time_text_list.append(caption.text)
            mid.append(time_text_list)

        mid1 = []
        #read en sub file
        for caption1 in webvtt.read(path+i+'.en.vtt'):
            # basic text preprocessing
            caption1.text=caption1.text.strip().replace('\n', ' ')
            caption1.text=caption1.text.replace('-', '')
            caption1.text=caption1.text.replace('- ', '')

            time_text_list1=[]
            time_text_list1.append(caption1.start[:-4])
            time_text_list1.append(caption1.end[:-4])
            time_text_list1.append(caption1.text)
            mid1.append(time_text_list1)
        
        if len(mid) == len(mid1):
            full.append(mid)
            text.append(mid1)

    # each line in youtube vtt files is often splitted sentence. Need to get them together as complete sentences.
    id_ko_line=[]
    en_line=[]
    time=[]

    for i in range(len(full)):
        middle = []
        middle1 = []
        middle2 =[]

        middle.append(full[i][0][0])
        middle.append(full[i][0][3])
        middle1.append(full[i][0][1]+"~"+full[i][0][2])
        middle2.append(text[i][0][2])

        # if sub does not end with ['?',".","!",'(',")", "~"], concat sub with following sub
        for n in range(len(full[i])-1):
            if middle[-1][-1] not in ['?',".","!",'(',")", "~"]:
                middle[-1] = middle[-1] + ' ' + full[i][n+1][3]
                middle1[-1] = middle1[-1] + "~" + full[i][n+1][2]
                middle2[-1] = middle2[-1] + ' ' + text[i][n+1][2]

            elif middle[-1][-1] in ['?',".","!",'(',")", "~"]:
                middle.append(full[i][n+1][3])
                middle1.append(full[i][n+1][1]+"~"+full[i][n+1][2])
                middle2.append(text[i][n+1][2])

        id_ko_line.append(middle)
        en_line.append(middle2)
        time.append(middle1)

    #concat time and sentence data
    final = []
    for i in range(len(time)):
        mid = []
        for t in range(len(time[i])):
            mid2 = []
            mid2.append(id_ko_line[i][0])
            mid2.append(time[i][t][:8])
            mid2.append(time[i][t][-8:])
            mid2.append(id_ko_line[i][t+1])
            mid2.append(en_line[i][t])
            mid.append(mid2)
        final.append(mid)

    # make video name + append it to 'final' list
    for f in range(len(final)):
        for u in range(len(final[f])):
            vid_name = final[f][u][0]+"_"+final[f][u][1].replace(":","")+'_'+final[f][u][2].replace(":","")
            final[f][u].append(vid_name)

    # final text preprocessing ; remove leading whitespace / double spaces
    for f in range(len(final)):
        for i in range(len(final[f])):
            final[f][i][3] = final[f][i][3].lstrip()
            final[f][i][3] = final[f][i][3].replace("  ", " ")

    # for better 1:1 matching btw en sub and ko sub
    fin = []
    for f in range(len(final)):
        for i in range(len(final[f])):
            if final[f][i][4][-1] in ['.', '!', '?', ')','~'] and final[f][i][4][0].isupper() == True:
                fin.append(final[f][i])
    return fin


def list_to_json(list, output_path):
    '''create text dataset with (video ID - kor sub - eng sub) pair'''
    # first, create dictionary with (video id - ko sub - eng sub) pair
    Dict = {}
    for f in list:
        Dict[f[-1]] = {}
        Dict[f[-1]]['ko'] = f[3]
        Dict[f[-1]]['en'] = f[4]
    
    # then, create json file based on 'Dict'
    with open(output_path+"text_data.json", "w", encoding='utf-8') as outfile:
        json.dump(Dict, outfile, indent=2 , ensure_ascii=False)


def create_video_data(outputpath, idlist, processedsub):
    '''create video data, trimmed based on timestamps'''
    # make directory for trimmed video clips
    if not os.path.exists(outputpath+'/video_data/'):
        os.makedirs(outputpath+'/video_data/')
        
    # get seconds from time.
    def get_sec(time_str):
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + int(s)

    # cut the video clips based on timestamps
    for i in range(len(processedsub)):

        input_video_path = './original_video/'+processedsub[i][-1][:-14]+".mp4"
        output_video_path = outputpath+'video_data/'+processedsub[i][-1]+'.mp4'

        video_file = VideoFileClip(input_video_path)
        fullduration = video_file.duration

        start = processedsub[i][1]
        start = get_sec(start)
        end = processedsub[i][2]
        end = get_sec(end)


        # duration has to be longer than 1 sec 
        if start != end:
                # end time should be smaller than full duration of original video
                if end > fullduration:
                    end = None
                else :
                    ## method 1 : video + audio
                    # with VideoFileClip(input_video_path) as video:
                        # new = video.subclip(start, end)
                        # new.write_videofile(output_video_path, codec='libx264')

                    # method 2 : only video (no audio)
                    ffmpeg_extract_subclip(input_video_path, start, end, targetname = output_video_path)


