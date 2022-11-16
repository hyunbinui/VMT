import argparse
from utils import *

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--idpath', type = str, required = True, help = 'youtube id list path')
    parser.add_argument('--output_path', type = str, default = './dataset/', help = 'text data output path')
    
    args = parser.parse_args()
    idpath = args.idpath
    output_path = args.output_path

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    id_list = txt_to_list(idpath)
    processed_sub = sub_processing(id_list)

    #create text data and video data
    list_to_json(processed_sub, output_path)
    create_video_data(output_path, id_list, processed_sub)


if __name__ == '__main__':
    main()