import os

counter = 6

videos_path = os.path.abspath('media/raw_videos2')

import pdb; pdb.set_trace()
for file in os.scandir('media/raw_videos2'):
    print(dir(file))
    file_name = os.path.abspath(file.path)
    os.rename(file_name, os.path.join(videos_path, 'VIDEO_' + str(counter) + '.mp4'))
    counter += 1