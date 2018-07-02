import os
def trimWav():
#change the range(10)

	in_path = os.listdir('./webData/wav')

	if not os.path.exists(r'''./webData/wavs'''):
		os.makedirs(r'''./webData/wavs''')    
	for z in in_path:
		r = './webData/wav/'+ z  
		out_path = './webData/wavs/out%03d{}'.format(z)
		subprocess.call(['ffmpeg', '-i', r, '-f', 'segment','-segment_time', '5', '-c', 'copy', out_path])
trimWav()		