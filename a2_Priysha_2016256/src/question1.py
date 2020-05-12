import matplotlib.pyplot as plot
from scipy.io import wavfile
import glob
import matplotlib.pyplot as plt
from librosa import display
from librosa import load
import numpy as np
import timeit
import pickle
from args import get_parser
import cmath

def DFTonly(data):
	N = len(data) #data can be a matrix here
	vec = np.linspace(0,N-1,num=N).reshape((N,1))
	row = np.linspace(0,N-1,num=N)
	M = np.zeros((vec*row).shape)
	M = M.astype('complex128')
	for i in range(M.shape[0]):
		for j in range(M.shape[1]):
			M[i][j] = cmath.rect(1.0, -2*(np.pi*vec[i]*row[j])/N)
	feature = np.dot(M, data)
	return feature

def FFT(data):
	N = len(data)
	M = int(N/2)
	even = np.zeros(M)
	odd = np.zeros(M)
	if N <=32:
		vec = np.linspace(0,N-1,num=N).reshape((N,1))
		row = np.linspace(0,N-1,num=N)
		M = np.zeros((vec*row).shape)
		M = M.astype('complex128')
		for i in range(M.shape[0]):
			for j in range(M.shape[1]):
				M[i][j] = cmath.rect(1.0, -2*(np.pi*vec[i]*row[j])/N)
		feature = np.dot(M, data)
		return feature
	for i in range(M):
		even[i] = data[2*i]
		odd[i] = data[(2*i)+1]
	even = FFT(even)
	odd = FFT(odd)
	freqbins = np.zeros(N)
	freqbins = freqbins.astype('complex128')
	for i in range(M): #iterating till Nyquist limit
		c = cmath.rect(1.0, -2*(np.pi*i)/N)
		val = c*odd[i]
		freqbins[i] = even[i] + val
		c = cmath.rect(1.0, -2*(np.pi*(i+M))/N)
		val = c*odd[i]
		freqbins[i + M] = even[i] + val
	return freqbins

def DFT(matrix, sr, window):
	hann = np.hanning(int(window*0.001*sr))
	matrix = matrix*hann.reshape((hann.shape[0],1))
	# print ("shape", matrix.shape)

	#own
	fftout = np.zeros(matrix.shape)
	fftout = fftout.astype('complex128')
	# fftout = DFTonly(matrix) #DFT only applied
	for i in range(matrix.shape[1]):
		# print (matrix[:,i].shape)
		fftout[:,i] = FFT(matrix[:,i])
	fftout = fftout[:int(len(fftout)/2) + 1]
	return np.square(np.abs(fftout))

	#<----------------------------inbuilt ---------------------------->
	# calc = np.fft.fft(matrix, axis=0)
	# # print (calc.shape)
	# # print (np.allclose(calc, fftout))
	# # print (calc)
	# # print (fftout)
	# calc = calc[:int(len(calc)/2) + 1]
	# # calc1 = np.fft.rfft(matrix, axis = 0)
	# # print (np.allclose(calc,calc1))
	# return np.square(np.abs(calc))

def split(data, sr, window, overlap):
	samples_perframe = int(window*0.001*sr) #25 msec
	overlap = int(overlap*0.001*sr) #10 msec
	frames = []
	count = 0
	while count+samples_perframe <= data.shape[0]:
		frame = data[count:count+samples_perframe]
		frames.append(frame)	
		count = count + samples_perframe - overlap
	if count!= data.shape[0]:
		zero = np.zeros(samples_perframe-(data.shape[0]-count))
		a = data[count:]
		a = np.hstack((a,zero))
		frames.append(a)
	return np.array(frames).T

def plot(samples, STFT, sr, i):
	print (i)
	signal_len = len(samples)
	freq, time = np.shape(STFT)
	print (freq)
	fig = plt.figure(figsize=(15,7.5))
	plt.pcolormesh(STFT, cmap=plt.cm.jet)
	plt.colorbar()
	plt.xlim([0, time-1])
	xlocs = np.linspace(0, time-1, 5)
	times = []
	for j in xlocs:
		times.append("%.2f" % (((j*signal_len/time) + 64)/sr))
	plt.ylim([0, freq])
	ylocs = np.int16(np.round(np.linspace(0, freq-1, 10)))
	freq = []
	for j in ylocs:
		freq.append(int(j*sr/(400+1)))
	plt.xticks(xlocs, times)
	plt.yticks(ylocs, freq)
	plt.xlabel("Time in seconds")
	plt.ylabel("Freq. in Hertz")
	plt.title('Spectogram features')
	plt.savefig("Kshitiz.png", bbox_inches="tight")
	plt.show()

#inbuilt
	# from scipy import signal
	# # print (i)
	# fs, data = wavfile.read(i)
	# print (i)
	# f, t, Sxx = signal.spectrogram(samples, sr)
	# plt.pcolormesh(t, f, Sxx)
	# plt.ylabel('Frequency [Hz]')
	# plt.xlabel('Time [sec]')
	# plt.show()

def start(paths, clas, args):
	print (clas)
	feature = []
	labels = clas*np.ones(len(paths))
	for i in paths:
		# start = timeit.default_timer()
		data, sr = load(i, sr=None)# sr : sampling rate, data : signal
		if args.argspass:
			sr = args.sr
		data = np.pad(data, (0, sr-len(data)), mode='constant', constant_values=0)
		# data = np.asarray(data, dtype=np.uint8)
		# sr, data = wavfile.read(i)
		# data = data.reshape((data.shape[0],1))
		matrix = split(data, sr, args.window, args.overlap)
		STFT = DFT(matrix, sr, args.window)
		if STFT.shape[0] < 201:
			STFT = np.vstack((STFT, np.zeros((201-STFT.shape[0], STFT.shape[1]))))
		if STFT.shape[0] > 201:
			STFT = STFT[:201,:]
		if STFT.shape[1] < 67:
			STFT = np.hstack((STFT, np.zeros((STFT.shape[0], 67-STFT.shape[1]))))
		if STFT.shape[1] > 67:
			STFT = STFT[:,:67]
		plot(data, STFT, sr, i)
		feature.append(STFT)
		# print (timeit.default_timer() - start)
	return feature, labels

args = get_parser()
# features = []
# labels = []
a, b = start(glob.glob("./training/zero/*.wav"), 0, args)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./training/one/*.wav"), 1, args)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./training/two/*.wav"), 2, args)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./training/three/*.wav"), 3, args)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./training/four/*.wav"), 4, args)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./training/five/*.wav"), 5, args)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./training/six/*.wav"), 6, args)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./training/seven/*.wav"), 7, args)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./training/eight/*.wav"), 8, args)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./training/nine/*.wav"), 9, args)
# features.extend(a)
# labels.extend(b)

# with open("trainingfeaturesSPEC.pkl", "wb") as f:
# 	pickle.dump(features, f)
# with open("traininglabelsSPEC.pkl", "wb") as f:
# 	pickle.dump(labels, f)

# features = []
# labels = []
# a, b = start(glob.glob("./validation/zero/*.wav"), 0, args)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./validation/one/*.wav"), 1, args)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./validation/two/*.wav"), 2, args)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./validation/three/*.wav"), 3, args)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./validation/four/*.wav"), 4, args)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./validation/five/*.wav"), 5, args)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./validation/six/*.wav"), 6, args)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./validation/seven/*.wav"), 7, args)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./validation/eight/*.wav"), 8, args)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./validation/nine/*.wav"), 9, args)
# features.extend(a)
# labels.extend(b)

# with open("validationfeaturesSPEC.pkl", "wb") as f:
# 	pickle.dump(features, f)
# with open("validationlabelsSPEC.pkl", "wb") as f:
# 	pickle.dump(labels, f)

# features = []
# labels = []
# a, b = start(glob.glob("./_background_noise_/*.wav"), -1, args)
# features.extend(a)
# labels.extend(b)

# with open("noisefeaturesSPEC.pkl", "wb") as f:
# 	pickle.dump(features, f)
# with open("noiselabelsSPEC.pkl", "wb") as f:
# 	pickle.dump(labels, f)

#<--------------------------------------------- Source for FFT ------------------------------------------>
#https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
#https://towardsdatascience.com/fast-fourier-transform-937926e591cb
#The FFT Algorithm - Simple Step by Step - https://www.youtube.com/watch?v=htCj9exbGo0&t=349s