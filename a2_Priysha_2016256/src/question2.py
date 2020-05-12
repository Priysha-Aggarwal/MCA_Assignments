import matplotlib.pyplot as plot
from scipy.io import wavfile
import glob
import matplotlib.pyplot as plt
from librosa import display
from librosa import load
from librosa import feature as librosa_feature
import numpy as np
import timeit
import pickle
from scipy.fftpack import dct

def DFT(matrix, sr):
	hann = np.hanning(int(25*0.001*sr))
	matrix = matrix*hann.reshape((hann.shape[0],1))
	calc = np.fft.fft(matrix, axis=0)
	calc = calc[:int(len(calc)/2) + 1]
	# calc1 = np.fft.rfft(matrix, axis = 0)
	# print (np.allclose(calc,calc1))
	return np.square(np.abs(calc))

def split(data, sr):
	samples_perframe = int(25*0.001*sr) #25 msec
	overlap = int(10*0.001*sr) #10 msec
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
	return np.array(frames).T #transpose or not?

def mel(f, inv=False):
	if inv==True:
		return 700*(np.power(10,f/2595) - 1)
	return 2595*np.log10(1+f/700)

def filters(sr):
	# low = mel(0)
	low = 0
	high = mel(sr/2)
	total_filters = 26
	total_points = total_filters + 2
	diff = (high-low)/((total_points-1)*1.0)
	mel_points = []
	fft_points = []
	for i in range(total_points):
		mel_points.append(low + i*diff)
		fft_points.append(mel(low + i*diff, True))
	mel_points = np.linspace(low, high, total_points)

	fft_bins = []
	samples_perframe = int(25*0.001*sr) #25 msec

	for i in fft_points:
		fft_bins.append(int(np.floor((samples_perframe + 1)*i/sr)))
	mel_filters = []
	for i in range(total_filters):
		start = fft_bins[i]
		mid = fft_bins[i+1]
		end = fft_bins[i+2]
		a = []
		for j in range(start):
			a.append(0)
		for j in range(mid-start):
			a.append(j/(1.0*(mid-start)))
		for j in range(end-mid):
			a.append((end-mid-j)/(end-mid))
		for j in range(int(samples_perframe/2) + 1 - end):
			a.append(0)
		mel_filters.append(a)
	return np.array(mel_filters)

def plot(samples, STFT, sr, i):
	signal_len = len(samples)
	time, freq = np.shape(STFT)
	print ("MFCC self, ", STFT.shape)
	fig = plt.figure(figsize=(7,4))
	plt.pcolormesh(STFT.T)
	plt.colorbar()
	plt.xlim([0, time-1])
	xlocs = np.linspace(0, time-1, 5)
	times = []
	for j in xlocs:
		times.append("%.2f" % (((j*signal_len/time) + 64)/sr))
	plt.xticks(xlocs, times)
	plt.yticks([])
	plt.xlabel("Time (in sec)")
	plt.ylabel("MFCC coefficients")
	plt.title('MFCC')
	plt.savefig("sampleMFCC.png", bbox_inches="tight")
	plt.show()

def start(paths, clas):
	print (clas)
	feature = []
	labels = clas*np.ones(len(paths))
	for i in paths:
		start = timeit.default_timer()
# <----------------------------------- READ ----------------------------------------->
		ipdata, sr = load(i, sr=None) #sr : sampling rate, data : signal
		# sr, data = wavfile.read(i)
		ipdata = np.pad(ipdata, (0, sr-len(ipdata)), mode='constant', constant_values=0)

	#inbuilt
		# print ("MFCC inbuilt", librosa_feature.mfcc(y=ipdata, sr=sr, n_mfcc=39).shape)
		# # plt.pcolormesh(librosa_feature.mfcc(y=ipdata, sr=sr, n_mfcc=39))
		# plt.colorbar()
		# plt.title('MFCC')
		# plt.tight_layout()
		# plt.show()
# <----------------------------------- PRE-EMPHASIS ----------------------------------------->
		samples = [ipdata[0]]
		for i in range(1,len(ipdata)):
			samples.append(ipdata[i]-0.95*ipdata[i-1])
		data = np.array(samples)
# <----------------------------------- FRAMING AND WINDOWING AND STFT ----------------------------------------->
		matrix = split(data, sr)
		STFT = DFT(matrix, sr) #POWER SPECTRUM
		if STFT.shape[0] < 201:
			STFT = np.vstack((STFT, np.zeros((201-STFT.shape[0], STFT.shape[1]))))
		if STFT.shape[0] > 201:
			STFT = STFT[:201,:]
		if STFT.shape[1] < 67:
			STFT = np.hstack((STFT, np.zeros((STFT.shape[0], 67-STFT.shape[1]))))
		if STFT.shape[1] > 67:
			STFT = STFT[:,:67]

# <----------------------------------- MEL FILTER BANK ----------------------------------------->
		mel_filters = filters(sr)
		filterbank_energies = np.dot(STFT.T, mel_filters.T)
		# print (filterbank_energies)
		filterbank_energies[np.where(filterbank_energies==0)] = filterbank_energies[np.where(filterbank_energies==0)] + 0.000000001

# <----------------------------------- Discrete Cosine Transform (DCT) ----------------------------------------->
# to decorrelate the filter bank coefficients and yield a compressed representation of the filter banks
#The reasons for discarding the other coefficients is that they represent fast changes in the filter bank coefficients and these fine details don’t contribute to Automatic Speech Recognition (ASR).
		filterbank_energies = dct(np.log(filterbank_energies), norm='ortho')[:,1:13]
		# print (np.shape(filterbank_energies)) #12 features per frame
		
#<---------------------------SPLIT INPUT DATA INTO FRAMES and FIND ENERGY ---------------------------------->
		matrix = np.square(split(ipdata, sr))
		energies = np.sum(matrix.T, axis=1)
		# energies = np.sum(STFT.T, axis = 1)
		energies = energies.reshape((energies.shape[0], 1))
		energies = np.vstack((energies, np.zeros((filterbank_energies.shape[0]-energies.shape[0],1))))
		filterbank_energies = np.hstack((filterbank_energies, energies))
		# print (filterbank_energies.shape) #13 features per frame

		concat = np.zeros(filterbank_energies.shape) #13 delta features
		concat2 = np.zeros(filterbank_energies.shape) #13 delta delta features
		zero = np.zeros((filterbank_energies.shape[0],1))
		pad = np.hstack((zero, filterbank_energies))
		pad = np.hstack((pad, zero))
		for j in range(1,pad.shape[0]-1):
			for k in range(1,pad.shape[1]-1):
				concat[j-1][k-1] = (pad[j][k+1] - pad[j][k-1])/2
		pad = np.hstack((zero, concat))
		pad = np.hstack((pad, zero))
		for j in range(1,pad.shape[0]-1):
			for k in range(1,pad.shape[1]-1):
				concat2[j-1][k-1] = (pad[j][k+1] - pad[j][k-1])/2
		
		mfcc = np.hstack((np.hstack((filterbank_energies, concat)), concat2)) #39 features per frame
		plot(data, mfcc, sr, i)
		feature.append(mfcc)
		# print (timeit.default_timer() - start)
	return feature, labels

# features = []
# labels = []
a, b = start(glob.glob("./training/zero/*.wav"), 0)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./training/one/*.wav"), 1)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./training/two/*.wav"), 2)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./training/three/*.wav"), 3)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./training/four/*.wav"), 4)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./training/five/*.wav"), 5)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./training/six/*.wav"), 6)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./training/seven/*.wav"), 7)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./training/eight/*.wav"), 8)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./training/nine/*.wav"), 9)
# features.extend(a)
# labels.extend(b)

# with open("trainingfeaturesMFCC.pkl", "wb") as f:
# 	pickle.dump(features, f)
# with open("traininglabelsMFCC.pkl", "wb") as f:
# 	pickle.dump(labels, f)

# features = []
# labels = []
# a, b = start(glob.glob("./validation/zero/*.wav"), 0)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./validation/one/*.wav"), 1)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./validation/two/*.wav"), 2)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./validation/three/*.wav"), 3)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./validation/four/*.wav"), 4)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./validation/five/*.wav"), 5)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./validation/six/*.wav"), 6)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./validation/seven/*.wav"), 7)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./validation/eight/*.wav"), 8)
# features.extend(a)
# labels.extend(b)
# a, b = start(glob.glob("./validation/nine/*.wav"), 9)
# features.extend(a)
# labels.extend(b)

# with open("validationfeaturesMFCC.pkl", "wb") as f:
# 	pickle.dump(features, f)
# with open("validationlabelsMFCC.pkl", "wb") as f:
# 	pickle.dump(labels, f)

# features = []
# labels = []
# a, b = start(glob.glob("./_background_noise_/*.wav"), -1)
# features.extend(a)
# labels.extend(b)

# with open("noisefeaturesMFCC.pkl", "wb") as f:
# 	pickle.dump(features, f)
# with open("noiselabelsMFCC.pkl", "wb") as f:
# 	pickle.dump(labels, f)