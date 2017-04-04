import matplotlib.pyplot as plt
import numpy as np
import proto as p


def load_files(file_names, units):
	data = {}
	index= 0
	for f in file_names:
		key = f.split('.')[0]+'_'+units[index]
		print key
		data[key] = np.load(f)[units[index]]
		index += 1
	return data

def calc_winlen_shift(samplingrate, winlen_ms, shift_ms):
	t = 1.0/samplingrate
	winlen = winlen_ms/t
	shift = shift_ms/t

	return winlen, shift

def plot_enframe(ex_frames, calc_frames):

	plt.figure(1)
	plt.subplot(211)
	plt.plot(ex_frames)

	plt.subplot(212)
	plt.plot(calc_frames)
	plt.show()

	return

def Assignment1():

	file_names = ['example.npz', 'tidigits.npz']
	units = ['example', 'tidigits']
	data = load_files(file_names,units)
	e = data['example_example'].item()
	ex_frames = e['frames']
	samples = e['samples']
	samplingrate = e['samplingrate']
	winlen, shift = calc_winlen_shift(samplingrate,0.020, 0.010)
	calc_frames = p.enframe(samples, winlen, shift)


	#plot_enframe(ex_frames,calc_frames)

	calc_preemph = p.preemp(calc_frames, p=0.97)
	ex_preemph = e['preemph']
	if np.array_equal(ex_preemph, calc_preemph):
		print 'yesai1'

	calc_windowed = p.windowing(calc_preemph)
	ex_windowed = e['windowed']
	if np.array_equal(np.round(ex_windowed,9), np.round(calc_windowed,9)): #Correct to 1e-9
		print 'yesai2'

	calc_fft = p.powerSpectrum(calc_windowed,512)
	ex_fft = e['spec']
	if np.array_equal(np.round(calc_fft,4), np.round(ex_fft,4)): # correct 1e-4
		print 'yesai3'

	calc_mel = p.logMelSpectrum(calc_fft, samplingrate)
	ex_mel = e['mspec']
	if np.array_equal(np.round(calc_mel,11), np.round(ex_mel,11)): # correct 1e-11
		print 'yesai4'

	calc_mfcc = p.cepstrum(calc_mel,13)
	ex_mfcc = e['mfcc']
	if np.array_equal(np.round(calc_mfcc,1), np.round(ex_mfcc,1)): # correct 1e-11
		print 'yesai5'

		return

def Assignment2():
	ti = np.load('tidigits.npz')['tidigits']
	for index,item in enumerate(ti):
		if index == 0:
			f = p.mfcc(item['samples'])
		else:
			f = np.vstack((f,p.mfcc(item['samples'])))
	corr = np.corrcoef(f)
	p.plot(corr)


if __name__ == '__main__':
	Assignment1()
	#Assignment2()

