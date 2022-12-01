import csv
import numpy as np

file_path = "C:\\Users\\xy.DESKTOP-6GVRBJT\\Documents\\Python\\twopstim\\twopstim\\stimuli\\OutputFiles\\_stimulus_output_2017_1_9_9_54_37.txt"

# fieldnames=('tcurr','boutInd','epoch','fulltime','phase','theta','data')
with open(file_path,'r') as csvfile:
	stim_file = next(csvfile)#skip header
	next(csvfile)#skip header
	next(csvfile)#skip header with wrong timing
	reader = csv.DictReader(csvfile)
	
	# flickering freuqnecy per epoch in hz
	freq_1 = 16.0
	freq_2 = 30.0
	freq_3 = 32.0
	freq_4 = 25.0
	
	# phasenlaengen
	phase_1 = 1/freq_1
	phase_2 = 1/freq_2
	phase_3 = 1/freq_3
	phase_4 = 1/freq_4
	
	# to analyze quickly
	max_length = 10000
	
	phase = [0]*max_length
	on_all = [0]*max_length
	epoch = [0]*max_length
	fulltime = [0]*max_length
	tcurr = [0]*max_length 
	framedur = [0]*max_length 
	
	ii = 0
	#print "Stimfile:\n"
	#print stim_file
	
	reader.fieldnames=['frame',"tcurr",'boutInd','epoch','fulltime','phase','theta','data']
	for row in reader:
		if ii < max_length:
			epoch[ii] = float(row['epoch'])
			#if epoch[ii] == 1.0:# Just for epoch 1
			phase[ii] = float(row['phase'])
			on_all[ii] = float(row['theta'])
			fulltime[ii] = float(row['fulltime'])
			tcurr[ii] = float(row['tcurr'])	
			if ii > 2: # calc framduration average
				framedur[ii] = fulltime[ii]-fulltime[ii-1]
			ii += 1
		
	# Average frameduration
	framedur_cut = framedur[:ii]
	frameduration = np.median(framedur_cut)
	#frameduration = sum(framedur_cut)/float(len(framedur_cut))
	print "\n\nFrameduration average: %f" %(frameduration)
	length_file = ii
	print "Length File: %d\n" %(length_file)
	
	# cut list
	on = on_all[:length_file]
	# stores no. of 1's or 0's in a row
	maximum_1s = [[0 for x in range(2)] for y in range(length_file)]
	maximum_0s = [[0 for x in range(2)] for y in range(length_file)]
	diff = [0]*length_file
	j = 0
	k = 0
	
	if length_file > max_length:
		length_file = max_length
	
	for ii in range(length_file-1):
		diff[ii] = on[ii+1]-on[ii]
		
		if(int(on[ii]) == 1):
			maximum_1s[j][0] += 1;
			#print "%d %d %1.3f  %d" %(on[ii],maximum_1s[j][0], tcurr[ii],j)
		if int(diff[ii]) == -1: # this phasis part has ended
			maximum_1s[j][1] = tcurr[ii] # record end time
			j += 1;
			 
			
		# Off phasis part
		if(int(on[ii]) == 0):
			maximum_0s[k][0] += 1;
			#print "%d %d %d %1.3f  %d" %(ii, on[ii],maximum_0s[k][0], tcurr[ii],k)
		if int(diff[ii]) == 1:
			maximum_0s[k][1] = tcurr[ii] # record end time
			k += 1;
			
	# Cut array, there are counting mistakes
	maximum_0s[0][0] = 0
	maximum_0s[k][0] = 0
	

	if "16_16" in stim_file:
		print "Stimulus with 16 Hz: Phasis-lenght= 1/16 = %f, -> ca. %1.1f frames per phasis" % (phase_1,phase_1/frameduration)
		print "Regularly, half a phasis should contain %1.1f frames." %(phase_1/frameduration/2)
		reg_dur_0 = phase_1/frameduration/2 # frames per half phasis
	elif "30_30" in stim_file:
		print "Stimulus with 30 Hz: Phasis-lenght= 1/30 = %f, -> ca. %1.1f frames per phasis" % (phase_2,phase_2/frameduration)
		print "Regularly, half a phasis should contain %1.1f frames." %(phase_2/frameduration/2)
		reg_dur_0 = phase_2/frameduration/2 # frames per half phasis
	elif "32_32" in stim_file:
		print "Stimulus with 32 Hz: Phasis-lenght= 1/32 = %f, -> ca. %1.1f frames per phasis\n" % (phase_3,phase_3/frameduration)
		print "Regularly, half a phasis should contain %1.1f frames." %(phase_3/frameduration/2)
		reg_dur_0 = phase_3/frameduration/2 # frames per half phasis
	elif "25_25" in stim_file:
		print "Stimulus with 25 Hz: Phasis-lenght= 1/25 = %f, -> ca. %1.1f frames per phasis\n" % (phase_4,phase_4/frameduration)
		print "Regularly, half a phasis should contain %1.1f frames." %(phase_4/frameduration/2)
		reg_dur_0 = phase_4/frameduration/2 # frames per half phasis
	
	
	for ii in range(length_file):
		if maximum_0s[ii][0] == 0 and ii > 0:
			end_index = ii;
			break
			
	maximum_0s = maximum_0s[:end_index]
	
	for ii in range(length_file):
		if maximum_1s[ii][0] == 0 and ii > 0:
			end_index = ii;
			break
			
	maximum_1s = maximum_1s[:end_index]
	
	# max along first axis
	max_0 = np.amax(maximum_0s,0)
	max_1 = np.amax(maximum_1s,0)
	
	print "\nMaximum 0s (No. of frames in a row which are black, e.g. off part of phasis)"
	print max_0
	print "Deviation of max: %2.1f %% error" %((max_0[0]-reg_dur_0)/reg_dur_0 * 100)
	print "Median: %2.1f  ,Average: %2.1f  ,Std Dev: %2.2f , in percent: %2.2f %%\n" %(np.median(maximum_0s,0)[0],np.average(maximum_0s,0)[0],np.std(maximum_0s,0)[0],np.std(maximum_0s,0)[0]/reg_dur_0 *100)

	print "Maximum 1s (No. of frames in a row which are bright, e.g. on part of phasis)"	
	print max_1
	print "Deviation of max: %2.1f %% error" %((max_1[0]-reg_dur_0)/reg_dur_0 * 100)
	print "Median: %2.1f  ,Average: %2.1f  ,Std Dev: %2.2f , in percent: %2.2f %%" %(np.median(maximum_1s,0)[0],np.average(maximum_1s,0)[0],np.std(maximum_1s,0)[0],np.std(maximum_1s,0)[0]/reg_dur_0 *100)
	
	