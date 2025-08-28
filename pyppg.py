from pyPPG import PPG, Fiducials, Biomarkers
from pyPPG.datahandling import load_data, plot_fiducials, save_data
import matplotlib.pyplot as plt
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.biomarkers as BM
import pyPPG.ppg_sqi as SQI
import numpy as np
import pandas as pd
import sys, os, json

# 데이터 읽어들이기
data_path = r"C:\Users\cream\OneDrive\Desktop\bioserver\white.csv"
df = pd.read_csv(data_path, on_bad_lines='skip')
raw_signal = df.iloc[1: , 2].to_numpy().flatten()
signal_df = pd.DataFrame(raw_signal, columns=['signal'])
file_name = 'signal_data.csv'
signal_df.to_csv(file_name, index=False, header=False)

# 초기 설정
start_sig = 1
end_sig = -1
savingfolder = 'temp_dir'
savingformat = 'csv'

data_path = os.path.abspath(file_name)

# 데이터 로드
signal = load_data(data_path=data_path)
signal.fs = 125
signal.v = signal.v [0:10000*signal.fs]

# 전처리 설정
signal.filtering = True # whether or not to filter the PPG signal
signal.fL=0.5000001 # Lower cutoff frequency (Hz)
signal.fH=5 # Upper cutoff frequency (Hz)
signal.order=4 # Filter order
signal.sm_wins={'ppg':50,'vpg':10,'apg':10,'jpg':10} # smoothing windows in millisecond for the PPG, PPG', PPG", and PPG'"

prep = PP.Preprocess(fL=signal.fL, fH=signal.fH, order=signal.order, sm_wins=signal.sm_wins)
signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)


# 기준점 찍힌 그래프
# Initialise the correction for fiducial points
corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
correction=pd.DataFrame()
correction.loc[0, corr_on] = True
signal.correction=correction

# Create a PPG class
s = PPG(signal)

fpex = FP.FpCollection(s=s)
fiducials = fpex.get_fiducials(s=s)

# Create a fiducials class
fp = Fiducials(fp=fiducials)

# Plot fiducial points
plot_fiducials(s, fp, savingfolder, legend_fontsize=12)

# Get PPG SQI
ppgSQI = round(np.mean(SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fp.sp)) * 100, 2)
print('Mean PPG SQI: ', ppgSQI, '%')

# Init the biomarkers package
bmex = BM.BmCollection(s=s, fp=fp)

# Extract biomarkers
bm_defs, bm_vals, bm_stats = bmex.get_biomarkers()
tmp_keys=bm_stats.keys()
print('Statistics of the biomarkers:')
for i in tmp_keys: print(i,'\n',bm_stats[i])

# Create a biomarkers class
bm = Biomarkers(bm_defs=bm_defs, bm_vals=bm_vals, bm_stats=bm_stats)

# Save PPG struct, fiducial points, biomarkers
# fp_new = Fiducials(fp.get_fp() + s.start_sig) # here the starting sample is added so that the results are relative to the start of the original signal (rather than the start of the analysed segment)
# save_data(s=s, fp=fp_new, bm=bm, savingformat=savingformat, savingfolder=savingfolder)
