# non-GUI 백엔드 설정
import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, redirect, url_for, session, abort
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, io, base64

from biosppy.signals import ppg as biosppy_ppg
from pyPPG import PPG, Fiducials, Biomarkers
import pyPPG.biomarkers as BM
import pyPPG.ppg_sqi as SQI
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
from pyPPG.datahandling import load_data, plot_fiducials

# 한글 폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)

# 세션 비밀 키
app.secret_key = 'a-very-secure-random-key-for-this-app'

# 업로드 폴더
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 샘플링 레이트 (디폴트 50Hz)
DEFAULT_SAMPLING_RATE = 50

# biosppy
def generate_biosppy_plot(signal, sr, filename):
    # 신호 처리
    output = biosppy_ppg.ppg(signal=signal, sampling_rate=sr, show=False)
    
    available_keys = output.keys()
    peak_key = 'peaks' if 'peaks' in available_keys else 'onsets'
    peak_count = len(output[peak_key]) if peak_key in available_keys else 0

    # 그래프 출력
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'BioSPPy Analysis: {filename}', fontsize=16)

    ts, filtered = output['ts'], output['filtered']
    ax1.plot(ts, signal[:len(ts)], label='Raw Signal')
    ax1.set_ylabel('Amplitude'); ax1.legend(); ax1.grid(True)
    
    ax2.plot(ts, filtered, label='Filtered Signal')
    if peak_count > 0:
        ax2.plot(ts[output[peak_key]], filtered[output[peak_key]], 'mo', markersize=5, label=f'{peak_key.capitalize()}s')
    ax2.set_ylabel('Amplitude'); ax2.legend(); ax2.grid(True)

    if 'heart_rate_ts' in available_keys and 'heart_rate' in available_keys:
        ts_data, hr_data = output['heart_rate_ts'], output['heart_rate']
        if hasattr(ts_data, 'size') and ts_data.size > 0:
            ax3.plot(ts_data, hr_data, label='Heart Rate')
    
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Heart Rate (bpm)'); ax3.legend(); ax3.grid(True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    ppg_sqi = 0
    return fig, peak_count, ppg_sqi

# pyPPG
def generate_pyppg_plot(signal, sr, filename, temp_dir):
    temp_file_path = None
    original_show = plt.show
    fig = None
    try:
        
        temp_file_path = os.path.join(temp_dir, f"temp_{os.urandom(8).hex()}.csv")
        pd.DataFrame(signal).to_csv(temp_file_path, index=False, header=False)

        # 데이터 로드
        signal = load_data(data_path=temp_file_path)
        signal.fs = sr
        signal.v = signal.v [0:10000*signal.fs]
        
        # 전처리 설정
        signal.filtering = True
        signal.fL=0.5
        signal.fH=12
        signal.order=4
        signal.sm_wins={'ppg':50,'vpg':10,'apg':10,'jpg':10}

        prep = PP.Preprocess(fL=signal.fL, fH=signal.fH, order=signal.order, sm_wins=signal.sm_wins)
        signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)
        
        # 기준점 설정
        corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
        correction=pd.DataFrame()
        correction.loc[0, corr_on] = True
        signal.correction=correction
        
        # Create a PPG class
        s = PPG(signal)
        fpex = FP.FpCollection(s=s)
        fiducials = fpex.get_fiducials(s=s)
        fp = Fiducials(fp=fiducials)
        peak_count = len(fp.sp)
        
        # PPG 품질 점수
        ppg_sqi = round(np.mean(SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fp.sp)) * 100, 2)
        
        # PPG 바이오마크
        """ bmex = BM.BmCollection(s=s, fp=fp)

        bm_defs, bm_vals, bm_stats = bmex.get_biomarkers()
        tmp_keys=bm_stats.keys()
        print('Statistics of the biomarkers:')
        for i in tmp_keys: print(i,'\n',bm_stats[i])

        bm = Biomarkers(bm_defs=bm_defs, bm_vals=bm_vals, bm_stats=bm_stats)"""
        
        # 그래프 출력
        plt.show = lambda: None        
        plot_fiducials(s, fp, legend_fontsize=12)
        fig = plt.gcf()
        fig.suptitle(f'pyPPG Fiducials Analysis: {filename}', fontsize=16)
        
    finally:
        plt.show = original_show
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    peak_count = 0
    return fig, peak_count, ppg_sqi

# 홈페이지
@app.route('/')
def homepage():
    return render_template('home.html')

# 업로드
@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('files')
    if not files or files[0].filename == '': return '<h3>파일이 선택되지 않았습니다.</h3><a href="/">돌아가기</a>'
    file_list = []
    for file in files:
        filename = file.filename
        if filename:
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            file.save(save_path)
            file_list.append(filename)
    session['uploaded_files'] = file_list
    return redirect(url_for('file_list'))

# 파일 목록
@app.route('/file_list')
def file_list():
    # 샘플링 레이트
    current_sr = request.args.get('sampling_rate', DEFAULT_SAMPLING_RATE, type=int)
    
    # 파일 리스트
    file_list = session.get('uploaded_files', [])
    return render_template('file_list.html', file_list=file_list, sampling_rate=current_sr)

# 그래프
@app.route('/graph/<path:filepath>')
def graph(filepath):
    
    # 라이브러리
    lib_choice = request.args.get('lib', 'biosppy')
    
    # 샘플링 레이트
    try: sampling_rate = request.args.get('sampling_rate', DEFAULT_SAMPLING_RATE, type=int)
    except (TypeError, ValueError): sampling_rate = DEFAULT_SAMPLING_RATE
    
    # 경로 지정    
    base_dir = os.path.abspath(app.config['UPLOAD_FOLDER'])
    full_path = os.path.abspath(os.path.join(base_dir, filepath))
    if not full_path.startswith(base_dir): abort(404)
    
    # 세팅   
    try:
        # 데이터 가공
        df = pd.read_csv(full_path, on_bad_lines='skip')
        signal = df.iloc[1:, 2].to_numpy().flatten()
        if signal.size < 50 or np.std(signal) < 1e-6:
            return "오류: CSV 파일의 데이터가 너무 짧거나, 신호에 변화가 없습니다."
        
        # 세그먼트 길이 설정
        max_length = len(signal)
        if max_length == 0:
            return "오류: CSV 파일에서 신호 데이터를 찾을 수 없습니다."

        # 세그먼트 지정(디폴트 = 1 ~ 끝)
        try:
            start_idx = int(request.args.get('start', 1))
        except (ValueError, TypeError):
            start_idx = 1

        try:
            end_idx_str = request.args.get('end')
            if end_idx_str is None or end_idx_str.strip() == '':
                end_idx = max_length
            else:
                end_idx = int(end_idx_str)
        except (ValueError, TypeError):
            end_idx = max_length
            
        start_idx = max(1, start_idx)
        end_idx = min(max_length, end_idx)

        if start_idx > end_idx: return f"오류: 시작점({start_idx})은 종료점({end_idx})보다 클 수 없습니다."

        signal = signal[start_idx - 1 : end_idx]
        
        if signal.size < 50 or np.std(signal) < 1e-6: return "오류: 선택된 범위의 데이터가 너무 짧거나, 신호에 변화가 없습니다."
        
        # 라이브러리 설정 (디폴트 "biosppy")
        if lib_choice == 'pyppg':
            fig, peak_count, ppg_sqi = generate_pyppg_plot(signal, sampling_rate, os.path.basename(filepath), app.config['UPLOAD_FOLDER'])
        else:
            fig, peak_count, ppg_sqi = generate_biosppy_plot(signal, sampling_rate, os.path.basename(filepath))
            
    except Exception as e: return f"파일 처리 중 오류가 발생했습니다. (오류: {e})"
    
    img = io.BytesIO()
    fig.savefig(img, format='png'); img.seek(0); plt.close(fig)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    return render_template(
            'graph.html', 
            filename=filepath, 
            plot_url=plot_url, 
            peak_count=peak_count, 
            ppg_sqi=ppg_sqi, 
            lib_choice=lib_choice, 
            sampling_rate=sampling_rate,
            start_idx=start_idx,
            end_idx=end_idx,
            max_length=max_length
        )


if __name__ == '__main__':
    app.run(debug=True)