import os
import random
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

speech_dir = "/Users/nitish_0184/Desktop/audio_anomalies/clips/speech"
music_dir = "/Users/nitish_0184/Desktop/audio_anomalies/clips/music"
audio_output_folder = "/Users/nitish_0184/Desktop/audio_anomalies/transformed/frequency_distortion_2d"
image_output_folder = "/Users/nitish_0184/Desktop/audio_anomalies/images/frequency_distortion_2d"

os.makedirs(audio_output_folder, exist_ok=True)
os.makedirs(image_output_folder, exist_ok=True)

def get_files(folder, limit):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.wav', '.mp3'))]
    return sorted(files)[:limit]

speech_files = get_files(speech_dir, 27000)
music_files = get_files(music_dir, 3000)
all_files = speech_files + music_files

def calculate_overlap(box1, box2):
    t_min_int = max(box1[0], box2[0])
    t_max_int = min(box1[1], box2[1])
    f_min_int = max(box1[2], box2[2])
    f_max_int = min(box1[3], box2[3])

    if t_min_int < t_max_int and f_min_int < f_max_int:
        intersection_area = (t_max_int - t_min_int) * (f_max_int - f_min_int)
        box2_area = (box2[1] - box2[0]) * (box2[3] - box2[2])
        return intersection_area / box2_area if box2_area > 0 else 0
    return 0

for idx, file_path in enumerate(all_files):
    filename = os.path.basename(file_path)
    y, sr = librosa.load(file_path, sr=None)
    
    D = librosa.stft(y)
    mag, phase = librosa.magphase(D)
    freqs = librosa.fft_frequencies(sr=sr)
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr)
    
    total_area_ref = 1250 * 8000
    boxes = []
    
    for attempt in range(50):
        f1_hz = random.uniform(200, 8000)
        f2_hz = random.uniform(200, 8000)
        t1_ms = random.uniform(0, 1250)
        t2_ms = random.uniform(0, 1250)
        
        f_min, f_max = min(f1_hz, f2_hz), max(f1_hz, f2_hz)
        t_min, t_max = min(t1_ms, t2_ms), max(t1_ms, t2_ms)
        
        if (f_max - f_min) < 400 or (t_max - t_min) < 35:
            continue
            
        area = (t_max - t_min) * (f_max - f_min)
        area_pct = (area / total_area_ref) * 100
        
        if area_pct <= 8:
            target_boxes = 3
        elif 8 < area_pct <= 16:
            target_boxes = 2
        else:
            target_boxes = 1
            
        new_box = (t_min, t_max, f_min, f_max)
        
        overlap_ok = True
        for existing_box in boxes:
            if calculate_overlap(existing_box, new_box) > 0.30:
                overlap_ok = False
                break
                
        if overlap_ok:
            boxes.append(new_box)
            
        if len(boxes) >= target_boxes:
            break

    modified_mag = mag.copy()
    effect_type = idx % 4
    
    for box in boxes:
        t_min, t_max, f_min, f_max = box
        
        t_min_idx = np.abs(times - (t_min / 1000.0)).argmin()
        t_max_idx = np.abs(times - (t_max / 1000.0)).argmin()
        f_min_idx = np.abs(freqs - f_min).argmin()
        f_max_idx = np.abs(freqs - f_max).argmin()
        
        box_mag = modified_mag[f_min_idx:f_max_idx, t_min_idx:t_max_idx]
        
        if box_mag.size == 0 or box_mag.shape[0] < 4:
            continue

        if effect_type == 0:
            pitch_shift_factor = random.uniform(0.25, 3.0)
            if pitch_shift_factor > 1.0:
                shift_bins = int((pitch_shift_factor - 1.0) * 10)
                shifted = np.roll(box_mag, shift_bins, axis=0)
                shifted[:shift_bins, :] = 0
            else:
                shift_bins = int((1.0 - pitch_shift_factor) * 10)
                shifted = np.roll(box_mag, -shift_bins, axis=0)
                shifted[-shift_bins:, :] = 0
            modified_mag[f_min_idx:f_max_idx, t_min_idx:t_max_idx] = shifted

        elif effect_type == 1:
            cutoff = int(box_mag.shape[0] * random.uniform(0.3, 0.7))
            box_mag[-cutoff:, :] = 0
            modified_mag[f_min_idx:f_max_idx, t_min_idx:t_max_idx] = box_mag

        elif effect_type == 2:
            midpoint = box_mag.shape[0] // 2
            top_half = box_mag[midpoint:, :]
            flipped_top = np.flipud(top_half)
            
            fold_size = min(flipped_top.shape[0], midpoint)
            box_mag[:fold_size, :] += flipped_top[:fold_size, :]
            box_mag[midpoint:, :] = 0
            modified_mag[f_min_idx:f_max_idx, t_min_idx:t_max_idx] = box_mag

        elif effect_type == 3:
            sheared_box = np.zeros_like(box_mag)
            max_shift = int(box_mag.shape[1] * 0.5)
            for i in range(box_mag.shape[0]):
                shift = int(np.sin(i / box_mag.shape[0] * np.pi) * max_shift)
                sheared_box[i, :] = np.roll(box_mag[i, :], shift)
            modified_mag[f_min_idx:f_max_idx, t_min_idx:t_max_idx] = sheared_box

    D_modified = modified_mag * phase
    y_anomalous = librosa.istft(D_modified)
    
    if np.max(np.abs(y_anomalous)) > 0:
        y_anomalous = y_anomalous / np.max(np.abs(y_anomalous))
        
    out_audio_path = os.path.join(audio_output_folder, f"{idx}_{filename}")
    sf.write(out_audio_path, y_anomalous, sr)

    plt.figure(figsize=(10, 6))
    S = librosa.feature.melspectrogram(y=y_anomalous, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'2D Frequency Distortion (Type {effect_type}): {filename}')
    
    ax = plt.gca()
    for box in boxes:
        t_min, t_max, f_min, f_max = box
        rect = patches.Rectangle(
            (t_min / 1000.0, f_min), 
            (t_max - t_min) / 1000.0, 
            (f_max - f_min), 
            linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)
        
    out_img_path = os.path.join(image_output_folder, f"{idx}_{filename.replace('.wav', '.png').replace('.mp3', '.png')}")
    plt.savefig(out_img_path, bbox_inches='tight')
    plt.close()

    #############################################################

    import os
import random
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

speech_dir = "/Users/nitish_0184/Desktop/audio_anomalies/clips/speech"
music_dir = "/Users/nitish_0184/Desktop/audio_anomalies/clips/music"
audio_output_folder = "/Users/nitish_0184/Desktop/audio_anomalies/transformed/hardware_noise_2d"
image_output_folder = "/Users/nitish_0184/Desktop/audio_anomalies/images/hardware_noise_2d"

os.makedirs(audio_output_folder, exist_ok=True)
os.makedirs(image_output_folder, exist_ok=True)

def get_files(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.wav', '.mp3'))]
    return sorted(files)

all_speech = get_files(speech_dir)
all_music = get_files(music_dir)

speech_files = all_speech[27000:54000]
music_files = all_music[3000:6000]
all_files = speech_files + music_files

def calculate_overlap(box1, box2):
    t_min_int = max(box1[0], box2[0])
    t_max_int = min(box1[1], box2[1])
    f_min_int = max(box1[2], box2[2])
    f_max_int = min(box1[3], box2[3])

    if t_min_int < t_max_int and f_min_int < f_max_int:
        intersection_area = (t_max_int - t_min_int) * (f_max_int - f_min_int)
        box2_area = (box2[1] - box2[0]) * (box2[3] - box2[2])
        return intersection_area / box2_area if box2_area > 0 else 0
    return 0

noise_classes = ['white', 'gaussian', 'pink', 'brownian', 'clicks', 'pops', 'hum', 'jitter']

for idx, file_path in enumerate(all_files):
    filename = os.path.basename(file_path)
    y, sr = librosa.load(file_path, sr=None)
    
    D = librosa.stft(y)
    mag, phase = librosa.magphase(D)
    freqs = librosa.fft_frequencies(sr=sr)
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr)
    
    total_area_ref = 1250 * 8000
    boxes = []
    
    for attempt in range(50):
        f1_hz = random.uniform(200, 8000)
        f2_hz = random.uniform(200, 8000)
        t1_ms = random.uniform(0, 1250)
        t2_ms = random.uniform(0, 1250)
        
        f_min, f_max = min(f1_hz, f2_hz), max(f1_hz, f2_hz)
        t_min, t_max = min(t1_ms, t2_ms), max(t1_ms, t2_ms)
        
        if (f_max - f_min) < 400 or (t_max - t_min) < 35:
            continue
            
        area = (t_max - t_min) * (f_max - f_min)
        area_pct = (area / total_area_ref) * 100
        
        if area_pct <= 8:
            target_boxes = 3
        elif 8 < area_pct <= 16:
            target_boxes = 2
        else:
            target_boxes = 1
            
        new_box = (t_min, t_max, f_min, f_max)
        
        overlap_ok = True
        for existing_box in boxes:
            if calculate_overlap(existing_box, new_box) > 0.30:
                overlap_ok = False
                break
                
        if overlap_ok:
            boxes.append(new_box)
            
        if len(boxes) >= target_boxes:
            break

    modified_mag = mag.copy()
    noise_type = noise_classes[idx % 8]
    
    for box in boxes:
        t_min, t_max, f_min, f_max = box
        
        t_min_idx = np.abs(times - (t_min / 1000.0)).argmin()
        t_max_idx = np.abs(times - (t_max / 1000.0)).argmin()
        f_min_idx = np.abs(freqs - f_min).argmin()
        f_max_idx = np.abs(freqs - f_max).argmin()
        
        box_mag = modified_mag[f_min_idx:f_max_idx, t_min_idx:t_max_idx]
        
        if box_mag.size == 0 or box_mag.shape[0] == 0:
            continue
            
        signal_power = np.mean(box_mag ** 2)
        if signal_power == 0:
            signal_power = 1e-10
            
        snr_db = random.uniform(15, 40)
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        raw_noise = np.zeros_like(box_mag)
        
        if noise_type in ['white', 'gaussian']:
            raw_noise = np.random.normal(0, 1, size=box_mag.shape)
            
        elif noise_type == 'pink':
            base_noise = np.random.normal(0, 1, size=box_mag.shape)
            box_freqs = freqs[f_min_idx:f_max_idx]
            pink_filter = 1.0 / np.sqrt(np.maximum(box_freqs, 1.0))
            raw_noise = base_noise * pink_filter[:, np.newaxis]
            
        elif noise_type == 'brownian':
            base_noise = np.random.normal(0, 1, size=box_mag.shape)
            box_freqs = freqs[f_min_idx:f_max_idx]
            brown_filter = 1.0 / np.maximum(box_freqs, 1.0)
            raw_noise = base_noise * brown_filter[:, np.newaxis]
            
        elif noise_type == 'clicks':
            num_clicks = max(1, int(box_mag.shape[1] * 0.05))
            click_cols = random.sample(range(box_mag.shape[1]), num_clicks)
            for c in click_cols:
                raw_noise[:, c] = np.random.normal(0, 1, size=box_mag.shape[0])
                
        elif noise_type == 'pops':
            num_pops = max(1, int(box_mag.shape[1] * 0.02))
            pop_cols = random.sample(range(box_mag.shape[1]), num_pops)
            pop_decay = np.linspace(1.0, 0.0, box_mag.shape[0]) ** 2
            for c in pop_cols:
                raw_noise[:, c] = np.random.normal(0, 1, size=box_mag.shape[0]) * pop_decay
                
        elif noise_type == 'hum':
            num_tones = random.randint(1, 3)
            for _ in range(num_tones):
                row = random.randint(0, box_mag.shape[0] - 1)
                raw_noise[row, :] = 1.0
                
        elif noise_type == 'jitter':
            for r in range(box_mag.shape[0]):
                shift = random.choice([-1, 0, 1])
                raw_noise[r, :] = np.roll(box_mag[r, :], shift) - box_mag[r, :]

        raw_power = np.mean(raw_noise ** 2)
        if raw_power > 0:
            noise_matrix = raw_noise * np.sqrt(noise_power / raw_power)
        else:
            noise_matrix = np.zeros_like(box_mag)

        modified_mag[f_min_idx:f_max_idx, t_min_idx:t_max_idx] += np.abs(noise_matrix)

    D_modified = modified_mag * phase
    y_anomalous = librosa.istft(D_modified)
    
    if np.max(np.abs(y_anomalous)) > 0:
        y_anomalous = y_anomalous / np.max(np.abs(y_anomalous))
        
    out_audio_path = os.path.join(audio_output_folder, f"{idx}_{noise_type}_{filename}")
    sf.write(out_audio_path, y_anomalous, sr)

    plt.figure(figsize=(10, 6))
    S = librosa.feature.melspectrogram(y=y_anomalous, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f decibels')
    plt.title(f'2D Hardware Noise ({noise_type}): {filename}')
    
    ax = plt.gca()
    for box in boxes:
        t_min, t_max, f_min, f_max = box
        rect = patches.Rectangle(
            (t_min / 1000.0, f_min), 
            (t_max - t_min) / 1000.0, 
            (f_max - f_min), 
            linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)
        
    out_img_path = os.path.join(image_output_folder, f"{idx}_{noise_type}_{filename.replace('.wav', '.png').replace('.mp3', '.png')}")
    plt.savefig(out_img_path, bbox_inches='tight')
    plt.close()

    #####################################


    import os
import random
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

speech_dir = "/Users/nitish_0184/Desktop/audio_anomalies/clips/speech"
music_dir = "/Users/nitish_0184/Desktop/audio_anomalies/clips/music"
audio_output_folder = "/Users/nitish_0184/Desktop/audio_anomalies/transformed/packet_drop"
image_output_folder = "/Users/nitish_0184/Desktop/audio_anomalies/images/packet_drop"

os.makedirs(audio_output_folder, exist_ok=True)
os.makedirs(image_output_folder, exist_ok=True)

def get_files(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.wav', '.mp3'))]
    return sorted(files)

all_speech = get_files(speech_dir)
all_music = get_files(music_dir)

speech_files = all_speech[54000:81000]
music_files = all_music[6000:9000]
all_files = speech_files + music_files

def check_1d_overlap(new_box, existing_boxes):
    t_min, t_max = new_box
    for ex_box in existing_boxes:
        ex_t_min, ex_t_max = ex_box
        if max(t_min, ex_t_min) < min(t_max, ex_t_max):
            return True
    return False

for idx, file_path in enumerate(all_files):
    filename = os.path.basename(file_path)
    y, sr = librosa.load(file_path, sr=None)
    
    total_duration_ms = (len(y) / sr) * 1000
    
    category = random.choice([1, 2, 3])
    
    if category == 1:
        num_anomalies = random.randint(1, 5)
        dt_range = (33, 75)
    elif category == 2:
        num_anomalies = random.randint(1, 4)
        dt_range = (75, 125)
    else:
        num_anomalies = 1
        dt_range = (126, min(500, total_duration_ms / 2))
        
    regions = []
    
    for attempt in range(50):
        dt = random.uniform(dt_range[0], dt_range[1])
        t1 = random.uniform(0, max(0.1, total_duration_ms - dt))
        t2 = t1 + dt
        
        if not check_1d_overlap((t1, t2), regions):
            regions.append((t1, t2))
            
        if len(regions) >= num_anomalies:
            break

    y_modified = y.copy()
    
    for t1, t2 in regions:
        start_sample = int((t1 / 1000) * sr)
        end_sample = int((t2 / 1000) * sr)
        
        y_modified[start_sample:end_sample] = 0.0

    out_audio_path = os.path.join(audio_output_folder, f"{idx}_packetdrop_{filename}")
    sf.write(out_audio_path, y_modified, sr)

    plt.figure(figsize=(10, 6))
    S = librosa.feature.melspectrogram(y=y_modified, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f decibels')
    plt.title(f'Packet Drop Injection: {filename}')
    
    ax = plt.gca()
    for t1, t2 in regions:
        rect = patches.Rectangle(
            (t1 / 1000.0, 0), 
            (t2 - t1) / 1000.0, 
            8000, 
            linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)
        
    out_img_path = os.path.join(image_output_folder, f"{idx}_packetdrop_{filename.replace('.wav', '.png').replace('.mp3', '.png')}")
    plt.savefig(out_img_path, bbox_inches='tight')
    plt.close()


    ####################################
    import os
import random
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

speech_dir = "/Users/nitish_0184/Desktop/audio_anomalies/clips/speech"
music_dir = "/Users/nitish_0184/Desktop/audio_anomalies/clips/music"
audio_output_folder = "/Users/nitish_0184/Desktop/audio_anomalies/transformed/db_jump_2d"
image_output_folder = "/Users/nitish_0184/Desktop/audio_anomalies/images/db_jump_2d"

os.makedirs(audio_output_folder, exist_ok=True)
os.makedirs(image_output_folder, exist_ok=True)

def get_files(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.wav', '.mp3'))]
    return sorted(files)

all_speech = get_files(speech_dir)
all_music = get_files(music_dir)

speech_files = all_speech[81000:108000]
music_files = all_music[9000:12000]
all_files = speech_files + music_files

def calculate_overlap(box1, box2):
    t_min_int = max(box1[0], box2[0])
    t_max_int = min(box1[1], box2[1])
    f_min_int = max(box1[2], box2[2])
    f_max_int = min(box1[3], box2[3])

    if t_min_int < t_max_int and f_min_int < f_max_int:
        intersection_area = (t_max_int - t_min_int) * (f_max_int - f_min_int)
        box2_area = (box2[1] - box2[0]) * (box2[3] - box2[2])
        return intersection_area / box2_area if box2_area > 0 else 0
    return 0

jump_classes = ['high_spike', 'low_drop', 'swell', 'flutter']

for idx, file_path in enumerate(all_files):
    filename = os.path.basename(file_path)
    y, sr = librosa.load(file_path, sr=None)
    
    D = librosa.stft(y)
    mag, phase = librosa.magphase(D)
    freqs = librosa.fft_frequencies(sr=sr)
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr)
    
    total_area_ref = 1250 * 8000
    boxes = []
    
    for attempt in range(50):
        f1_hz = random.uniform(200, 8000)
        f2_hz = random.uniform(200, 8000)
        t1_ms = random.uniform(0, 1250)
        t2_ms = random.uniform(0, 1250)
        
        f_min, f_max = min(f1_hz, f2_hz), max(f1_hz, f2_hz)
        t_min, t_max = min(t1_ms, t2_ms), max(t1_ms, t2_ms)
        
        if (f_max - f_min) < 400 or (t_max - t_min) < 35:
            continue
            
        area = (t_max - t_min) * (f_max - f_min)
        area_pct = (area / total_area_ref) * 100
        
        if area_pct <= 8:
            target_boxes = 3
        elif 8 < area_pct <= 16:
            target_boxes = 2
        else:
            target_boxes = 1
            
        new_box = (t_min, t_max, f_min, f_max)
        
        overlap_ok = True
        for existing_box in boxes:
            if calculate_overlap(existing_box, new_box) > 0.30:
                overlap_ok = False
                break
                
        if overlap_ok:
            boxes.append(new_box)
            
        if len(boxes) >= target_boxes:
            break

    modified_mag = mag.copy()
    jump_type = jump_classes[idx % 4]
    
    for box in boxes:
        t_min, t_max, f_min, f_max = box
        
        t_min_idx = np.abs(times - (t_min / 1000.0)).argmin()
        t_max_idx = np.abs(times - (t_max / 1000.0)).argmin()
        f_min_idx = np.abs(freqs - f_min).argmin()
        f_max_idx = np.abs(freqs - f_max).argmin()
        
        box_mag = modified_mag[f_min_idx:f_max_idx, t_min_idx:t_max_idx]
        
        if box_mag.size == 0 or box_mag.shape[0] == 0 or box_mag.shape[1] == 0:
            continue
            
        num_cols = box_mag.shape[1]
        
        if jump_type == 'high_spike':
            db_shift = random.uniform(15, 35)
            linear_gain = 10 ** (db_shift / 20)
            modified_mag[f_min_idx:f_max_idx, t_min_idx:t_max_idx] = box_mag * linear_gain
            
        elif jump_type == 'low_drop':
            db_shift = random.uniform(-35, -15)
            linear_gain = 10 ** (db_shift / 20)
            modified_mag[f_min_idx:f_max_idx, t_min_idx:t_max_idx] = box_mag * linear_gain
            
        elif jump_type == 'swell':
            x = np.linspace(0, np.pi, num_cols)
            swell_curve = np.sin(x)
            db_shift = random.uniform(10, 25)
            linear_gains = 10 ** ((swell_curve * db_shift) / 20)
            modified_mag[f_min_idx:f_max_idx, t_min_idx:t_max_idx] = box_mag * linear_gains[np.newaxis, :]
            
        elif jump_type == 'flutter':
            cycles = random.uniform(3, 10)
            x = np.linspace(0, 2 * np.pi * cycles, num_cols)
            flutter_curve = (np.sin(x) + 1) / 2 
            db_shift = random.uniform(10, 20)
            linear_gains = 10 ** ((flutter_curve * db_shift) / 20)
            modified_mag[f_min_idx:f_max_idx, t_min_idx:t_max_idx] = box_mag * linear_gains[np.newaxis, :]

    D_modified = modified_mag * phase
    y_anomalous = librosa.istft(D_modified)
    
    if np.max(np.abs(y_anomalous)) > 0:
        y_anomalous = y_anomalous / np.max(np.abs(y_anomalous))
        
    out_audio_path = os.path.join(audio_output_folder, f"{idx}_{jump_type}_{filename}")
    sf.write(out_audio_path, y_anomalous, sr)

    plt.figure(figsize=(10, 6))
    S = librosa.feature.melspectrogram(y=y_anomalous, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f decibels')
    plt.title(f'2D Decibel Jump ({jump_type}): {filename}')
    
    ax = plt.gca()
    for box in boxes:
        t_min, t_max, f_min, f_max = box
        rect = patches.Rectangle(
            (t_min / 1000.0, f_min), 
            (t_max - t_min) / 1000.0, 
            (f_max - f_min), 
            linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)
        
    out_img_path = os.path.join(image_output_folder, f"{idx}_{jump_type}_{filename.replace('.wav', '.png').replace('.mp3', '.png')}")
    plt.savefig(out_img_path, bbox_inches='tight')
    plt.close()
    ######################################


    import os
import random
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d

speech_dir = "/Users/nitish_0184/Desktop/audio_anomalies/clips/speech"
music_dir = "/Users/nitish_0184/Desktop/audio_anomalies/clips/music"
audio_output_folder = "/Users/nitish_0184/Desktop/audio_anomalies/transformed/time_stretch_2d"
image_output_folder = "/Users/nitish_0184/Desktop/audio_anomalies/images/time_stretch_2d"

os.makedirs(audio_output_folder, exist_ok=True)
os.makedirs(image_output_folder, exist_ok=True)

def get_files(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.wav', '.mp3'))]
    return sorted(files)

all_speech = get_files(speech_dir)
all_music = get_files(music_dir)

speech_files = all_speech[108000:135000]
music_files = all_music[12000:15000]
all_files = speech_files + music_files

def calculate_overlap(box1, box2):
    t_min_int = max(box1[0], box2[0])
    t_max_int = min(box1[1], box2[1])
    f_min_int = max(box1[2], box2[2])
    f_max_int = min(box1[3], box2[3])

    if t_min_int < t_max_int and f_min_int < f_max_int:
        intersection_area = (t_max_int - t_min_int) * (f_max_int - f_min_int)
        box2_area = (box2[1] - box2[0]) * (box2[3] - box2[2])
        return intersection_area / box2_area if box2_area > 0 else 0
    return 0

for idx, file_path in enumerate(all_files):
    filename = os.path.basename(file_path)
    y, sr = librosa.load(file_path, sr=None)
    
    D = librosa.stft(y)
    mag, phase = librosa.magphase(D)
    freqs = librosa.fft_frequencies(sr=sr)
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr)
    
    total_area_ref = 1250 * 8000
    boxes = []
    
    for attempt in range(50):
        f1_hz = random.uniform(200, 8000)
        f2_hz = random.uniform(200, 8000)
        t1_ms = random.uniform(0, 1250)
        t2_ms = random.uniform(0, 1250)
        
        f_min, f_max = min(f1_hz, f2_hz), max(f1_hz, f2_hz)
        t_min, t_max = min(t1_ms, t2_ms), max(t1_ms, t2_ms)
        
        if (f_max - f_min) < 400 or (t_max - t_min) < 35:
            continue
            
        area = (t_max - t_min) * (f_max - f_min)
        area_pct = (area / total_area_ref) * 100
        
        if area_pct <= 8:
            target_boxes = 3
        elif 8 < area_pct <= 16:
            target_boxes = 2
        else:
            target_boxes = 1
            
        new_box = (t_min, t_max, f_min, f_max)
        
        overlap_ok = True
        for existing_box in boxes:
            if calculate_overlap(existing_box, new_box) > 0.30:
                overlap_ok = False
                break
                
        if overlap_ok:
            boxes.append(new_box)
            
        if len(boxes) >= target_boxes:
            break

    modified_mag = mag.copy()
    
    for box in boxes:
        t_min, t_max, f_min, f_max = box
        
        t_min_idx = np.abs(times - (t_min / 1000.0)).argmin()
        t_max_idx = np.abs(times - (t_max / 1000.0)).argmin()
        f_min_idx = np.abs(freqs - f_min).argmin()
        f_max_idx = np.abs(freqs - f_max).argmin()
        
        box_mag = modified_mag[f_min_idx:f_max_idx, t_min_idx:t_max_idx]
        
        if box_mag.size == 0 or box_mag.shape[0] < 2 or box_mag.shape[1] < 4:
            continue
            
        num_freqs, num_frames = box_mag.shape
        rate = random.uniform(0.5, 2.0)
        new_frames = max(2, int(num_frames / rate))
        
        x_old = np.linspace(0, 1, num_frames)
        f_interp = interp1d(x_old, box_mag, axis=1, kind='linear', fill_value="extrapolate")
        
        x_new = np.linspace(0, 1, new_frames)
        stretched_mag = f_interp(x_new)
        
        if new_frames >= num_frames:
            modified_mag[f_min_idx:f_max_idx, t_min_idx:t_max_idx] = stretched_mag[:, :num_frames]
        else:
            padded_mag = np.zeros_like(box_mag)
            padded_mag[:, :new_frames] = stretched_mag
            modified_mag[f_min_idx:f_max_idx, t_min_idx:t_max_idx] = padded_mag

    D_modified = modified_mag * phase
    y_anomalous = librosa.istft(D_modified)
    
    if np.max(np.abs(y_anomalous)) > 0:
        y_anomalous = y_anomalous / np.max(np.abs(y_anomalous))
        
    out_audio_path = os.path.join(audio_output_folder, f"{idx}_timestretch_{filename}")
    sf.write(out_audio_path, y_anomalous, sr)

    plt.figure(figsize=(10, 6))
    S = librosa.feature.melspectrogram(y=y_anomalous, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f decibels')
    plt.title(f'2D Time Stretch / Compression: {filename}')
    
    ax = plt.gca()
    for box in boxes:
        t_min, t_max, f_min, f_max = box
        rect = patches.Rectangle(
            (t_min / 1000.0, f_min), 
            (t_max - t_min) / 1000.0, 
            (f_max - f_min), 
            linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)
        
    out_img_path = os.path.join(image_output_folder, f"{idx}_timestretch_{filename.replace('.wav', '.png').replace('.mp3', '.png')}")
    plt.savefig(out_img_path, bbox_inches='tight')
    plt.close()



    ##################################